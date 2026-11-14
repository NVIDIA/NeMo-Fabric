#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock OpenClaw CLI and Gateway used by adapter tests."""

import json
import os
import signal
import sys
import time
import urllib.request
from pathlib import Path
from uuid import uuid4
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


args = sys.argv[1:]
if args[:1] != ["gateway"] and (delay := os.environ.get("FAKE_OPENCLAW_COMMAND_DELAY")):
    time.sleep(float(delay))
if args == ["--version"]:
    print("OpenClaw 2026.9.34 (fabric test)")
    raise SystemExit(0)
if args[:3] == ["config", "validate", "--json"]:
    with open(os.environ["OPENCLAW_CONFIG_PATH"], encoding="utf-8") as stream:
        value = json.load(stream)
    with open(os.environ["FAKE_OPENCLAW_CAPTURE"], "w", encoding="utf-8") as stream:
        json.dump(value, stream)
    print(json.dumps({"valid": True}))
    raise SystemExit(0)
if args[:2] == ["gateway", "call"]:
    config = json.loads(Path(os.environ["OPENCLAW_CONFIG_PATH"]).read_text())
    method = args[2]
    params = json.loads(args[args.index("--params") + 1])
    request = urllib.request.Request(
        f"http://127.0.0.1:{config['gateway']['port']}/_rpc/{method}",
        data=json.dumps(params).encode(),
        headers={
            "Authorization": "Bearer " + os.environ["OPENCLAW_GATEWAY_TOKEN"],
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request) as response:
        print(response.read().decode())
    raise SystemExit(0)
if args[:2] != ["gateway", "run"]:
    raise SystemExit(2)

port = int(args[args.index("--port") + 1])
token = os.environ["OPENCLAW_GATEWAY_TOKEN"]
state_file = Path(os.environ["OPENCLAW_STATE_DIR"]) / "mock-native-sessions.json"
native_sessions = json.loads(state_file.read_text()) if state_file.exists() else {}


def save_sessions():
    state_file.write_text(json.dumps(native_sessions))


def session_rpc(method, params):
    if method == "sessions.create":
        key = params["key"].lower()
        if key in native_sessions:
            raise ValueError("session already exists")
        entry = {
            "key": key,
            "sessionId": str(uuid4()),
            "messages": [],
            "abortedLastRun": False,
        }
        native_sessions[key] = entry
        save_sessions()
        return {
            "ok": True,
            "key": key,
            "sessionId": entry["sessionId"],
            "entry": entry,
            "runStarted": False,
        }
    if method == "sessions.resolve":
        matches = [
            entry
            for entry in native_sessions.values()
            if entry["sessionId"] == params["sessionId"]
        ]
        if not matches:
            if params.get("allowMissing") is True:
                return {"ok": False}
            raise ValueError("No matching native session")
        return {"ok": True, "key": matches[0]["key"], "agentId": params["agentId"]}
    entry = native_sessions.get(params.get("key", params.get("sessionKey")))
    if method == "chat.history":
        return {
            "sessionKey": entry["key"] if entry else None,
            "sessionId": entry["sessionId"] if entry else None,
            "totalMessages": len(entry["messages"]) if entry else 0,
        }
    if method == "sessions.describe":
        return {"session": entry}
    if method == "sessions.get":
        return {
            "messages": entry["messages"][-params.get("limit", 200) :] if entry else []
        }
    raise ValueError("unknown method")


if readonly_path := os.environ.get("FAKE_OPENCLAW_CONFIG_READONLY"):
    with open(readonly_path, "w", encoding="utf-8") as stream:
        stream.write(os.environ.get("OPENCLAW_CONFIG_READONLY", ""))
if pid_path := os.environ.get("FAKE_OPENCLAW_PID"):
    with open(pid_path, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))


def stop_gateway(*_unused):
    if delay := os.environ.get("FAKE_OPENCLAW_SHUTDOWN_DELAY"):
        time.sleep(float(delay))
    if stopped_path := os.environ.get("FAKE_OPENCLAW_STOPPED"):
        with open(stopped_path, "w", encoding="utf-8") as stream:
            stream.write("stopped")
    raise SystemExit(0)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *unused):
        pass

    def do_GET(self):
        if self.path == "/v1/chat/completions":
            self.send_response(405)
            self.send_header("Allow", "POST")
            self.end_headers()
            return
        if self.path == "/fabric/shared-config/health":
            if self.headers.get("Authorization") != f"Bearer {token}":
                self.send_response(401)
                self.end_headers()
                return
            mode = os.environ.get("FAKE_OPENCLAW_SHARED_CONFIG", "ready")
            if mode == "missing":
                self.send_response(404)
                self.end_headers()
                return
            with open(os.environ["OPENCLAW_CONFIG_PATH"], encoding="utf-8") as stream:
                generated = json.load(stream)
            agent_id = next(iter(generated["agents"]["entries"]))
            body = json.dumps(
                {
                    "ready": mode != "inactive",
                    "agent_id": agent_id.lower() if mode != "wrong-agent" else "other",
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/healthz":
            body = json.dumps({"ok": True, "status": "live"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/readyz":
            ready = True
            if state_path := os.environ.get("FAKE_OPENCLAW_HEALTH_STATE"):
                with open(state_path, encoding="utf-8") as stream:
                    ready = json.load(stream)["ready"]
            body = json.dumps(
                {"ready": ready, "failing": [] if ready else ["telegram"]}
            ).encode()
            self.send_response(200 if ready else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/startupz":
            body = json.dumps(
                {"ok": True, "status": "started", "version": "2026.9.34"}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/v1/models":
            self.send_response(404)
            self.end_headers()
            return
        if self.headers.get("Authorization") != f"Bearer {token}":
            self.send_response(401)
            self.end_headers()
            return
        body = json.dumps(
            {
                "object": "list",
                "data": [
                    {"id": "openclaw/default", "object": "model"},
                    {"id": "openclaw/reviewer", "object": "model"},
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.startswith("/_rpc/"):
            if self.headers.get("Authorization") != f"Bearer {token}":
                self.send_response(401)
                self.end_headers()
                return
            params = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                result = session_rpc(self.path.removeprefix("/_rpc/"), params)
            except ValueError:
                self.send_response(400)
                self.end_headers()
                return
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        if self.headers.get("Authorization") != f"Bearer {token}":
            self.send_response(401)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        with open(os.environ["FAKE_OPENCLAW_REQUEST"], "w", encoding="utf-8") as stream:
            json.dump(request, stream)
        if requests_path := os.environ.get("FAKE_OPENCLAW_REQUESTS"):
            with open(requests_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(request) + "\n")
        response_text = "OpenClaw response"
        if key := self.headers.get("x-openclaw-session-key"):
            entry = native_sessions[key]
            entry["messages"].append(
                {"role": "user", "content": request["messages"][-1]["content"]}
            )
            if os.environ.get("FAKE_OPENCLAW_MEMORY_REPLY"):
                response_text = json.dumps(entry["messages"])
            entry["messages"].append(
                {"role": "assistant", "content": response_text, "stopReason": "stop"}
            )
            save_sessions()
        chunks = [
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": response_text},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        ]
        body = (
            "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            + "data: [DONE]\n\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
signal.signal(signal.SIGTERM, stop_gateway)
server.serve_forever()
