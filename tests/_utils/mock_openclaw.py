#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock OpenClaw CLI and Gateway used by adapter tests."""

import json
import os
import signal
import sys
import threading
import time
import urllib.request
from pathlib import Path
from socketserver import TCPServer
from uuid import uuid4
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from websockets.frames import Opcode
from websockets.exceptions import InvalidState
from websockets.server import ServerProtocol


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
    if issues := os.environ.get("FAKE_OPENCLAW_CONFIG_ISSUES"):
        print(json.dumps({"ok": False, "valid": False, "issues": json.loads(issues)}))
        print("noise on standard error", file=sys.stderr)
        raise SystemExit(1)
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
# The real Gateway broadcasts agent and chat events to every operator connection.
event_streams = set()


def save_sessions():
    state_file.write_text(json.dumps(native_sessions))


def plugins_rpc():
    # Each entry is "active" when enabled and "unloaded" otherwise.
    # FAKE_OPENCLAW_PLUGINS maps an id to a runtime state ("absent" leaves it
    # out, "catalog" lists it as not installed); FAKE_OPENCLAW_PLUGIN_ERRORS
    # maps an id to a load error. FAKE_OPENCLAW_PLUGIN_ORIGINS maps an id to
    # the origin the Gateway reports.
    if log_path := os.environ.get("FAKE_OPENCLAW_PLUGIN_COMMANDS"):
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write("plugins.list\n")
    config_path = Path(os.environ["OPENCLAW_CONFIG_PATH"])
    plugins = json.loads(config_path.read_text(encoding="utf-8")).get("plugins", {})
    if "$include" in plugins:
        include = config_path.parent / plugins["$include"]
        plugins = json.loads(include.read_text(encoding="utf-8"))
    states = {
        key: "active" if entry.get("enabled", True) else "unloaded"
        for key, entry in plugins.get("entries", {}).items()
    }
    errors = json.loads(os.environ.get("FAKE_OPENCLAW_PLUGIN_ERRORS", "{}"))
    states.update({key: "unloaded" for key in errors})
    states.update(json.loads(os.environ.get("FAKE_OPENCLAW_PLUGINS", "{}")))
    origins = json.loads(os.environ.get("FAKE_OPENCLAW_PLUGIN_ORIGINS", "{}"))
    return {
        "plugins": [
            {
                "id": key,
                "enabled": state == "active",
                **({"installed": False} if state == "catalog" else {}),
                **({"origin": origins[key]} if key in origins else {}),
                "runtime": {
                    "state": "unloaded" if state == "catalog" else state,
                    **({"error": errors[key]} if key in errors else {}),
                },
            }
            for key, state in states.items()
            if state != "absent"
        ]
    }


def session_rpc(method, params):
    if method == "plugins.list":
        return plugins_rpc()
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
    if method == "sessions.abort":
        return {"ok": True, "status": "no-active-run", "abortedRunId": None}
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

    def websocket(self):
        protocol = ServerProtocol()
        header = (
            self.requestline
            + "\r\n"
            + "".join(f"{key}: {value}\r\n" for key, value in self.headers.items())
            + "\r\n"
        )
        protocol.receive_data(header.encode())
        request = protocol.events_received()[0]
        protocol.send_response(protocol.accept(request))
        for data in protocol.data_to_send():
            self.wfile.write(data)

        lock = threading.Lock()

        def send(value):
            with lock:
                protocol.send_text(json.dumps(value).encode())
                for data in protocol.data_to_send():
                    self.wfile.write(data)

        send(
            {
                "type": "event",
                "event": "connect.challenge",
                "payload": {"nonce": "mock-challenge"},
            }
        )
        event_streams.add(send)
        try:
            while data := self.rfile.read1(65536):
                with lock:
                    protocol.receive_data(data)
                    frames = protocol.events_received()
                    for pending in protocol.data_to_send():
                        self.wfile.write(pending)
                for frame in frames:
                    if frame.opcode == Opcode.TEXT:
                        request = json.loads(frame.data)
                        if request["method"] == "connect":
                            authenticated = request["params"].get("auth") == {
                                "token": token
                            }
                            payload = {"type": "hello-ok", "protocol": 4}
                        else:
                            assert authenticated
                            payload = session_rpc(request["method"], request["params"])
                        send(
                            {
                                "type": "res",
                                "id": request["id"],
                                "ok": authenticated,
                                "payload": payload,
                            }
                        )
                    elif frame.opcode == Opcode.CLOSE:
                        return
        except (ConnectionError, OSError):
            pass
        finally:
            event_streams.discard(send)

    def do_GET(self):
        if self.headers.get("Upgrade", "").lower() == "websocket":
            self.websocket()
            return
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
        if events := int(os.environ.get("FAKE_OPENCLAW_TURN_EVENTS", "0")):
            # A slow model turn streams events to the session connections.
            seconds = float(os.environ.get("FAKE_OPENCLAW_TURN_SECONDS", "0"))
            for sequence in range(events):
                for stream_event in list(event_streams):
                    try:
                        stream_event(
                            {"type": "event", "event": "agent", "seq": sequence}
                        )
                    except (OSError, InvalidState):
                        event_streams.discard(stream_event)
                time.sleep(seconds / events)
        response_text = "OpenClaw response"
        if key := self.headers.get("x-openclaw-session-key"):
            entry = native_sessions[key]
            entry["messages"].append(
                {
                    "role": "user",
                    "content": request["messages"][-1]["content"],
                    "__openclaw": {"id": str(uuid4())},
                }
            )
            if os.environ.get("FAKE_OPENCLAW_MEMORY_REPLY"):
                response_text = json.dumps(entry["messages"])
            entry["messages"].append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "stopReason": "stop",
                    "__openclaw": {"id": str(uuid4())},
                }
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


class GatewayServer(ThreadingHTTPServer):
    def server_bind(self):
        # Avoid reverse DNS delays for this loopback-only test server.
        TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


server = GatewayServer(("127.0.0.1", port), Handler)
signal.signal(signal.SIGTERM, stop_gateway)
server.serve_forever()
