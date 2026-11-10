#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock OpenClaw CLI and Gateway used by adapter tests."""

import hashlib
import json
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def workshop_skills() -> Path:
    """Where the agent saves the skills it writes, inside OpenClaw's state."""
    return (
        Path(os.environ["OPENCLAW_STATE_DIR"])
        / "agents"
        / "default"
        / "agent"
        / "workshop-skills"
    )


def skill_reply(request: dict) -> str | None:
    """Write or use a skill the way the agent does, for retention tests.

    `author-skill <name>` saves a new skill in OpenClaw's state.
    `use-skill <name> <nonce>` discovers it there, reads it, and replies with
    its path, its hash, and an output that only this request can produce.
    """
    messages = [m for m in request.get("messages", []) if m.get("role") == "user"]
    words = str(messages[-1].get("content", "")).split() if messages else []
    if words[:1] == ["author-skill"] and len(words) == 2:
        skill = workshop_skills() / words[1] / "SKILL.md"
        skill.parent.mkdir(parents=True, exist_ok=True)
        skill.write_text(
            f"---\nname: {words[1]}\ndescription: Echo a nonce.\n---\n\n"
            "Reply with the skill name and the nonce.\n",
            encoding="utf-8",
        )
    elif words[:1] != ["use-skill"] or len(words) != 3:
        return None
    found = sorted(workshop_skills().glob("*/SKILL.md"))
    skill = next((path for path in found if path.parent.name == words[1]), None)
    if skill is None:
        return "SKILL_MISSING"
    return json.dumps(
        {
            "path": str(skill),
            "sha256": hashlib.sha256(skill.read_bytes()).hexdigest(),
            "output": f"{words[1]}:{words[2]}" if words[0] == "use-skill" else None,
        }
    )


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
if args[:2] != ["gateway", "run"]:
    raise SystemExit(2)

port = int(args[args.index("--port") + 1])
token = os.environ["OPENCLAW_GATEWAY_TOKEN"]
if readonly_path := os.environ.get("FAKE_OPENCLAW_CONFIG_READONLY"):
    with open(readonly_path, "w", encoding="utf-8") as stream:
        stream.write(os.environ.get("OPENCLAW_CONFIG_READONLY", ""))
if state_path := os.environ.get("FAKE_OPENCLAW_STATE_CAPTURE"):
    with open(state_path, "w", encoding="utf-8") as stream:
        stream.write(os.environ["OPENCLAW_STATE_DIR"])
if pid_path := os.environ.get("FAKE_OPENCLAW_PID"):
    with open(pid_path, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))


def stop_gateway(*_unused):
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
        chunks = [
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": skill_reply(request) or "OpenClaw response"
                        },
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
