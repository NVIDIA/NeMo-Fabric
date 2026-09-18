#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock OpenClaw CLI and Gateway used by adapter tests."""

import json
import os
import signal
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


args = sys.argv[1:]
if args == ["--version"]:
    print("OpenClaw 2099.1.0")
    raise SystemExit(0)
if args[:3] == ["config", "validate", "--json"]:
    with open(os.environ["OPENCLAW_CONFIG_PATH"], encoding="utf-8") as stream:
        value = json.load(stream)
    with open(
        os.environ["FAKE_OPENCLAW_CAPTURE"], "w", encoding="utf-8"
    ) as stream:
        json.dump(value, stream)
    print(json.dumps({"valid": True}))
    raise SystemExit(0)
if args[:3] == ["plugins", "list", "--json"]:
    print(json.dumps({"plugins": []}))
    raise SystemExit(0)
if args[:3] == ["gateway", "call", "nemoRelay.status"]:
    print(json.dumps({"ok": True}))
    raise SystemExit(0)
if args[:2] != ["gateway", "run"]:
    raise SystemExit(2)

port = int(args[args.index("--port") + 1])
token = os.environ["OPENCLAW_GATEWAY_TOKEN"]
if readonly_path := os.environ.get("FAKE_OPENCLAW_CONFIG_READONLY"):
    with open(readonly_path, "w", encoding="utf-8") as stream:
        stream.write(os.environ.get("OPENCLAW_CONFIG_READONLY", ""))
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
        if self.path != "/readyz":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if self.headers.get("Authorization") != f"Bearer {token}":
            self.send_response(401)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        with open(
            os.environ["FAKE_OPENCLAW_REQUEST"], "w", encoding="utf-8"
        ) as stream:
            json.dump(request, stream)
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "OpenClaw "},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "response"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        body += "data: [DONE]\n\n"
        body = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
signal.signal(signal.SIGTERM, stop_gateway)
server.serve_forever()
