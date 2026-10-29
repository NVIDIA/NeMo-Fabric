# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local-model integration gate using real coding harnesses and Relay >=0.11.

RUN_FABRIC_RELAY_HTTP_INTEGRATION=1 FABRIC_TEST_NEMO_RELAY_COMMAND=/path/to/nemo-relay \
    uv run pytest tests/e2e/test_relay_http_models.py
"""

import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from _utils.mock_api_server import _messages_text_events, _responses_text_events
from nemo_fabric import (
    EnvironmentConfig,
    Fabric,
    FabricConfig,
    FabricRuntimeError,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
    RelayAtifConfig,
    RelayObservabilityConfig,
    RuntimeConfig,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FABRIC_RELAY_HTTP_INTEGRATION") != "1",
    reason="set RUN_FABRIC_RELAY_HTTP_INTEGRATION=1 and select a Relay >=0.11 CLI",
)


@pytest.fixture(name="http_model_endpoint")
def http_model_endpoint_fixture():
    # Use this machine's non-loopback address so Relay's remote-HTTP policy runs.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect(("192.0.2.1", 80))
        host = route.getsockname()[0]
    captured = []

    class ModelHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append({key.lower(): value for key, value in self.headers.items()})
            events = (
                _responses_text_events(payload, "echo response")
                if self.path == "/v1/responses"
                else _messages_text_events(payload, "echo response")
            )
            body = "".join(events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer((host, 0), ModelHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield host, f"http://{host}:{server.server_port}/v1", captured
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize("adapter_id", ["codex", "claude"])
@pytest.mark.parametrize(
    "trust", ["allowed", "omitted", "wrong-host", "relay-disabled"]
)
async def test_relay_http_model_forwarding(
    http_model_endpoint, tmp_path, adapter_id, trust
):
    host, base_url, captured = http_model_endpoint
    settings = {}
    if trust in {"allowed", "wrong-host"}:
        settings["insecure_http_model_hosts"] = [
            host if trust == "allowed" else "other.invalid"
        ]
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path / "xdg")
    os.environ["NEMO_RELAY_CONFIG_SCOPE"] = "user"
    config = FabricConfig(
        metadata=MetadataConfig(name="relay-http-smoke"),
        harness=HarnessConfig(
            adapter_id=f"nvidia.fabric.{adapter_id}",
            resolution="preinstalled",
            settings=settings,
        ),
        models={
            "default": ModelConfig(
                provider="fabric-test",
                model="fabric-echo",
                api_key_env="FABRIC_TEST_API_KEY",
                base_url=base_url,
            )
        },
        runtime=RuntimeConfig(
            input_schema="text" if adapter_id == "codex" else "chat",
            output_schema="message",
            artifacts=tmp_path / "artifacts",
        ),
        environment=EnvironmentConfig(
            provider="local",
            workspace=tmp_path,
            artifacts=tmp_path / "artifacts",
            env={"FABRIC_TEST_API_KEY": "dummy-private-model-key"},
        ),
    )
    if trust != "relay-disabled":
        config.enable_relay(
            output_dir=tmp_path / "telemetry",
            observability=RelayObservabilityConfig(
                atif=RelayAtifConfig(
                    enabled=True, output_directory=tmp_path / "telemetry"
                )
            ),
        )
    if trust in {"omitted", "wrong-host"}:
        with pytest.raises(
            FabricRuntimeError, match=f"{adapter_id}_relay_start_failed"
        ):
            await Fabric().run(config, base_dir=tmp_path, input="Reply echo response.")
        assert captured == []
        return

    result = await Fabric().run(config, base_dir=tmp_path, input="Reply echo response.")
    assert result.status == "succeeded", result.to_mapping()
    assert result.output["response"] == "echo response"
    assert captured
    if trust == "relay-disabled":
        assert any(
            headers.get("authorization") == "Bearer dummy-private-model-key"
            or headers.get("x-api-key") == "dummy-private-model-key"
            for headers in captured
        )
        return
    for headers in captured:
        assert (
            not {"authorization", "x-api-key", "cookie", "proxy-authorization"}
            & headers.keys()
        )
    assert any(
        artifact["kind"] == "atif" for artifact in result.output["relay_artifacts"]
    )
