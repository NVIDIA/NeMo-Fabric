<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric OpenClaw Adapter

This adapter supports two lifecycle shapes:

- Managed runtime: Each NeMo Fabric runtime owns an isolated local OpenClaw
  Gateway. Use this shape for evaluation, benchmark runs using the Harbor framework, and other bounded jobs.
- Service: One prepared or attached Gateway serves multiple independent NeMo
  Fabric runtimes. OpenClaw chat channels run beside the NeMo Fabric runtime API and keep
  their own OpenClaw sessions.

NeMo Fabric-originated invocations use the Gateway's OpenAI-compatible Chat
Completions endpoint. Telegram and other channels connect directly to OpenClaw;
they do not pass through a NeMo Fabric runtime.

The following diagram shows multiple NeMo Fabric runtimes sharing one OpenClaw
Gateway while Telegram uses a separate channel path:

![OpenClaw service architecture with multiple NeMo Fabric runtimes and Telegram channels](../../../assets/OpenClaw%20Chat%20Channels.png)

## Install

Install the adapter in every NeMo Fabric environment:

```bash
pip install "nemo-fabric[openclaw]"
```

Managed runtimes and NeMo Fabric-owned services also require the pinned OpenClaw
version:

```bash
# Requires Node.js >=24.22.1 <25 or >=26.12.1, and npm 11.16+.
npm install --global openclaw@2026.9.34 --allow-scripts=openclaw
```

An attachment to a caller-owned Gateway does not require a local OpenClaw
executable. OpenClaw is not a Python dependency of the adapter. For managed or
prepared use, if `openclaw` is not in `PATH`, set
`harness.settings.openclaw_command` to the absolute path of the executable. A
bare command is resolved through `PATH`; a relative path containing a directory
is resolved from the NeMo Fabric base directory.

### Upgrade a Gateway With Retained State

When upgrading a caller-owned Gateway from OpenClaw 2026.9.4, stop the Gateway and run `openclaw backup create --verify` with the Gateway's existing `OPENCLAW_STATE_DIR` and `OPENCLAW_CONFIG_PATH` before upgrading. The backup must include the active config, its include files, and all configured agent directories, including paths outside the state directory. Protect the archive because it can contain credentials. Use the 2026.9.34 executable with the same path settings to run `openclaw doctor --fix`, then restart the Gateway. Version 2026.9.34 rejects the older session database until this migration completes. Keep the verified backup for rollback.

Managed runtimes and NeMo Fabric-owned services use temporary state directories, so each new Gateway starts with fresh state.

## Configuration

### Supported Normalized Configuration

The adapter supports normalized model configuration, replacement system
instructions, workspace, skills, tool policy, and MCP servers. Refer to the
[OpenClaw adapter descriptor](./openclaw.fabric-adapter.json) for the
authoritative field-level contract.

The adapter maps these fields into a generated `openclaw.json` before starting
the Gateway. A non-empty `tools.enabled` list maps to `tools.allow`, an empty
list maps to a wildcard `tools.deny` policy, and `tools.blocked` maps to
`tools.deny`. It uses a generated one-time token, loopback binding, and an
isolated temporary OpenClaw state directory. The Gateway is stopped and its
temporary configuration is removed when the NeMo Fabric runtime stops.

For custom OpenAI-compatible providers, set `models.<role>.base_url`; the
adapter generates an OpenClaw custom provider using the
`openai-completions` API adapter. Set `api_key_env` when that provider requires
an API key.

### Harness Settings

Only OpenClaw's default `openclaw` agent runtime is supported.
The following settings are specific to the OpenClaw adapter:

| Setting | Default | Description |
| --- | --- | --- |
| `openclaw_command` | `openclaw` | Executable name or path. |
| `agent_id` | `default` | OpenClaw agent for NeMo Fabric-originated invocations. |
| `channel_config` | None | OpenClaw-native `channels` and `bindings` for a prepared service. Every binding must target `agent_id`. Managed runtimes reject this setting. |
| `port_range` | Random available range | Inclusive consumer-allocated range with `start` and `end` fields; it must span at least 111 ports. |
| `startup_timeout_seconds` | `30` | Gateway startup timeout. |
| `shutdown_timeout_seconds` | `5` | Graceful shutdown timeout. |
| `connect_timeout_seconds` | `10` | Local HTTP connection timeout. |
| `read_timeout_seconds` | None | Local HTTP response timeout. When unset, the invocation timeout bounds the response. |

OpenClaw's [derived-port mapping](https://docs.openclaw.ai/gateway/multiple-gateways#port-mapping-derived) uses `base_port + 2` for browser control and allocates browser CDP ports from `base_port + 11` through `base_port + 110`. Set `port_range` when the consumer reserves a port range before configuring NeMo Fabric; the adapter selects a base whose complete derived footprint fits inside that range. The adapter verifies only that `base_port` and `base_port + 2` are available before startup. OpenClaw allocates the CDP ports on demand instead of reserving the complete range, so the adapter cannot guarantee that those ports will still be available when OpenClaw needs them. Keep the configured range reserved for the lifetime of the runtime when using OpenClaw browser features.

Relay telemetry and native OpenTelemetry are not supported.

## Gateway Health

Use `Fabric.check()` with a prepared or attached service handle to observe the Gateway before starting any NeMo Fabric runtimes. Gateway checks use the local SDK and the selected Gateway.

| Level | Native observation |
|---|---|
| `live` | `GET /healthz` returns the expected liveness response. |
| `active` | Authenticated `GET /v1/models` returns a valid model inventory. |
| `ready` | `GET /readyz` reports native Gateway readiness, including configured channel health. |

Requesting `ready` evaluates `live`, `active`, and `ready` in order under one deadline, stopping at the first nonpassing result. A configured channel failure can fail Gateway readiness while the first two levels pass. Native readiness follows OpenClaw's channel policies and can use cached observations.

Check an owned Gateway with the following code, or pass the handle returned by `attach_service()`:

```python
from nemo_fabric import CheckRequest, Fabric

fabric = Fabric()
async with await fabric.prepare_service(config, base_dir=base_dir) as service:
    report = await fabric.check(
        service=service.handle,
        request=CheckRequest(
            scope="service", service_id=service.service_id, level="ready"
        ),
    )
    print(report.outcome)
```

Checks can run while connected runtimes are invoking. Timeouts return `unknown/probe_timeout`. Release during a check returns `unknown/observation_changed`; checks after release return `unknown/service_unavailable`.

## Runtime Health

Use a Gateway service check before starting runtimes or to monitor a shared
Gateway that serves several runtimes. Use a runtime check after starting a
runtime to verify the selected agent and Chat Completions endpoint. A Gateway
can pass `active` while a runtime fails `active`: for example, the Gateway
returns a valid model inventory that no longer includes the runtime's selected
agent. Both `ready` checks include configured channel health.

Use runtime-scoped `Fabric.check()` to check the agent selected by a NeMo Fabric runtime. These checks support owned Gateways and runtimes connected to a prepared or attached service. The SDK uses the same request for local and OpenShell environments; OpenShell forwards the request through `fabric-agent-ctl check`.

| Level | Native evidence |
|---|---|
| `live` | `GET /healthz` returns the liveness response. An owned Gateway process is running. |
| `active` | `GET /startupz` confirms startup; authenticated `GET /v1/models` includes `openclaw/<agent_id>`; `GET /v1/chat/completions` returns the native `405` response with `Allow: POST`. |
| `ready` | `GET /readyz` returns HTTP 200 with `ready: true`, including configured channel health. |

The `GET /v1/chat/completions` check confirms that the Chat Completions endpoint is enabled. The model inventory can remain available through the Responses API when Chat Completions is disabled. Malformed native responses return `unknown/probe_error`; a missing selected agent or disabled request endpoint returns `failed/endpoint_unavailable`. An interface disabled after an invocation failure cannot pass `active`.

Check a runtime with the following code:

```python
from nemo_fabric import CheckRequest, Fabric

fabric = Fabric()
environment = await fabric.prepare_environment(config, base_dir=base_dir)
try:
    async with await fabric.start_runtime_in(
        config, environment, base_dir=base_dir
    ) as runtime:
        report = await fabric.check(
            environment=environment,
            request=CheckRequest(
                scope="runtime", runtime_id=runtime.runtime_id, level="ready"
            ),
        )
        print(report.outcome)
finally:
    await fabric.release_environment(environment)
```

Levels run in order under one deadline and stop at the first nonpassing result. A busy runtime can pass; a timeout returns `unknown/probe_timeout`. Reported runtime activity covers NeMo Fabric invocations. Stop or ownership changes invalidate observations.

## Use a NeMo Fabric-Owned Service

`prepare_service()` maps the normalized `FabricConfig` to `openclaw.json`,
starts one Gateway, and returns a process-local service handle. Starting a
runtime with that service creates a separate NeMo Fabric runtime and OpenClaw
session without starting another Gateway.

```python
from nemo_fabric import Fabric

fabric = Fabric()
async with await fabric.prepare_service(config, base_dir=base_dir) as service:
    async with await fabric.start_runtime(
        config,
        base_dir=base_dir,
        service=service,
    ) as runtime:
        result = await runtime.invoke(input="Review the workspace changes.")
```

`service.handle` contains sanitized identity and connection information. It
never contains the Gateway token or channel credentials. The following example
shows a sanitized service handle:

```json
{
  "service_id": "service-...",
  "service_binding": "fabric-service-binding-...",
  "adapter_id": "nvidia.fabric.openclaw",
  "service_type": "openclaw_gateway",
  "ownership": "fabric_owned",
  "connection": {
    "gateway_url": "http://127.0.0.1:18789",
    "agent_id": "default"
  },
  "metadata": {
    "openclaw_version": "2026.9.34"
  }
}
```

You can start multiple runtimes with the same service. Each runtime uses its
own `runtime_id` as the OpenClaw Chat Completions `user`, so turns remain in a
runtime-specific session. Stop all connected runtimes before releasing the
service. NeMo Fabric rejects release while a connected runtime is active.

Service handles use a process-local registry. Reconstructing a service after
the NeMo Fabric process exits is not supported yet; cold resume will define that
behavior separately.

## Attach to a Caller-Owned Gateway

Use `attach_service()` when another deployment system starts and supervises
the Gateway. Supply an endpoint and the name of an environment variable that
contains the Gateway token. Do not put the token value in the reference.

```python
from nemo_fabric import Fabric
from nemo_fabric import ServiceReference

reference = ServiceReference.from_mapping(
    {
        "adapter_id": "nvidia.fabric.openclaw",
        "service_type": "openclaw_gateway",
        "connection": {
            "gateway_url": "https://openclaw.example.com",
            "gateway_token_env": "OPENCLAW_GATEWAY_TOKEN",
            "agent_id": "default",
        },
    }
)

fabric = Fabric()
async with await fabric.attach_service(config, reference) as service:
    async with await fabric.start_runtime(config, service=service) as runtime:
        result = await runtime.invoke(input="Review the workspace changes.")
```

The attached configuration can contain request-level model sampling and a
system instruction. Remove deployment-owned model endpoints, API-key
references, tools, MCP servers, skills, the OpenClaw command, channel
configuration, port range, and startup or shutdown timeouts because NeMo Fabric
cannot verify or apply those fields to an existing Gateway. Connection and read
timeouts and `agent_id` remain valid client-side settings. Releasing an
attachment closes only NeMo Fabric-owned clients and private connection material; it
does not stop the Gateway.

## Configure Chat Channels on a Prepared Service

Add OpenClaw-native `channels` and `bindings` before calling
`prepare_service()`. NeMo Fabric limits every binding to the one agent selected by
`agent_id`; the pinned OpenClaw version validates the nested channel settings.
Use `SecretRef` values instead of putting credentials directly in
`FabricConfig`.

For example, two Telegram bot accounts can route to the same configured agent:

```python
config.harness.settings["channel_config"] = {
    "channels": {
        "telegram": {
            "defaultAccount": "support",
            "accounts": {
                "support": {
                    "botToken": {
                        "source": "env",
                        "provider": "default",
                        "id": "TELEGRAM_SUPPORT_BOT_TOKEN",
                    },
                    "dmPolicy": "allowlist",
                    "allowFrom": ["tg:123456789"],
                },
                "alerts": {
                    "botToken": {
                        "source": "env",
                        "provider": "default",
                        "id": "TELEGRAM_ALERTS_BOT_TOKEN",
                    },
                    "dmPolicy": "allowlist",
                    "allowFrom": ["tg:123456789"],
                },
            },
        }
    },
    "bindings": [
        {
            "agentId": "default",
            "match": {"channel": "telegram", "accountId": "*"},
        }
    ],
}
```

The `channels` and `bindings` values use OpenClaw field names without a NeMo
Fabric-specific Telegram schema. OpenClaw starts the channels with the Gateway.
Channel messages create OpenClaw-owned sessions and continue without a NeMo
Fabric runtime. NeMo Fabric runtimes connected to the same service use separate
sessions through Chat Completions.

A prepared service snapshots this configuration when it starts. Use an
attached, caller-owned Gateway when channel configuration must be managed or
updated independently of the NeMo Fabric service lifecycle.

## Cleanup

The adapter runs the Gateway in an isolated process group, continuously checks
its startup endpoint, and forwards `SIGINT` and `SIGTERM` on POSIX systems.
On Linux, the adapter uses `setpriv --pdeathsig SIGTERM` when `setpriv` is
available and can run, so that the kernel signals the Gateway after abrupt
adapter termination. A sandbox that forbids applying capabilities, such as an
OpenShell sandbox, makes `setpriv` fail; the adapter then starts the Gateway
without it. Windows uses a kill-on-close Job Object for the Gateway process
tree. Without `setpriv`, Linux has the same catchable-signal protection as
macOS but cannot handle `SIGKILL`.

## Test the Service Lifecycle

Run these commands from the repository root. Install the supported OpenClaw
version first as described in [Install](#install).

Run the adapter and service tests:

```bash
uv run pytest tests/adapters/test_openclaw.py
cargo test -p nemo-fabric-core service_hosts_share_across_runtimes_and_require_ordered_release
```

Run the code-review example with one Gateway and two NeMo Fabric runtimes:

```bash
export NVIDIA_API_KEY="<your-key>"
uv run python -m examples.code_review_agent \
  --variant openclaw \
  --service \
  --runtime-count 2 \
  --input "Review the workspace changes."
```

For a live Telegram check, export the bot token and supply your numeric Telegram
user ID. Telegram is available as soon as the service starts and remains
available while the Fabric runtimes run. After their invocations finish, the
example keeps only the Gateway service alive for two more minutes:

```bash
export TELEGRAM_BOT_TOKEN="<your-token>"

uv run python -m examples.code_review_agent \
  --variant openclaw \
  --service \
  --runtime-count 2 \
  --telegram-token-env TELEGRAM_BOT_TOKEN \
  --telegram-allow-from "<your-numeric-user-id>" \
  --service-duration-seconds 120 \
  --input "Review the workspace changes."
```

When the example reports that the Telegram channel is active, open a direct
chat with the bot, send `/start` if this is your first interaction, and then
send a message. You do not need to wait for the Fabric runtimes to stop. To
specifically verify that Telegram is independent of those runtimes, send
another message after the example reports that they have stopped and before
the displayed service-only interval expires. The command prints its final JSON
output after the Gateway and channel stop at the end of that interval.

For Harbor, use managed mode so the Gateway and agent tools share the task
workspace, as shown in the
[calculator example](../../../examples/harbor/calculator/README.md#4-openclaw).
