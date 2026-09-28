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
# Requires Node.js >=24.16.0 <25 or >=26.1.0, and npm 11.16+.
npm install --global openclaw@2026.9.4 --allow-scripts=openclaw
```

An attachment to a caller-owned Gateway does not require a local OpenClaw
executable. OpenClaw is not a Python dependency of the adapter. For managed or
prepared use, if `openclaw` is not in `PATH`, set
`harness.settings.openclaw_command` to the absolute path of the executable. A
bare command is resolved through `PATH`; a relative path containing a directory
is resolved from the NeMo Fabric base directory.

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
temporary configuration is removed when the NeMo Fabric runtime stops, unless
`state_dir` retains it as described in [Retain Native State](#retain-native-state).

For custom providers, set `models.<role>.base_url` and, optionally,
`models.<role>.api`: `openai-completions` (the default), `openai-responses`, or
`anthropic-messages`. The adapter generates an OpenClaw custom provider for that
endpoint. Set `api_key_env` when the provider requires an API key. For such a
model, `models.<role>.settings.model_metadata` adds OpenClaw catalog fields such
as `contextWindow` and `reasoning`, and `models.<role>.settings.reasoning_effort`
on the selected role sets OpenClaw's default thinking level.

The selected role, `models.default` or the sole role, is the agent's primary
model. Every other role also becomes a model that OpenClaw can switch to, named
by its role as an alias. A role with its own endpoint gets a provider named
`<provider>-<role>`, and roles with identical configuration share one model.

### Harness Settings

Only OpenClaw's default `openclaw` agent runtime is supported.
The following settings are specific to the OpenClaw adapter:

| Setting | Default | Description |
| --- | --- | --- |
| `openclaw_command` | `openclaw` | Executable name or path. |
| `agent_id` | `default` | OpenClaw agent for NeMo Fabric-originated invocations. |
| `channel_config` | None | OpenClaw-native `channels` and `bindings` for a prepared service. Every binding must target `agent_id`. Managed runtimes reject this setting. |
| `port` | None | Fixed Gateway base port. `port + 2` must also be free. Mutually exclusive with `port_range`. |
| `port_range` | Random available range | Inclusive consumer-allocated range with `start` and `end` fields; it must span at least 111 ports. |
| `state_dir` | Temporary directory | Directory that retains native state across runtimes. Refer to [Retain Native State](#retain-native-state). |
| `native_config` | None | Additional `openclaw.json` sections. Refer to [Native Configuration](#native-configuration). |
| `startup_timeout_seconds` | `30` | Gateway startup timeout. |
| `shutdown_timeout_seconds` | `10` | Graceful shutdown timeout. |
| `connect_timeout_seconds` | `10` | Local HTTP connection timeout. |
| `read_timeout_seconds` | `600` | Local HTTP response timeout. |

OpenClaw's [derived-port mapping](https://docs.openclaw.ai/gateway/multiple-gateways#port-mapping-derived) uses `base_port + 2` for browser control and allocates browser CDP ports from `base_port + 11` through `base_port + 110`. Set `port_range` when the consumer reserves a port range before configuring NeMo Fabric; the adapter selects a base whose complete derived footprint fits inside that range. The adapter verifies only that `base_port` and `base_port + 2` are available before startup. OpenClaw allocates the CDP ports on demand instead of reserving the complete range, so the adapter cannot guarantee that those ports will still be available when OpenClaw needs them. Keep the configured range reserved for the lifetime of the runtime when using OpenClaw browser features.

Relay telemetry and native OpenTelemetry are not supported.

### Native Configuration

`harness.settings.native_config` adds OpenClaw configuration that NeMo Fabric
does not derive, such as `plugins`, `tools.web`, `tools.toolSearch`,
`diagnostics`, `agents.defaults.heartbeat`, and `gateway.controlUi`. These
sections can override adapter defaults, such as the disabled control UI and
telemetry, but a value that differs from one NeMo Fabric owns fails before
startup. NeMo Fabric owns the Gateway listener (`gateway.mode`, `port`, `bind`,
`auth`, and `http`), `models`, `agents.entries`, the agent workspace and
models, and whatever `tools`, `mcp`, and `skills` values it generates.
Planning rejects the listener, `models`, and agent fields directly.

The control UI requires `state_dir`, because the browser signs in with the
retained interface token. When `gateway.controlUi.allowedOrigins` is omitted,
the adapter allows the Gateway's loopback origins.

### Retain Native State

Set `harness.settings.state_dir` to keep OpenClaw's native state, such as
sessions, memory, and channel pairing, across runtimes. The directory holds
`openclaw.json`, which the adapter regenerates at each start and OpenClaw treats
as read-only, `gateway.log` with the Gateway's output, and `interface-token`,
the Gateway credential for API and control UI clients. The token is created
once with owner-only permissions. Only one runtime can use a state directory at
a time, and `state_dir` is supported only on POSIX hosts. Relative paths resolve
from the NeMo Fabric base directory. Change the configuration through
`FabricConfig` and `native_config` rather than by editing `openclaw.json`.

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
    "openclaw_version": "2026.9.4"
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
available so that the kernel signals the Gateway after abrupt adapter
termination. Windows uses a kill-on-close Job Object for the Gateway process
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
