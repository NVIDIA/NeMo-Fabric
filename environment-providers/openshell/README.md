<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenShell Environment Provider

This optional host process connects NVIDIA NeMo Fabric to OpenShell through its Rust SDK. The application uses the normal NeMo Fabric SDK; OpenShell owns sandbox isolation, network policy, and deletion. The agent and its adapter run inside the task image.

## Architecture

The diagrams show the host and sandbox responsibilities, followed by how an SDK
`invoke` call travels through the provider to the runtime in the sandbox.

![Fabric agent execution in OpenShell](../../assets/openshell_architecture.png)

[Edit the architecture diagram](../../assets/openshell_architecture.excalidraw).

![SDK invocation flow](../../assets/openshell_invocation.png)

[Edit the invocation diagram](../../assets/openshell_invocation.excalidraw).

The SDK starts the host-side provider when the first OpenShell operation runs.
Fabric launches `fabric-environment-openshell serve --stdio` as a child process,
using `NEMO_FABRIC_OPEN_SHELL_PROVIDER` when set or finding the binary on `PATH`.
It exchanges JSON with that process through standard input and output. Fabric
reuses a provider process for environment operations and keeps runtime command
processes separate by runtime ID.

For a Fabric-owned sandbox, the provider passes `settings.command` to OpenShell
when creating it; for a caller-owned sandbox, it verifies the existing startup
command. The examples use `fabric-agent-ctl serve` as that long-running sandbox
command. The provider then uses OpenShell exec calls to run short-lived
`fabric-agent-ctl start`, `invoke`, and `stop` commands against the task runtime
server.

Build from the repository root:

```sh
just build-openshell-provider
export NEMO_FABRIC_OPEN_SHELL_PROVIDER="$PWD/target/debug/fabric-environment-openshell"
```

Use a registered OpenShell gateway and match the CLI, gateway, and runtime components to **0.1.2**. The provider pins that release's SDK to commit `6648bd0c290efbc41ba131ee9831ee45cd431f94`. Initial support has been tested only on Linux.

Environment configuration uses `provider="openshell"` and `control_location="in_env_control"`. The gateway connection selects a named workspace or explicitly uses OpenShell's default workspace. Workspace and artifact roots are absolute POSIX paths inside the sandbox; `runtime.artifacts` remains the host collection destination. Select an immutable image digest or local Docker image ID. Task images contain the SDK, `fabric-agent-ctl`, the selected adapter, and the agent/harness.

When `settings.sandbox_name` is omitted, the provider derives a stable name from the Fabric environment ID so it can reconcile a create request whose response times out.

With an existing Fabric-ready configuration, use the normal SDK lifecycle:

```python
from nemo_fabric import EnvironmentReference, Fabric

fabric = Fabric()
environment = await fabric.attach_environment(
    config,
    EnvironmentReference.from_mapping({
        "provider": "openshell",
        "resource": {"sandbox_name": sandbox_name, "sandbox_id": sandbox_id},
    }),
)
try:
    async with await fabric.start_runtime_in(config, environment) as runtime:
        result = await runtime.invoke(input=agent_input)
finally:
    await fabric.release_environment(environment)
```

The host sends a task-local `FabricConfig` to `fabric-agent-ctl start`, then full
`RunRequest` values to invoke. The task SDK owns runtime IDs, state, and normalized
results, including failed results with partial output. Host gateway settings and
host discovery paths stay outside the sandbox. Installed task descriptors are
discovered normally; custom descriptor directories use
`environment.settings.task_discovery_paths` with absolute POSIX paths. Skills and
MCP command paths in the agent configuration must refer to task-visible files.

Declared artifacts are copied to
`runtime.artifacts/<runtime_id>/<invocation_id>/` on the host. Each reference
becomes a readable host path; earlier invocation files remain intact. Collection
accepts up to 16 regular files, 256 KiB per file, and 512 KiB total. Files are read
through directory descriptors with symlink and traversal checks; directories and
special files are rejected. Collection never truncates files or copies the
workspace. Relay file references retain their correlation metadata.

Invocation is buffered, with a task deadline of at most 300 seconds. Streaming is
rejected before startup. One OpenShell sandbox can own multiple bridge runtime
IDs, including starting another runtime after one stops. Missing or invalid
start responses retain an unknown binding against the immutable sandbox ID,
including after reattachment, and prevent another start because the runtime
identity cannot be recovered safely. Commands are not replayed. Explicit stop
and Fabric-owned deletion remain available for cleanup. Bridge invocations are
buffered; direct local SDK runtimes retain streaming and service support.

The two ownership paths have separate responsibilities:

| Operation | Resource ownership | Release behavior |
| --- | --- | --- |
| `prepare_environment` | Fabric creates the sandbox | Confirm deletion of the bound sandbox |
| `attach_environment` | Caller supplies an existing sandbox name and ID | Detach without deleting or connecting to the gateway |

Attachment verifies readiness, image, startup command, and any declared policy. Later operations check the immutable sandbox ID, so a same-name replacement is rejected. Fabric-owned release checks typed deletion acknowledgements and waits for the original ID; an accepted acknowledgement alone does not prove deletion. Readiness cleanup also preserves replacements.

Gateway credentials use environment-variable references, such as `connection.token_env` and `connection.ca_cert_env`. Tokens require verified HTTPS. Literal credential fields are rejected. Routine provider and transport errors exclude raw SDK exceptions and process stderr. Provider timeout settings accept 1–600 seconds. Calls are bounded and are not automatically replayed after a lost response. Cancelling preparation releases a confirmed resource; an unconfirmed cleanup error includes its sandbox identity for recovery.

The provider communicates privately over JSON and stdio. This interface is internal to NeMo Fabric. The public task command schemas are described in the [command bridge guide](../../sdk/python/nemo-fabric-runtime/src/nemo_fabric/host/README.md).
