<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenShell Environment Provider

This optional host process connects NVIDIA NeMo Fabric to OpenShell through its Rust SDK. The application uses the normal Fabric SDK; OpenShell owns sandbox isolation, network policy, and deletion. The agent and its adapter run inside the task image.

Build from the repository root:

```sh
just build-openshell-provider
export NEMO_FABRIC_OPEN_SHELL_PROVIDER="$PWD/target/debug/fabric-environment-openshell"
```

Use a registered OpenShell gateway and match the CLI, gateway, and runtime components to **0.1.2**. The provider pins that release's SDK to commit `6648bd0c290efbc41ba131ee9831ee45cd431f94`. Linux is the initial qualification target.

Environment configuration uses `provider="openshell"` and `control_location="in_env_control"`. The gateway connection selects a named workspace or explicitly uses OpenShell's default workspace. Workspace and artifact roots are absolute POSIX paths inside the sandbox; `runtime.artifacts` remains the host collection destination. Select an immutable image digest or local Docker image ID. Task images contain the SDK, `fabric-agent-ctl`, the selected adapter, and the agent/harness.

The two ownership paths have separate responsibilities:

| Operation | Resource ownership | Release behavior |
| --- | --- | --- |
| `prepare_environment` | Fabric creates the sandbox | Confirm deletion of the bound sandbox |
| `attach_environment` | Caller supplies an existing sandbox name and ID | Detach without deleting or connecting to the gateway |

Attachment verifies readiness, image, startup command, and any declared policy. Later operations check the immutable sandbox ID, so a same-name replacement is rejected. Fabric-owned release checks typed deletion acknowledgements and waits for the original ID; an accepted acknowledgement alone does not prove deletion. Readiness cleanup also preserves replacements.

Gateway credentials use environment-variable references, such as `connection.token_env` and `connection.ca_cert_env`. Tokens require verified HTTPS. Literal credential fields are rejected. Routine provider and transport errors exclude raw SDK exceptions and process stderr. Provider timeout settings accept 1–600 seconds. Calls are bounded and are not automatically replayed after a lost response. Cancelling preparation releases a confirmed resource; an unconfirmed cleanup error includes its sandbox identity for recovery.

The provider communicates privately over JSON and stdio. This interface is internal to Fabric. The public task command schemas are described in the [command bridge guide](../../sdk/python/nemo-fabric-runtime/src/nemo_fabric/host/README.md).
