<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Run a NeMo Fabric Agent in OpenShell

These examples use OpenShell as the sandbox and policy boundary while keeping
the normal NeMo Fabric SDK lifecycle in the consumer. The agent and its adapter
run inside the sandbox with `fabric-agent-ctl serve`; the consumer calls the
provider through the SDK. Start with the [minimal SDK echo example](../openshell_sdk/README.md)
to see the complete host-to-sandbox path, then explore the [Code Review](../code_review_agent/README.md)
example.

## Requirements

The shared launcher currently requires Linux, Docker using Linux containers,
Rust/Cargo, `uv`, Python 3, `curl`, `openssl`, and `tar`. It builds the OpenShell
environment provider and a small SDK image, downloads a pinned OpenShell
release, starts a local gateway, then builds and runs the workload image.
Builds and test output are kept under `.tmp/`.

Set `UV_PROJECT_ENVIRONMENT` to an installed NeMo Fabric environment if it is
not at `.tmp/host-stack-tests`. The environment needs NeMo Fabric and the
native extension. The SDK/provider examples use the Rust OpenShell provider
binary downloaded by the launcher and do not require the OpenShell Python SDK.
The raw/direct [Code Review command-line example](../code_review_agent/README.md#rawdirect-openshell-code-review)
uses the OpenShell Python SDK; install the version matching the pinned gateway
only when using that example:

```bash
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.tmp/host-stack-tests}"
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" openshell==0.1.2
```

Set `OPENSHELL_VERSION` to test another release and
`OPENSHELL_EXAMPLE_PORT` if port `18080` is already in use.

## Workload interface

`run-demo.sh` accepts a workload hook script. A workload provides its name,
supported ownership modes, image staging, and consumer invocation. It can set
`SERVER_ENV_ARGS` for the long-lived `fabric-agent-ctl serve` process; use this
when an adapter is installed in a separate Python environment. The shared
launcher builds the images, starts the gateway, creates the requested sandbox,
passes its handle to the consumer, and cleans up resources it created. A workload
may define `configure_workload_gateway` to configure gateway-side providers
before a sandbox starts; it may set `MODEL_ENV_ARGS` for sandbox creation options
such as a registered provider profile.

`OPENSHELL_EXAMPLE_MODE` accepts `caller_owned`, `fabric_owned`, or `both`.
Caller-owned mode creates a sandbox before invoking the SDK consumer and
confirms it remains after detach. Fabric-owned mode lets the provider create
the sandbox through `prepare_environment` and confirms
`release_environment` removes it. A workload can support either or both modes.

## Agent readiness

A Fabric-ready agent needs a registered adapter and a runtime image that
contains the adapter, its harness and dependencies, the task, and the
`fabric-agent-ctl` runtime server. Keep sandbox lifecycle and model access in
the consumer/provider configuration. The agent itself should not need
OpenShell-specific calls.

For another agent, adapt the example workload hooks and provide:

- an adapter descriptor and importable adapter implementation;
- a reproducible image build containing the agent and declared task files;
- an OpenShell policy that grants only the required network and filesystem
  access; and
- a normal SDK consumer that starts, invokes, and stops the runtime.

The SDK echo example exercises state across two turns, normalized results,
declared artifact collection, and both environment ownership modes. The Code
Review example exercises a packaged third-party harness, model endpoint policy,
and optional Hermes Relay evidence.
