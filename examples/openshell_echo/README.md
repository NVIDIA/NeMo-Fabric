<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SDK Echo in OpenShell

This is the smallest end-to-end example of running an NVIDIA NeMo Fabric-ready
agent in OpenShell through the normal NeMo Fabric SDK. The consumer calls the SDK to
attach or prepare an environment, check the runtime server, start and invoke a runtime, then stop and release it. The consumer starts the runtime after the server check passes.

The task image contains the SDK runtime server, adapter, and descriptor. The
shared [launcher](../openshell/README.md) builds the image, starts a local
OpenShell gateway, and runs both ownership modes by default. No model endpoint
or inference key is needed.

From the repository root, use a host Python environment that has the NeMo
Fabric SDK and its native extension. Then run:

```bash
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
bash examples/openshell/run-demo.sh examples/openshell_echo/demo.sh \
  --input '{"message":"hello from the host SDK"}'
```

To run one lifecycle mode, set `OPENSHELL_EXAMPLE_MODE=caller_owned` or
`OPENSHELL_EXAMPLE_MODE=fabric_owned`. The shared launcher checks that the
caller-owned sandbox remains after SDK detach and that the NeMo Fabric-owned
sandbox is deleted after SDK release.

The consumer writes the server health report and two normalized results to
`.tmp/openshell-echo/{caller_owned,fabric_owned}-results.json`. The second
invocation returns a declared receipt; the provider collects it into the
host-side artifact directory.

The server check uses `Fabric.check(environment=environment, request=CheckRequest(scope="server"))`, which maps to `fabric-agent-ctl check` inside the sandbox. Use the Pi adapter to exercise runtime `live`, `active`, and `ready` checks.
