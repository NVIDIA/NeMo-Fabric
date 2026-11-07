<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deadline Qualification Setup

Use existing authorization and supplied endpoint choices. Ask only for missing
model ID, API protocol, credential variable name, host harness interpreters,
clean pinned Hermes source checkout, or sandbox network policy. Export credential
values through the environment; plans contain variable names, never their values.
A confirmed no-auth endpoint can use a nonsecret placeholder `OPENAI_API_KEY`.

Verify the served model ID through `/models`, tool-calling compatibility and
reachability from both host and sandbox. Reuse the same served model for both
compatible harnesses. OpenShell native TCP policy needs a DNS hostname, not an
IP literal; verify forward/reverse resolution and restrict `allowed_ips`, port
and the client Python binary to the intended endpoint. A separate sandbox URL
may route to that same endpoint. Keep policy files local when they contain
private infrastructure addresses.

Preview the full denominator before executing: Hermes/Deep Agents ×
host/OpenShell × smoke/finite/unlimited/timeout. Finite and unlimited cases each
use two sequential 165-second MCP calls within the invocation, with real model
calls before and after the tool work. Each successful review must quote its
unique completion receipt and report the actual example's `answer()` value.
The short-deadline case must report `timeout` at `invoke` and clean up.

Use compatible harness dependency environments already prepared with their
supported installations. The runner creates fresh SDK environments and adds
only the selected dependency site-packages paths; it does not activate their
editable SDK `.pth` files. It installs the current SDK, evidence, contract,
common adapters, catalog, collector and selected adapter wheels first. A clean
Hermes source checkout at the image's pinned revision is explicitly recorded.
The separate host/task interpreters need not have identical Python versions.

For a release, select the exact candidate commit in a disposable checkout.
Ordinary Rust/Python/TypeScript suites and documentation checks remain required
by `RELEASING.md`; this gate supplies the focused live-deadline evidence.
