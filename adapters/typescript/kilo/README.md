<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kilo Code Adapter

This package connects NVIDIA NeMo Fabric to Kilo Code through Kilo's local server and JavaScript SDK.

Install the adapter and its consumer-managed harness packages together:

```bash
npm install nemo-fabric-adapters-kilo @kilocode/cli@7.7.12 @kilocode/sdk@7.7.12
```

The adapter maps normalized model configuration, replacement system instructions, maximum turns, tool policy, MCP servers, and skill directories. It starts an isolated `kilo serve` process and keeps one Kilo session warm for the Fabric runtime lifetime.

The initial adapter does not advertise streaming, cancellation, updates, service mode, or Relay telemetry. HTTP MCP endpoints must use HTTPS unless they are loopback addresses. The adapter disables project Kilo configuration and uses an isolated profile so the Fabric configuration remains authoritative.
