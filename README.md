<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Fabric)](https://github.com/NVIDIA/NeMo-Fabric/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Fabric/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Fabric?color=green)](https://github.com/NVIDIA/NeMo-Fabric/releases)
[![PyPI](https://img.shields.io/pypi/v/nemo-fabric?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-fabric/)
[![Crates.io](https://img.shields.io/crates/v/nemo-fabric-core?label=nemo-fabric-core&color=B7410E&logo=rust)](https://crates.io/crates/nemo-fabric-core)
[![Crates.io](https://img.shields.io/crates/v/nemo-fabric-cli?label=nemo-fabric-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-fabric-cli)

<p align="center">
  <img src="assets/fabric-hero.svg" alt="Diagram showing NeMo Fabric connecting applications, evaluation systems, and reinforcement learning rollouts to Hermes Agent, Codex, Claude Code, LangChain Deep Agents, and custom agents, with results, artifacts, and telemetry as outputs." width="1000">
</p>

NeMo Fabric gives applications, evaluation systems, and rollout platforms a consistent interface for running agent harnesses and custom agents.

- **Compose runs:** Configure models, tools, skills, plugins, MCP servers, subagents, and runtime settings consistently.
- **Validate before execution:** Detect unsupported configurations before starting a runtime.
- **Switch harnesses:** Reuse the consumer integration while adapters handle harness-specific behavior.
- **Compare outcomes:** Collect normalized results, errors, artifacts, lifecycle events, and telemetry references.

## Supported Agent Harnesses

NeMo Fabric includes adapters for the following harnesses and execution targets. Select a harness name for its adapter documentation:

| Agent Harness | Tool Policy | MCP | Skills | Telemetry | Native Interfaces |
| --- | :---: | :---: | :---: | :---: | :---: |
| [Claude Code](adapters/python/claude/README.md) | ✅ | ✅ | ✅ | ✅ | — |
| [Cline](adapters/typescript/cline/README.md) | ✅ | ✅ | ✅ | — | — |
| [Codex](adapters/python/codex/README.md) | — | ✅ | ✅ | ✅ | — |
| [Factory Droid](adapters/typescript/droid/README.md) | ✅ | ✅ | ✅ | — | — |
| [Hermes Agent](adapters/python/hermes/README.md) | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Kilo Code](adapters/typescript/kilo/README.md) | ✅ | ✅ | ✅ | — | — |
| [AWS Kiro](adapters/typescript/kiro/README.md) | — | ✅ | — | — | — |
| [AWS Strands](adapters/typescript/strands/README.md) | — | ✅ | — | — | — |
| [LangChain Deep Agents](adapters/python/deepagents/README.md) | ✅ | ✅ | ✅ | ✅ | — |
| [mini-SWE-agent](adapters/python/mini-swe-agent/README.md) | — | — | — | ✅ | — |
| [NOOA](adapters/python/nooa/README.md) | — | ✅ | ✅ | ✅ | — |
| [OpenClaw](adapters/python/openclaw/README.md) | ✅ | ✅ | ✅ | — | ✅ |
| [OpenHands](adapters/python/openhands/README.md) | ✅ | ✅ | ✅ | — | — |
| [OpenCode](adapters/typescript/opencode/README.md) | — | ✅ | ✅ | — | — |
| [Pi](adapters/typescript/pi/README.md) | ✅ | — | ✅ | ✅ | — |
| [Qwen Code](adapters/typescript/qwen/README.md) | ✅ | ✅ | ✅ | — | — |
| [Remote Agent](adapters/python/remote-agent/README.md) | — | — | — | ✅ | — |

Native interfaces: harness-native API, dashboard, or TUI access enabled through Fabric.

✅ Supported · — Not supported. Support may come through normalized configuration, native harness behavior, or a specific target. Refer to each adapter guide or the [complete compatibility matrix](adapters/README.md#configuration-compatibility) for details and limitations.

## Ecosystem

NeMo Fabric connects supported agent harnesses to benchmarking, training, and deployment workflows.

### Benchmarking and Training

- **[Harbor Framework](https://github.com/harbor-framework/harbor):** Compare harnesses, models, and agent configurations on the same benchmark tasks.
- **[NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym):** Evaluate interchangeable harnesses in shared task environments and collect training rollouts.
- **[NVIDIA NeMo Polar](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server):** Switch harnesses while keeping the same task and reinforcement learning rollout workflow.

### Deployment and Management

- **[NVIDIA NemoClaw](https://github.com/NVIDIA/NemoClaw):** Deploy and manage sandboxed agents through a common workflow across harnesses.
- **[NVIDIA NeMo Helix](https://github.com/NVIDIA-NeMo/nemo-helix):** Run, evaluate, and improve agents through platform-managed workflows.

Refer to the [ecosystem guide](docs/about-nemo-fabric/ecosystem.mdx) for integration status and details.

## Extensibility

Integrate with NeMo Fabric as a consumer or an agent harness:

- **Consumer integration:** Use the Python SDK to run supported agent harnesses from your application, evaluation system, or rollout platform. Start with the [consumer integration guide](docs/integrations/consumer/overview.mdx).
- **Harness integration:** Build an adapter to make your harness, custom agent, or remote service available through NeMo Fabric’s common execution interface. Start with the [adapter contract](docs/adapter-contract/README.md) and [adapter examples](docs/adapter-contract/examples.md).

[Integration skills](skills/README.md) provide coding-agent guidance for both paths.

## Documentation

- **[Overview](docs/about-nemo-fabric/overview.mdx):** Understand the execution model and key concepts.
- **[Quickstart](docs/getting-started/quickstart.mdx):** Run your first agent.
- **[Consumer integrations](docs/integrations/consumer/overview.mdx):** Use NeMo Fabric from applications, evaluation systems, and rollout platforms.
- **[Supported harnesses](adapters/README.md):** Configure and run agent harnesses already integrated with NeMo Fabric.
- **[Harness integrations](docs/adapter-contract/README.md):** Make your harness or custom agent available through NeMo Fabric.
- **[Contribution guide](CONTRIBUTING.md):** Build, test, and contribute to NeMo Fabric.

## Roadmap

Planned areas of development include:

- **Task visibility and control:** Monitor agent progress, guide ongoing work, and manage pauses, approvals, and continuation.
- **Extensible agents and subagents:** Extend custom tools, skills, and middleware to support richer agent workflows and delegation.
- **Execution observability:** Provide connected run histories, traces, diagnostics, and retained evidence across successful and failed executions.
- **Usage, cost, and comparison:** Standardize model-usage accounting, add transparent cost estimates, and enable meaningful comparisons across task attempts.
- **Broader agent coverage:** Expand supported agents and evaluation workflows, including NOOA Atom and Goose.
- **Simpler integration and setup:** Improve onboarding, configuration validation, and adapter environment setup.
