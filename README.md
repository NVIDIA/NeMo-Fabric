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
  <img src="assets/fabric-hero.png" alt="Diagram showing NeMo Fabric connecting applications, evaluation systems, and reinforcement learning rollouts to Hermes Agent, Codex, Claude Code, LangChain Deep Agents, and custom agents, with results, artifacts, and telemetry as outputs." width="1000">
</p>

NeMo Fabric gives applications, evaluation systems, and rollout platforms a consistent interface for running agent harnesses and custom agents.

- **Compose runs:** Configure models, tools, skills, MCP servers, subagents, and runtime settings consistently.
- **Validate before execution:** Detect unsupported configurations before starting a runtime.
- **Switch harnesses:** Reuse the consumer integration while adapters handle harness-specific behavior.
- **Compare outcomes:** Collect normalized results, errors, artifacts, lifecycle events, and telemetry references.

## Supported Agent Harnesses

NeMo Fabric includes adapters for the following harnesses and execution targets. Select a harness name for its adapter documentation:

| Agent Harness | Tool Policy | MCP | Skills | Telemetry |
| --- | :---: | :---: | :---: | :---: |
| [Claude Code](adapters/python/claude/README.md) | ✅ | ✅ | ✅ | ✅ |
| [Cline](adapters/typescript/cline/README.md) | ✅ | ✅ | ✅ | — |
| [Codex](adapters/python/codex/README.md) | — | ✅ | ✅ | ✅ |
| [Hermes Agent](adapters/python/hermes/README.md) | ✅ | ✅ | ✅ | ✅ |
| [Kilo Code](adapters/typescript/kilo/README.md) | ✅ | ✅ | ✅ | — |
| [LangChain Deep Agents](adapters/python/deepagents/README.md) | ✅ | ✅ | ✅ | ✅ |
| [mini-SWE-agent](adapters/python/mini-swe-agent/README.md) | — | — | — | ✅ |
| [NOOA](adapters/python/nooa/README.md) | — | ✅ | ✅ | ✅ |
| [OpenClaw](adapters/python/openclaw/README.md) | ✅ | ✅ | ✅ | — |
| [OpenCode](adapters/typescript/opencode/README.md) | — | ✅ | ✅ | — |
| [Pi](adapters/typescript/pi/README.md) | ✅ | — | ✅ | ✅ |
| [Qwen Code](adapters/typescript/qwen/README.md) | ✅ | ✅ | ✅ | — |
| [Remote Agent](adapters/python/remote-agent/README.md) | — | — | — | ✅ |

✅ Supported · — Not supported. Support may come through normalized configuration, native harness behavior, or a specific target. Refer to each adapter guide or the [complete compatibility matrix](adapters/README.md#configuration-compatibility) for details and limitations.

## NVIDIA Ecosystem

NeMo Fabric is an execution layer for NVIDIA products that build, evaluate, deploy, and improve agents. The ecosystem includes shipped integrations with [NVIDIA NeMo Helix](https://github.com/NVIDIA-NeMo/nemo-helix), concrete integrations with [NVIDIA NemoClaw](https://github.com/NVIDIA/NemoClaw/) and [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/gym), and emerging rollout integrations such as [NVIDIA Polar](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server). [Explore the NeMo Fabric ecosystem](docs/about-nemo-fabric/ecosystem.mdx).

## Extensibility

NeMo Fabric separates consumers from execution targets through a stable, versioned contract. This makes it extensible in two directions:

- **Integrate a consumer:** Applications, evaluation systems, and rollout platforms use the Python SDK and typed `FabricConfig` to run different harnesses without implementing each harness integration. Start with the [consumer integration overview](docs/integrations/consumer/overview.mdx) or [consumer integration skills](skills/README.md).
- **Add an execution target:** Adapters translate the NeMo Fabric contract into the native behavior of a harness, framework, custom agent, or remote service. Start with the [adapter contract](docs/adapter-contract/README.md) and [adapter examples](docs/adapter-contract/examples.md).

## Documentation

- [Quickstart](docs/getting-started/quickstart.mdx): Run a first agent through the Python SDK.
- [Installation](docs/getting-started/install.mdx): Install the runtime, adapters, harness dependencies, and telemetry components, and choose a deployment scenario.
- [Documentation overview](docs/about-nemo-fabric/overview.mdx): Understand the execution model and choose an interface.
- [Adapter reference](adapters/README.md): Compare adapter configuration, runtime, and observability support.
- [Contribution guide](CONTRIBUTING.md): Build, test, and contribute to the project.

## Roadmap

- **Expand the harness ecosystem:** Add high-priority first- and third-party harnesses through the normalized lifecycle and capability contracts.
- **Sandbox-native execution:** Run NeMo Fabric in secure environments through providers such as [NVIDIA OpenShell](https://docs.nvidia.com/openshell/about/overview).
- **Deepen NVIDIA ecosystem integrations:** Make NeMo Fabric a standard execution boundary for agent configuration, evaluation, rollouts, artifacts, and telemetry.
- **Enhance session lifecycle management:** Add explicit cold-start, resume, and fork operations with stable session identifiers and capability-aware errors.
- **Compose subagents:** Configure supported subagents through validated, portable NeMo Fabric configuration.
