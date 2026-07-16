<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Fabric CLI Experiments

This is a working note for experimentation and team discussion. It is not a
committed public interface or release contract.

## Intent

Keep `nemo-fabric` as a thin SDK-backed runner:

- every run starts from a complete, typed `FabricConfig`;
- the CLI only selects where that config comes from;
- planning and execution go through the public Python SDK;
- `base_dir` only resolves relative resource paths; and
- Fabric does not load a persisted YAML, TOML, or JSON configuration.

Profiles, overlays, config discovery, and compatibility loaders are out of
scope. Fabric has not been released, so this experiment makes a clean break.

## Current Variation

### 1. Preset

```bash
nemo-fabric run --preset hermes --input "Say hello"
```

A small, complete config maintained with the CLI. Useful for smoke tests and
quick harness probes. Presets do not inherit or merge.

### 2. Example

```bash
nemo-fabric run \
  --example examples.code_review_agent \
  --variant hermes \
  --input "Review this workspace"
```

A runnable SDK example with complete variants. Examples are Python source and
assets that people can read, copy, and edit. A variant is a factory, not a
profile.

### 3. User factory

```bash
nemo-fabric run \
  --factory my_agent.config:build_config \
  --base-dir . \
  --input "Review this workspace"
```

The unrestricted customization path. The callable takes no arguments and
returns a complete `FabricConfig`.

`--factory` is the current spelling of the earlier `--custom` idea because it
makes the Python contract explicit.

All three selectors feed the same commands:

```text
plan   doctor   run   chat
```

## Variations to Discuss

| Question | Lean option | Other reasonable experiments |
| --- | --- | --- |
| Selector grammar | `--preset`, `--example`, `--factory` | source subcommands; `preset:name` references |
| Editable starting points | copy an example's Python source | `example init`; `preset eject`; repository-only samples |
| Example distribution | bundle a small runnable example in the wheel | repository examples only; plugin/entry-point discovery |
| Preset customization | keep presets complete and intentionally small | a few typed flags for model/workspace; no arbitrary `--set` |
| Custom source name | `--factory module:callable` | `--custom module:callable`; `--agent module:callable` |
| Interactive surface | retain `chat` for harness probing | ship only `plan`, `doctor`, and `run` |
| Output | JSON as the automation contract | explicit human-readable output mode |

## Boundaries

- No `agent.yaml` or profile directories.
- No Fabric config loader for YAML, TOML, or JSON.
- No preset inheritance or core merge semantics.
- No implicit user/project/system config discovery.
- JSON request input is still valid; it describes an invocation, not an agent.
- Adapter-generated harness files remain valid implementation details.
- Private serialization across Python/Rust or Harbor process boundaries is not
  a public authoring format.

## Current Bias

Start with the three explicit selectors. Keep presets tiny, make examples the
editable learning surface, and use Python factories for real customization.
Add scaffolding or plugin discovery only after the basic run path proves
useful.
