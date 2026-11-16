<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# Fabric 0.7 Documentation Stack

All three phases target 0.7. Source baseline: upstream main `4678ba4606d91e842c016a97f5b721f7812e98e6`. The original discussion mockup is PR #611; this stack selectively carries the approved About and Getting Started changes and preserves current-main guide content elsewhere.

| Phase | Branch | Parent |
| --- | --- | --- |
| About NeMo Fabric | `docs/overview-refresh` | Upstream main |
| Getting Started | `docs/getting-started-refresh` | `docs/overview-refresh` |
| Consumers and Harnesses | `docs/consumer-harness-hierarchy` | `docs/getting-started-refresh` |

## Approved Content Changes

- About: reviewed Overview and Ecosystem rewrites; Supported Harnesses mirrors the README comparison and links to existing guides.
- Getting Started: landing pages, connected Quickstart tutorials, installation guidance, and source-setup migration. The first tutorial supplies an explicit empty tool allowlist for the Deep Agents requirements on current main; the skill/MCP continuation clears that policy before enabling Hermes tools.
- Source Card links stay relative to repository files. Publication resolves those links to version-relative routes. Explicit slug metadata preserves existing card and navigation routes. All moved installation heading anchors remain as compatibility links, and the Rust folder keeps its original /reference/api/rust-library-reference route.

## Existing Content and Destinations

| Source | Destination | Treatment |
| --- | --- | --- |
| `docs/sdk/python.mdx` | Consumers > Python SDK | Existing guide body unchanged; no SDK task-page split. |
| `docs/sdk/health-checks.mdx` | Existing SDK route, linked from the SDK and Troubleshooting | Body unchanged. |
| `docs/sdk/system-stats.mdx` | Existing SDK route, linked from the SDK and Troubleshooting | Body unchanged. |
| Existing consumer Harbor guide | Consumers > Consumer Integrations > Benchmarking & Evaluation > Harbor Framework | Body unchanged, including latest main's Execution Deadlines section. |
| Existing harness guides | Harnesses > Harness Integrations | Bodies unchanged; discovery table adds the NOOA documentation link. |
| Hermes and OpenClaw overview/interface pages | Existing paths, grouped under each harness | No content migration or rewrite. |
| Adapter Contract and its six tutorials | Harnesses > Adapter Contract | Bodies unchanged, including latest main's `config_schema` validation guidance. Numbered tutorial navigation; no additional overview child. |
| CLI: Install the CLI | Consumers > CLI (Experimental) > Install | Exact section extraction. |
| CLI: Experiment With Presets | Consumers > CLI (Experimental) > Presets | Exact section extraction; original command arguments retained. |
| CLI: Experiment With Examples | Consumers > CLI (Experimental) > Examples | Exact section extraction. |
| CLI: Scaffold Examples and its Python/Rust subsections | Consumers > CLI (Experimental) > Scaffold an Application | Exact section extraction. |
| CLI introduction and boundaries | CLI (Experimental) section landing | Preserved; former section anchors link to the extracted pages. |
| SDK API Inventory | Existing SDK section, linked from Reference > SDK API Inventory | No extraction or duplicated table. |
| SDK compatibility matrix | Existing SDK section, linked from Reference > Harness Compatibility | No extraction or rewritten support claims. |
| SDK contract-boundary/schema explanation | Existing SDK section, linked from Reference > Configuration and Schemas | No extraction or rewritten contracts. |
| Artifacts page | Reference > Packages and Distribution | Body and existing route retained. |
| Installation: Inspect Adapter Metadata Without Harness SDKs | Harnesses Overview subsection | Existing subsection moved intact; old installation anchor links to the destination. No standalone catalog page. |
| Installation: Isolated Task Environment | Install a Harness Adapter | Existing deployment explanation and commands retained. |
| Installation: source-build steps | Contributing > Development Setup and Building from Source | Setup migration needed by notebook/OpenShell routes; explicit current Node.js prerequisite. |
| Root `CONTRIBUTING.md` | Contributing Overview | Existing content published on the site; only source-link rebasing and MDX comment formatting. Root file unchanged. |
| Generated Python/Rust references | Reference | Content and source paths retained; no hand edits. |
| Release Notes and legal pages | Existing sections | Unchanged. |

## Minimal New Routing Entries

- Consumers Overview and Harnesses Overview route readers to existing guides.
- Deployments and Benchmarking & Evaluation are navigation groups, not additional pages.
- NemoClaw, Helix, Gym, and Polar entries link to upstream instructions and Ecosystem; they introduce no new local workflow procedures.
- NOOA links to its existing adapter README and Harbor example.
- Testing, Documentation Contributions, and Troubleshooting are brief indexes to existing guidance, with no newly authored commands or diagnostic examples.
- Every nested section with landing content uses the section itself; no duplicate Overview child. Explicit Overview children are reserved for top-level groups.

## Deferred From the Mockup

SDK splits/rewrites, merged or expanded health/stats guides, expanded consumer workflows, harness execution/streaming rewrites, additional adapter-authoring guidance, new diagnostic examples, and unrelated command/capability corrections are excluded.

## Validation

Each phase runs `just no_uv=true docs` with the existing local documentation environment. The publication helper has focused tests. Additional checks compare protected guide bodies against the baseline, compare all four migrated CLI sections exactly, parse authored Python snippets, and reject nested Overview children. Compiler-dependent generated-reference churn is discarded after validation.

Live model-backed tutorial execution and every-harness installation are outside this documentation reorganization's local checks.
