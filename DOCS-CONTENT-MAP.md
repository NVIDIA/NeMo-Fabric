<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# Fabric 0.7 Documentation Stack

All three phases target 0.7. Source baseline: upstream main `ec383ae67ddb6ed25eed848a497a0dd90ee7b72e`. The original discussion mockup is PR #611; this stack selectively carries the approved About and Getting Started changes and preserves current-main guide content elsewhere.

| Phase | Branch | Parent |
| --- | --- | --- |
| About NeMo Fabric | `docs/overview-refresh` | Upstream main |
| Getting Started | `docs/getting-started-refresh` | `docs/overview-refresh` |
| Consumers, Harnesses, and SDK Subsections | `docs/consumer-harness-hierarchy` | `docs/getting-started-refresh` |

## Approved Content Changes

- About: reviewed Overview and Ecosystem rewrites; Supported Harnesses mirrors the README comparison and links to existing guides.
- Getting Started: landing pages, connected Quickstart tutorials, installation guidance, and source-setup migration. The first tutorial supplies an explicit empty tool allowlist for the Deep Agents requirements on current main; the skill/MCP continuation clears that policy before enabling Hermes tools.
- Source Card links stay relative to repository files. Publication resolves those links to version-relative routes. Explicit slug metadata preserves existing card and navigation routes. All moved installation heading anchors remain as compatibility links, and the Rust folder keeps its original /reference/api/rust-library-reference route.

## Existing Content and Destinations

| Source | Destination | Treatment |
| --- | --- | --- |
| `docs/sdk/python.mdx` | Consumers > Python SDK | Introduction and first-run example retained on the section landing; all remaining main content moved intact into ten direct subsections. See the SDK preservation map below. |
| `docs/sdk/health-checks.mdx` | Consumers > Python SDK > Health Checks | Existing route and entire detailed guide retained; original SDK Runtime Health, Gateway Service Health, and Sandbox Command Bridge blocks appended. |
| `docs/sdk/system-stats.mdx` | Consumers > Python SDK > Resource Usage and Stats | Existing route and entire detailed walkthrough retained; original SDK resource snapshots, invocation stats, sampled peaks/means, and cgroup blocks appended. |
| Existing consumer Harbor guide | Consumers > Consumer Integrations > Benchmarking & Evaluation > Harbor Framework | Body unchanged, including latest main's Execution Deadlines section. |
| Existing harness guides | Harnesses > Harness Integrations | Individual guide bodies unchanged; the landing page adds the NOOA documentation link and the relocated metadata-inspection subsection. |
| Hermes and OpenClaw overview/interface pages | Existing paths, grouped under each harness | No content migration or rewrite. |
| Adapter Contract and its six tutorials | Harnesses > Adapter Contract | Bodies unchanged, including latest main's `config_schema` validation guidance. Numbered tutorial navigation; no additional overview child. |
| CLI: Install the CLI | Consumers > CLI (Experimental) > Install | Exact section extraction. |
| CLI: Experiment With Presets | Consumers > CLI (Experimental) > Presets | Exact section extraction; original command arguments retained. |
| CLI: Experiment With Examples | Consumers > CLI (Experimental) > Examples | Exact section extraction. |
| CLI: Scaffold Examples and its Python/Rust subsections | Consumers > CLI (Experimental) > Scaffold an Application | Exact section extraction. |
| CLI introduction and boundaries | CLI (Experimental) section landing | Preserved; former section anchors link to the extracted pages. |
| SDK API Inventory | Python SDK > Execution Model, linked from Reference > SDK API Inventory | Complete table moved once, without duplication. |
| SDK compatibility matrix | Python SDK > Configure Agents, linked from Reference > Harness Compatibility | Complete matrix moved once; support claims unchanged. |
| SDK contract-boundary/schema explanation | Python SDK > SDK Contract Boundaries, linked from Reference > Configuration and Schemas | Complete original block retained; contracts unchanged. |
| Artifacts page | Reference > Packages and Distribution | Body and existing route retained. |
| Installation: Inspect Adapter Metadata Without Harness SDKs | Harness Integrations > Inspect Adapter Metadata | Existing subsection body moved intact; installation links point directly to it, and the former Harnesses Overview anchor remains as a compatibility link. No standalone catalog page. |
| Installation: Isolated Task Environment | Install a Harness Adapter | Existing deployment explanation and commands retained. |
| Installation: source-build steps | Contributing > Development Setup and Building from Source | Setup migration needed by notebook/OpenShell routes; explicit current Node.js prerequisite. |
| Root `CONTRIBUTING.md` | Contributing Overview | Existing content published on the site; only source-link rebasing and MDX comment formatting. Root file unchanged. |
| Generated Python/Rust references | Reference | Content and source paths retained; no hand edits. |
| Release Notes and legal pages | Existing sections | Unchanged. |

## Minimal New Routing Entries

- Consumers Overview and Harnesses Overview route readers to existing guides.
- Deployments and Benchmarking & Evaluation use their own section landing pages with brief selection guidance and links to the existing integration entries. They add no setup procedures.
- NemoClaw, Helix, Gym, and Polar entries link to upstream instructions and Ecosystem; they introduce no new local workflow procedures.
- NOOA links to its existing adapter README and Harbor example.
- Testing, Documentation Contributions, and Troubleshooting are brief indexes to existing guidance, with no newly authored commands or diagnostic examples.
- Every nested section with landing content uses the section itself; no duplicate Overview child. Explicit Overview children are reserved for top-level groups.

The subsequently approved SDK split supersedes the original decision to keep one SDK page and hidden health/stats entries. This is a preservation-only extraction from main, not adoption of the SDK rewrites in #611.

## Deferred From the Mockup

SDK content rewrites, expanded consumer workflows, harness execution/streaming rewrites, additional adapter-authoring guidance, new diagnostic examples, and unrelated command/capability corrections are excluded.

## Python SDK Preservation Map and Later PR Notes

Baseline: upstream main `ec383ae67ddb6ed25eed848a497a0dd90ee7b72e` (verified again before implementation). Ranges below are inclusive lines in that commit, including original front matter and comments. All 2,120 lines across the three source files are assigned exactly once. Only front matter, necessary heading levels, link destinations, and brief navigation/setup pointers change; the technical prose, examples, commands, tables and diagrams remain intact. No SDK behavior or API changes.

The parent Python SDK opens its existing landing page. Its ten children are flat subsections with no intermediate chapter groups. The Python SDK section is collapsed by default. Health and stats keep their existing public routes. Every former Python SDK heading anchor remains on the landing page beside a link to the new destination; authored incoming links point directly to the relocated sections. Generated references are untouched and their old links continue to resolve. Reference routing entries point to the single relocated source table/block.

| Destination | Main source ranges | Preserved material |
| --- | --- | --- |
| Python SDK landing | `python.mdx:1–23`, `python.mdx:24–137`, `python.mdx:1710–1712` | Introduction, full first-run and separate-host metadata examples, Next Steps. Section landing, no Overview child. |
| Execution Model | `python.mdx:138–208`, `python.mdx:790–812` | Full execution-model diagram, five-object concepts table, hosting semantics, and API inventory table. |
| Configure Agents | `python.mdx:374–789`, `python.mdx:1503–1531`, `python.mdx:1552–1599` | Entire configuration block, compatibility matrix, environment forwarding, MCP authentication/headers, tools/skills, Relay/telemetry and file output, dynamic plugins, feature support and custom fields. Keep the interdependent examples together. |
| Requests and Runtimes | `python.mdx:237–266`, `python.mdx:868–901`, `python.mdx:902–1038`, `python.mdx:1406–1439` | Single/multi-turn execution, retained state, Hermes SessionRef resume, recorded runtime status and server states, ownership/recovery/cleanup/error behavior, application-owned parallelism. |
| Environments and Shared Services | `python.mdx:209–236`, `python.mdx:813–867`, `python.mdx:1532–1551` | Explicit environment lifecycle, service preparation and attachment examples, install/runtime responsibilities. |
| Streaming | `python.mdx:1232–1279`, `python.mdx:1280–1405` | Full native OpenAI and Relay streaming examples, constraints, backpressure, authentication and correlation. |
| Results and Errors | `python.mdx:1440–1489`, `python.mdx:1490–1502`, `python.mdx:1600–1632` | RunResult usage/cache/cost/correlation semantics, events, errors/deadlines/evidence and cleanup behavior. |
| Health Checks | `python.mdx:267–373`, `health-checks.mdx:1–104` | Entire detailed guide plus entire SDK health block, including diagram, runtime/service examples and sandbox command bridge. Visible direct child. |
| Resource Usage and Stats | `python.mdx:1039–1231`, `system-stats.mdx:1–273` | Entire stats walkthrough and collection-window diagram plus SDK snapshots, invocation stats, sampled peaks/means, native cgroup constraints, examples and caveats. Visible direct child. |
| Software Execution Evidence | `python.mdx:1713–1743` | Selected software inventories, collection bounds/lifetime, dispatch/shared services and calling-application reports. |
| SDK Contract Boundaries | `python.mdx:1633–1709` | Schemas/models, versioning, configuration extensibility, resilience, capacity and backpressure. |

### Notes to Carry Into the Subsequent PR

- Describe this as a content-preserving SDK split added to phase three, alongside the Consumers/Harnesses organization work. Do not describe the SDK, health, or stats material as unchanged in location or hidden navigation.
- Explicitly call out the retained execution-model diagram, concepts/API inventory tables, recorded runtime status, state retention, Hermes resume, lifecycle/error details, invocation stats, sampled peaks/means and cgroup examples.
- Health Checks and Resource Usage and Stats each retain their entire main guide and the supporting SDK sections/examples, rather than replacing either with an introduction or link.
- Preserve existing routes and old SDK heading links. Repository/source links are updated where content moved; generated API references use compatibility pointers. No generated-reference edits, product changes, or new consumer/harness procedures.
- Validation: exact source-range audit covers all 2,120 source lines; all 23 tables and 45 fenced blocks match main (3 Mermaid diagrams, 42 code/output blocks; 32 SDK Python snippets parse). `just no_uv=true docs` passed with 0 errors and 12 existing warnings (generated-reference comment parsing and the unavailable FDR redirect comparison). All three Mermaid diagrams parsed and rendered through Mermaid CLI. The generated-link and publication-helper tests passed: 18 tests. All 105 authored Python snippets parse; authored fragment checks and the combined preview strict broken-link check pass. Browser checks confirm all ten new routes, detailed health/stats headings, rendered diagrams, and legacy heading navigation; phase-one/two SDK previews retain their monolithic guide. Fresh independent review found no material issues. Generated-reference churn was discarded.
- No live model calls or all-harness execution are required for this text-preserving extraction. These examples retain the same prerequisites and behavioral limits as main.
- Keep these notes for the later PR body, using the repository PR template when a PR is actually requested. Nothing is pushed or published during local preview.

## Review Follow-up for the Later PR

- Rebased the entire stack onto main `ec383ae6` after #632 landed during publication preparation. Preserve its new Configure Skills guide, harness/adapter-contract updates, and complete SDK skills table, example, compatibility row, and caveats. The existing Configure Skills route and the additional former SDK heading anchor remain valid.
- Narrow the Overview provenance claim to selected software versions and available infrastructure software versions; keep the availability/coverage caveat.
- Replace obsolete landing-page/template assertions with checks of reader destinations, configured navigation, and published Card targets. The test works at each phase of the stack.
- Require whitespace-only suffixes on closing code fences in the Card-link publication helper. Six regression cases cover backticks, tildes, marker-like content, shorter/mismatched markers, and valid longer closers.
- Add descriptions to all 20 new pages identified by the review. Their existing guide and routing bodies remain unchanged.
- Shared fixes are committed in phase one and inherited by the later phases. Metadata additions stay in phase three, where those pages are introduced. The focused documentation/publication suite passes all 36 checks on each of the three phases. The final `just no_uv=true docs` build passes with zero errors and the same 12 existing warnings; the refreshed combined preview passes configuration and strict broken-link checks (only the FDR HTTP 403 redirect-comparison warning remains). The SDK preservation audit, all 105 authored Python snippets, 125 fragment links, and 438 repository links into docs pass. Generated-reference churn is discarded.

## Validation

Each phase runs `just no_uv=true docs` with the existing local documentation environment. The publication helper has focused tests. Additional checks compare protected guide bodies against the baseline, compare all four migrated CLI sections exactly, parse authored Python snippets, and reject nested Overview children. Compiler-dependent generated-reference churn is discarded after validation.

Live model-backed tutorial execution and every-harness installation are outside this documentation reorganization's local checks.
