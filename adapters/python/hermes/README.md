<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Hermes Agent Adapter

Start with [hermes.mdx](../../../docs/integrations/harness/hermes/hermes.mdx) for shared setup and an overview of all four invocation paths.

This adapter hosts Hermes Agent SDK, API, browser, and terminal invocation paths within one NeMo Fabric runtime. All four use normalized `FabricConfig`, translated into Hermes's native configuration and agent settings.

**Experimental terminal interface:** `interfaces.terminal` (including `terminal.port`), the `tui.json` / `tui.token` attachment files, and the `nemo_fabric_adapters.hermes.terminal` helper may change across releases as the [#579 launcher contract](https://github.com/NVIDIA/NeMo-Fabric/issues/579) is adopted. Compatibility is not guaranteed for these surfaces; see the [terminal guide](../../../docs/integrations/harness/hermes/hermes-tui.mdx).

## Execution Paths

The adapter supports the following client paths:

| Path | Execution backend | Guide |
| --- | --- | --- |
| SDK | Embedded Hermes `AIAgent` in the adapter host; enabled by default | [Hermes SDK](../../../docs/integrations/harness/hermes/hermes-sdk.mdx) |
| HTTP API | Managed native API gateway | [Hermes API](../../../docs/integrations/harness/hermes/hermes-api.mdx) |
| Dashboard/browser TUI | Managed TUI service with browser assets and a terminal bridge | [Hermes Dashboard](../../../docs/integrations/harness/hermes/hermes-dashboard.mdx) |
| Terminal TUI | Terminal renderer attached to the same TUI service; terminal-only mode runs without serving the browser dashboard | [Hermes Terminal TUI](../../../docs/integrations/harness/hermes/hermes-tui.mdx) |

Browser and terminal clients share one backend process and use separate conversations. Neither requires the API gateway.

Set `harness.settings.sdk.enabled` to `false` for native-only execution, with at least one native interface enabled. Your application can still start, check, and stop that runtime through the NeMo Fabric SDK. SDK invocation returns a failed `RunResult` with `hermes_sdk_disabled`, leaving native interfaces and health checks available.

## Runtime Health

`Fabric.check()` reports normalized `live`, `active`, and `ready` results for the SDK agent when enabled, or for the native backends when the SDK is disabled:

| Level | SDK Enabled (Including Mixed Runtimes) | SDK Disabled: Managed Native Backends |
| --- | --- | --- |
| `live` | Session/database ownership and fresh native diagnostics | Owned API/TUI processes have not exited |
| `active` | Available `run_conversation()` interface and fresh native diagnostics | Process checks plus API `/health` or TUI `/api/status`; TUI also requires an authenticated RPC |
| `ready` | A valid `get_activity_summary()` response after session/interface checks | The same native process and endpoint checks used for `active` |

Hermes's `/health` and `/api/status` are native endpoints. Startup probes every enabled interface. After startup, mixed-runtime health and `idle`/`busy` activity describe the SDK agent; a failed API/TUI child does not fail SDK runtime health. Probe native endpoints and interface logs separately to diagnose those services. Native-only runtimes check native backends and report activity as `unknown` because native turns are not fully aggregated.

Refer to the [SDK health-check overview](../../../docs/sdk/health-checks.mdx) and [Hermes API health guidance](../../../docs/integrations/harness/hermes/hermes-api.mdx#health-testing-and-limitations) for the distinction between native health and normalized checks.

## Install

The adapter package supports Python 3.11 through 3.14. Use [Hermes Agent v0.22.7](https://github.com/NousResearch/hermes-agent/releases/tag/v0.22.7) with Python 3.14 and Relay 0.11 on supported platforms. This release includes dependency security fixes absent from v2026.9.24. Hermes Agent 0.20 and later is not installable from PyPI. Follow the [Hermes Agent installation guide](https://hermes-agent.nousresearch.com/docs/installation), select v0.22.7, then install the NeMo Fabric packages into the same Python environment.

The following table shows which NeMo Fabric components each package expression
provides. None of these expressions installs Hermes Agent:

| Installation | Runtime | Adapter | Harness | NeMo Relay Python Package |
| --- | --- | --- | --- | --- |
| `pip install nemo-fabric nemo-fabric-adapters-hermes` | Yes | Yes | No | No |
| `pip install "nemo-fabric[hermes-agent]" "nemo-fabric-adapters-hermes[relay]"` | Yes | Yes | No | Yes |
| `pip install "nemo-fabric-adapters-hermes[full]"` | No | Yes | No | Yes |
| `pip install "nemo-fabric-adapters-hermes[relay]"` | No | Yes | No | Yes |
| `pip install nemo-fabric-adapters-hermes` | No | Yes | No | No |

NeMo Fabric tests Hermes Agent v0.22.7, including Relay telemetry and streaming, on Python 3.14. The adapter's `relay` and `full` extras accept `nemo-relay>=0.10,<0.12`; Hermes v0.22.7 requires `>=0.11,<0.12` on supported platforms. Hermes v0.22.5 with Relay 0.10 remains supported. Use a separate Python 3.14 Hermes environment when the application uses another Python version, and set `ADAPTER_PYTHON` to its interpreter. Hermes Agent declares its Python package version as `0.0.0`; identify the installed release by its Git tag and commit (`b200f5bc60e31588ba2cb60e288029c4a54ba5fa`) when reviewing scanner results. The adapter checks the Hermes integration and Relay version at startup when telemetry is enabled.

Create its isolated development environment:

```bash
just install-hermes-agent
export ADAPTER_PYTHON="$PWD/.venv-hermes/bin/python"
```

For split runtime and adapter environments, configure `ADAPTER_PYTHON` and use
matching NeMo Fabric release versions. Refer to the
[installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install#install-an-adapter-and-harness-without-the-runtime).

Hermes Agent v0.22.7 supports ordinary `Runtime.invoke()` calls, Relay telemetry, and `Runtime.invoke_stream()`.

The invocation paths have different additional requirements:

| Path | Requirements Beyond Shared Installation |
| --- | --- |
| SDK | No interface-specific packages or frontend builds; install dependencies required by selected tools and MCP servers. |
| HTTP API | `aiohttp` pinned by Hermes v0.22.5 (`3.14.5`), already included by `just install-hermes-agent`; see [API installation](../../../docs/integrations/harness/hermes/hermes-api.mdx#installation). |
| Dashboard/browser TUI | Python web-serving dependencies, built browser and terminal assets, and a supported Node runtime; see [dashboard installation](../../../docs/integrations/harness/hermes/hermes-dashboard.mdx#installation). |
| Terminal TUI | Python web-serving dependencies for the backend, plus a built terminal renderer and supported Node runtime for the client; terminal-only mode needs no browser assets. See [terminal installation](../../../docs/integrations/harness/hermes/hermes-tui.mdx#installation). |

Native interfaces require a POSIX host. Runtime startup does not build or install frontends. The [manual TUI example](../../../examples/interfaces/hermes/README.md#4-terminal-tui) provides the setup commands and interaction flow.

## What It Maps

The adapter receives a normalized payload from NeMo Fabric and materializes a native Hermes Agent configuration for:

- selected model provider, model name, base URL, temperature, `top_p`, and
  per-model `max_tokens` through `models`;
- replacement `instructions.system` and `runtime.max_turns`;
- workspace and explicit environment variables through `environment`;
- invocation timeout through `runtime.timeout_seconds`;
- `skills.paths` and `skills.directories` as Hermes Agent's
  `skills.external_dirs`, and `skills.bundled` as `skills.disabled`
  (see [Skills](#skills));
- NeMo Fabric MCP servers as Hermes Agent MCP server config;
- `tools.enabled` and `tools.blocked` as Hermes-native toolset selection and
  blocking policy;
- optional NeMo Relay telemetry plugin configuration.

Tool selectors are Hermes toolset names because that is the native policy
surface Hermes exposes.

The runtime starts even when some configured MCP servers are unavailable.
Hermes Agent connects each server separately at startup and keeps the tools
of the servers that connected.
Every result from the runtime reports the servers Hermes Agent currently lists
as failed under `output.mcp.unavailable_servers` (`name`, `transport`, and a
`reason` of `failed`), and the adapter writes a warning to standard error when a
server first becomes unavailable. The report reflects Hermes Agent's state
after each turn, so a server that Hermes Agent has since connected is no longer
reported, and a server that is still connecting is not reported. The adapter
does not cache failures: each new runtime connects every server again. Hermes
Agent records only an error message for a failed connection, which is omitted
because it can carry credentials, so the report has no `error_type` and does
not distinguish timeouts.
An OAuth server that cannot be authenticated still fails the invocation with
`hermes_mcp_authentication_failed`.

System instructions support `instructions.system.mode: replace`. Use `runtime.max_turns` to control the agent loop.

The descriptor validates the following `harness.settings` fields:

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `sdk.enabled` | boolean | `true` | Initialize the embedded Hermes `AIAgent`. Disable only with an enabled native interface; invocation then fails with `hermes_sdk_disabled` while runtime management remains available. |
| `reasoning_config` | object | `{"effort": "none"}` | Configures Hermes model reasoning. The closed object accepts an optional `enabled` boolean and an optional `effort` value of `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`. |
| `plugins_enabled` | array of nonempty strings | `[]` | List of native Hermes plugin identifiers to enable. Configure NeMo Relay through NeMo Fabric telemetry settings. |
| `save_trajectories` | boolean | `false` | Enables Hermes-native JSONL conversation trajectory saving. This is separate from normalized NeMo Fabric telemetry. |
| `max_tokens` | positive integer | Hermes native default | Used when the selected model has no explicit `max_tokens`. A model-level limit takes precedence; if both are omitted, Hermes uses its native default. |
| `terminal_timeout` | positive number | `60` | Limits a Hermes terminal tool operation in seconds; it does not enable a TUI. |
| `interfaces.api`, `interfaces.dashboard`, `interfaces.terminal` | object | None | Serve Hermes's API or shared dashboard/terminal TUI backend, with an exact `port` and optional `enabled`. Dashboard and terminal ports must match when both are enabled; API requires a distinct port; see [Hermes invocation paths](../../../docs/integrations/harness/hermes/hermes.mdx). |
| `startup_timeout_seconds` | number, greater than 0 and at most 75 | `60` | Time from the start of the runtime for every enabled interface to pass its health check. |
| `auxiliary.title_generation.enabled` | boolean | Hermes native default; `false` with Relay | Native automatic session-title generation. An explicit value wins over the Relay default. |
| `auxiliary.background_review.enabled` | boolean | Hermes native default | Native automatic background memory and skill review after turns. Explicit native `/refine` remains available. |
| `tools.tool_search.enabled` | `auto`, `on`, `off`, or boolean | Hermes native default | Native tool search; `true` and `false` are aliases for `on` and `off`. `off` presents the selected tool schemas directly. |

For example:

```python
from nemo_fabric import HarnessConfig

harness = HarnessConfig(
    adapter_id="nvidia.fabric.hermes",
    settings={
        "reasoning_config": {"enabled": True, "effort": "medium"},
        "plugins_enabled": ["disk-cleanup"],
        "save_trajectories": True,
        "max_tokens": 1024,
        "terminal_timeout": 90,
    },
)
```

### Native Harness Controls

The `auxiliary` and `tools` settings use Hermes's native `config.yaml` structure. The adapter writes only the fields you set, before any SDK agent or native interface starts, so all four execution paths read them. Omitting a field keeps the current behavior. For title generation, an explicit value always wins; when it is omitted, a Relay-enabled runtime keeps writing `false` and other runtimes keep Hermes's default. `harness.settings.tools` is Hermes's native tools section and is separate from NeMo Fabric's normalized top-level `tools.enabled` and `tools.blocked`.

These controls reduce automatic background model calls. They do not stop context compaction calls and do not correlate concurrent sessions; refer to the [Hermes guide](../../../docs/integrations/harness/hermes/hermes.mdx#native-harness-controls).

## Skills

[Configure Skills](../../../docs/integrations/harness/skills.mdx) describes
the `skills` fields. `skills.paths` is passed to Hermes Agent's
`skills.external_dirs` as written, without checks. `skills.directories` is
checked at startup and reported, and can't be combined with `skills.paths`.
Mount permissions decide whether the agent can change a skill. The mount or
file system enforces read-only access, not NeMo Fabric, which doesn't change
permissions. On a read-only mount, Hermes's write fails and Hermes returns the
error to the agent as the tool's result:

```yaml
skills:
  directories:
    - folder: /sandbox/team-skills          # a folder of skills, read-only mount
    - folder: /sandbox/project-skills       # a folder the agent edits, read-write mount
    - skill: /opt/skills/release-review     # one chosen skill
  duplicates: first_wins                    # optional: reject, first_wins, last_wins
  bundled:
    exclude: [xurl]                         # optional bundled-skill selection
```

Before you mount one read-write folder into several sandboxes, read
[Sharing a Writable Skills Volume](../../../docs/integrations/harness/skills.mdx#sharing-a-writable-skills-volume).

Hermes-specific behavior:

- Hermes loads every `SKILL.md` below an entry at any depth, starting at the
  entry's own folder and including skills nested inside another skill's
  folder. It skips folders such as `.git`, `node_modules`, `venv`, and
  `site-packages`, and a skill's own `references`, `templates`, `assets`,
  and `scripts` folders.
- Without `duplicates`, Hermes refuses two different skills with the same
  name, so neither loads. It also refuses a skill whose name is another
  skill's folder name; the report lists that skill under `skipped` with
  reason `ambiguous`. With `first_wins` or `last_wins`, the adapter passes
  only the selected skill folders; start fails if a dropped copy is nested
  inside a selected skill.
- When one entry lies inside another, such as a `skill` entry inside a
  `folder` entry, the adapter passes only the outer folder, because Hermes
  refuses a skill it reaches through two folders.
- A skill in the Hermes home's `skills` folder, bundled or written by the
  agent, outranks a configured skill with the same name. Hermes's
  `skill_manage` tool writes every new skill there, never into a
  `skills.directories` entry, because the adapter doesn't set Hermes's
  `skills.create_dir`. With `runtime.keep_state` the home is the agent's state
  folder, so the next runtime for the same agent loads those skills. Hermes
  edits a configured skill in place when its mount is writable; on a
  read-only mount the edit fails and the file is unchanged.
- An SDK-only runtime starts with no bundled skills in a new home: the
  embedded SDK agent doesn't copy them. Runtimes with the API, dashboard, or
  terminal interface get Hermes's own copy at start. Whenever `skills.bundled`
  is set, the adapter copies them at startup, as Hermes does. The copy always
  includes `hermes-agent`, which Hermes always loads.
- Because Hermes always loads `hermes-agent`, NeMo Fabric rejects a bundled
  selection that leaves it out: `include` must name it (so `include: []`
  fails), `exclude` can't name it, and `enabled: false` isn't supported (the
  descriptor sets `supports_disable: false`). Use `include: [hermes-agent]`
  for the fewest bundled skills. A rejected `include` or `exclude` fails
  startup with `hermes_skill_invalid` and the reason
  `essential_bundled_skill`.
- `skills.bundled` hides unselected bundled skills by name through Hermes's
  `skills.disabled`, which hides every skill with that name. Names that a
  `skills.paths` or `skills.directories` skill provides stay enabled, and
  Hermes doesn't copy those bundled skills. Names of the agent's own skills in
  the Hermes home, such as skills it wrote in a kept state folder, also stay
  enabled. Hermes tells them apart from bundled copies by its
  `skills/.bundled_manifest` file. Hermes never copies a bundled skill over
  the agent's own skill in the same place. When the agent's skill is
  elsewhere in the home, Hermes copies the bundled skill beside it and
  refuses both as ambiguous; the adapter warns at start.

With `skills.directories` or `skills.bundled`, every result carries
`output.skills` (`loaded`, `empty_entries`, `skipped`, `duplicates`, and
`bundled`). `duplicates` lists every name that more than one skill folder
Hermes sees provides, including a bundled copy and the agent's own skill in
the Hermes home with no configured copy, with each copy's origin and path and
Hermes's outcome. `bundled.selected` lists the selected bundled skills that load,
`bundled.unavailable` lists selected skills that don't with a reason, and the
Hermes-specific `bundled.essential` lists the skills Hermes always loads,
which `selected` also includes. Failures stop startup with
`hermes_skill_not_found` or `hermes_skill_invalid`, with `field`, `reason`,
and `path` or `name` metadata. The
[Hermes guide](../../../docs/integrations/harness/hermes/hermes.mdx#skills)
covers each case.

## Configuration Mapping and Ownership

Managed native processes use the adapter's launcher to apply explicit instructions and behavioral settings before constructing agents. Explicit enabled tools, including an empty list, override native selection; a nonempty blocked list overrides native exclusions. Unspecified tool policies can differ between SDK and native paths.

The default Hermes home is `<artifact root>/.fabric/hermes/runtimes/<runtime-id>`. With `runtime.keep_state`, the adapter uses the agent state folder described in the [Hermes guide](../../../docs/integrations/harness/hermes/hermes.mdx#shared-runtime-home). The adapter writes native configuration and optional NeMo Relay plugin configuration there. When native interfaces are enabled, an owner-only `interfaces/agent-config.json` snapshot supplies the constructor mapping and is removed on stop. Model credentials are provided through child environments rather than public attachment metadata.

The adapter leaves the installed Hermes source unchanged. Independently started processes do not automatically receive its mapping. Refer to [Shared Configuration and Ownership](../../../docs/integrations/harness/hermes/hermes.mdx#shared-configuration-and-ownership) for precedence and managed API gateway scope.

## Cold Resume

`SessionRef` resume requires `sdk.enabled: true`, which is the default. The run plan reports the adapter's declared `session_resume` capability. Native-only runtimes can retain their Hermes home with `runtime.keep_state` and start without issuing a session reference. Hermes rejects explicit SDK session selection in native-only mode with `session_resume_unsupported`.

With `runtime.keep_state`, save `Runtime.session_ref.to_mapping()` in caller-owned
storage. Pass `session_ref=SessionRef.from_mapping(saved_reference)`
to `Fabric.start_runtime()` to reopen the exact native conversation. The caller
must also retain the full artifact state tree, including NeMo Fabric's private
relative bindings in `.fabric/state/nvidia.fabric.hermes/.sessions`. A corrected
agent name can select the same reference. A different adapter cannot. A missing or incompatible session returns an explicit error.

The adapter uses Hermes's native resume checks, conversation projection, and
reopen operation. Hermes owns native identifiers and compression segments.
NeMo Fabric supplies the current startup configuration and instructions and creates
a new runtime ID. Hermes can restore its saved system prompt and pinned tools.
NeMo Fabric does not override those native resume rules. `RunResult.session_ref`
confirms the selected conversation.

An uncertain earlier turn produces a `session_recovery_required` startup
warning. NeMo Fabric preserves history, including unfinished requests. The model
may act on those requests after acknowledgment. Inspect the
native and external state before issuing a new `RunRequest` with
`acknowledge_recovery=True`. This gate applies to NeMo Fabric-controlled invocations.
NeMo Fabric does not change native API or dashboard session selection or behavior.

The state lock remains held while shutdown is unconfirmed. On Unix, native API
and dashboard processes inherit the lock so a surviving writer still prevents
replacement after its host exits. Retry cleanup after a caller timeout and confirm that the runtime has stopped. See the [SDK resume guide](../../../docs/sdk/python.mdx#resume-a-hermes-conversation)
for selection, recovery, and storage details.

## Execution Model

The runtime owns the enabled execution backends:

- **SDK agent:** Enabled by default and initialized during startup with its parent `SessionDB` and SDK-owned MCP connections. Ordered SDK invocations reuse the agent and conversation until stop closes them. Setting `sdk.enabled: false` skips these resources.
- **API gateway:** Starts when `interfaces.api` is enabled. Clients use the per-start `interfaces/api.token` with `Authorization: Bearer <token>` and receive native HTTP responses.
- **Shared TUI service:** Starts in dashboard mode when `interfaces.dashboard` is enabled, or headless `hermes serve` mode for terminal-only use. Browser and terminal clients attach to the same service with separate conversations. Terminal attachment uses private `interfaces/tui.json` and `interfaces/tui.token` files; the helper runs the prebuilt renderer without starting another backend.

SDK invocations return normalized NeMo Fabric `RunResult` objects. API, browser, and terminal responses remain native. With NeMo Relay enabled, the adapter finalizes and flushes SDK invocation evidence and managed native turn segments while preserving conversation history. Standard ATOF/ATIF formats do not imply normalized native results or complete session-scoped artifact collection.

Detaching a client leaves the runtime running. Runtime stop shuts down its owned native services, removes private attachment files and the configuration snapshot, and closes any embedded SDK resources. Sharing persisted files does not establish cross-runtime resume.

See [Limitations and Planned Work](../../../docs/integrations/harness/hermes/hermes.mdx#limitations-and-planned-work) for approvals, active-turn detach behavior, remote attachment, process cleanup, and native activity/inventory boundaries.

## Maintaining The Adapter

Keep `hermes.fabric-adapter.json` aligned with the Python implementation:

- `contract_version` must match the adapter contract supported by NeMo Fabric core.
- `adapter_id` is the stable id selected by `harness.adapter_id`.
- `adapter_kind` is `python` because NeMo Fabric can invoke it through Python.
- `runner.module` names the persistent host module that NeMo Fabric invokes with
  `python -m`.
- `requirements` supplies dependency checks to NeMo Fabric diagnostics; keep
  required env vars, binaries, or packages current.
- `config.accepts` must match the NeMo Fabric sections this adapter maps into Hermes Agent.
- `telemetry.providers` declares provider-specific outputs and integration modes
  the adapter can produce or forward.

Do not put end-user agent settings in this directory. Users vary harness,
model, skills, MCP, tools, telemetry, and runtime behavior through complete
typed `FabricConfig` values and ordinary Python composition. The adapter
descriptor describes adapter capabilities; it is not an agent configuration.
Add descriptor fields only when NeMo Fabric core or the SDK actually uses them.
