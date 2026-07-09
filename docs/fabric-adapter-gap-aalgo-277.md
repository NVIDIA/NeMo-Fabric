# Fabric Adapter Gap Analysis (AALGO-277 Part A, Step 1)

> **Status:** Draft — code-verified 2026-07-06 against `NeMo-Fabric`, `nemo-platform`, and `NeMo-Agent-Toolkit`.
>
> **Scope:** Document Fabric adapter additions required for `react-optimize.yml` / `calculator-optimize.yml` parity and map each gap to AALGO-277 acceptance criteria. This is prerequisite runtime work (Part A); it does not implement adapters.

---

## 1. Summary

AALGO-277 numeric optimize cannot replace `nat optimize` until trial execution can run Fabric-backed ReAct agents with the same tools, model binding, evaluator semantics, and trace correlation as today's NAT path.

**Current Fabric adapters (verified):**

| Adapter | Path | Kind | Harness | Telemetry |
|---------|------|------|---------|-----------|
| Hermes SDK | `adapters/hermes-sdk/` | `python` (inline) | `hermes` | relay, atif, otel, openinference |
| Hermes CLI | `adapters/hermes-cli/` | `process` | `hermes` | relay, atif, otel, openinference |
| Codex CLI | `adapters/codex-cli/` | `process` | `codex` | relay (via common utils) |

**Missing for strict parity:**

| Adapter | Priority for AALGO-277 | Blocks |
|---------|------------------------|--------|
| `langchain-react` | **P0** | `react-optimize`, `calculator-optimize` E2E |
| Built-in parity tools (harness-owned) | **P0** | Same |
| OpenAI/IGW per-trial overrides | **P0** | `temperature` / `top_p` HPO |
| `tunable_rag_evaluator` parity (Evaluator) | **P0** | Trial scoring |
| `claude-code-cli` | P1 (OpenClaw follow-on) | OpenClaw parity only |
| `cursor-cli` | P2 (optional scope) | Cursor-backed agents |
| `langgraph` | P2 (BYO agents) | LangGraph wrapper users |

Generic Fabric `run()` on Hermes/Codex is **not** sufficient: the parity fixtures require a NAT-shaped ReAct loop, specific built-in tools, and a weighted judge scorer — none of which exist in Fabric today.

---

## 2. Existing Adapter Baseline

### 2.1 Descriptor contract

All adapters ship a `fabric-adapter.json` descriptor consumed by Fabric planning. Verified fields in use:

```json
{
  "adapter_id": "nvidia.fabric.<harness>.<surface>",
  "harness": "<harness-name>",
  "adapter_kind": "python | process",
  "runner": { "...": "module/callable or command/script" },
  "requirements": { "binaries": [], "env": [] },
  "config": { "accepts": ["models", "tools", "telemetry", ...] },
  "telemetry": { "supports": ["relay", "atif", ...] }
}
```

Reference implementations:

- **Inline Python:** `adapters/hermes-sdk/fabric-adapter.json` → `nemo_fabric_adapters.hermes_sdk.adapter:run`
- **Process stdin JSON:** `adapters/codex-cli/fabric-adapter.json` → `python3 adapter.py` with `FABRIC_INVOCATION` or stdin payload

Shared adapter utilities live in `adapters/common/src/nemo_fabric_adapters/common/utils.py`.

### 2.2 RunRequest / RunResult contract (no schema change needed)

Fabric already exposes the invocation surface required by optimize trials:

| Field | Location | Optimize use |
|-------|----------|--------------|
| `RunRequest.request_id` | `crates/fabric-core/src/runtime.rs` | Per-row correlation key |
| `RunRequest.context` | same | `{experiment_id, trial_number, row_id}` |
| `RunRequest.overrides` | same | Per-trial `temperature`, `top_p`, etc. |
| `RunResult.runtime_id` | `RunResult` | ATIF namespace (Relay) |
| `RunResult.invocation_id` | `RunResult` | Per-invocation evidence |
| `ArtifactManifest` | `RunResult` | ATIF/ATOF paths |

Relay ATIF output is already namespaced by `runtime_id` via `normalize_relay_output_dirs()` in common utils — no Fabric core change required for trial trace correlation.

### 2.3 Telemetry.extra (no Fabric schema change needed)

`AtifConfig.extra` is passed through `_relay_api_atif_config()` from profile/config `telemetry.atif.extra`. The harness/optimize plugin can set per-trial metadata (`experiment_id`, `trial_number`, optional `row_id`) in profile telemetry before dispatch. Optional Fabric convenience (auto-propagate `RunRequest.context` → Relay `extra`) is **not** required for acceptance criterion 5.

### 2.4 Deferred harnesses (Fabric MVP plan)

`POC-TO-MVP-PLAN.md` explicitly defers Codex, Claude Code, Cursor, OpenClaw, and Deep Agents until after the Hermes slice is stable. Codex CLI adapter exists but is not on the optimize hot path. AALGO-277 strict parity depends on **`langchain-react`**, not on completing all deferred harnesses.

---

## 3. Required Adapter Additions

### 3.1 `adapters/langchain-react/` (P0 — blocking)

**Purpose:** Run Fabric-native ReAct agents equivalent to NAT `workflow._type: react_agent` without NAT registration, Builder, or config models.

**Adapter descriptor (proposed):**

```json
{
  "adapter_id": "nvidia.fabric.langchain.react",
  "harness": "langchain-react",
  "adapter_kind": "python",
  "runner": {
    "module": "nemo_fabric_adapters.langchain_react.adapter",
    "callable": "run"
  },
  "requirements": {
    "env": []
  },
  "config": {
    "accepts": ["models", "tools", "telemetry"]
  },
  "telemetry": {
    "supports": ["relay", "atif", "otel", "openinference"]
  }
}
```

**Fabric agent package shape** (post `nat_to_fabric` conversion):

```yaml
schema_version: fabric.agent/v1alpha1
harness:
  adapter_id: nvidia.fabric.langchain.react
  settings:
    workflow:
      tool_names: [wiki, clock]           # or [calculator, current_datetime]
      llm_name: default
      parse_agent_response_max_retries: 3
      max_tool_calls: 15
      use_native_tool_calling: false      # true for calculator-optimize
    tools:                                # harness-resolved built-ins
      wiki: { kind: wiki_search }
      clock: { kind: current_datetime }
      calculator: { kind: function_group, group: calculator }
models:
  default:
    provider: openai
    model: nvidia-nemotron-3-nano-30b-a3b
    temperature: 0.0
    top_p: 1.0
telemetry:
  provider: relay
  atif:
    enabled: true
    extra:
      experiment_id: "<set per trial by harness>"
      trial_number: "<set per trial by harness>"
```

Exact nesting can follow `harness.settings` conventions used by Hermes; the adapter must read workflow + models from `effective_config`, not NAT YAML.

**Behavior to port from NAT (NAT-free extraction):**

| NAT source | Capability |
|------------|------------|
| `nvidia_nat_langchain/.../react_agent/agent.py` (`ReActAgentGraph`) | LangGraph ReAct loop |
| `.../react_agent/register.py` | Workflow config fields, recursion limit, history trim |
| `.../react_agent/output_parser.py` | Text-mode parsing, `FINAL_ANSWER_PATTERN` |
| `.../react_agent/prompt.py` | System prompt construction |

**Config fields that must be honored:**

| Field | react-optimize | calculator-optimize |
|-------|----------------|---------------------|
| `workflow.tool_names` | `[wiki, clock]` | `[calculator, current_datetime]` |
| `workflow.llm_name` | `llm` | `llm` |
| `workflow.parse_agent_response_max_retries` | 3 | 3 |
| `workflow.max_tool_calls` | default 15 | default 15 |
| `workflow.use_native_tool_calling` | false (default) | **true** |
| `llms.llm.temperature` | swept 0.0–0.8 | swept 0.0–0.8 |
| `llms.llm.top_p` | swept 0.5–1.0 | swept 0.5–1.0 |

**Per-trial model overrides:**

Before constructing the OpenAI-compatible chat client, the adapter (or `nemo_harness` caller) must merge `RunRequest.overrides` into the resolved model config:

```python
# Pseudocode — applied in adapter or harness react/llm.py
model_cfg = resolve_model(effective_config, profile)
if request.overrides:
    model_cfg = deep_merge(model_cfg, request.overrides)  # temperature, top_p, ...
client = ChatOpenAI(base_url=..., api_key=..., **model_cfg)
```

IGW `base_url` and `api_key` injection remain an **agents/platform facade** responsibility before Fabric dispatch (same as today's deploy-time injection into NAT configs).

**Output normalization:**

Return `RunResult.output` as a string (final agent answer) plus full message trace in artifacts when Relay is enabled. Evaluator `tunable_rag_evaluator` parity consumes `generated_answer` text per row.

**Acceptance tests:**

- Golden runs on converted `react-optimize` and `calculator-optimize` Fabric packages against `react-eval-data.json` / `calculator-eval-data.json`.
- Per-trial override changes `temperature`/`top_p` observable in LLM calls (mock or trace assertion).

---

### 3.2 Built-in parity tools (P0 — harness-owned, not a separate adapter)

These are **not** NAT function registrations. They ship as normal Fabric/harness tools resolved by `langchain-react` (or `nemo_harness/tools.py`).

| Tool | NAT source | Parity fixture |
|------|------------|----------------|
| `wiki_search` | `nvidia_nat_langchain/.../tools/wikipedia_search.py` | `react-agent.yml` → `functions.wiki` |
| `current_datetime` | NAT core `current_datetime` function | both fixtures |
| Calculator group | `add`, `subtract`, `multiply`, `divide`, `compare` | `calculator-agent.yml` → `function_groups.calculator` |

**Calculator note:** Platform example installs `nemo-agents-example-calculator` for NAT `function_groups.calculator`. Fabric path must bundle equivalent arithmetic tools without NAT entry points.

**Acceptance:** Agent answers wiki/datetime questions (react) and arithmetic questions (calculator) with the same tool surface as NAT baselines.

---

### 3.3 `adapters/claude-code-cli/` (P1 — OpenClaw parity)

**Purpose:** Mirror Codex process-adapter lifecycle for Claude Code CLI (OpenClaw).

**Reference implementations to mirror:**

| Pattern | Source |
|---------|--------|
| Process adapter lifecycle | `adapters/codex-cli/src/nemo_fabric_adapters/codex_cli/adapter.py` |
| NAT workflow semantics (subprocess, Relay) | `nemo-platform/plugins/nemo-agents/vendor/claude_code_agent_adapter/` |

**Proposed descriptor:**

```json
{
  "adapter_id": "nvidia.fabric.claude.code.cli",
  "harness": "claude-code",
  "adapter_kind": "process",
  "runner": {
    "command": "python3",
    "script": "src/nemo_fabric_adapters/claude_code_cli/adapter.py",
    "stdin_payload": "fabric_request"
  },
  "requirements": {
    "binaries": ["claude"]
  },
  "config": {
    "accepts": ["models", "telemetry"]
  },
  "telemetry": {
    "supports": ["relay", "atif"]
  }
}
```

**Fabric config mapping (mirror Codex):**

| Fabric field | Claude Code CLI |
|--------------|-----------------|
| `environment.workspace` | subprocess `cwd` / `--add-dir` |
| `harness.settings.permission_mode` | `--permission-mode` |
| `harness.settings.model` | `--model` |
| `harness.settings.sandbox` | sandbox / permission settings |
| `harness.settings.timeout_seconds` | invocation timeout |
| `models.default` | optional model override |
| `telemetry` | Relay hooks via common utils |

**Not required for AALGO-277 strict parity** (react/calculator optimize fixtures use ReAct, not Claude Code).

---

### 3.4 `adapters/cursor-cli/` (P2 — optional)

**Purpose:** Surface the vendored NAT Cursor adapter as a Fabric process adapter.

**Reference:** `nemo-platform/plugins/nemo-agents/vendor/cursor_agent_adapter/src/nat_cursor_agent_adapter/register.py`

**Proposed descriptor:**

```json
{
  "adapter_id": "nvidia.fabric.cursor.cli",
  "harness": "cursor",
  "adapter_kind": "process",
  "requirements": { "binaries": ["cursor-agent"] },
  "config": { "accepts": ["models", "telemetry"] },
  "telemetry": { "supports": ["relay", "atif"] }
}
```

**Fabric mapping:**

| NAT field | Fabric equivalent |
|-----------|-------------------|
| `working_directory` | `environment.workspace` |
| `mode` (`plan` / `ask`) | `harness.settings.mode` |
| `model` | `models.default.model` or `harness.settings.model` |
| `sandbox` | `harness.settings.sandbox` |
| `trust_workspace` | `harness.settings.trust_workspace` |
| `timeout_seconds` | `harness.settings.timeout_seconds` |

**Not required for AALGO-277 strict parity.**

---

### 3.5 `adapters/langgraph/` (P2 — BYO LangGraph)

**Purpose:** Load and invoke user-supplied compiled LangGraph agents.

**NAT source to port:** `nvidia_nat_langchain/src/nat/plugins/langchain/langgraph_workflow.py`

**Config surface:**

| Field | Behavior |
|-------|----------|
| `dependencies` | Add paths to `sys.path` before import |
| `graph` | `module.py:symbol` dynamic import |
| `env` | `.env` file or inline env dict |
| input | `messages` list → `ainvoke` |
| output | Normalize `messages` from graph state |

**Proposed descriptor:**

```json
{
  "adapter_id": "nvidia.fabric.langgraph",
  "harness": "langgraph",
  "adapter_kind": "python",
  "config": { "accepts": ["telemetry"] },
  "telemetry": { "supports": ["relay", "atif"] }
}
```

**Not required for AALGO-277 strict parity** (fixtures use `react_agent`, not `langgraph_wrapper`).

---

## 4. Cross-Cutting Concerns (Not Separate Adapters)

### 4.1 OpenAI / IGW model binding

| Concern | Owner | Notes |
|---------|-------|-------|
| `base_url`, `api_key` | Agents platform / IGW preflight | Injected before optimize job dispatch (today: deploy-time into NAT YAML) |
| `model_name` | Fabric `models.default` | From converted agent package |
| `temperature`, `top_p` | `RunRequest.overrides` | Applied per Optuna trial by harness before LLM construction |
| Judge LLM (`judge_llm`) | Evaluator | Not swept; used only in `tunable_rag_evaluator` scoring |

### 4.2 Evaluator: `tunable_rag_evaluator` compatibility (P0)

**Not a Fabric adapter** — Evaluator plugin scope. Required for trial objective values.

**NAT reference:** `nvidia_nat_langchain/.../eval/tunable_rag_evaluator.py`

**Semantics to preserve:**

- Judge prompt + optional `default_scoring: true` rubric
- Structured JSON output: `coverage`, `correctness`, `relevance` (0.0–1.0)
- Weighted `average_score` = `0.5*coverage + 0.3*correctness + 0.2*relevance` (fixture defaults)
- Row input: `question`, expected `answer` description, `generated_answer` from harness

**Existing Evaluator SDK metrics (verified in `nemo-evaluator`):**

| Metric | Approximate NAT overlap |
|--------|-------------------------|
| `answer_accuracy` | Partial — single score, not 3-component weighted |
| `context_relevance` | Different semantics (RAGAS) |
| `faithfulness` | RAGAS groundedness |
| `response_relevancy` | Closer to `relevance` but not identical rubric |

**Recommendation:** Add a dedicated Evaluator metric (e.g. `tunable_rag_average_score`) that ports NAT's prompt, JSON schema, weights, and `average_score` aggregation. Do not assume RAGAS metrics are drop-in replacements.

### 4.3 Trial trace tagging (plugin-owned)

| Mechanism | Required? | Owner |
|-----------|-----------|-------|
| `RunRequest.request_id` per row | Yes | `nemo_harness` |
| `RunRequest.context = {experiment_id, trial_number, row_id}` | Yes | `nemo_harness` |
| `telemetry.atif.extra = {experiment_id, trial_number}` | Recommended | Optimize/harness profile |
| Persist `{experiment_id, trial_number, row_id} → runtime_id, atif_path` | Yes | Jobs results sidecar |
| Fabric auto-propagate `context` → Relay `extra` | No (nice-to-have) | Fabric core |

Relay already writes ATIF under `artifacts/relay/<runtime_id>/` — correlation is a plugin concern.

### 4.4 NAT-to-Fabric converter (out of band)

User-run helper `nat_to_fabric(nat_config) -> FabricAgentPackage` converts legacy YAML once. Runtime APIs accept only `fabric.agent/v1alpha1`. Not part of adapter implementation but **blocks golden tests** for converted fixtures.

---

## 5. Acceptance Criteria Mapping

| AALGO-277 criterion | Fabric / harness dependency | Adapter(s) | Evaluator / other |
|---------------------|----------------------------|------------|-------------------|
| **1.** `react-optimize.yml` runs E2E | ReAct loop + wiki + datetime + IGW LLM + per-trial overrides | `langchain-react`, built-in tools | `tunable_rag_evaluator` parity, `evaluate_trial` (Part A step 3) |
| **1.** `calculator-optimize.yml` runs E2E | ReAct + `use_native_tool_calling: true` + calculator group | same | same |
| **2.** Trial count / best-trial scores match NAT | Deterministic execution (not adapter-specific) | Per-trial overrides must reach LLM | Same scorer semantics |
| **3.** Multi-objective Pareto | None beyond trial execution | Fabric returns row outputs | Evaluator returns all metric scores (Part B) |
| **4.** `sampler: grid` exhaustive coverage | None — Fabric executes each trial config | — | Optuna backend (Part B) |
| **5.** ATIF traces tagged per trial | `request_id`, `runtime_id`, artifact manifest, optional `telemetry.extra` | All adapters using Relay | Trial trace sidecar in Jobs results |
| Artifact filenames (`optimized_config.yml`, etc.) | Not Fabric adapters | — | Customizer + Jobs (Part B) |

**Strict parity minimum adapter set:** `langchain-react` + built-in tools + OpenAI override path + Relay refs. Claude Code, Cursor, and LangGraph adapters are **follow-on** unless product expands acceptance scope.

---

## 6. Implementation Order (Adapter Workstream)

```text
Phase A1a — langchain-react adapter skeleton + descriptor
Phase A1b — Port ReAct graph/LLM/output from NAT (no NAT imports)
Phase A1c — Built-in tools (wiki, datetime, calculator)
Phase A1d — RunRequest.overrides for temperature/top_p
Phase A1e — Relay/ATIF artifact refs in RunResult
Phase A1f — Golden tests with converted react/calculator packages

Parallel (P1, non-blocking for strict parity):
  claude-code-cli adapter (Codex pattern)
  cursor-cli adapter (vendored NAT port)
  langgraph adapter (langgraph_wrapper port)
```

---

## 7. Explicit Non-Goals (Adapter Layer)

- Prompt GA / numeric Optuna study loop (Customizer Part B)
- NAT subprocess or `nvidia-nat-config-optimizer` hot path
- Fabric schema changes for trial tagging (plugin-owned)
- Hypervolume Pareto selection
- Per-LLM default `OptimizableField` search spaces in Fabric
- Completing all harnesses listed in Fabric MVP deferrals before `langchain-react` ships

---

## 8. Open Questions

| # | Question | Impact |
|---|----------|--------|
| 1 | Can any existing Evaluator RAGAS metric exactly reproduce NAT `tunable_rag_evaluator` weighted `average_score`? | Scorer implementation choice |
| 2 | Final `harness.settings` nesting for workflow/tools in `fabric.agent/v1alpha1` | Converter + adapter config contract |
| 3 | Should `langchain-react` live under `NeMo-Fabric/adapters/` or co-locate with `nemo_harness`? | Repo layout (plan suggests both: adapter descriptor in Fabric, react runtime in harness) |
| 4 | ATIF attribute names for Intake (`nemo.optimizer.experiment_id`, etc.) | Trace query UX |

---

## 9. References

| Artifact | Path |
|----------|------|
| Plan (Part A step 1) | `.cursor/plans/fabric_optimize_bridge_7c242065.plan.md` |
| react-optimize fixture | `nemo-platform/plugins/nemo-agents/examples/react-agent/react-optimize.yml` |
| calculator-optimize fixture | `nemo-platform/plugins/nemo-agents/examples/calculator-agent/.../calculator-optimize.yml` |
| NAT ReAct agent | `NeMo-Agent-Toolkit/packages/nvidia_nat_langchain/.../react_agent/` |
| NAT tunable RAG evaluator | `NeMo-Agent-Toolkit/packages/nvidia_nat_langchain/.../eval/tunable_rag_evaluator.py` |
| NAT LangGraph wrapper | `NeMo-Agent-Toolkit/packages/nvidia_nat_langchain/.../langgraph_workflow.py` |
| Codex Fabric adapter | `NeMo-Fabric/adapters/codex-cli/` |
| Fabric RunRequest | `NeMo-Fabric/crates/fabric-core/src/runtime.rs` |
| Relay ATIF namespacing | `NeMo-Fabric/adapters/common/.../utils.py` → `normalize_relay_output_dirs` |
