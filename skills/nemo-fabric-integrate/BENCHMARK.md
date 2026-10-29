# Skill Benchmark: nemo-fabric-integrate

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-fabric-integrate`
- Validation status: `passed`
- Evaluation date: 2026-10-29
- Evaluator version: `1.6.4`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 6 evaluation tasks (4 positive, 2 negative)
- Dataset digest: `sha256:4e12e11bb429ad195c00ceab9cd53bbf6351f32173f03b0ae827531ba2441a9e` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nemo-fabric-integrate | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| nemo-fabric-integrate | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| nemo-fabric-integrate | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| nemo-fabric-integrate | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| nemo-fabric-integrate | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |

- Controls for `nemo-fabric-integrate`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 80.9% — baseline ran, but no comparable score was available; uplift unavailable | 73.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 66.7% → 58.3% (-8.4 points) |
| Correctness | 33.3% → 86.7% (+53.4 points) | 30.0% → 80.0% (+50.0 points) |
| Discoverability | 93.8% — baseline ran, but no comparable score was available; uplift unavailable | 85.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 33.6% → 52.0% (+18.4 points) | 24.6% → 71.8% (+47.2 points) |
| Efficiency | 72.2% — baseline ran, but no comparable score was available; uplift unavailable | 71.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 92.8% → 85.0% (-7.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 91.5% → 84.2% (-7.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,866,365 | 1,758,271 | +108,094 | +6.15% | skill 6/6; base 6/6 |
| claude-code | nemo-fabric-integrate-001-python-service | 220,122 | 143,355 | +76,767 | +53.55% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-002-evaluation-platform | 494,401 | 144,541 | +349,860 | +242.05% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-003-adapter-contribution-negative | 119,747 | 144,252 | -24,505 | -16.99% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-004-overview-negative | 81,363 | 70,825 | +10,538 | +14.88% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-005-streaming-boundary | 555,508 | 143,682 | +411,826 | +286.62% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-006-system-instruction-modes | 395,224 | 1,111,616 | -716,392 | -64.45% | skill 1/1; base 1/1 |
| codex | All cases | 2,729,291 | 327,655 | +2,401,636 | +732.98% | skill 6/6; base 6/6 |
| codex | nemo-fabric-integrate-001-python-service | 1,441,005 | 54,940 | +1,386,065 | +2522.87% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-002-evaluation-platform | 1,043,157 | 71,197 | +971,960 | +1365.17% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-003-adapter-contribution-negative | 55,455 | 70,523 | -15,068 | -21.37% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-004-overview-negative | 32,543 | 21,566 | +10,977 | +50.90% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-005-streaming-boundary | 69,978 | 55,244 | +14,734 | +26.67% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-006-system-instruction-modes | 87,153 | 54,185 | +32,968 | +60.84% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,595,656 | 2,085,926 | +2,509,730 | +120.32% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nemo-fabric-integrate/SKILL.md`)
- **MEDIUM** QUALITY/quality_reliability: MCP skill lacks connection/error guidance (`skills/nemo-fabric-integrate/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Large skill (6600 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/nemo-fabric-integrate/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nemo-fabric-integrate/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nemo-fabric-integrate/SKILL.md`)
- 7 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
