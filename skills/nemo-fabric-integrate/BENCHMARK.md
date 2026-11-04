# Skill Benchmark: nemo-fabric-integrate

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-fabric-integrate`
- Validation status: `passed`
- Evaluation date: 2026-11-04
- Evaluator version: `1.6.6`
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
| nemo-fabric-integrate | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nemo-fabric-integrate | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nemo-fabric-integrate | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nemo-fabric-integrate | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nemo-fabric-integrate | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |

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
| Overall | 80.0% — baseline ran, but no comparable score was available; uplift unavailable | 74.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 83.3% (-16.7 points) | 100.0% → 75.0% (-25.0 points) |
| Correctness | 53.3% → 90.0% (+36.7 points) | 30.0% → 70.0% (+40.0 points) |
| Discoverability | 90.0% — baseline ran, but no comparable score was available; uplift unavailable | 87.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 43.0% → 63.9% (+20.9 points) | 31.8% → 63.2% (+31.4 points) |
| Efficiency | 72.9% — baseline ran, but no comparable score was available; uplift unavailable | 75.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 91.0% → 85.9% (-5.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 91.1% → 86.6% (-4.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,750,807 | 3,023,579 | +727,228 | +24.05% | skill 6/6; base 6/6 |
| claude-code | nemo-fabric-integrate-001-python-service | 2,239,195 | 909,068 | +1,330,127 | +146.32% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-002-evaluation-platform | 316,699 | 144,299 | +172,400 | +119.47% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-003-adapter-contribution-negative | 188,549 | 107,436 | +81,113 | +75.50% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-004-overview-negative | 81,420 | 70,663 | +10,757 | +15.22% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-005-streaming-boundary | 632,103 | 180,714 | +451,389 | +249.78% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-006-system-instruction-modes | 292,841 | 1,611,399 | -1,318,558 | -81.83% | skill 1/1; base 1/1 |
| codex | All cases | 1,647,781 | 306,543 | +1,341,238 | +437.54% | skill 6/6; base 6/6 |
| codex | nemo-fabric-integrate-001-python-service | 103,825 | 41,056 | +62,769 | +152.89% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-002-evaluation-platform | 1,269,692 | 71,369 | +1,198,323 | +1679.05% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-003-adapter-contribution-negative | 60,452 | 54,816 | +5,636 | +10.28% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-004-overview-negative | 36,561 | 30,286 | +6,275 | +20.72% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-005-streaming-boundary | 67,472 | 67,986 | -514 | -0.76% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-006-system-instruction-modes | 109,779 | 41,030 | +68,749 | +167.56% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,398,588 | 3,330,122 | +2,068,466 | +62.11% | skill 12/12; base 12/12 |

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
