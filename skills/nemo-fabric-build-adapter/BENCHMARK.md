# Skill Benchmark: nemo-fabric-build-adapter

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-fabric-build-adapter`
- Validation status: `passed`
- Evaluation date: 2026-10-27
- Evaluator version: `1.6.2`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 7 evaluation tasks (5 positive, 2 negative)
- Dataset digest: `sha256:b0644fe48c189bd6fcbf458a022b58e155d5b001adaf4b153c9ea36fd4cae5f7` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nemo-fabric-build-adapter | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| nemo-fabric-build-adapter | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| nemo-fabric-build-adapter | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| nemo-fabric-build-adapter | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| nemo-fabric-build-adapter | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |

- Controls for `nemo-fabric-build-adapter`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 77.4% — baseline ran, but no comparable score was available; uplift unavailable | 70.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 85.7% → 71.4% (-14.3 points) | 92.9% → 71.4% (-21.5 points) |
| Correctness | 42.9% → 91.4% (+48.5 points) | 42.9% → 88.6% (+45.7 points) |
| Discoverability | 92.0% — baseline ran, but no comparable score was available; uplift unavailable | 86.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 23.2% → 52.7% (+29.5 points) | 24.3% → 45.6% (+21.3 points) |
| Efficiency | 79.4% — baseline ran, but no comparable score was available; uplift unavailable | 61.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 94.0% → 84.7% (-9.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 88.7% → 81.7% (-7.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 9,863,255 | 1,694,860 | +8,168,395 | +481.95% | skill 7/7; base 7/7 |
| claude-code | nemo-fabric-build-adapter-001-new-python-adapter | 7,596,890 | 791,892 | +6,804,998 | +859.33% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-build-adapter-002-native-openai-streaming | 120,847 | 144,306 | -23,459 | -16.26% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-build-adapter-003-consumer-integration-negative | 110,541 | 143,607 | -33,066 | -23.03% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-build-adapter-004-first-party-maintenance-negative | 114,666 | 179,842 | -65,176 | -36.24% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-build-adapter-005-system-instruction-composition | 1,636,296 | 144,028 | +1,492,268 | +1036.10% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-build-adapter-006-relay-request-correlation | 163,905 | 181,959 | -18,054 | -9.92% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-build-adapter-007-warm-session-continuation | 120,110 | 109,226 | +10,884 | +9.96% | skill 1/1; base 1/1 |
| codex | All cases | 4,215,789 | 905,170 | +3,310,619 | +365.75% | skill 7/7; base 7/7 |
| codex | nemo-fabric-build-adapter-001-new-python-adapter | 3,704,195 | 529,094 | +3,175,101 | +600.10% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-002-native-openai-streaming | 68,277 | 97,142 | -28,865 | -29.71% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-003-consumer-integration-negative | 41,588 | 60,503 | -18,915 | -31.26% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-004-first-party-maintenance-negative | 48,099 | 54,560 | -6,461 | -11.84% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-005-system-instruction-composition | 111,300 | 54,549 | +56,751 | +104.04% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-006-relay-request-correlation | 158,162 | 54,998 | +103,164 | +187.58% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-007-warm-session-continuation | 84,168 | 54,324 | +29,844 | +54.94% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 14,079,044 | 2,600,030 | +11,479,014 | +441.50% | skill 14/14; base 14/14 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/nemo-fabric-build-adapter/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nemo-fabric-build-adapter/SKILL.md`)
- **MEDIUM** QUALITY/quality_reliability: MCP skill lacks connection/error guidance (`skills/nemo-fabric-build-adapter/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nemo-fabric-build-adapter/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nemo-fabric-build-adapter/SKILL.md`)
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
