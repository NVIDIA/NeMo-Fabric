# Skill Benchmark: nemo-fabric-build-adapter

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-fabric-build-adapter`
- Evaluation date: 2026-09-22
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (5 positive, 2 negative)
- Dataset digest: `sha256:19a320bb2ed23f27f8d70495a39ca9ba694d273b4906ad17e23f48ee82798cc3` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

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
| Overall | Not available | 70.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 65.0% → 42.9% (-22.1 points) |
| Correctness | Not available | 56.0% → 80.0% (+24.0 points) |
| Discoverability | Not available | 85.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 38.9% → 80.3% (+41.4 points) |
| Efficiency | Not available | 65.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 34,446,212 | 8,262,408 | N/A | N/A | skill 10/21; base 11/11 |
| claude-code | nemo-fabric-build-adapter-001-new-python-adapter | 19,903,828 | 4,052,394 | +15,851,434 | +391.16% | skill 3/3; base 3/3 |
| claude-code | nemo-fabric-build-adapter-002-native-openai-streaming | 68,230 | 94,897 | -26,667 | -28.10% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-build-adapter-003-consumer-integration-negative | 9,158,296 | 623,470 | +8,534,826 | +1368.92% | skill 2/2; base 2/2 |
| claude-code | nemo-fabric-build-adapter-004-first-party-maintenance-negative | 248,402 | 262,408 | -14,006 | -5.34% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-build-adapter-005-system-instruction-composition | 4,782,657 | 3,004,514 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | nemo-fabric-build-adapter-006-relay-request-correlation | 144,412 | 98,085 | +46,327 | +47.23% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-build-adapter-007-warm-session-continuation | 140,387 | 126,640 | +13,747 | +10.86% | skill 1/1; base 1/1 |
| codex | All cases | 10,413,655 | 14,336,674 | N/A | N/A | skill 7/7; base 10/10 |
| codex | nemo-fabric-build-adapter-001-new-python-adapter | 1,352,286 | 1,357,166 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nemo-fabric-build-adapter-002-native-openai-streaming | 82,357 | 74,279 | +8,078 | +10.88% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-003-consumer-integration-negative | 123,104 | 117,647 | +5,457 | +4.64% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-004-first-party-maintenance-negative | 7,948,560 | 12,276,783 | N/A | N/A | skill 1/1; base 2/2 |
| codex | nemo-fabric-build-adapter-005-system-instruction-composition | 571,897 | 313,925 | +257,972 | +82.18% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-006-relay-request-correlation | 259,525 | 83,253 | +176,272 | +211.73% | skill 1/1; base 1/1 |
| codex | nemo-fabric-build-adapter-007-warm-session-continuation | 75,926 | 113,621 | -37,695 | -33.18% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 44,859,867 | 22,599,082 | N/A | N/A | skill 17/28; base 21/21 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

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
