# Skill Benchmark: nemo-fabric-integrate

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-fabric-integrate`
- Evaluation date: 2026-10-08
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 6 evaluation tasks (4 positive, 2 negative)
- Dataset digest: `sha256:4e12e11bb429ad195c00ceab9cd53bbf6351f32173f03b0ae827531ba2441a9e` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
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
| Overall | 77.1% — baseline ran, but no comparable score was available; uplift unavailable | 72.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 83.3% → 83.3% (±0.0 points) | 66.7% → 58.3% (-8.4 points) |
| Correctness | 46.7% → 93.3% (+46.6 points) | 36.7% → 80.0% (+43.3 points) |
| Discoverability | 85.0% — baseline ran, but no comparable score was available; uplift unavailable | 77.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 34.7% → 60.1% (+25.4 points) | 37.1% → 61.3% (+24.2 points) |
| Efficiency | 63.7% — baseline ran, but no comparable score was available; uplift unavailable | 85.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 10,600,942 | 14,732,230 | -4,131,288 | -28.04% | skill 6/6; base 6/6 |
| claude-code | nemo-fabric-integrate-001-python-service | 784,956 | 7,448,869 | -6,663,913 | -89.46% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-002-evaluation-platform | 4,431,805 | 287,995 | +4,143,810 | +1438.85% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-003-adapter-contribution-negative | 452,804 | 278,069 | +174,735 | +62.84% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-004-overview-negative | 68,421 | 30,000 | +38,421 | +128.07% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-005-streaming-boundary | 4,169,583 | 2,561,457 | +1,608,126 | +62.78% | skill 1/1; base 1/1 |
| claude-code | nemo-fabric-integrate-006-system-instruction-modes | 693,373 | 4,125,840 | -3,432,467 | -83.19% | skill 1/1; base 1/1 |
| codex | All cases | 5,954,906 | 3,664,192 | +2,290,714 | +62.52% | skill 6/6; base 6/6 |
| codex | nemo-fabric-integrate-001-python-service | 98,533 | 148,007 | -49,474 | -33.43% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-002-evaluation-platform | 612,823 | 733,541 | -120,718 | -16.46% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-003-adapter-contribution-negative | 4,508,195 | 2,551,188 | +1,957,007 | +76.71% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-004-overview-negative | 18,169 | 17,947 | +222 | +1.24% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-005-streaming-boundary | 106,206 | 84,751 | +21,455 | +25.32% | skill 1/1; base 1/1 |
| codex | nemo-fabric-integrate-006-system-instruction-modes | 610,980 | 128,758 | +482,222 | +374.52% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 16,555,848 | 18,396,422 | -1,840,574 | -10.01% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

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
- **MEDIUM** QUALITY/quality_efficiency: Large skill (5805 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/nemo-fabric-integrate/SKILL.md`)
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
