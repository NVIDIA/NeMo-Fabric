## Description: <br>
Use this skill when integrating NVIDIA NeMo Fabric into a consumer application, service, evaluation harness, or platform through the typed Python SDK — translating the consumer's own application, job, or deployment config into an in-memory FabricConfig, choosing the single-invocation convenience API or an explicitly started runtime, validating with plan and doctor, and consuming normalized results, artifacts, and telemetry. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to integrate NVIDIA NeMo Fabric into their own applications, services, evaluation harnesses, or platforms through the public Python SDK, mapping their configuration to a typed FabricConfig, selecting a runtime lifecycle, validating with plan and doctor, and consuming normalized results, artifacts, and telemetry. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Mapping Consumer Config To FabricConfig](references/config-mapping.md) <br>
- [Results, Evidence, And Errors](references/results-and-errors.md) <br>
- [SDK API Inventory](references/sdk-api-inventory.md) <br>
- [NeMo Fabric Python SDK](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/sdk/python.mdx) <br>
- [NeMo Fabric Installation Guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/getting-started/install.mdx) <br>
- [NeMo Fabric Overview](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/about-nemo-fabric/overview.mdx) <br>
- [Hermes Integration Guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/integrations/harness/hermes.mdx) <br>
- [System Stats](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/sdk/system-stats.mdx) <br>
- [nemo_fabric.client API Reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.client.md) <br>
- [nemo_fabric.runtime API Reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.runtime.md) <br>
- [NeMo Fabric Adapter Catalog](https://github.com/NVIDIA/NeMo-Fabric/blob/main/sdk/python/nemo-fabric-adapter-catalog/README.md) <br>
- [code_review_agent Example](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/code_review_agent) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Shell commands] <br>
**Output Format:** [Markdown with inline Python and shell code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (4 positive, 2 negative), 1 attempt per task, each task attempt run in its own isolated sandbox pod in a `k8s-sandbox` environment; validation status `passed`; evaluator version `1.6.6`; dataset digest `sha256:4e12e11bb429ad195c00ceab9cd53bbf6351f32173f03b0ae827531ba2441a9e` (skill-evaluator-dataset-snapshot/1); evaluated 2026-11-04. Tier 2 and Tier 3 evidence required for publication. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

**Publication Recommendation:** Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

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

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 91.0% → 85.9% (-5.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 91.1% → 86.6% (-4.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Skill Version(s): <br>
0.7.0 (source: Cargo.toml workspace version; pyproject.toml reports placeholder 0.0.0; git SHA 446d887, committed 2026-11-04) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
