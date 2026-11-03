## Description: <br>
Use this skill when integrating NVIDIA NeMo Fabric into a consumer application, service, evaluation harness, or platform through the typed Python SDK — translating the consumer's own application, job, or deployment config into an in-memory FabricConfig, choosing the single-invocation convenience API or an explicitly started runtime, validating with plan and doctor, and consuming normalized results, artifacts, and telemetry. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to integrate NVIDIA NeMo Fabric into their own applications, services, evaluation harnesses, or platforms through the public Python SDK, mapping their configuration to a typed FabricConfig, selecting a runtime lifecycle, validating with plan and doctor, and handling normalized results, artifacts, telemetry, and errors. <br>

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
- [config-mapping.md](references/config-mapping.md) <br>
- [results-and-errors.md](references/results-and-errors.md) <br>
- [sdk-api-inventory.md](references/sdk-api-inventory.md) <br>
- [Python SDK guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/sdk/python.mdx) <br>
- [Installation guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/getting-started/install.mdx) <br>
- [NeMo Fabric overview](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/about-nemo-fabric/overview.mdx) <br>
- [Resource usage and stats](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/sdk/system-stats.mdx) <br>
- [Hermes integration guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/integrations/harness/hermes.mdx) <br>
- [Adapter catalog guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/sdk/python/nemo-fabric-adapter-catalog/README.md) <br>
- [API reference: client](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.client.md) <br>
- [API reference: runtime](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.runtime.md) <br>
- [code_review_agent example](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/code_review_agent) <br>
- [Harbor example](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/harbor) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline Python and shell code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (4 positive, 2 negative), 1 attempt per task, each run in an isolated k8s-sandbox pod; evaluator version 1.6.6, evaluated 2026-11-03. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? (security signal, 100%) <br>
- Correctness: Is the answer correct? (accuracy signal, 100%) <br>
- Discoverability: Was the right skill loaded when needed? (skill_execution signal, 100%) <br>
- Effectiveness: Did the skill help complete the task? (goal_accuracy 50% + behavior_check 50%) <br>
- Efficiency: Did it avoid wasted tool calls and token usage? (skill_efficiency 50% + token_efficiency 50%) <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 81.0% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Security | 83.3% → 100.0% (+16.7 points) | Not available |
| Correctness | 40.0% → 80.0% (+40.0 points) | Not available |
| Discoverability | 93.8% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Effectiveness | 34.7% → 58.1% (+23.4 points) | Not available |
| Efficiency | 73.3% — baseline ran, but no comparable score was available; uplift unavailable | Not available |

## Skill Version(s): <br>
0.0.0 (source: pyproject.toml; git SHA 199725f, committed 2026-11-03) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
