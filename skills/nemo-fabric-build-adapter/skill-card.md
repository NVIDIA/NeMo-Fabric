## Description: <br>
Build, migrate, review, and maintain third-party NVIDIA NeMo Fabric adapters against the public adapter contract. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers building, migrating, reviewing, or maintaining third-party NVIDIA NeMo Fabric adapters against the published southbound adapter contract. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NVIDIA NeMo Fabric Adapter Contract](https://github.com/NVIDIA/NeMo-Fabric/tree/main/docs/adapter-contract) <br>
- [Adapter Contract JSON Schemas](https://github.com/NVIDIA/NeMo-Fabric/tree/main/schemas/adapter-contract) <br>
- [LangGraph Custom-Agent Example](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/langgraph_custom_agent) <br>
- [NeMo Agent Toolkit Shared Adapter](https://github.com/NVIDIA/NeMo-Fabric/tree/main/external/nat) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (5 positive, 2 negative) with 3 attempts per task in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was selected and activated when needed. <br>
- Effectiveness: Whether the skill helped achieve the user's goal and expected workflow behavior. <br>
- Efficiency: Whether the skill avoided wasted tool calls and excessive token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 70.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 65.0% → 42.9% (-22.1 points) |
| Correctness | Not available | 56.0% → 80.0% (+24.0 points) |
| Discoverability | Not available | 85.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 38.9% → 80.3% (+41.4 points) |
| Efficiency | Not available | 65.2% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
da842c0 (source: git SHA, committed 2026-09-22) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
