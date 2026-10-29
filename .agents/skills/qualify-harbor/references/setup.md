<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Interactive Harbor Setup

Use this conversation when starting a new qualification run. A supplied plan or explicit provider choice takes precedence; reuse known answers and existing authorization. Planning alone does not build images, spend provider credit, or launch a task.

## Choose How Models Are Supplied

Start with: **“Use one shared BYOM endpoint for compatible adapters, or individual provider credentials?”** Recommend the shared endpoint when the user already has one served. Explain that native-provider exceptions can be added afterward. Do not begin with command flags, 18 adapter forms, or a request for secret values.

Establish the adapter scope from the request. Offer all compatible adapters or a named subset if scope is missing. Keep every catalog row in the eventual plan, including adapters outside the selected subset. Derive version/source, execution host and task from existing context where possible; propose the current committed checkout and the runbook baseline when appropriate instead of asking users to supply Git hashes. Ask for a host or version decision when it cannot be inferred safely.

### One Shared BYOM Endpoint

Collect only missing connection details: reachable base URL, exact served model ID, supported API protocols, and credential environment-variable name. For example, ask for the URL and model first, then resolve authentication and protocol uncertainty. Reuse a known NIM/provider configuration instead of asking again.

A Chat Completions endpoint does not establish Responses or Anthropic Messages support. Check the pinned adapter documentation/model schema and the endpoint's documented capabilities or authorized preflight evidence. Keep uncertain protocols as “needs verification.” Do not send paid test prompts merely to complete the questionnaire.

If the user requested only compatible BYOM adapters, skip native-only/protocol-incompatible routes without requesting their keys again. Offer optional native exceptions in the preview only when broader coverage is wanted.

Some custom-model routes need metadata beyond a URL and key, such as protocol, token budgets or accepted input formats. First verify that the pinned adapter and public Harbor bridge can express the required fields, then ask only for missing shared metadata. If adapter docs, schema or bridge transport disagree, report a blocked contract mismatch rather than inventing plan fields, fabricated model aliases or zero prices. Pi gateway models are one case to check; a successful known-model profile does not prove arbitrary model registration.

Use the same endpoint, full model ID and credential reference wherever the adapter supports them. Select its real transport/provider, such as an explicit `openai` override for a vendor-prefixed model ID, rather than rewriting the served ID. Prefer the SDK bridge when the requested profiles require its provider, model-settings or workflow options; use the requested bridge when compatible and explain any required change before executing.

### Individual Provider Credentials

Ask which providers/models the user wants for the selected adapters. Group compatible adapters by provider so one credential reference can serve several adapters; permit per-adapter overrides. Show available credential **names and presence only** on the execution host, and ask for unresolved model/provider choices. Do not assume that a native account subscription provides a usable API key or that credentials on the local machine exist on a remote execution host.

For missing credentials, offer secure provisioning through the user's normal secret source or skipping the affected adapters. If skipping is already authorized, record the reason and continue. Do not repeatedly prompt for skipped providers. Label user-reported presence separately from an actual execution-host check. A populated environment variable proves availability, not validity or account access. In this mode, the global model/endpoint arguments are only a template for plan generation: configure every selected row explicitly so the seed provider/model cannot leak into unrelated adapters.

## Keep Secrets Outside The Conversation

Ask for variable names such as `MODEL_API_KEY`, `ANTHROPIC_API_KEY` or `KIRO_API_KEY`, never key values. Check only whether the named variable is nonempty in the environment that will launch Harbor, without displaying its value. Explain that Harbor forwards selected variables into disposable task containers; avoid forwarding unrelated credentials.

Use the existing managed environment transport. Plans, previews, commands shown to the user, settings, image builds, committed files and reports contain references only. Do not ask users to paste or upload keys. If a value arrives unsolicited, do not repeat it in the plan or summary.

For a confirmed unauthenticated local endpoint, explain the nonsecret placeholder required by clients and record that this is not authentication coverage. Do not replace a missing real provider credential with a dummy key or infer unauthenticated access from a missing variable.

## Resolve Adapter Routes For The User

Use the exact catalog and installed-version contracts, with the profiles in [the runbook](runbook.md). The table below identifies decisions, not a static allowlist; new adapters still come from the complete catalog.

| Route | Setup behavior |
| --- | --- |
| Direct compatible API | Apply the shared endpoint/key reference or the chosen provider profile. Verify the required protocol; in this gate Claude needs Anthropic Messages and Codex normally uses Responses. |
| Kiro | Native Kiro provider and advertised model only; `KIRO_API_KEY` is required. In BYOM mode, offer native credentials as an optional exception or skip. Never create an experimental BYOM bridge. |
| Droid custom BYOK | Prepare the supported native custom-model profile/image described in the runbook. Its profile can use the shared endpoint, but normalized `base_url` is not the route. Preserve its credential-reference contract and record the profile/image identity. |
| Remote Agent | Prepare the compatible HTTP profile in the runbook. Explain that HTTP-contract coverage does not demonstrate autonomous remote workspace/tool execution. |
| Registered workflow | Select a catalog target appropriate to the task, such as NOOA CodingAgent for a coding task, and validate its settings. Do not substitute a different adapter to make the row pass. Ask only when the target choice is genuinely ambiguous. |
| Runtime or budgeting prerequisites | Prepare a supported interpreter/source image where needed, such as Hermes. For an unpriced Mini-SWE model, make the cost-tracking opt-out and positive turn limit explicit in the preview; do not silently report zero cost or remove the bound. |

Compatibility, credentials and installation readiness are separate. A valid provider key does not satisfy a missing harness/profile; a supported BYOM profile does not prove that the endpoint implements the required protocol. Prepare supported profiles rather than treating missing setup as unsupported capability. A complete supported recipe makes a row eligible for preparation; an installed, checked image/profile establishes execution readiness. Do not label an unbuilt image or an unverified endpoint ready to run.

## Preview Coverage, Then Save The Plan

Summarize the selected package/source version, bridge, execution host, single task, provider cost estimate/basis and execution bounds. Include a compact table with one row per catalog adapter:

| Adapter | Model / route | Credential reference | Readiness / reason |
| --- | --- | --- | --- |
| A selected Chat Completions adapter | Shared model endpoint | `MODEL_API_KEY`: availability checked | Ready to run; endpoint/profile checks complete |
| Kiro without native credentials | Native provider required; BYOM unsupported | `KIRO_API_KEY`: missing | Skip; credential prerequisite |
| An adapter outside a requested subset | Outside selected scope | Not forwarded | Excluded |

The example is a layout, not a coverage result. Use “needs verification/setup” while prerequisites remain unresolved, “ready to prepare” for a complete recipe, and “ready to run” only after actual preparation and checks. Separate these from “skipped.” Ask only for unresolved choices, such as adding a native provider exception, provisioning a missing key, accepting a bounded unpriced-model policy, or proceeding with a subset. Reuse prior run/spending authorization; do not add another approval step when the concrete plan is already authorized. An unanswered required choice remains pending, not implicitly accepted.

A conversational preview can be incomplete. If the exact source/package identity, real task path/hash, model selections or required planning inputs are missing, show the preview and ask for the next missing input; do not invent placeholders or claim an executable plan was validated. Respect plan-only/no-call requests during this setup.

Once those inputs are available, generate `plan.json` using `qualify.py plan` and apply the chosen per-row profiles on the user's behalf. Preserve catalog records and rows, target bindings, the original task/verifier/resources, credential names, reasons, cost basis and bounds. Validate the final plan with the gate's existing `validate_plan` before preparation. Do not add new plan fields just to record this conversation or store secret values. Show the saved plan path and the resulting readiness/coverage counts. Human-facing “skip” retains the gate’s existing actions: `blocked` for missing prerequisites, `unsupported` for an unavailable route, and `excluded` outside the requested subset. Use `run` for selected rows with a complete preparation recipe; do not introduce a `skipped` plan action or launch them before preparation succeeds.

Once prerequisites and authorization are settled, follow the runbook to prepare and execute. A missing native key produces skipped coverage, not a Fabric integration defect; retain the gate's prerequisite-blocked outcome and do not claim a full-catalog pass. Keep model task reward and complete usage availability informational under the existing integration criterion. Replaying a configured plan should not restart onboarding.
