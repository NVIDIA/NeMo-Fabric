<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Codex streaming POC — stub (pending a usable model path)

**Harness:** `nvidia.fabric.codex` · gateway (Relay CLI) mode · part of the
Codex + Claude child POC.

**Status: pending a usable model path.** Codex runs against the OpenAI Responses
API through the Relay gateway. Every route to a real capture is currently blocked:

- **ChatGPT subscription / SSO** — auth itself **works** (no 401, no quota error;
  the same mechanism that unblocked [Claude](../claude/findings.md)), but the
  installed Codex CLI is **too old for the account's default model**
  (`gpt-5.6-sol` → *"requires a newer version of Codex — please upgrade"*), and the
  older ChatGPT-account models (`gpt-5-codex`, `gpt-5`, `codex-mini-latest`,
  `o4-mini`) are rejected as *"not supported when using Codex with a ChatGPT
  account."* Fix is a machine-side `npm install -g @openai/codex@latest`.
- **API key** — the available `OPENAI_API_KEY` is valid (`GET /v1/models` → 200)
  but **out of quota** (`429 insufficient_quota`).
- **NVIDIA inference bypass** — `inference-api.nvidia.com` (litellm-backed) rejects
  Codex's OpenAI-proprietary tool types (`type=namespace`, `type=web_search`) with
  `400`, and `config_overrides` did not strip them.

The relay-only `invoke_stream` prototype ([../common/](../common/README.md)) is
harness-agnostic and already validated on the other gateway path (Claude), so no
Codex-specific prototype work is pending — only a real run once one model path opens.

**To complete once a model path opens** (upgraded CLI, a funded key, or a
compatible endpoint): run the harness through `common/run_harness.py`, and capture
native evidence by teeing
`AsyncTurnHandle.stream()` — record every notification's `method` + `payload`
before the terminal-result collector — at
[adapters/codex/.../adapter.py:927](../../adapters/codex/src/nemo_fabric_adapters/codex/adapter.py#L927),
alongside the Relay listener. Then add `native-events.jsonl`, `events.atof.jsonl`,
and the native→ATOF mapping here.

Cross-harness recommendation: [../synthesis/README.md](../synthesis/README.md).
