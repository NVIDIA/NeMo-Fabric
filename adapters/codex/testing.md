<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Testing the Codex Adapter

Run the unit and opt-in real SDK tests separately:

```bash
uv run pytest tests/adapters/test_codex_adapter.py -q
RUN_FABRIC_CODEX_INTEGRATION=1 uv run pytest tests/e2e/test_codex.py -q
RUN_FABRIC_CODEX_RELAY_INTEGRATION=1 \
  FABRIC_TEST_NEMO_RELAY_COMMAND=/path/to/nemo-relay \
  uv run pytest tests/e2e/test_codex.py -q
```

Set `FABRIC_TEST_CODEX_BIN=/path/to/codex` on either opt-in command to validate
an explicit app-server override instead of the SDK-pinned runtime.

The SDK test uses the current Codex authentication state and exercises both the
single-invocation convenience API and multiple turns against one started
runtime. The Relay test additionally requires an external gateway binary and
verifies model responses, stable thread identity across turns, ATOF, and ATIF;
gateway startup alone is not a passing result. The semantic regression also
requires decoded LLM request content, a model, token usage, and the expected
agent response in ATIF.
