<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Claude Adapter Tests

The default suite uses deterministic mock Claude Code and Relay CLIs and
requires no credentials. Test a current `nemo-relay` CLI with the mock Claude
client, or run the live integrations on an authenticated developer host:

```bash
FABRIC_NEMO_RELAY_COMMAND="$(command -v nemo-relay)" uv run --no-sync pytest tests/e2e/test_claude.py -q -k real_relay_gateway
RUN_FABRIC_CLAUDE_INTEGRATION=1 uv run --no-sync pytest tests/e2e/test_claude.py -q -k live
RUN_FABRIC_CLAUDE_RELAY_INTEGRATION=1 uv run --no-sync pytest tests/e2e/test_claude.py -q -k live_claude_relay
```

The first command uses the mock Claude client and does not require credentials.
Set `FABRIC_TEST_CLAUDE_MODEL` to override the default live-test model,
`claude-sonnet-4-5`.
The live Relay test applies the same semantic artifact contract as Codex: ATOF
must contain structured LLM requests and token usage, and ATIF must contain the
expected agent response.
