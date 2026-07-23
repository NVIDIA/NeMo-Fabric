<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# POC Native-Tee Patches

Exact, `git apply`-able patches for the **temporary** native-tee seam each harness
findings file uses to capture `native-events.jsonl` (the SDK stream *before* Relay).
They exist so the captures are reproducible and non-destructive — apply, run, reverse
-apply — rather than hand-editing adapters and running `git checkout --` (which would
discard unrelated local edits).

| patch | adapter seam |
|---|---|
| `hermes-native-tee.patch` | wraps `hermes_cli.plugins.PluginManager.invoke_hook` after `discover_plugins` |
| `deepagents-native-tee.patch` | tees each `(namespace, mode, chunk)` in the `agent.astream` loop |
| `claude-native-tee.patch` | sets `include_partial_messages` and records each `StreamEvent` |
| `codex-native-tee.patch` | tees each `AsyncTurnHandle.stream()` notification |

Each is gated on `POC_NATIVE_RECORD`, so it is inert unless that variable is set.

```bash
git apply streaming-poc/patches/<harness>-native-tee.patch
# ... run the capture (see the harness findings.md) ...
git apply -R streaming-poc/patches/<harness>-native-tee.patch
```

These seams are POC-only instrumentation and must not ship in the adapters.
