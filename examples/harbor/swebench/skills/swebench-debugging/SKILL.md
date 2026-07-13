---
name: swebench-debugging
description: Apply a focused reproduce-fix-test workflow to SWE-Bench repository tasks.
---

# SWE-Bench debugging

1. Read the task and locate the smallest relevant implementation and test surface.
2. Reproduce the reported behavior before editing when the environment permits it.
3. Make the narrowest implementation change that addresses the root cause.
4. Run the focused regression test first, then the nearest existing test module.
5. Inspect the final diff for generated files, debug output, and unrelated changes.

Do not modify tests merely to make an incorrect implementation pass.
