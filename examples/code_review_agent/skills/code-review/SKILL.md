---
name: code-review
description: Review Python changes for correctness risks and missing test coverage.
---

# Code Review

1. Read the changed implementation and its nearest tests.
2. Identify correctness, error-handling, and compatibility risks before style issues.
3. Cite the smallest relevant file and line range for each finding.
4. Suggest the narrowest correction and a focused regression test.

If no correctness issue is present, say so directly and identify any remaining
test-coverage uncertainty.
