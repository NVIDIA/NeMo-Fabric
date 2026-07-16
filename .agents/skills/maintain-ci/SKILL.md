---
name: maintain-ci
description: Maintain and review NeMo Fabric GitHub Actions workflows with minimum permissions, pinned action SHAs, deterministic caching, lockfile-backed tools, and local validation
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Maintain GitHub Actions CI

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when a change touches `.github/workflows/*.yml` or
`.github/workflows/*.yaml`, or when reviewing CI behavior for security,
reliability, or reproducibility.

## Standards

- Put `permissions:` on each job that needs token access.
- Avoid workflow-level permissions unless the repository intentionally centralizes
  them and the inheritance tradeoff is documented.
- Keep third-party actions pinned to full commit SHAs and preserve the readable
  version comment after the SHA.
- Prefer action-native or ecosystem-native caching over generic
  `actions/cache`.
- Use lockfiles or dependency manifests to drive cache invalidation.
- Keep deploy and publish permissions isolated to the jobs that need them.
- Read both caller and callee when a workflow uses `workflow_call`.
- Keep documentation publish and preview credentials isolated to the Fern docs
  workflow.
- Keep local commands aligned with the corresponding `justfile` recipes when
  they provide equivalent behavior.

## Permission Model

- `contents: read` is the default minimum for checkout-based build, test, docs,
  and packaging jobs.
- `pull-requests: read` is required for PR metadata lookup jobs.
- `pages: write` and `id-token: write` should be limited to Pages deployment
  jobs and any caller that invokes them through a reusable workflow.
- For reusable workflows, the caller must grant every permission the called
  jobs require. The callee cannot elevate beyond what the caller provides.

## Caching

- Prefer `astral-sh/setup-uv` cache support with `cache-dependency-glob`
  anchored to `uv.lock`.
- Prefer `Swatinem/rust-cache` with explicit `shared-key` and `workspaces`
  instead of ad hoc target-directory caching.
- Avoid caching generated outputs that can hide stale behavior unless the repo
  already relies on them deliberately.

## Review Checklist

- [ ] Each job has the minimum permissions it needs
- [ ] Reusable workflow callers grant only the scopes their callees require
- [ ] Every external action is pinned to a full SHA
- [ ] Cache settings are tied to lockfiles, manifests, or explicit tool versions
- [ ] Secrets are only passed to the jobs that consume them
- [ ] Python, Rust, and documentation jobs remain aligned with their lockfiles
      and `justfile` recipes
- [ ] Concurrency, branch filters, and documentation publish guards still
      reflect repository intent

## Validation

Start with the narrowest useful checks:

```bash
just --fmt --check
```

Use ripgrep to inspect the workflow graph before editing:

```bash
rg -n "uses:|permissions:|secrets:|concurrency:|cache|just " .github/workflows
```

If local lint passes but the question is whether GitHub will authorize the run,
inspect GitHub's permission model and the upstream action or reusable workflow
source instead of assuming local success proves remote success.

## Canonical References

- `.github/workflows/ci_python.yml`
- `.github/workflows/ci_rust.yml`
- `.github/workflows/fern-docs.yml`
- `Cargo.lock`
- `uv.lock`
- `docs/package-lock.json`
- `justfile`
- `maintain-packaging`
- `validate-change`
