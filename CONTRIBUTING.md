<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to NeMo Fabric

Thank you for your interest in contributing to NeMo Fabric. This guide covers
the development workflow, coding standards, and pull request process.

## Development Setup

This section collects the setup steps needed before building, testing, or
contributing changes.

### Package Installation

NeMo Fabric is not currently available on PyPI. To consume the Python packages,
build wheels from a source checkout:

```bash
just wheels
uv pip install --find-links dist "nemo-fabric[runtime]"
```

Adapters are distributed as optional extras. For example, install the Hermes
SDK adapter with:

```bash
uv pip install --find-links dist "nemo-fabric[adapters-hermes-sdk]"
```

Refer to the [installation guide](docs/getting-started/install.mdx) for the
complete list of adapters and installation options.

### Source Development

Install these tools before you start:

- **Rust** (stable toolchain) -- install with [rustup](https://rustup.rs/)
- **Python** >= 3.11
- **uv** -- follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- **just** >= 1.50.0 -- `cargo install just --locked`

Clone the repository, create a virtual environment, and build the Rust and
Python packages:

```bash
git clone https://github.com/NVIDIA/NeMo-Fabric.git
cd NeMo-Fabric

uv venv --seed .venv --python 3.13
source .venv/bin/activate
uv sync --all-groups --all-extras
just no_uv=true build-all
```

Verify the checkout by running the test suites described in
[Testing Requirements](#testing-requirements).

## Release Tagging

Versioned release tags must use raw Rust-compatible SemVer without a leading
`v`.

- Use `0.1.0` for stable releases.
- Use `0.1.0-rc.1` for prereleases.
- Do not create tags such as `v0.1.0` or `v0.1.0-rc.1`.

This keeps release tags aligned with Cargo package versions and lets the release
tooling translate the version consistently for Python packages.

## Code Style

These style requirements keep contributions consistent across Rust, Python,
and general repository files.

### Rust

Use these commands and conventions when changing the core runtime, CLI, or
native Python extension:

- **Formatting**: `cargo fmt --all`
- **Format check**: `cargo fmt --all -- --check`
- **Compilation check**: `cargo check --workspace --locked`

### Python

Follow the existing style in the Python SDK, adapters, examples, and tests.
Use type annotations for public APIs and keep native binding declarations in
sync with their Rust implementations.

### General

Use the naming conventions appropriate to each language: Rust and Python use
`snake_case` for functions and variables, Rust types use `PascalCase`, and
Python classes use `PascalCase`.

## Testing Requirements

**Run tests for every language surface affected by your changes.** If a change
touches the Rust core or public schemas, run both the Rust and Python suites
because the Python SDK and adapters depend on the native core contract.

Run the affected test targets through the repository `justfile`:

```bash
# Rust workspace
just test-rust

# Python SDK, adapters, integrations, and examples
just test-python

# Both suites
just test-all
```

If the virtual environment is already synchronized, use `no_uv=true` to avoid
reinstalling dependencies:

```bash
just no_uv=true test-python
just no_uv=true test-all
```

When adding functionality, include tests in the corresponding Rust crate or in
the relevant area under `tests/`. Public contract changes must keep the checked-in
JSON Schema snapshots and native Python binding declarations synchronized.

## Documentation Checklist

If your change affects public behavior, adapters, examples, or workspace
structure, update the corresponding documentation in the same branch.

Before opening a PR, check the following:

1. Confirm that `README.md` reflects the current workspace, supported adapters,
   and top-level documentation.
2. Update the relevant SDK or API reference docs for public API changes.
3. Update the relevant adapter or example `README.md` when that surface changes.
4. Update embedded examples, integration docs, and adapter-support notes to
   reflect the current behavior.
5. For docs site changes, run `just docs` to regenerate the Python and Rust API
   references and validate the Fern configuration.

For documentation-heavy changes, prefer small targeted commits so the history
clearly separates entry-point changes, reference changes, examples, and
maintenance updates.

## DCO Sign-Off

Every commit in a pull request must include a Developer Certificate of Origin
sign-off.

Use the `-s`/`--signoff` flag with `git commit` when creating commits, or add a `Signed-off-by:` trailer when
fixing an older commit before review.

## Pull Request Process

This section describes how to prepare and submit changes for review.

### Before Submitting

Complete these checks before opening or updating a pull request:

1. Open or identify an issue describing the proposed enhancement or bug fix.
   External contributors should use a GitHub issue; NVIDIA contributors may use a GitHub or Linear issue.
2. Run the relevant test suites and confirm they pass.
3. Verify the affected packages compile with `just build-rust`,
   `just build-python`, or `just build-all`.
4. Update the relevant documentation entry points and references.
5. Rebase your branch on the latest `main` to avoid merge conflicts.

### PR Description

Complete the pull request template:

- **Overview**: Summarize the change and explain why it is needed.
- **Where should the reviewer start?**: Point to the most important file, test,
  or design decision.
- **Related issues**: Link the issue with the appropriate action keyword.
- **Testing**: List the checks you ran and any tests you added.
- **Breaking changes**: Call out public API or configuration changes that affect
  existing users.

### Review Expectations

- All PRs require at least one approving review before merge.
- Reviewers may request changes for code quality, test coverage,
  documentation, or design concerns.
- Address review feedback by pushing additional commits; do not force-push
  during review.
- CI must pass before merging.

## Commit Message Conventions

Use the following format for commit messages:

```text
type: short description of the change

Optional longer description explaining the motivation and context.
```

Valid types:

| Type | Purpose |
|------|---------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `test` | Test additions or modifications |
| `refactor` | Code restructuring without behavior changes |
| `chore` | Build, CI, or tooling changes |
| `perf` | Performance improvements |

Examples:

```text
feat: add typed runtime configuration overlays
fix: preserve adapter errors in run results
docs: clarify Hermes SDK adapter installation
test: cover concurrent Python runtime invocations
```

Keep the first line under 72 characters. Use the body for additional context
when the change is not self-explanatory.

## SPDX License Headers

All source files must include an SPDX license header. Use the appropriate
comment syntax for the file type.

**Rust:**

```rust
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
```

**Python:**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

**HTML / Markdown:**

```html
<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
```

For MDX files, use a JSX comment:

```mdx
{/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 */}
```

**TOML / YAML / shell:**

```text
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

Reviewers will check SPDX headers during review.
