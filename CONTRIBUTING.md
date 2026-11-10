<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to NVIDIA NeMo Fabric

Thank you for your interest in contributing to NeMo Fabric. This guide covers
the development workflow, coding standards, and pull request process.

## Contribution Access

Anyone with a GitHub account may create or comment on an issue in this public
repository. Pull request creation is limited to approved users in the
`NeMo-Fabric-developers` group. General changes require CODEOWNER approval from
`NeMo-Fabric-reviewers`, a subset of the developer group. Documentation and
dependency changes use their specialized approver groups.

For non-security contributions, external contributors should open an issue with
a reproducer, proposed design, or patch description. A NeMo Fabric maintainer
can adopt the work into a pull request when appropriate. Do not report security
vulnerabilities through a public issue or pull request; follow the [security
policy](SECURITY.md). Public issue and pull request comments remain welcome and
do not require developer access.

## Development Setup

This section collects the setup steps needed before building, testing, or
contributing changes.

### Source Development

Install these tools before you start:

- **Rust** (stable toolchain) -- install with [rustup](https://rustup.rs/)
- **Python** >= 3.11
- **Node.js** >= 20.18.3 with npm
- **uv** -- follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- **just** >= 1.50.0 -- `cargo install just --locked`

Clone the repository, create a virtual environment, and build the Rust,
Python, and TypeScript packages:

```bash
git clone https://github.com/NVIDIA/NeMo-Fabric.git
cd NeMo-Fabric

uv venv --seed .venv --python 3.13
source .venv/bin/activate
just install-hermes-agent
uv sync --all-groups --all-extras
just no_uv=true build-all
```

In the above, replace `3.13` with the Python version you want to use for development.

Verify the checkout by running the test suites described in [Testing](#testing).

### Package Installation

To build Python packages (wheels) locally from a source checkout:

```bash
just wheels
```

The resulting wheel files are placed in the `dist/` directory. Install the
packages into a virtual environment with:

```bash
uv pip install --find-links ./dist nemo-fabric
```

Hermes Agent 0.20 and later is not installable from PyPI. For local development,
check out the pinned Hermes source and synchronize it into the project
environment with:

```bash
just install-hermes-agent
```

The recipe checks out Hermes Agent under `external/hermes-agent` and installs
it as the editable source declared in `pyproject.toml`. End users should follow
the [Hermes Agent installation guide](https://hermes-agent.nousresearch.com/docs/installation)
and install `nemo-fabric-adapters-hermes` without a harness extra.

Refer to the [installation guide](docs/getting-started/install.mdx) for the
complete list of adapters and installation options.



## Release Tagging

Release tags use SemVer with a leading `v`.

- Use `v0.1.0` for stable releases.
- Use `v0.1.0-rc.1` for prereleases.
- Do not use tags such as `0.1.0` or `0.1.0-rc.1`.

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

### TypeScript

Use strict TypeScript for the adapter-contract binding. Preserve the JSON wire
property names, run the checked-in generator instead of editing generated
declarations, and keep production dependencies out of the contract package.

### General

Use the naming conventions appropriate to each language. Rust and Python use
`snake_case` for functions and variables. Rust, Python, and TypeScript types use
`PascalCase`. TypeScript contract properties preserve the wire `snake_case`
names.

## Testing

**Run tests for every language surface affected by your changes.** If a change
touches the Rust core or public adapter-contract schemas, run the Rust, Python,
and TypeScript suites because both language bindings depend on the generated
wire contract.

Run the affected test targets through the repository `justfile`:

```bash
# Rust workspace
just test-rust

# Python SDK, adapters, integrations, and examples
just test-python

# TypeScript adapter contract
just test-typescript

# All supported language surfaces
just test-all
```

If the virtual environment is already synchronized, use `no_uv=true` to avoid
reinstalling dependencies:

```bash
just no_uv=true test-python
just no_uv=true test-all
```

When adding functionality, include tests in the corresponding Rust crate or in
the relevant area under `tests/`. Public contract changes must keep the
checked-in JSON Schema snapshots, Python representations, and generated
TypeScript declarations synchronized.

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
* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.
  * Any contribution which contains commits that are not Signed-Off will not be accepted.
* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```
* Full text of the DCO (https://developercertificate.org/):
  ```
    Developer Certificate of Origin
    Version 1.1
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.
    Developer's Certificate of Origin 1.1
    By making a contribution to this project, I certify that:
    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or
    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or
    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.
    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```


## Pull Request Process

This section describes how to prepare and submit changes for review.

### PR Cleanup Policy

The PR Cleanup workflow initially runs in **dry-run mode**: it reports proposed
closures without posting comments or closing PRs. Once enforcement is enabled:

- Each author may have up to **3 open PRs, including drafts**, across all target
  branches. Cleanup keeps the three oldest by original creation time and closes
  newer excess PRs. Maintainers of the
  [`nemo-fabric-developers` team](https://github.com/orgs/NVIDIA/teams/nemo-fabric-developers)
  are exempt from this cap; repository Maintain or Admin access alone does not
  grant an exemption. Team roles are checked on every run.
- PRs with **more than 28 days without activity**, including drafts, are closed
  regardless of author or team role. Human comments and edits, reviews and review
  comments, reopening, and pushes count as activity. Bot comments, CI results,
  labels, and draft/ready transitions alone do not count.

Each closure explains its reason. To resume an inactive PR, reopen it when ready
and tag maintainers for review. To reopen a PR closed for exceeding the cap,
first merge or close another PR. Reopening preserves original creation order
and triggers policy evaluation again. Branches and discussions are preserved.

Cleanup runs when PRs open, reopen, or receive commits, and daily. It acts after
PR creation, so an excess PR may briefly appear and trigger CI.

#### Maintainer Setup and Rollout

The configuration is in `.github/pr-cleanup.json`; the implementation and tests
are in `scripts/ci/pr_cleanup.py` and `tests/scripts/test_pr_cleanup.py`.

1. Install an organization-approved GitHub App with organization **Members:
   read** permission and access to this repository. Set repository variable
   `PR_CLEANUP_APP_ID` and secret `PR_CLEANUP_APP_PRIVATE_KEY`. The workflow
   obtains a short-lived membership-read token; repository writes use the
   separate `GITHUB_TOKEN`. Missing or failed membership reads skip cap
   enforcement and report an error, rather than assuming nobody is exempt.
2. Confirm the organization/repository Actions event policy permits
   `pull_request_target`. The workflow executes only trusted default-branch
   code, never PR-head code. Allow the pinned actions in the organization's
   action policy. Manual dispatch is restricted to the default branch.
3. Merge with `PR_CLEANUP_APPLY` unset. Inspect dry-run summaries, announce the
   policy, and then set that repository variable to `true` to enable automatic
   comments and closures. Manual dispatch defaults to reporting only; selecting
   its `apply` input explicitly requests enforcement for that run, regardless
   of the automatic setting. Existing excess PRs are included in enforcement.
4. To pause closures, unset `PR_CLEANUP_APPLY` and avoid manual apply runs, or
   disable the workflow. Previously closed PRs are not automatically reopened.

Push timing is observed and saved in the workflow's `pr-cleanup-state` artifact;
commit author dates are not used as push timestamps. New PRs, missing/expired
state, or an observation gap of at least 28 days start a fresh 28-day observation
period. Consequently, the first deployment does not immediately close old PRs
for inactivity. Dry-run observation history carries forward into enforcement.
Keep artifact retention at least 28 days and do not delete the latest history
unless you intend to restart observation. Comment and review history is read
with pagination before an inactivity decision. Failed lookups skip affected
closures and appear in the workflow summary. Rechecks reduce races with human
activity, but GitHub does not offer atomic read-and-close operations.

Local read-only inspection, with an authenticated `gh` CLI and an organization
membership-read token available in `PR_CLEANUP_MEMBERS_TOKEN`:

```bash
GH_TOKEN="$(gh auth token)" python3 scripts/ci/pr_cleanup.py --repo NVIDIA/NeMo-Fabric
python3 -m unittest discover -s tests/scripts -p test_pr_cleanup.py -v
```

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
feat: add typed runtime diagnostics
fix: preserve adapter errors in run results
docs: clarify Hermes Agent adapter installation
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
