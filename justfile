# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

export REPO_ROOT := justfile_directory()

# Skip dependency synchronization when the project environment is already fully synced.
no_uv := "false"

# Remove local Rust and Python build and test artifacts.
clean:
    #!/usr/bin/env bash
    shopt -s globstar nullglob
    cargo clean
    rm -rf \
        .coverage \
        .pytest_cache \
        python/.pytest_cache \
        **/__pycache__ \
        **/*.egg-info \
        **/*.so \
        **/coverage.xml \
        **/dist \
        docs/node_modules \
        target/

# Build the Rust workspace using the locked dependency set.
build-rust:
    cargo build --workspace --locked
    cargo install --path crates/fabric-cli --locked --force

# Build and install the Python distribution and native runtime in the project environment.
# --set [no_uv=true|false]
build-python:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "{{ no_uv }}" == "true" ]]; then
        uv pip install --python .venv/bin/python --no-deps --reinstall \
            --editable ./python \
            --editable .
    else
        uv sync --no-default-groups \
            --reinstall-package nemo-fabric \
            --reinstall-package nemo-fabric-runtime
    fi

# Build all supported language packages.
build-all: build-rust build-python

# Create or update the lockfile for every Python project.
lock-python:
    #!/usr/bin/env bash
    set -euo pipefail
    projects=(
        .
        python
        adapters/common
        adapters/codex-cli
        adapters/hermes-cli
        adapters/hermes-sdk
    )
    for project in "${projects[@]}"; do
        uv lock --project "$project"
    done

# Generate the Python and Rust API references and validate the Fern configuration.
# --set [no_uv=true|false]
docs:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "{{ no_uv }}" != "true" ]]; then
        uv sync --group docs
    fi
    npm ci --prefix docs --ignore-scripts
    PATH="{{ REPO_ROOT }}/.venv/bin:$PATH" bash scripts/generate_api_docs.sh
    uv run --no-sync python scripts/docs/generate_rust_library_reference.py
    npx --prefix docs --no-install fern check

# Run the Python test suite with the same optional integrations used by CI.
# --set [no_uv=true|false]
test-python:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "{{ no_uv }}" != "true" ]]; then
        uv sync --group test --no-group dev --extra harbor --extra hermes --extra relay --extra runtime
    fi
    uv run --no-sync pytest

# Run the Rust workspace test suite using the locked dependency set.
test-rust:
    cargo test --workspace --locked

# Run all Rust and Python tests.
test-all: test-rust test-python

# Build wheels for every Python project into the repository dist directory.
wheels:
    #!/usr/bin/env bash
    set -euo pipefail
    projects=(
        .
        python
        adapters/common
        adapters/codex-cli
        adapters/hermes-cli
        adapters/hermes-sdk
    )
    uv build --wheel --clear --out-dir dist "${projects[0]}"
    for project in "${projects[@]:1}"; do
        uv build --wheel --out-dir dist "$project"
    done
