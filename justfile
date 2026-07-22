# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

export REPO_ROOT := justfile_directory()

# Skip dependency synchronization when the project environment is already fully synced.
no_uv := "false"
# When set, versioning and packaging targets use this exact release version.
ref_name := ""
# Linux wheel artifacts target this minimum glibc version for compatibility.
linux_glibc_version := "2.17"

python_projects := ". python adapters/common adapters/claude adapters/codex adapters/deepagents adapters/hermes"

bash_helpers := '''
set -euo pipefail

uv_python_executable() {
    (
        cd "$REPO_ROOT"
        uv python find
    )
}

activate_project_venv() {
    local venv_dir="$REPO_ROOT/.venv"
    local venv_bin=""
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        venv_bin="$REPO_ROOT/.venv/bin"
    elif [[ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
        venv_bin="$REPO_ROOT/.venv/Scripts"
    else
        echo "ERROR: expected project virtualenv Python executable under .venv" >&2
        exit 1
    fi
    if command -v cygpath >/dev/null 2>&1; then
        venv_bin="$(cygpath -u "$venv_bin")"
    fi
    export VIRTUAL_ENV="$venv_dir"
    export PATH="$venv_bin:$PATH"
    unset PYTHONHOME
}

prepend_ziglang_to_path() {
    local python_executable="$1"
    local zig_dir=""
    zig_dir="$("$python_executable" - <<'PY'
from pathlib import Path
import importlib.util

spec = importlib.util.find_spec("ziglang")
if spec is None or spec.origin is None:
    raise SystemExit("ERROR: expected ziglang from the locked uv environment")

zig = Path(spec.origin).resolve().parent / "zig"
if not zig.exists():
    raise SystemExit(f"ERROR: expected zig binary at {zig}")

print(zig.parent)
PY
    )"
    export PATH="$zig_dir:$PATH"
}

linux_manylinux_compatibility() {
    local glibc_version="${linux_glibc_version:-2.17}"
    printf 'manylinux_%s\n' "${glibc_version//./_}"
}

python_wheel_build_args() {
    local os_name=""
    os_name="$(uname -s)"
    case "$os_name" in
        Linux)
            printf '%s\0' --compatibility "$(linux_manylinux_compatibility)" --zig
            ;;
        Darwin|CYGWIN*|MINGW*|MSYS*)
            printf '%s\0' --compatibility pypi
            ;;
        *)
            echo "ERROR: unsupported OS for wheels: $os_name" >&2
            exit 1
            ;;
    esac
}

semver_to_pep440() {
    local python_executable=""
    python_executable="$(uv_python_executable)"

    "$python_executable" - "$1" <<'PY'
import re
import sys

pattern = re.compile(
    r"^(?P<release>\d+\.\d+\.\d+)"
    r"(?:-(?P<pre_label>alpha|beta|rc)(?:\.(?P<pre_num>\d+))?)?"
    r"(?:\+(?P<local>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
match = pattern.fullmatch(sys.argv[1])
if not match:
    raise SystemExit(
        "Unsupported package version format. Expected SemVer with optional "
        "alpha/beta/rc prerelease and optional build metadata."
    )

pep440 = match.group("release")
pre_label = match.group("pre_label")
if pre_label:
    pre_map = {"alpha": "a", "beta": "b", "rc": "rc"}
    pre_num = match.group("pre_num") or "0"
    pep440 += f"{pre_map[pre_label]}{pre_num}"

local = match.group("local")
if local:
    normalized_local = ".".join(
        part.lower() for part in re.split(r"[._-]+", local) if part
    )
    if not normalized_local:
        raise SystemExit("Python package local version metadata cannot be empty")
    pep440 += f"+{normalized_local}"

print(pep440)
PY
}

set_cargo_workspace_version() {
    local version="$1"
    local python_executable=""
    python_executable="$(uv_python_executable)"

    "$python_executable" - "$version" <<'PY'
from pathlib import Path
import re
import sys

version = sys.argv[1]
if version.startswith("v"):
    raise SystemExit("Release tags must not start with 'v'; use raw SemVer such as 0.1.0")
if not re.fullmatch(
    r"\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)(?:\.\d+)?)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?",
    version,
):
    raise SystemExit(
        f"Unsupported release version '{version}'; use 0.1.0 or a supported "
        "prerelease such as 0.1.0-rc.1"
    )

path = Path("Cargo.toml")
text = path.read_text()
section = ""
output = []
changed = []
found_workspace_version = False
found_nemo_fabric_core = False

for line in text.splitlines(keepends=True):
    section_match = re.match(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$", line)
    if section_match:
        section = section_match.group(1)

    updated = line
    if section == "workspace.package":
        updated, count = re.subn(
            r'^(version\s*=\s*")([^"]+)(".*)$',
            rf"\g<1>{version}\g<3>",
            line,
        )
        if count == 1:
            found_workspace_version = True
            if updated != line:
                changed.append("workspace.package.version")
    elif section == "workspace.dependencies":
        updated, count = re.subn(
            r'^(nemo-fabric-core\s*=\s*\{[^}]*\bversion\s*=\s*")([^"]+)(".*)$',
            rf"\g<1>{version}\g<3>",
            line,
        )
        if count == 1:
            found_nemo_fabric_core = True
            if updated != line:
                changed.append("workspace.dependencies.nemo-fabric-core.version")

    output.append(updated)

missing = []
if not found_workspace_version:
    missing.append("workspace.package.version")
if not found_nemo_fabric_core:
    missing.append("workspace.dependencies.nemo-fabric-core.version")
if missing:
    raise SystemExit(f"Failed to find expected Cargo version fields: {', '.join(missing)}")

path.write_text("".join(output))
if changed:
    print(f"Cargo.toml version set to {version}: {', '.join(changed)}")
else:
    print(f"Cargo.toml already set to {version}")
PY

    local metadata_file=""
    metadata_file="$(mktemp)"
    if ! cargo metadata --no-deps --format-version 1 > "$metadata_file"; then
        rm -f "$metadata_file"
        return 1
    fi
    if ! "$python_executable" - "$version" "$metadata_file" <<'PY'
import json
import sys
from pathlib import Path

version = sys.argv[1]
metadata = json.loads(Path(sys.argv[2]).read_text())
workspace_members = set(metadata["workspace_members"])
mismatched = []
checked = 0

for package in metadata["packages"]:
    if package["id"] not in workspace_members:
        continue
    checked += 1
    if package["version"] != version:
        mismatched.append(f"{package['name']}={package['version']}")

if checked == 0:
    raise SystemExit("Cargo metadata did not include any Fabric workspace packages")
if mismatched:
    raise SystemExit(
        f"Cargo workspace packages do not all resolve to {version}: {', '.join(mismatched)}"
    )
print(f"Cargo metadata resolves {checked} Fabric workspace packages to {version}")
PY
    then
        rm -f "$metadata_file"
        return 1
    fi
    rm -f "$metadata_file"
}

set_python_project_versions() {
    local version=""
    local python_executable=""
    version="$(semver_to_pep440 "$1")"
    python_executable="$(uv_python_executable)"

    "$python_executable" - "$version" <<'PY'
from pathlib import Path
import re
import sys
import tomllib

version = sys.argv[1]
project_paths = (
    Path("pyproject.toml"),
    *sorted(Path("adapters").glob("**/pyproject.toml")),
)
pin_pattern = re.compile(r'(nemo-fabric-[a-z0-9-]+\s*==\s*)([^"\s,;]+)')

for path in project_paths:
    text = path.read_text()
    updated, count = re.subn(
        r'^version\s*=\s*"[^"]+"$',
        f'version = "{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit(f"Failed to find exactly one project version in {path}")
    updated = pin_pattern.sub(rf"\g<1>{version}", updated)
    if updated != text:
        path.write_text(updated)
        print(f"{path} version and internal pins updated to {version}")
    else:
        print(f"{path} already set to {version}")

runtime_path = Path("python/pyproject.toml")
runtime = tomllib.loads(runtime_path.read_text())
project = runtime.get("project", {})
if "version" in project or "version" not in project.get("dynamic", []):
    raise SystemExit(
        "python/pyproject.toml must keep a dynamic version derived from Cargo.toml"
    )

mismatched_pins = []
for path in project_paths:
    for match in pin_pattern.finditer(path.read_text()):
        if match.group(2) != version:
            mismatched_pins.append(f"{path}: {match.group(0)}")
if mismatched_pins:
    raise SystemExit(
        "Internal Python dependency pins are not synchronized: "
        + ", ".join(mismatched_pins)
    )
print("python/pyproject.toml continues to derive its version from Cargo.toml")
PY
}

set_project_version() {
    local version="$1"
    set_cargo_workspace_version "$version"
    set_python_project_versions "$version"
}
'''

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
        target/ \
        **/build/

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
        editable_projects=()
        for project in {{ python_projects }}; do
            editable_projects+=(--editable "$project")
        done
        uv pip install --python .venv/bin/python --no-deps --reinstall \
            --group adapters \
            "${editable_projects[@]}"
    else
        uv sync --no-default-groups --group adapters --extra claude --extra runtime \
            --reinstall-package nemo-fabric \
            --reinstall-package nemo-fabric-runtime
    fi

# Build all supported language packages.
build-all: build-rust build-python

# Create or update the lockfile for every Python project.
lock-python:
    #!/usr/bin/env bash
    set -euo pipefail
    projects=({{ python_projects }})
    for project in "${projects[@]}"; do
        uv lock --project "$project"
    done

# [version] or --set ref_name=<version>
set-version version="":
    #!/usr/bin/env bash
    {{ bash_helpers }}
    version="{{ version }}"
    if [[ -z "$version" ]]; then
        version="{{ ref_name }}"
    fi
    if [[ -z "$version" ]]; then
        echo "Error: version is required for set-version" >&2
        exit 1
    fi
    cd "$REPO_ROOT"
    set_project_version "$version"
    just lock-python

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
    npx --prefix docs --no-install fern check --warnings
    npx --prefix docs --no-install fern docs broken-links --strict

# Launch Jupyter Lab for the onboarding notebooks under examples/notebooks/.
# Jupyter is fetched on demand so it stays out of the project lockfile.
notebooks:
    uv run --no-sync --with jupyterlab --with ipykernel jupyter lab examples/notebooks

# Run the Python test suite with the same optional integrations used by CI.
# --set [no_uv=true|false]
test-python:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "{{ no_uv }}" != "true" ]]; then
        uv sync --group test --no-group dev --extra claude --extra codex --extra deepagents --extra harbor --extra hermes --extra relay --extra runtime
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
    {{ bash_helpers }}
    linux_glibc_version="{{ linux_glibc_version }}"
    uv sync --inexact --only-group package
    activate_project_venv
    if [[ "$(uname -s)" == "Linux" ]]; then
        prepend_ziglang_to_path "$(uv_python_executable)"
    fi
    projects=({{ python_projects }})
    uv build --wheel --clear --out-dir dist .
    for project in "${projects[@]}"; do
        if [[ "$project" == "." || "$project" == "python" ]]; then
            # Exclude the top-level package as we already built that
            # Exclude the python package as that needs special handling for maturin
            continue
        fi
        uv build --wheel --out-dir dist "$project"
    done
    # uv build forces Maturin compatibility off, producing non-portable linux_* tags.
    build_args=()
    while IFS= read -r -d '' arg; do
        build_args+=("$arg")
    done < <(python_wheel_build_args)
    (
        cd python
        maturin build \
            --release \
            --locked \
            "${build_args[@]}" \
            --out ../dist
    )
