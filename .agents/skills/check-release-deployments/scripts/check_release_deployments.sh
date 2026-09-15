#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 <tag>" >&2
    exit 1
fi

tag="$1"
repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

cargo_version="$(just normalize-release-tag "$tag")"
python_version="$(just release-tag-to-py-version "$tag")"
node_version="$(echo "$tag" | sed 's/^v//')"
github_repository='NVIDIA/nemo-fabric'
check_cargo_packages=true
check_python_packages=true
check_node_packages=true
cargo_skip_reason=''
python_skip_reason=''
node_skip_reason=''

request_status() {
    local url="$1"
    local status=""

    sleep 0.1
    status="$(
        curl --location --silent --show-error --output /dev/null \
            --write-out '%{http_code}' --user-agent 'Mozilla/5.0' "$url" 2>/dev/null
    )" || status="${status:-000}"
    printf '%s\n' "$status"
}

pypi_deployment_status() {
    local package_name="$1"
    local version="$2"
    local response=""
    local body=""
    local status=""

    sleep 0.1
    response="$(
        curl --location --silent --show-error \
            --header 'Accept: application/vnd.pypi.simple.v1+json' \
            --write-out $'\n%{http_code}' \
            "https://pypi.org/simple/${package_name}/" 2>/dev/null
    )" || {
        printf '000\n'
        return
    }

    status="${response##*$'\n'}"
    body="${response%$'\n'*}"
    if [[ "$status" != '200' ]]; then
        printf '%s\n' "$status"
        return
    fi

    if ! jq -e '.versions | type == "array"' >/dev/null <<<"$body"; then
        printf '000\n'
    elif jq -e --arg version "$version" '.versions | index($version) != null' \
        >/dev/null <<<"$body"; then
        printf '200\n'
    else
        printf '404\n'
    fi
}

deployment_status() {
    case "$1" in
        200) printf '☑' ;;
        404) printf '○' ;;
        *) printf '%s' "$1" ;;
    esac
}

print_github_release_pipeline() {
    local trigger_type='release'
    local nightly_date=""
    local nightly_created=""
    local tag_sha=""
    local runs=""
    local run=""
    local workflow=""
    local run_id=""
    local run_status=""
    local conclusion=""
    local run_url=""

    if [[ "$node_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+-alpha\.([0-9]{8})$ ]]; then
        trigger_type='nightly'
        nightly_date="${BASH_REMATCH[1]}"
        nightly_created="${nightly_date:0:4}-${nightly_date:4:2}-${nightly_date:6:2}"
    fi

    printf '## GitHub Release Pipeline\n\n'
    printf '| Workflow | Tag | Run | Status |\n'
    printf '| --- | --- | --- | --- |\n'

    if ! gh api "repos/${github_repository}/git/ref/tags/${tag}" >/dev/null 2>&1; then
        printf '| all workflows | `%s` | — | ○ tag not found |\n' "$tag"
        return
    fi

    sleep 0.1
    tag_sha="$(
        gh api "repos/${github_repository}/commits/${tag}" --jq '.sha' 2>/dev/null
    )" || {
        printf '| all workflows | `%s` | — | 000 |\n' "$tag"
        return
    }

    sleep 0.1
    if [[ "$trigger_type" == 'nightly' ]]; then
        runs="$(
            gh run list --repo "$github_repository" --commit "$tag_sha" \
                --workflow nightly-alpha-tag.yml --created "$nightly_created" \
                --limit 100 \
                --json databaseId,workflowName,status,conclusion,url 2>/dev/null
        )" || {
            printf '| all workflows | `%s` | — | 000 |\n' "$tag"
            return
        }
    else
        runs="$(
            gh run list --repo "$github_repository" --branch "$tag" \
                --commit "$tag_sha" --event push --limit 100 \
                --json databaseId,workflowName,status,conclusion,url 2>/dev/null
        )" || {
            printf '| all workflows | `%s` | — | 000 |\n' "$tag"
            return
        }
    fi

    if ! jq -e 'length > 0' >/dev/null <<<"$runs"; then
        printf '| all workflows | `%s` | — | ○ run not found |\n' "$tag"
        return
    fi

    while IFS= read -r run; do
        workflow="$(jq -r '.workflowName' <<<"$run")"
        if [[ "$workflow" == 'Request NVSkills CI' ]]; then
            continue
        fi
        run_id="$(jq -r '.databaseId' <<<"$run")"
        run_status="$(jq -r '.status' <<<"$run")"
        conclusion="$(jq -r '.conclusion // empty' <<<"$run")"
        run_url="$(jq -r '.url' <<<"$run")"
        if [[ "$run_status" == 'completed' && "$conclusion" == 'success' ]]; then
            conclusion='☑ success'
        elif [[ -n "$conclusion" ]]; then
            conclusion="$conclusion"
        else
            conclusion="$run_status"
        fi
        printf '| `%s` | `%s` | [%s](%s) | %s |\n' \
            "$workflow" "$tag" "$run_id" "$run_url" "$conclusion"

        if [[ "$conclusion" == 'failure' ]]; then
            case "$workflow" in
                'Publish Rust crates')
                    check_cargo_packages=false
                    cargo_skip_reason='Publish Rust crates failed'
                    ;;
                'Python')
                    check_python_packages=false
                    python_skip_reason='Python failed'
                    ;;
                'Publish TypeScript package')
                    check_node_packages=false
                    node_skip_reason='Publish TypeScript package failed'
                    ;;
            esac
        fi
    done < <(jq -c '.[]' <<<"$runs")
}

print_result() {
    local ecosystem="$1"
    local package_name="$2"
    local version="$3"
    local url="$4"
    local status=""

    status="$(request_status "$url")"
    printf '| %s | `%s` | `%s` | %s |\n' \
        "$ecosystem" \
        "$package_name" \
        "$version" \
        "$(deployment_status "$status")"
}

print_github_release_pipeline

printf '\n## Package Deployments\n\n'
printf '| Ecosystem | Package | Version | Status |\n'
printf '| --- | --- | --- | --- |\n'

if [[ "$check_cargo_packages" == true ]]; then
    while IFS= read -r cargo_file; do
        crate_name="$(python -c "import tomllib; print(tomllib.load(open('$cargo_file', 'rb'))['package']['name'])")"
        print_result \
            'Cargo' \
            "$crate_name" \
            "$cargo_version" \
            "https://crates.io/api/v1/crates/${crate_name}/${cargo_version}"
    done < <(
        find crates -name Cargo.toml -type f \
            ! -path 'crates/fabric-python/Cargo.toml' | sort
    )
else
    printf '| Cargo | — | — | skipped: %s |\n' "$cargo_skip_reason"
fi

if [[ "$check_python_packages" == true ]]; then
    while IFS= read -r package_path; do
        project_file="${package_path}/pyproject.toml"
        package_name="$(python -c "import tomllib; print(tomllib.load(open('$project_file', 'rb'))['project']['name'])")"
        status="$(pypi_deployment_status "$package_name" "$python_version")"
        printf '| Python | `%s` | `%s` | %s |\n' \
            "$package_name" \
            "$python_version" \
            "$(deployment_status "$status")"
    done < <(just python-package-paths)
else
    printf '| Python | — | — | skipped: %s |\n' "$python_skip_reason"
fi

if [[ "$check_node_packages" == true ]]; then
    while IFS= read -r package_file; do
        package_name="$(cat "$package_file" | jq .name | sed 's/\"//g')"
        print_result \
            'Node.js' \
            "$package_name" \
            "$node_version" \
            "https://registry.npmjs.org/${package_name}/${node_version}"
    done < <(git ls-files | grep package.json | grep -v docs/ | grep -v adapters/typescript/package.json || true)
else
    printf '| Node.js | — | — | skipped: %s |\n' "$node_skip_reason"
fi
