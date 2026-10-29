# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan and run a bounded Harbor -> Fabric -> harness integration qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[4]
CATALOG = Path(
    "sdk/python/nemo-fabric-adapter-catalog/src/nemo_fabric_adapter_catalog/catalog.json"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def sanitize(text: str, credential_names: tuple[str, ...] = ()) -> str:
    for name, value in os.environ.items():
        if (
            value
            and len(value) >= 4
            and (
                name in credential_names
                or name
                in {
                    "DOCKER_AUTH_CONFIG",
                    "AWS_ACCESS_KEY_ID",
                    "AZURE_STORAGE_CONNECTION_STRING",
                }
                or re.search(
                    r"(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)$", name, re.I
                )
            )
        ):
            text = text.replace(value, "[REDACTED]")
    text = re.sub(
        r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?(?:bearer\s+)?)[^\s\"',}]+",
        r"\1[REDACTED]",
        text,
    )
    return re.sub(r"(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[REDACTED]@", text)


def sanitize_data(value: object, credential_names: tuple[str, ...]) -> object:
    """Redact string values without changing JSON types or structure."""
    if isinstance(value, str):
        return sanitize(value, credential_names)
    if isinstance(value, dict):
        return {
            key: sanitize_data(item, credential_names) for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_data(item, credential_names) for item in value]
    return value


def sanitize_artifact(text: str, suffix: str, credential_names: tuple[str, ...]) -> str:
    """Preserve machine-readable trial evidence when scrubbing credential values."""
    if suffix in {".jsonl", ".ndjson"}:
        return "\n".join(
            sanitize_artifact(line, ".json", credential_names).removesuffix("\n")
            for line in text.split("\n")
        )
    if suffix == ".json":
        try:
            value = json.loads(text)
        except ValueError:
            return sanitize(text, credential_names)
        cleaned = sanitize_data(value, credential_names)
        if cleaned == value:
            return text
        return json.dumps(cleaned, ensure_ascii=False) + "\n"
    return sanitize(text, credential_names)


def process_env(key_env: str | None = None) -> dict[str, str]:
    names = (
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
    )
    env = {name: os.environ[name] for name in names if name in os.environ}
    env.update(PYTHONNOUSERSITE="1", UV_NO_CONFIG="1", PIP_CONFIG_FILE=os.devnull)
    if key_env:
        env[key_env] = os.environ[key_env]
    return env


def execute(
    command: list[str],
    log: Path,
    timeout: int,
    *,
    key_env: str | None = None,
    cwd: Path | None = None,
) -> int:
    """Bound the entire process group; redact before writing persistent output."""
    credential_names = (key_env,) if key_env else ()
    with log.open("a", encoding="utf-8") as stream:
        stream.write(sanitize(shlex.join(command), credential_names) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=process_env(key_env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=True,
        )

        def drain() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(sanitize(line, credential_names))
                stream.flush()

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            code = 130 if isinstance(error, KeyboardInterrupt) else 124
            for sig, grace in (
                (signal.SIGINT, 30),
                (signal.SIGTERM, 10),
                (signal.SIGKILL, 5),
            ):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=grace)
                    break
                except subprocess.TimeoutExpired:
                    continue
        reader.join(timeout=5)
        return code


def plan_catalog(plan: dict) -> dict:
    return plan.get("catalog_snapshot") or load(ROOT / CATALOG)


def task_identity(path: Path) -> dict:
    """Identify exactly one Harbor task, including its unmodified verifier."""
    path = path.resolve()
    for relative in (
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "tests/test.sh",
    ):
        if not (path / relative).is_file():
            raise ValueError(f"single Harbor task missing {relative}: {path}")
    config = tomllib.loads((path / "task.toml").read_text())
    files = {}
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path)
        if relative.parts[0] == "solution" or "__pycache__" in relative.parts:
            continue
        if item.is_symlink():
            raise ValueError(f"task symlinks are unsupported: {relative}")
        if item.is_file():
            files[str(relative)] = hashlib.sha256(item.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "name": config.get("task", {}).get("name", path.name),
        "sha256": fingerprint(files),
    }


def stage_task(
    plan: dict, destination: Path, image: str, task_python: str = "python3"
) -> None:
    """Keep task assets and tests; use the prepared task plus harness image."""
    shutil.copytree(
        plan["task"]["path"],
        destination,
        ignore=shutil.ignore_patterns("solution", "__pycache__"),
    )
    # Harbor must build our overlay rather than pulling the benchmark-only image.
    config_path = destination / "task.toml"
    config = config_path.read_text()
    config = re.sub(r"(?m)^docker_image\s*=.*$", "", config)
    config_path.write_text(config)
    (destination / "environment/Dockerfile").write_text(f"FROM {image}\n")
    # Inspect the agent's final install before the verifier installs its own deps.
    # Run the upstream verifier unchanged and preserve its exit status.
    verifier = destination / "tests/test.sh"
    verifier.rename(destination / "tests/upstream-test.sh")
    python = shlex.quote(task_python)
    verifier.write_text(
        "#!/bin/bash\n"
        f"{python} /qualification/identity.py > /logs/verifier/qualification-identity.json\n"
        f"{python} /qualification/identity.py --selection /tmp/fabric-run-*.json > /logs/verifier/qualification-selection.json\n"
        "bash /tests/upstream-test.sh\n"
    )


def valid_install_sources(identity: dict) -> bool:
    if not identity.get("editable"):
        return True
    source = identity.get("editable_sources", {}).get("hermes-agent", {})
    revision = source.get("revision")
    return (
        identity["editable"] == ["hermes-agent"]
        and isinstance(revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", revision) is not None
        and revision == identity.get("harness_sources", {}).get("hermes-agent")
        and source.get("dirty") is False
    )


def unchanged_install(expected: dict, observed: dict) -> bool:
    """Protect Fabric/harness identity; record other task dependency changes only."""
    if not valid_install_sources(observed):
        return False
    for key, value in expected.items():
        if key == "packages":
            if any(
                observed.get(key, {}).get(name) != version
                for name, version in value.items()
                if name.startswith("nemo-fabric")
                or name
                in {
                    "deepagents",
                    "hermes-agent",
                    "mini-swe-agent",
                    "nooa",
                    "nooa-bench",
                    "nooa-cli",
                    "openhands-sdk",
                    "openhands-tools",
                }
            ):
                return False
        elif observed.get(key) != value:
            return False
    return True


def validate_plan(plan: dict) -> None:
    if plan["format_version"] != 1 or plan["mode"] not in {"source", "registry"}:
        raise ValueError("unsupported qualification plan")
    if (
        plan["concurrency"] != 1
        or plan["attempts"] != 1
        or not 30 <= plan["runtime_timeout_seconds"] <= 1800
    ):
        raise ValueError(
            "qualification requires concurrency=1, attempts=1 and a 30..1800 second deadline"
        )
    if task_identity(Path(plan["task"]["path"])) != plan["task"]:
        raise ValueError("task changed; regenerate the qualification plan")
    reference = load(ROOT / CATALOG)
    catalog = plan_catalog(plan)
    if plan["mode"] == "source" and catalog != reference:
        raise ValueError("source mode requires the checkout's unmodified catalog")
    expected = set(reference["adapters"])
    ids = [row["adapter_id"] for row in plan["adapters"]]
    if (
        set(catalog["adapters"]) != expected
        or len(ids) != len(set(ids))
        or set(ids) != expected
    ):
        raise ValueError(
            "plan must contain every catalog adapter exactly once; regenerate it"
        )
    if plan["version"] != catalog["catalog_version"]:
        raise ValueError(
            "catalog must match the exact package version under test; use --catalog-json for a published prerelease catalog"
        )
    if (
        plan["source_sha"]
        != subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    ):
        raise ValueError("source HEAD changed; regenerate the qualification plan")
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "crates",
            "sdk",
            "adapters",
            "adapter-contract",
            "Cargo.toml",
            "Cargo.lock",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/ci/generate_adapter_catalog.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
        env=process_env(),
    )
    for row in plan["adapters"]:
        if row["model"].startswith(("/", "./", "../", "~/")):
            raise ValueError(
                "serve a local checkpoint first, then use its advertised model ID and HTTP endpoint"
            )
        record = catalog["adapters"][row["adapter_id"]]
        source = reference["adapters"][row["adapter_id"]]
        if (
            row["catalog"] != record
            or fingerprint(load(ROOT / record["source"])) != record["sha256"]
            or any(
                record[key] != source[key]
                for key in ("source", "ecosystem", "descriptor")
            )
            or record["package"]["name"] != source["package"]["name"]
        ):
            raise ValueError(
                "catalog/source descriptor drift; regenerate catalog and plan"
            )
        if row["action"] not in {"run", "blocked", "unsupported", "excluded"}:
            raise ValueError("invalid plan action")
        if row["action"] != "run" and not row.get("reason"):
            raise ValueError("non-run entries require a reason")
        provider = row.get("model_provider")
        if provider is not None and (
            not isinstance(provider, str)
            or not provider
            or provider != provider.strip()
        ):
            raise ValueError(
                "model_provider must be non-empty without surrounding whitespace"
            )
        key = row.get("api_key_env")
        task_python = row.get("task_python")
        if task_python is not None and (
            not isinstance(task_python, str) or not task_python.strip()
        ):
            raise ValueError("task_python must name a compatible task interpreter")
        if key and (
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            or not re.search(r"(?:API_KEY|TOKEN|SECRET|PASSWORD)$", key)
        ):
            raise ValueError(
                "use a Harbor-redactable credential variable such as MODEL_API_KEY"
            )

        def validate_settings(value: object) -> None:
            if isinstance(value, dict):
                for name, child in value.items():
                    if not isinstance(name, str):
                        raise ValueError("settings JSON objects require string keys")
                    if re.search(
                        r"(?:api.?key|auth.?token|password|secret)$", name, re.I
                    ):
                        raise ValueError(
                            "credentials belong in an environment variable, not harness settings"
                        )
                    validate_settings(child)
            elif isinstance(value, list):
                for child in value:
                    validate_settings(child)

        validate_settings(row["harness_settings"])
        target_id = row.get("workflow_target_id")
        workflow_settings = row.get("workflow_settings")
        if target_id is None:
            if workflow_settings is not None:
                raise ValueError("workflow_settings requires workflow_target_id")
            if row["action"] == "run" and record["descriptor"].get("target_types") == [
                "workflow"
            ]:
                raise ValueError("workflow-only adapters require workflow_target_id")
        else:
            if (
                plan["bridge"] != "sdk"
                or not isinstance(target_id, str)
                or not target_id
                or target_id != target_id.strip()
            ):
                raise ValueError(
                    "workflow_target_id requires a non-empty ID and the SDK bridge"
                )
            target = catalog["targets"].get(target_id)
            target_source = reference["targets"].get(target_id)
            if (
                target is None
                or target_source is None
                or any(
                    target[key] != target_source[key]
                    for key in ("source", "ecosystem", "descriptor")
                )
                or target["package"]["name"] != target_source["package"]["name"]
                or fingerprint(load(ROOT / target["source"])) != target["sha256"]
                or target["descriptor"]["type"] != "workflow"
                or target["descriptor"]["adapter_id"] != row["adapter_id"]
                or "workflow" not in record["descriptor"].get("target_types", [])
            ):
                raise ValueError(
                    "workflow target metadata must match its catalog and adapter"
                )
            if workflow_settings is not None:
                if not isinstance(workflow_settings, dict) or any(
                    not isinstance(key, str) for key in workflow_settings
                ):
                    raise ValueError("workflow_settings must be a JSON object")
                try:
                    json.dumps(workflow_settings, allow_nan=False)
                except (TypeError, ValueError, OverflowError, RecursionError):
                    raise ValueError(
                        "workflow_settings must contain JSON values"
                    ) from None
                validate_settings(workflow_settings)
        model_settings = row.get("model_settings")
        if model_settings is not None:
            if plan["bridge"] != "sdk" or not isinstance(model_settings, dict):
                raise ValueError("model_settings requires an object and the SDK bridge")
            validate_settings(model_settings)
        max_turns = row.get("max_turns")
        if max_turns is not None and (
            isinstance(max_turns, bool)
            or not isinstance(max_turns, int)
            or max_turns <= 0
        ):
            raise ValueError("max_turns must be a positive integer")
        url = row.get("base_url")
        if url:
            parsed = urlsplit(url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "endpoint must be an HTTP(S) URL without credentials, query, or fragment"
                )
            if parsed.scheme == "http" and parsed.hostname not in {
                "localhost",
                "127.0.0.1",
                "::1",
                *plan["trusted_http_hosts"],
            }:
                raise ValueError(
                    "non-loopback HTTP requires an explicit --trusted-http-host"
                )
    cost = plan["estimated_cost_usd"]
    if (
        not isinstance(cost, (int, float))
        or not math.isfinite(cost)
        or cost < 0
        or not plan["cost_basis"]
    ):
        raise ValueError("record a finite nonnegative cost estimate and its basis")


def make_plan(args: argparse.Namespace) -> dict:
    catalog = load(getattr(args, "catalog_json", None) or ROOT / CATALOG)
    rows = []
    prerequisites = {
        "nvidia.fabric.hermes": "Requires a supported Hermes runtime/source image; select its compatible task_python and record pinned, clean source provenance.",
        "nvidia.fabric.remote-agent": "Requires an explicitly configured compatible HTTP endpoint/profile; workspace/tool execution is separate coverage.",
        "nvidia.fabric.kiro": "Requires Kiro CLI, KIRO_API_KEY, and a Kiro-advertised provider/model. Skip when KIRO_API_KEY is absent; BYOM is unsupported.",
        "nvidia.fabric.droid": "Requires Factory authentication or a prepared native Droid BYOK profile/image; custom endpoints use native settings, not model base_url.",
    }
    for adapter_id, record in sorted(catalog["adapters"].items()):
        reason = prerequisites.get(adapter_id, "")
        unsupported = record["descriptor"].get("target_types") == ["workflow"]
        targets = [
            target_id
            for target_id, target in catalog["targets"].items()
            if target["descriptor"]["adapter_id"] == adapter_id
        ]
        if unsupported:
            reason = "This adapter requires a registered workflow target. Select workflow_target_id and optional workflow_settings with the SDK bridge before marking the row runnable."
        if (
            adapter_id == "nvidia.fabric.claude"
            and args.endpoint_kind == "chat-completions"
        ):
            reason = "Claude requires an Anthropic Messages-compatible endpoint; this plan selects Chat Completions."
        accepts = record["descriptor"].get("config", {}).get("accepts", [])
        settings = {}
        if adapter_id == "nvidia.fabric.codex":
            # Harbor supplies the disposable container boundary; nested bwrap
            # requires user namespaces that many Docker hosts prohibit.
            settings = {"sandbox": "danger-full-access", "approval_mode": "deny_all"}
        if adapter_id == "nvidia.fabric.claude":
            settings = {"permission_mode": "bypassPermissions"}
        if adapter_id == "nvidia.fabric.langchain.deepagents":
            settings = {"deepagents": {"backend": {"type": "local_shell"}}}
        if adapter_id == "nvidia.fabric.qwen":
            settings = {"permission_mode": "yolo"}
        properties = (
            record["descriptor"].get("settings_schema", {}).get("properties", {})
        )
        if "insecure_http_model_hosts" in properties and args.trusted_http_host:
            settings["insecure_http_model_hosts"] = args.trusted_http_host
        rows.append(
            {
                "adapter_id": adapter_id,
                "catalog": record,
                "action": "unsupported"
                if unsupported
                else "blocked"
                if reason
                else "run",
                "reason": reason,
                "registered_targets": targets,
                "model": args.model,
                "base_url": args.base_url,
                "api_key_env": "KIRO_API_KEY"
                if adapter_id == "nvidia.fabric.kiro"
                else args.api_key_env,
                "harness_settings": settings,
                "enabled_tools": [
                    "ls",
                    "read_file",
                    "write_file",
                    "edit_file",
                    "glob",
                    "grep",
                    "execute",
                ]
                if adapter_id == "nvidia.fabric.langchain.deepagents"
                else None,
                "telemetry": "none",
                "capabilities": {
                    name: "untested"
                    if supported
                    else "not_declared"
                    if name == "subagents"
                    else "unsupported"
                    for name, supported in {
                        "mcp": "mcp" in accepts,
                        "skills": "skills" in accepts,
                        "telemetry_atif": any(
                            "atif" in provider.get("outputs", [])
                            for provider in record["descriptor"]
                            .get("telemetry", {})
                            .get("providers", {})
                            .values()
                        ),
                        "subagents": "subagents"
                        in json.dumps(record["descriptor"].get("settings_schema", {})),
                    }.items()
                },
            }
        )
    return {
        "format_version": 1,
        "task": task_identity(args.task),
        "mode": args.mode,
        "version": args.version,
        "catalog_snapshot": catalog,
        "catalog_sha256": fingerprint(catalog),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "harbor_version": args.harbor_version,
        "bridge": args.bridge,
        "endpoint_kind": args.endpoint_kind,
        "backend": getattr(args, "backend", "operator-managed inference"),
        "trusted_http_hosts": args.trusted_http_host,
        "estimated_cost_usd": args.estimated_cost_usd,
        "cost_basis": args.cost_basis,
        "concurrency": 1,
        "attempts": 1,
        "runtime_timeout_seconds": args.timeout,
        "created_at": now(),
        "adapters": rows,
    }


def harbor_command(
    plan: dict, row: dict, python: str, task: Path, jobs: Path, name: str
) -> list[str]:
    command = [
        python,
        "-m",
        "harbor.cli.main",
        "run",
        "--path",
        str(task),
        "--agent",
        "fabric"
        if plan["bridge"] == "native"
        else "nemo_fabric.integrations.harbor:FabricAgent",
        "--model",
        row["model"],
        "--job-name",
        name,
        "--jobs-dir",
        str(jobs),
        "--n-concurrent",
        "1",
        "--n-attempts",
        "1",
    ]
    options = {
        "fabric_adapter_id": row["adapter_id"],
        "fabric_workspace": "/app",
        "fabric_config_base_dir": "/app",
        "fabric_harness_settings": row["harness_settings"],
        "fabric_runtime_timeout_seconds": plan["runtime_timeout_seconds"],
        "fabric_timeout_sec": plan["runtime_timeout_seconds"] + 60,
        "fabric_telemetry": row["telemetry"],
    }
    if row.get("task_python") is not None:
        options["fabric_python"] = row["task_python"]
    if row.get("model_provider") is not None:
        options["fabric_model_provider"] = row["model_provider"]
    if row.get("model_settings") is not None:
        options["fabric_model_settings"] = row["model_settings"]
    if row.get("workflow_target_id") is not None:
        options["fabric_workflow_target_id"] = row["workflow_target_id"]
        options["fabric_workflow_settings"] = row.get("workflow_settings") or {}
    if row.get("max_turns") is not None:
        options["fabric_max_turns"] = row["max_turns"]
    if row.get("base_url"):
        options["fabric_model_base_url"] = row["base_url"]
    if row.get("api_key_env"):
        key = row["api_key_env"]
        options["fabric_model_api_key_env"] = key
        command.extend(["--ae", f"{key}=${{{key}}}"])
    if row.get("enabled_tools") is not None:
        options["fabric_enabled_tools"] = row["enabled_tools"]
    if row["catalog"]["ecosystem"] == "typescript":
        record = row["catalog"]
        options["fabric_discovery_paths"] = [
            f"/opt/harness/node_modules/{record['package']['name']}/{Path(record['source']).name}"
        ]
    if row["adapter_id"] == "nvidia.fabric.claude":
        options["fabric_environment_env"] = {"IS_SANDBOX": "1"}
    for key, value in options.items():
        command.extend(
            [
                "--ak",
                f"{key}={json.dumps(value) if isinstance(value, (dict, list)) else value}",
            ]
        )
    return command


def one(directory: Path, pattern: str) -> Path:
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one {pattern}, found {len(paths)}")
    return paths[0]


def classify(text: str) -> str:
    for category, pattern in (
        (
            "adapter_admission",
            r"fingerprint|descriptor.*(drift|mismatch)|capabilit|unsupported.*(provider|model)|model_schema|settings_schema|config.*invalid",
        ),
        (
            "auth",
            r"unauthorized|authentication|api.key.*(missing|invalid|required)|\b(?:401|403)\b",
        ),
        (
            "install",
            r"no matching distribution|modulenotfound|no module named|harness_unavailable|cannot find (module|package)|enoent",
        ),
        (
            "provider_model",
            r"model.*(not found|unknown|not supported)|\b404\b|messages.compatible|provider",
        ),
        ("timeout", r"timeout|timed out|deadline"),
        (
            "infrastructure",
            r"no space left|connection refused|docker.*(error|failed)|temporary failure|rate.limit|\b(?:429|502|503)\b",
        ),
    ):
        if re.search(pattern, text, re.I):
            return category
    return "transport_runner"


def normalized_accounting(path: Path) -> dict:
    from harbor.models.agent.context import AgentContext
    from nemo_fabric.integrations.harbor.fabric_agent import (
        populate_context_from_result,
    )

    context = AgentContext()
    populate_context_from_result(context, path)
    return {
        key: getattr(context, key)
        for key in ("n_input_tokens", "n_output_tokens", "n_cache_tokens", "cost_usd")
    }


def admission_fingerprint(
    spec: dict, descriptor: dict, target_descriptor: dict | None = None
) -> str | None:
    from nemo_fabric import FabricConfig, inspect_adapter

    options = (
        {"target_descriptor": target_descriptor}
        if target_descriptor is not None
        else {}
    )
    return inspect_adapter(
        FabricConfig.from_mapping(spec["config"]), descriptor, **options
    ).descriptor_sha256


def valid_selection(selected: dict, row: dict, target: dict | None = None) -> bool:
    origins = selected.get("provenance") or []
    if selected.get("adapter_id") != row["adapter_id"] or not origins:
        return False
    workflow = selected.get("workflow")
    if row.get("workflow_target_id") is None:
        if workflow is not None:
            return False
    elif (
        target is None
        or not isinstance(workflow, dict)
        or workflow.get("target_id") != row["workflow_target_id"]
        or workflow.get("entrypoint") != target["descriptor"]["spec"]["entrypoint"]
        or not workflow.get("provenance")
        or not all(
            origin.get("source") == "installed_package"
            and origin.get("path", "").startswith(
                selected.get("python_prefix", "") + "/share/nemo-fabric/adapters/"
            )
            and Path(origin["path"]).name == Path(target["source"]).name
            for origin in workflow["provenance"]
        )
    ):
        return False
    filename = Path(row["catalog"]["source"]).name
    if row["catalog"]["ecosystem"] == "typescript":
        expected = (
            f"/opt/harness/node_modules/{row['catalog']['package']['name']}/{filename}"
        )
        return all(
            origin.get("source") == "explicit_local" and origin.get("path") == expected
            for origin in origins
        )
    prefix = selected.get("python_prefix", "") + "/share/nemo-fabric/adapters/"
    return all(
        origin.get("source") == "installed_package"
        and origin.get("path", "").startswith(prefix)
        and Path(origin["path"]).name == filename
        for origin in origins
    )


def selection_payload(plan: dict, row: dict) -> dict:
    from nemo_fabric.integrations.harbor.fabric_agent import build_harbor_config

    paths = (
        [
            f"/opt/harness/node_modules/{row['catalog']['package']['name']}/{Path(row['catalog']['source']).name}"
        ]
        if row["catalog"]["ecosystem"] == "typescript"
        else []
    )
    provider_options = (
        {"model_provider_name": row["model_provider"]}
        if row.get("model_provider") is not None
        else {}
    )
    if row.get("model_settings") is not None:
        provider_options["model_settings"] = row["model_settings"]
    if row.get("workflow_target_id") is not None:
        provider_options["workflow_target_id"] = row["workflow_target_id"]
        provider_options["workflow_settings"] = row.get("workflow_settings") or {}
    config = build_harbor_config(
        adapter_id=row["adapter_id"],
        workspace="/app",
        harness_settings=row["harness_settings"],
        model_name=row["model"],
        model_base_url=row.get("base_url"),
        model_api_key_env=row.get("api_key_env"),
        timeout_seconds=plan["runtime_timeout_seconds"],
        max_turns=row.get("max_turns"),
        enabled_tools=row.get("enabled_tools"),
        telemetry=row["telemetry"],
        discovery_paths=tuple(paths),
        **provider_options,
    )
    return {"config": config.to_mapping(), "config_base_dir": "/app"}


def inspect_results(
    job: Path,
    row: dict,
    expected_identity: dict,
    expected_selection: dict,
    target: dict | None = None,
) -> dict:
    """Keep transport, reward, selection, usage and telemetry as separate evidence."""
    summary = {"checks": {}, "failures": [], "limitations": [], "evidence": []}

    def fail(category: str, reason: str) -> None:
        summary["failures"].append({"category": category, "reason": sanitize(reason)})

    # Verifier evidence remains meaningful even when result transport fails.
    try:
        reward_path = one(job, "*/verifier/reward.txt")
        reward = float(reward_path.read_text().strip())
        if not math.isfinite(reward):
            raise ValueError("Task reward is not finite")
        summary["reward"] = reward
        summary["checks"]["verifier_completed"] = True
        summary["checks"]["task_success"] = summary["reward"] == 1.0
    except (OSError, ValueError) as error:
        summary["checks"]["verifier_completed"] = False
        fail("harbor_verifier", str(error))
    try:
        task_identity = load(one(job, "*/verifier/qualification-identity.json"))
        summary["checks"]["task_install_identity"] = unchanged_install(
            expected_identity, task_identity
        )
        if not summary["checks"]["task_install_identity"]:
            fail(
                "adapter_admission",
                "Fabric/harness identity differs from inspected image or has editable installs",
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        fail("adapter_admission", f"Task identity unavailable: {error}")
    try:
        observed_selection = load(one(job, "*/verifier/qualification-selection.json"))
        summary["task_selection"] = observed_selection
        summary["checks"]["task_descriptor_provenance"] = (
            valid_selection(observed_selection, row, target)
            and observed_selection == expected_selection
        )
        if not summary["checks"]["task_descriptor_provenance"]:
            fail(
                "adapter_admission",
                "Task descriptor origin differs from the installed preflight selection",
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        fail("adapter_admission", f"Task selection unavailable: {error}")
    try:
        job_result = load(job / "result.json")
        stats = job_result["stats"]
        summary["checks"]["harbor_lifecycle"] = (
            stats["n_completed_trials"] == 1 and stats["n_errored_trials"] == 0
        )
        if not summary["checks"]["harbor_lifecycle"]:
            fail(
                "harbor_lifecycle",
                "Harbor did not complete exactly one trial without exceptions",
            )
        trial = load(one(job, "*/result.json"))
        exception = trial.get("exception_info")
        if exception:
            message = str(exception.get("exception_message", exception))
            fail(classify(message), message[:2000])
        result_path = one(job, "*/agent/fabric-result-*.json")
        summary["evidence"].append(str(result_path))
        document = load(result_path)
        if "runner_error" in document:
            fail(classify(json.dumps(document)), json.dumps(document["runner_error"]))
            return summary
        from nemo_fabric import RunResult

        result = RunResult.from_mapping(document).to_mapping()
        summary["checks"]["result_transport"] = True
        summary["observed"] = {
            key: result.get(key)
            for key in (
                "status",
                "adapter_id",
                "harness",
                "runtime_id",
                "invocation_id",
                "request_id",
            )
        }
        summary["checks"]["selection"] = result.get("adapter_id") == row["adapter_id"]
        if not summary["checks"]["selection"]:
            fail("adapter_admission", "Observed adapter differs from selected adapter")
        if result["status"] != "succeeded" or result.get("error"):
            fail(
                classify(json.dumps(result.get("error"))),
                json.dumps(result.get("error")),
            )
        spec = load(one(job, "*/agent/fabric-run-*.json"))
        target_id = row.get("workflow_target_id")
        workflow = spec["config"].get("workflow")
        summary["checks"]["workflow_selection"] = (
            workflow is None
            if target_id is None
            else isinstance(workflow, dict)
            and workflow.get("target_id") == target_id
            and (workflow.get("settings") or {}) == (row.get("workflow_settings") or {})
        )
        if not summary["checks"]["workflow_selection"]:
            fail(
                "adapter_admission",
                "Uploaded workflow differs from the selected target/settings",
            )
        expected_admission = admission_fingerprint(
            spec,
            row["catalog"]["descriptor"],
            target["descriptor"] if target is not None else None,
        )
        summary["admission_sha256"] = {
            "expected_normalized": expected_admission,
            "uploaded": spec.get("adapter_descriptor_sha256"),
            "catalog_raw": row["catalog"]["sha256"],
            **({"target_catalog_raw": target["sha256"]} if target is not None else {}),
        }
        summary["checks"]["descriptor_fingerprint"] = (
            expected_admission is not None
            and spec.get("adapter_descriptor_sha256") == expected_admission
        )
        if not summary["checks"]["descriptor_fingerprint"]:
            fail(
                "adapter_admission",
                "Uploaded fingerprint differs from the core-normalized catalog descriptor",
            )
        context = trial["agent_result"] or {}
        usage = result.get("usage") or {}
        summary["usage"] = {
            key: usage.get(key)
            for key in (
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "cost_usd",
            )
        }
        summary["harbor_usage"] = {
            key: context.get(key)
            for key in (
                "n_input_tokens",
                "n_output_tokens",
                "n_cache_tokens",
                "cost_usd",
            )
        }
        expected_usage = normalized_accounting(result_path)
        known = all(
            isinstance(expected_usage[key], int)
            and not isinstance(expected_usage[key], bool)
            and expected_usage[key] >= 0
            for key in ("n_input_tokens", "n_output_tokens")
        )
        summary["checks"]["usage_available"] = known
        if not known:
            summary["limitations"].append(
                "Harness complete token usage is unavailable; unknown counters remain unknown"
            )
        metadata = context.get("metadata", {}).get("fabric", {})
        summary["checks"]["harbor_result_transport"] = metadata.get(
            "adapter_id"
        ) == result.get("adapter_id") and metadata.get("invocation_id") == result.get(
            "invocation_id"
        )
        if not summary["checks"]["harbor_result_transport"]:
            fail(
                "transport_runner",
                "Harbor context did not retain Fabric invocation identity",
            )
        summary["checks"]["usage_transport"] = summary["harbor_usage"] == expected_usage
        if not summary["checks"]["usage_transport"]:
            fail(
                "usage",
                "Harbor accounting differs from Fabric's normalized invocation accounting",
            )
        if (
            known
            and expected_usage["n_input_tokens"] + expected_usage["n_output_tokens"]
            == 0
        ):
            summary["limitations"].append(
                "Harness reported no positive inference usage; accounting does not establish model activity"
            )
        if row["telemetry"] == "relay":
            validation = load(one(job, "*/agent/telemetry-validation.json"))
            trajectory = one(job, "*/agent/trajectory.json")
            from harbor.models.trajectories.trajectory import Trajectory

            Trajectory.model_validate_json(trajectory.read_text())
            summary["checks"]["telemetry_atif"] = validation["status"] == "succeeded"
            if not summary["checks"]["telemetry_atif"]:
                fail("telemetry_export", "Relay export validation failed")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        fail("transport_runner", str(error))
    return summary


def cleanup(prefix: str, log: Path) -> bool:
    """Remove only Compose resources carrying this invocation's unique task prefix."""
    cleanup_failed = False
    probes = subprocess.run(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label=org.nvidia.fabric.qualification.run={prefix}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.split()
    for resource in probes:
        if execute(["docker", "rm", "-f", resource], log, 30):
            cleanup_failed = True
    for kind, listing, deletion in (
        ("container", ["ps", "-aq"], ["rm", "-f"]),
        ("network", ["network", "ls", "-q"], ["network", "rm"]),
    ):
        found = subprocess.run(
            ["docker", *listing, "--filter", "label=com.docker.compose.project"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.split()
        for resource in found:
            inspected = subprocess.check_output(
                [
                    "docker",
                    kind,
                    "inspect",
                    "--format",
                    "{{json .Config.Labels}}"
                    if kind == "container"
                    else "{{json .Labels}}",
                    resource,
                ],
                text=True,
                timeout=30,
            )
            labels = json.loads(inspected)
            if (labels or {}).get("com.docker.compose.project", "").startswith(prefix):
                if execute(["docker", *deletion, resource], log, 30):
                    cleanup_failed = True
    return not cleanup_failed


def write_report(output: Path, report: dict) -> None:
    rows = report["adapters"]
    report["all_harnesses_passed"] = bool(rows) and all(
        row["outcome"] == "passed" for row in rows
    )
    report["full_catalog_passed"] = (
        report["scope"] == "full_catalog"
        and report["all_harnesses_passed"]
        and report.get("state", "completed") == "completed"
    )
    report["updated_at"] = now()
    report["pass_criteria"] = (
        "integration; task reward and harness usage availability are informational"
    )
    credential_names = tuple(
        name
        for row in [*rows, *report["plan"].get("adapters", [])]
        if (name := row.get("selected", row).get("api_key_env"))
    )
    save(output / "report.json", sanitize_data(report, credential_names))
    lines = [
        f"Harbor qualification: {'PASS' if report['full_catalog_passed'] else 'NOT QUALIFIED'}",
        f"Scope: {report['scope']}; denominator: {len(rows)} catalog adapters",
        f"Mode: {report['plan']['mode']}; source: {report['plan']['source_sha']}",
        f"Task: {report['plan']['task']['name']}; SHA256: {report['plan']['task']['sha256']}",
        "",
        "| Adapter | Integration | Task reward | Classification / limitations |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        reason = row.get("reason") or "; ".join(
            f"{f['category']}: {f['reason']}" for f in row.get("failures", [])
        )
        reason = "; ".join(filter(None, [reason, *row.get("limitations", [])]))
        reward = row.get("reward", "n/a")
        lines.append(
            f"| {row['adapter_id']} | {row['outcome']} | {reward} | {reason.replace('|', '/').replace(chr(10), ' ')[:600]} |"
        )
    lines.extend(
        [
            "",
            "Integration pass requires lifecycle, provenance, result/accounting transport and cleanup. Task reward and harness usage availability are informational; missing verifier evidence remains a failure.",
            "Reproduce with the recorded commands and retained environment/plan. Raw task artifacts are private; inspect and redact before sharing.",
            "MCP, skills, subagents and telemetry are untested unless explicitly exercised; one task does not establish benchmark or all-feature equivalence.",
        ]
    )
    temporary = output / "summary.tmp"
    temporary.write_text(sanitize("\n".join(lines) + "\n", credential_names))
    temporary.replace(output / "summary.md")


def run(plan: dict, environment: dict, output: Path, selected: list[str]) -> int:
    validate_plan(plan)
    if environment["plan"] != plan:
        raise ValueError("environment was prepared for a different plan")
    unknown = set(selected) - {row["adapter_id"] for row in plan["adapters"]}
    if unknown:
        raise ValueError(f"unknown subset adapters: {sorted(unknown)}")
    output.mkdir(mode=0o700)
    prefix = "fq-" + uuid.uuid4().hex[:12]
    report = {
        "format_version": 1,
        "started_at": now(),
        "finished_at": None,
        "state": "running",
        "scope": "developer_subset" if selected else "full_catalog",
        "plan": {
            key: value
            for key, value in plan.items()
            if key not in {"adapters", "catalog_snapshot"}
        },
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "host": {"hostname": os.uname().nodename, "platform": sys.platform},
        "adapters": [],
    }
    for row in plan["adapters"]:
        report["adapters"].append(
            {
                "adapter_id": row["adapter_id"],
                "outcome": "blocked",
                "reason": "Not reached",
                "capabilities": row["capabilities"],
                "registered_targets": row.get("registered_targets", []),
                "selected": {
                    "package": row["catalog"]["package"],
                    "descriptor_sha256": row["catalog"]["sha256"],
                    "model": row["model"],
                    "provider": row.get("model_provider")
                    or (
                        row["model"].split("/", 1)[0]
                        if "/" in row["model"]
                        else "openai"
                    ),
                    "base_url": row.get("base_url"),
                    "api_key_env": row.get("api_key_env"),
                    "harness_settings": row["harness_settings"],
                    "model_settings": row.get("model_settings"),
                    "workflow_target_id": row.get("workflow_target_id"),
                    "workflow_settings": row.get("workflow_settings"),
                    "max_turns": row.get("max_turns"),
                    "task_python": row.get("task_python"),
                    "telemetry": row["telemetry"],
                },
            }
        )
    write_report(output, report)
    save(output / "plan.json", plan)
    host_python = environment.get("python")
    if not host_python or not Path(host_python).is_file():
        for row, planned in zip(report["adapters"], plan["adapters"]):
            if planned["action"] != "run":
                row.update(outcome=planned["action"], reason=planned["reason"])
                continue
            row["reason"] = (
                "Host installation unavailable; inspect environment.json and base/host installation logs"
            )
        report.update(state="completed", finished_at=now())
        write_report(output, report)
        return 1
    host_identity_path = output / "host-identity.json"
    identity_script = Path(__file__).with_name("identity.py")
    if execute([host_python, str(identity_script)], output / "host-identity.log", 45):
        raise ValueError("host package identity inspection failed")
    identity_text = subprocess.check_output(
        [host_python, str(identity_script)], env=process_env(), text=True, timeout=45
    )
    host_identity = json.loads(identity_text)
    save(host_identity_path, host_identity)
    report["host"]["identity"] = host_identity
    runtime_wheel = next(
        name
        for name in environment["wheel_sha256"]
        if name.startswith("nemo_fabric_runtime-")
    )
    expected_runtime_hash = environment["wheel_sha256"][runtime_wheel]
    if (
        host_identity["editable"]
        or host_identity["packages"].get("nemo-fabric-runtime") != plan["version"]
        or host_identity["packages"].get("harbor") != plan["harbor_version"]
        or host_identity["catalog_sha256"] != fingerprint(plan_catalog(plan))
        or host_identity["wheel_sha256"].get("nemo-fabric-runtime")
        != expected_runtime_hash
    ):
        raise ValueError(
            "host has editable installs or differs from planned Fabric/Harbor versions"
        )
    for index, (row, result) in enumerate(
        zip(plan["adapters"], report["adapters"], strict=True)
    ):
        if selected and row["adapter_id"] not in selected:
            result.update(
                outcome="excluded",
                reason="Outside explicitly selected developer subset",
            )
            write_report(output, report)
            continue
        if row["action"] != "run":
            result.update(
                outcome=row["action"],
                reason=row["reason"],
                failures=[
                    {
                        "category": "provider_model"
                        if "provider" in row["reason"].lower()
                        else "prerequisite",
                        "reason": row["reason"],
                    }
                ],
            )
            write_report(output, report)
            continue
        if row.get("api_key_env") and not os.environ.get(row["api_key_env"]):
            result.update(
                reason=f"Missing credential environment variable {row['api_key_env']}",
                failures=[
                    {"category": "auth", "reason": "Required credential is unavailable"}
                ],
            )
            write_report(output, report)
            continue
        install = environment["adapters"].get(row["adapter_id"], {})
        if install.get("status") != "ready":
            result.update(
                outcome="failed",
                reason="Installation failed or was not completed",
                failures=[
                    {
                        "category": "install",
                        "reason": install.get("log", "No prepared image"),
                    }
                ],
            )
            write_report(output, report)
            continue
        directory = output / str(index)
        directory.mkdir()
        task = directory / f"{prefix}-{index}"
        image = install["image"]
        stage_task(plan, task, image, row.get("task_python", "python3"))
        jobs = directory / "jobs"
        name = f"qualification-{index}"
        command = harbor_command(plan, row, host_python, task, jobs, name)
        result.update(
            reason="Running",
            started_at=now(),
            command=shlex.join(command),
            artifact_directory=str(directory),
        )
        write_report(output, report)
        try:
            task_identity = json.loads(
                subprocess.check_output(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--network",
                        "none",
                        "--label",
                        f"org.nvidia.fabric.qualification.run={prefix}",
                        "--entrypoint",
                        "cat",
                        image,
                        "/opt/qualification-identity.json",
                    ],
                    text=True,
                    timeout=45,
                )
            )
            save(directory / "image-identity.json", task_identity)
            result["image_id"] = subprocess.check_output(
                ["docker", "image", "inspect", "--format", "{{.Id}}", image],
                text=True,
                timeout=30,
            ).strip()
            result["checks"] = {
                "host_task_runtime_version": task_identity["packages"].get(
                    "nemo-fabric-runtime"
                )
                == plan["version"]
            }
            package = row["catalog"]["package"]["name"]
            observed_packages = (
                task_identity["packages"]
                if row["catalog"]["ecosystem"] == "python"
                else task_identity["npm"]
            )
            result["checks"]["task_adapter_version"] = (
                observed_packages.get(package) == row["catalog"]["package"]["version"]
            )
            result["task_identity"] = task_identity
            if (
                not result["checks"]["host_task_runtime_version"]
                or not valid_install_sources(task_identity)
                or not result["checks"]["task_adapter_version"]
                or task_identity["catalog_sha256"] != fingerprint(plan_catalog(plan))
                or task_identity["wheel_sha256"].get("nemo-fabric-runtime")
                != expected_runtime_hash
            ):
                raise ValueError(
                    "task runtime differs from host or has editable imports"
                )
            probe = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--label",
                    f"org.nvidia.fabric.qualification.run={prefix}",
                    "--entrypoint",
                    row.get("task_python", "python"),
                    "-i",
                    image,
                    "/qualification/identity.py",
                    "--selection",
                    "-",
                ],
                input=json.dumps(selection_payload(plan, row)),
                capture_output=True,
                text=True,
                timeout=60,
                env=process_env(),
                check=False,
            )
            (directory / "selection.log").write_text(sanitize(probe.stderr))
            if probe.returncode:
                raise ValueError(
                    "adapter admission: installed descriptor preflight failed; inspect selection.log"
                )
            selected_adapter = json.loads(probe.stdout)
            result["preflight_selection"] = selected_adapter
            target = plan_catalog(plan)["targets"].get(row.get("workflow_target_id"))
            if not valid_selection(selected_adapter, row, target):
                raise ValueError(
                    "adapter admission: source or unexpected descriptor origin selected"
                )
            code = execute(
                command,
                directory / "harbor.log",
                plan["runtime_timeout_seconds"] + 600,
                key_env=row.get("api_key_env"),
                cwd=directory,
            )
            evidence = inspect_results(
                jobs / name, row, task_identity, selected_adapter, target
            )
            evidence["checks"] = {**result["checks"], **evidence["checks"]}
            result.update(evidence)
            if row["telemetry"] == "relay":
                result["capabilities"]["telemetry_atif"] = (
                    "passed" if result["checks"].get("telemetry_atif") else "failed"
                )
            result["exit_code"] = code
            if code:
                log_text = (directory / "harbor.log").read_text()[-20000:]
                result["failures"].append(
                    {
                        "category": "cancelled"
                        if code == 130
                        else "timeout"
                        if code == 124
                        else classify(log_text),
                        "reason": f"Harbor exit {code}; inspect harbor.log",
                    }
                )
            result.update(
                outcome="failed" if result["failures"] else "passed", reason=""
            )
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            result.update(
                outcome="failed",
                reason=sanitize(str(error)),
                failures=[
                    {
                        "category": "adapter_admission"
                        if "adapter admission" in str(error)
                        else "prerequisite",
                        "reason": sanitize(str(error)),
                    }
                ],
            )
            code = 1
        finally:
            try:
                result["cleanup"] = (
                    "passed" if cleanup(prefix, directory / "cleanup.log") else "failed"
                )
            except (OSError, subprocess.SubprocessError, ValueError):
                result["cleanup"] = "failed"
            if result["cleanup"] != "passed":
                result["outcome"] = "failed"
                result.setdefault("failures", []).append(
                    {
                        "category": "cleanup",
                        "reason": "Owned Compose resource cleanup failed",
                    }
                )
            # Harbor redacts its configuration. Scrub known secrets from all
            # retained text as well, including failing harness stdout.
            for path in directory.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    try:
                        text = path.read_text(encoding="utf-8")
                    except UnicodeError:
                        continue
                    cleaned = sanitize_artifact(
                        text,
                        path.suffix,
                        (row["api_key_env"],) if row.get("api_key_env") else (),
                    )
                    if cleaned != text:
                        path.write_text(cleaned, encoding="utf-8")
            result["finished_at"] = now()
            write_report(output, report)
        print(f"{row['adapter_id']}: {result['outcome']}", flush=True)
        if code == 130:
            report["state"] = "cancelled"
            break
    report.update(
        state="cancelled" if report["state"] == "cancelled" else "completed",
        finished_at=now(),
    )
    write_report(output, report)
    return (
        0
        if report["full_catalog_passed"]
        or (
            selected
            and all(
                row["outcome"] == "passed"
                for row in report["adapters"]
                if row["adapter_id"] in selected
            )
        )
        else 1
    )


def stop(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main() -> None:
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument(
        "--task",
        type=Path,
        required=True,
        help="Exactly one Harbor task directory; never a dataset root",
    )
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--mode", choices=("source", "registry"), required=True)
    plan.add_argument("--version", required=True)
    plan.add_argument("--harbor-version", required=True)
    plan.add_argument(
        "--catalog-json",
        type=Path,
        help="Catalog resource extracted from the exact published catalog wheel; required when published package metadata differs from the checkout",
    )
    plan.add_argument("--bridge", choices=("native", "sdk"), default="native")
    plan.add_argument("--model", required=True)
    plan.add_argument("--base-url")
    plan.add_argument(
        "--backend",
        default="operator-managed inference",
        help="Serving backend/version or provider deployment identity",
    )
    plan.add_argument("--api-key-env", default="MODEL_API_KEY")
    plan.add_argument(
        "--endpoint-kind",
        choices=("chat-completions", "anthropic-messages", "openai-and-anthropic"),
        default="chat-completions",
    )
    plan.add_argument("--trusted-http-host", action="append", default=[])
    plan.add_argument("--estimated-cost-usd", type=float, required=True)
    plan.add_argument("--cost-basis", required=True)
    plan.add_argument(
        "--timeout", type=int, choices=range(30, 1801), metavar="30..1800", default=240
    )
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--environment", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "plan":
        value = make_plan(args)
        validate_plan(value)
        if args.output.exists():
            parser.error("plan output already exists")
        save(args.output, value)
        print(
            f"Planned {len(value['adapters'])} adapters; estimated provider cost ${value['estimated_cost_usd']:.2f}"
        )
    else:
        raise SystemExit(
            run(
                load(args.plan),
                load(args.environment),
                args.output.resolve(),
                args.only,
            )
        )


if __name__ == "__main__":
    main()
