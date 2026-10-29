# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write package provenance without collecting environment variables or credentials."""

from __future__ import annotations

import importlib.metadata
from importlib.resources import files
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit


def selection(payload: dict) -> dict:
    from nemo_fabric import Fabric, FabricConfig
    from nemo_fabric.capabilities import _adapter_descriptor_sha256

    plan = Fabric().plan(
        FabricConfig.from_mapping(payload["config"]),
        base_dir=payload["config_base_dir"],
    )
    resolved = plan.to_mapping()
    descriptor = resolved["adapter_descriptor"]
    selected = {
        "adapter_id": descriptor["descriptor"]["adapter_id"],
        "python_prefix": sys.prefix,
        "provenance": descriptor["provenance"],
        "runner": descriptor["descriptor"]["runner"],
        "admission_sha256": _adapter_descriptor_sha256(plan),
    }
    target = resolved.get("adapter_target_descriptor")
    if target is not None:
        selected["workflow"] = {
            "target_id": target["descriptor"]["id"],
            "entrypoint": target["descriptor"]["spec"]["entrypoint"],
            "provenance": target["provenance"],
        }
    return selected


def identity() -> dict:
    import nemo_fabric

    packages = {}
    editable = []
    editable_sources = {}
    archives = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        packages[name] = distribution.version
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        if direct.get("dir_info", {}).get("editable"):
            editable.append(name)
            url = urlsplit(direct.get("url", ""))
            if url.scheme == "file":
                source = Path(unquote(url.path))
                revision = subprocess.run(
                    ["git", "-C", str(source), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                status = subprocess.run(
                    ["git", "-C", str(source), "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                editable_sources[name] = {
                    "url": direct["url"],
                    "revision": revision.stdout.strip()
                    if revision.returncode == 0
                    else None,
                    "dirty": status.returncode != 0 or bool(status.stdout.strip()),
                }
        if name.startswith("nemo-fabric"):
            archives[name] = (
                direct.get("archive_info", {}).get("hashes", {}).get("sha256")
            )
    npm = (
        subprocess.run(
            ["npm", "ls", "--prefix", "/opt/harness", "--json", "--depth=0"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if shutil.which("npm")
        else None
    )
    # npm may include local paths and diagnostic prose; retain versions only.
    dependencies = json.loads(npm.stdout or "{}").get("dependencies", {}) if npm else {}
    global_npm = (
        subprocess.run(
            ["npm", "ls", "--global", "--json", "--depth=0"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if npm
        else None
    )
    global_dependencies = (
        json.loads(global_npm.stdout or "{}").get("dependencies", {})
        if global_npm
        else {}
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": str(Path(sys.executable).resolve()),
        "runtime_origin": nemo_fabric.__file__,
        "packages": dict(sorted(packages.items())),
        "editable": sorted(editable),
        "editable_sources": editable_sources,
        "wheel_sha256": archives,
        "catalog_sha256": hashlib.sha256(
            json.dumps(
                json.loads(
                    files("nemo_fabric_adapter_catalog")
                    .joinpath("catalog.json")
                    .read_text()
                ),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest(),
        "npm": {name: info.get("version") for name, info in dependencies.items()},
        "npm_global": {
            name: info.get("version") for name, info in global_dependencies.items()
        },
        "harness_sources": json.loads(
            Path("/opt/qualification-harness-sources.json").read_text()
        )
        if Path("/opt/qualification-harness-sources.json").exists()
        else {},
    }


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--selection":
        payload = (
            json.load(sys.stdin)
            if sys.argv[2] == "-"
            else json.loads(Path(sys.argv[2]).read_text())
        )
        result = selection(payload)
    else:
        result = identity()
    print(json.dumps(result, indent=2, sort_keys=True))
