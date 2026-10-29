# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build isolated host/task installs for the Harbor qualification plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tarfile
from pathlib import Path

from qualify import ROOT, execute, load, save, stop, validate_plan


def base_environment(plan: dict, work: Path, host_python: str) -> dict:
    context = work / "base"
    context.mkdir()
    archive = context / "source.tar"
    with archive.open("wb") as stream:
        subprocess.run(
            ["git", "archive", plan["source_sha"]],
            cwd=ROOT,
            stdout=stream,
            check=True,
            timeout=120,
        )
    source = context / "source"
    source.mkdir()
    with tarfile.open(archive) as bundle:
        bundle.extractall(source, filter="data")
    archive.unlink()
    shutil.copyfile(Path(__file__).parent / "identity.py", context / "identity.py")
    shutil.copyfile(
        Path(__file__).parents[1] / "assets/Dockerfile", context / "Dockerfile"
    )
    run_id = hashlib.sha256(str(work.resolve()).encode()).hexdigest()[:12]
    tag = f"fabric-qualification-base:{plan['source_sha'][:12]}-{plan['mode']}-{run_id}"
    task_base = tag + "-task"
    if execute(
        [
            "docker",
            "build",
            "--progress=plain",
            "-t",
            task_base,
            str(Path(plan["task"]["path"]) / "environment"),
        ],
        work / "task-build.log",
        900,
    ):
        raise RuntimeError("task environment build failed; inspect task-build.log")
    code = execute(
        [
            "docker",
            "build",
            "--progress=plain",
            "--build-arg",
            f"FABRIC_VERSION={plan['version']}",
            "--build-arg",
            f"INSTALL_MODE={plan['mode']}",
            "--build-arg",
            f"TASK_IMAGE={task_base}",
            "-t",
            tag,
            str(context),
        ],
        work / "base-build.log",
        3600,
    )
    if code:
        raise RuntimeError("base installation failed; inspect base-build.log")
    wheels = work / "wheels"
    container = subprocess.check_output(
        ["docker", "create", tag], text=True, timeout=30
    ).strip()
    try:
        subprocess.run(
            ["docker", "cp", f"{container}:/wheels", str(wheels)],
            check=True,
            timeout=120,
        )
    finally:
        subprocess.run(
            ["docker", "rm", container], check=True, capture_output=True, timeout=30
        )
    venv = work / "host"
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    for command in (
        [uv, "venv", "--python", host_python, str(venv)],
        [str(venv / "bin/python"), "-m", "ensurepip"],
        [
            str(venv / "bin/python"),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--retries",
            "1",
            "--timeout",
            "30",
            "--find-links",
            str(wheels),
            *[
                str(path)
                for path in sorted(wheels.glob("*.whl"))
                if path.name.split(f"-{plan['version']}-")[0]
                in {
                    "nemo_fabric",
                    "nemo_fabric_runtime",
                    "nemo_fabric_adapter_catalog",
                    "nemo_fabric_adapter_contract",
                    "nemo_fabric_adapters_common",
                }
            ],
            f"harbor=={plan['harbor_version']}",
        ],
    ):
        if execute(command, work / "host-install.log", 900):
            raise RuntimeError("host installation failed; inspect host-install.log")
    return {
        "base_image": tag,
        "task_base_image": task_base,
        "task_base_image_id": subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", task_base],
            text=True,
            timeout=30,
        ).strip(),
        "python": str(venv / "bin/python"),
        "wheels": str(wheels),
        "wheel_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in wheels.glob("*.whl")
        },
    }


def adapter_dockerfile(plan: dict, row: dict, base: str) -> str:
    record = row["catalog"]
    leaf = Path(record["source"]).parent.name
    name = record["package"]["name"]
    version = record["package"]["version"]
    lines = [f"FROM {base}"]
    if record["ecosystem"] == "python":
        manifest = load_manifest(
            ROOT / Path(record["source"]).parent / "pyproject.toml"
        )
        extra = (
            "[harness]"
            if "harness" in manifest["project"].get("optional-dependencies", {})
            else ""
        )
        requirement = (
            f"{name}{extra} @ file:///wheels/{name.replace('-', '_')}-{version}-py3-none-any.whl"
            if plan["mode"] == "source"
            else f"{name}{extra}=={version}"
        )
        lines.append(f'RUN pip install --find-links /wheels "{requirement}"')
    else:
        directory = f"/src/adapters/typescript/{leaf}"
        if plan["mode"] == "source":
            lines.extend(
                [
                    "RUN npm ci --prefix /src/adapter-contract/typescript --ignore-scripts",
                    f"RUN npm ci --prefix /src/adapters/typescript --workspace {name} --include-workspace-root --ignore-scripts",
                    "RUN npm run build --prefix /src/adapter-contract/typescript",
                    "RUN npm run build --prefix /src/adapters/typescript --workspace nemo-fabric-adapters-common",
                    f"RUN npm run build --prefix /src/adapters/typescript --workspace {name}",
                    "RUN mkdir /packs && npm pack --prefix /src/adapter-contract/typescript /src/adapter-contract/typescript --pack-destination /packs --ignore-scripts",
                    "RUN npm pack /src/adapters/typescript/common --pack-destination /packs --ignore-scripts",
                    f"RUN npm pack {directory} --pack-destination /packs --ignore-scripts",
                    "RUN mkdir -p /opt/harness && npm install --prefix /opt/harness /packs/*.tgz --omit=dev",
                ]
            )
        else:
            lines.append(
                f"RUN mkdir -p /opt/harness && npm install --prefix /opt/harness {name}@{version} --omit=dev"
            )
        manifest = load(ROOT / Path(record["source"]).parent / "package.json")
        peers = manifest.get("peerDependencies", {})
        tested = manifest.get("devDependencies", {})
        if peers:
            specs = [f"{peer}@{tested.get(peer, pin)}" for peer, pin in peers.items()]
            lines.append("RUN npm install --prefix /opt/harness " + " ".join(specs))
    # These harnesses deliberately are not dependencies of their adapter packages.
    installs = {
        "nvidia.fabric.droid": "npm install --global droid@0.233.0",
        "nvidia.fabric.openhands": (
            'pip install "openhands-sdk==1.50.0" "openhands-tools==1.50.0" '
            '"litellm==1.100.1" "mcp==1.28.1" && pip check'
        ),
        "nvidia.fabric.openclaw": "npm install --global openclaw@2026.9.4 --allow-scripts=openclaw",
        "nvidia.fabric.cline": "npm install --prefix /opt/harness @cline/sdk@0.0.83",
        "nvidia.fabric.strands": "npm install --global @strands-agents/cli@0.2.1",
        "nvidia.fabric.opencode": "npm install --global bun@1.4.2",
    }
    if row["adapter_id"] in installs:
        lines.append("RUN " + installs[row["adapter_id"]])
    lines.extend(
        [
            "RUN rm -rf /src /packs",
            "COPY identity.py /qualification/identity.py",
            "RUN python /qualification/identity.py > /opt/qualification-identity.json",
            'LABEL org.nvidia.fabric.qualification="true"',
            f'LABEL org.nvidia.fabric.source="{plan["source_sha"]}"',
            "WORKDIR /app",
        ]
    )
    return "\n".join(lines) + "\n"


def load_manifest(path: Path) -> dict:
    import tomllib

    return tomllib.loads(path.read_text())


def main() -> None:
    os.umask(0o077)
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument(
        "--host-python",
        default="3.12",
        help="Host interpreter version or existing executable; must support Harbor (Python >=3.12)",
    )
    args = parser.parse_args()
    plan = load(args.plan)
    validate_plan(plan)
    work = args.work.resolve()
    work.mkdir(mode=0o700)  # Never overwrite another run or environment.
    environment = {"plan": plan, "adapters": {}}
    save(work / "environment.json", environment)
    try:
        environment.update(base_environment(plan, work, args.host_python))
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        environment["error"] = str(error)
        save(work / "environment.json", environment)
        raise SystemExit(1) from error
    save(work / "environment.json", environment)
    for index, row in enumerate(plan["adapters"]):
        if row["action"] != "run":
            continue
        directory = work / str(index)
        directory.mkdir()
        tag = (
            row.get("image")
            or f"fabric-qualification:{environment['base_image'].split(':', 1)[1]}-{index}"
        )
        if not row.get("image"):
            shutil.copyfile(
                Path(__file__).with_name("identity.py"), directory / "identity.py"
            )
            (directory / "Dockerfile").write_text(
                adapter_dockerfile(plan, row, environment["base_image"])
            )
            code = execute(
                ["docker", "build", "--progress=plain", "-t", tag, str(directory)],
                directory / "install.log",
                1800,
            )
        else:
            code = execute(
                ["docker", "image", "inspect", tag], directory / "install.log", 30
            )
        environment["adapters"][row["adapter_id"]] = {
            "image": tag,
            "status": "failed" if code else "ready",
            "install_exit_code": code,
            "log": str(directory / "install.log"),
        }
        save(work / "environment.json", environment)
        if code == 130:
            raise SystemExit(130)
        print(
            f"{row['adapter_id']}: {'install failed' if code else 'ready'}", flush=True
        )


if __name__ == "__main__":
    main()
