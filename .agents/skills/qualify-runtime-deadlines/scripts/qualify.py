#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare, execute and verify source-revision deadline qualification."""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
HARNESSES = ("hermes", "deepagents")
LOCATIONS = ("host", "openshell")
CASES = ("smoke", "finite", "unlimited", "timeout")
HERMES_REVISION = "bf0203a794a07aac3ff764972ab9087baa4e4f4e"
PROJECTS = (
    "sdk/python/nemo-fabric-runtime",
    "sdk/python/nemo-fabric-evidence",
    "sdk/python/nemo-fabric-adapter-catalog",
    "sdk/python/nemo-fabric-collector",
    "sdk/python/nemo-fabric",
    "adapter-contract/python",
    "adapters/python/common",
    "adapters/python/hermes",
    "adapters/python/deepagents",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def selected(value, allowed):
    result = value.split(",")
    require(
        result and len(result) == len(set(result)) and set(result) <= set(allowed),
        "invalid coverage selection",
    )
    return result


def command(argv, env, log, timeout=1800):
    with Path(log).open("w") as output:
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    require(returncode == 0, f"command failed; inspect {log}")


def python_json(python, code):
    return json.loads(
        subprocess.check_output(
            [str(python), "-c", code],
            text=True,
            env={
                k: v
                for k, v in os.environ.items()
                if k not in ("PYTHONPATH", "PYTHONHOME")
            },
        )
    )


def check_checkout(revision):
    require(
        git("rev-parse", "HEAD") == revision,
        "selected revision does not match this checkout",
    )
    require(
        not git("status", "--porcelain", "--untracked-files=no"),
        "tracked source changes invalidate qualification",
    )


def make_plan(args):
    revision = git("rev-parse", f"{args.revision}^{{commit}}")
    check_checkout(revision)
    require(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env),
        "invalid credential variable name",
    )
    for url in (args.model_url, args.sandbox_model_url or args.model_url):
        parsed = urlsplit(url)
        require(
            parsed.scheme in ("http", "https")
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment,
            "model URL must not contain credentials or query parameters",
        )
    source = Path(args.hermes_source).resolve()
    require(
        git("rev-parse", "HEAD", cwd=source) == HERMES_REVISION,
        "Hermes checkout does not match the task image pin",
    )
    require(
        not git("status", "--porcelain", cwd=source),
        "Hermes source checkout must be clean",
    )
    plan = dict(
        schema_version=1,
        revision=revision,
        model_url=args.model_url,
        sandbox_model_url=args.sandbox_model_url or args.model_url,
        model=args.model,
        api_key_env=args.api_key_env,
        policy=str(Path(args.policy).resolve()),
        hermes_source=str(source),
        hermes_revision=HERMES_REVISION,
        dependency_python={
            name: str(Path(getattr(args, name + "_python")).absolute())
            for name in HARNESSES
        },
        cases=selected(args.cases, CASES),
        locations=selected(args.locations, LOCATIONS),
    )
    plan["policy_sha256"] = digest(plan["policy"])
    output = Path(args.output).resolve()
    require(not output.exists(), "choose a new plan path")
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, plan)
    print(
        f"Saved {len(HARNESSES) * len(plan['cases']) * len(plan['locations'])} inference cells to {output}"
    )


def prepare_hosts(plan, output, env):
    wheels = output / "wheels"
    wheels.mkdir()
    for index, project in enumerate(PROJECTS):
        arguments = ["uv", "build", "--wheel", "--out-dir", str(wheels)]
        if project == "sdk/python/nemo-fabric-runtime":
            arguments += ["--config-setting", "build-args=--locked"]
        command([*arguments, project], env, output / f"build-{index}.log")
    installation = dict(
        revision=plan["revision"],
        wheels={str(path): digest(path) for path in wheels.glob("*.whl")},
        hosts={},
    )
    require(len(installation["wheels"]) == len(PROJECTS), "missing SDK/adapter wheel")
    for harness in HARNESSES:
        dependency_python = plan["dependency_python"][harness]
        dependency_site = python_json(
            dependency_python,
            "import json,sysconfig; print(json.dumps(sysconfig.get_path('purelib')))",
        )
        host = output / f"host-{harness}"
        command(
            ["uv", "venv", "--python", dependency_python, str(host)],
            env,
            output / f"venv-{harness}.log",
        )
        python = host / "bin/python"
        command(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-deps",
                *installation["wheels"],
            ],
            env,
            output / f"install-{harness}.log",
        )
        site = Path(
            python_json(
                python,
                "import json,sysconfig; print(json.dumps(sysconfig.get_path('purelib')))",
            )
        )
        # A path entry does not execute .pth files in the dependency environment.
        paths = [dependency_site] + (
            [plan["hermes_source"]] if harness == "hermes" else []
        )
        (site / "qualification-dependencies.pth").write_text("\n".join(paths) + "\n")
        probe = python_json(
            python,
            """import importlib,importlib.metadata as m,json,sys
names=['nemo_fabric','nemo_fabric._native','nemo_fabric.models','nemo_fabric_adapter_contract','nemo_fabric_adapters.common','nemo_fabric.runtime','nemo_fabric_adapters.HARNESS_NAME.adapter']
print(json.dumps({'python':sys.executable,'modules':{name:importlib.import_module(name).__file__ for name in names},'versions':{name:m.version(name) for name in ['nemo-fabric-runtime','nemo-fabric-evidence','nemo-fabric-adapters-hermes','nemo-fabric-adapters-deepagents']},'harness_version':m.version('hermes-agent' if 'HARNESS_NAME'=='hermes' else 'deepagents'),'direct_url':json.loads(m.distribution('nemo-fabric-runtime').read_text('direct_url.json'))}))""".replace(
                "HARNESS_NAME", harness
            ),
        )
        require(
            all(
                Path(path).resolve().is_relative_to(host)
                for path in probe["modules"].values()
            ),
            "ambient SDK/adapter imports do not qualify",
        )
        require(
            not probe["direct_url"].get("dir_info", {}).get("editable"),
            "editable SDK installs do not qualify",
        )
        probe["sha256"] = {path: digest(path) for path in probe["modules"].values()}
        installation["hosts"][harness] = probe
    save(output / "installation.json", installation)
    return installation


def validate_report(report, revision, release=False):
    require(
        report["schema_version"] == 1 and report["revision"] == revision,
        "qualification revision mismatch",
    )
    require(report["status"] == "passed", "qualification did not pass")
    full = set(report["cases"]) == set(CASES) and set(report["locations"]) == set(
        LOCATIONS
    )
    require(not release or full, "developer subsets cannot satisfy the release gate")
    required_artifacts = {
        report[key]
        for key in ("installation", "inference", "transport_junit", "inference_junit")
    }
    require(required_artifacts <= set(report["artifacts"]), "missing evidence hashes")
    for path, expected in report["artifacts"].items():
        require(
            Path(path).is_file() and digest(path) == expected,
            f"missing or changed evidence: {path}",
        )
    installation = json.loads(Path(report["installation"]).read_text())
    require(installation["revision"] == revision, "SDK installation revision mismatch")
    require(set(installation["hosts"]) == set(HARNESSES), "missing host installations")
    require(len(installation["wheels"]) == len(PROJECTS), "missing SDK/adapter wheels")
    for path, expected in installation["wheels"].items():
        require(digest(path) == expected, "SDK wheel changed after qualification")
    for host in installation["hosts"].values():
        require(
            set(host["sha256"]) == set(host["modules"].values())
            and len(host["modules"]) == 7,
            "incomplete SDK import evidence",
        )
        prefix = Path(host["python"]).absolute().parent.parent
        require(
            all(
                Path(path).resolve().is_relative_to(prefix)
                for path in host["modules"].values()
            ),
            "ambient SDK import evidence",
        )
        require(
            not host["direct_url"].get("dir_info", {}).get("editable"),
            "editable SDK installation",
        )
        for path, expected in host["sha256"].items():
            require(
                digest(path) == expected, "installed SDK changed after qualification"
            )
    for name, expected_names in {
        "transport": {
            "test_live_invocations_outlast_300_seconds_with_finite_and_no_timeout",
            "test_live_finite_deadline_keeps_timeout_reason",
        },
        "inference": {"test_code_review_deadline_matrix"},
    }.items():
        cases = list(ET.parse(report[name + "_junit"]).getroot().iter("testcase"))
        require(
            {case.attrib["name"] for case in cases} == expected_names
            and len(cases) == len(expected_names),
            "missing or duplicate workload tests",
        )
        require(
            not any(
                child.tag in ("failure", "error", "skipped")
                for case in cases
                for child in case
            ),
            "failed or skipped workload test",
        )
    records = json.loads(Path(report["inference"]).read_text())["records"]
    expected = set(itertools.product(HARNESSES, report["locations"], report["cases"]))
    actual = [(row["harness"], row["location"], row["mode"]) for row in records]
    require(
        len(actual) == len(expected) and set(actual) == expected,
        "missing or duplicate inference cells",
    )
    for row in records:
        require(
            row["status"] == "passed"
            and row["cleanup"] == "passed"
            and not row.get("cleanup_errors"),
            "inference or cleanup failure",
        )
        mode, elapsed = row["mode"], row["elapsed_seconds"]
        if mode == "timeout":
            error = row.get("error") or row.get("result", {}).get("error", {})
            require(
                error.get("code") == "timeout"
                and error.get("stage") == "invoke"
                and 14 <= elapsed < 45,
                "invalid timeout evidence",
            )
            require(row["timeout_seconds"] == 15, "wrong enforced deadline")
        else:
            result = row["result"]
            require(
                result["status"] == "succeeded"
                and result["output"]["completed"] is True,
                "review did not complete",
            )
            text = json.dumps(result["output"]["response"])
            require(
                row["expected_receipt"]
                and row["expected_receipt"] in text
                and "41" in text,
                "missing real-source completion receipt",
            )
            if mode in ("finite", "unlimited"):
                require(
                    elapsed > 320
                    and row["timeout_seconds"] == (900 if mode == "finite" else None),
                    "long invocation/deadline not established",
                )
    return full


def run_plan(args):
    plan = json.loads(Path(args.plan).read_text())
    check_checkout(plan["revision"])
    full = set(plan["cases"]) == set(CASES) and set(plan["locations"]) == set(LOCATIONS)
    require(not args.release or full, "release runs require the full matrix")
    require(
        digest(plan["policy"]) == plan["policy_sha256"],
        "network policy changed after planning",
    )
    require(
        git("rev-parse", "HEAD", cwd=plan["hermes_source"]) == plan["hermes_revision"]
        and not git("status", "--porcelain", cwd=plan["hermes_source"]),
        "Hermes source changed after planning",
    )
    credential = os.environ.get(plan["api_key_env"])
    require(credential, "export the selected credential variable before running")
    request = Request(
        plan["model_url"].rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + credential},
    )
    with urlopen(request, timeout=15) as response:
        require(
            plan["model"] in {item["id"] for item in json.load(response)["data"]},
            "served model ID does not match plan",
        )
    output = Path(args.output).resolve()
    require(not output.exists(), "choose a new output directory")
    output.mkdir(parents=True)
    save(output / "plan.json", plan)
    report = dict(
        schema_version=1,
        revision=plan["revision"],
        cases=plan["cases"],
        locations=plan["locations"],
        status="failed",
        artifacts={},
    )
    env = dict(os.environ)
    # A source wheel must be rebuilt from this checkout, not an ambient target override.
    env.pop("CARGO_TARGET_DIR", None)
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    try:
        installation = prepare_hosts(plan, output, env)
        report["installation"] = str(output / "installation.json")
        consumer = str(
            Path(installation["hosts"]["deepagents"]["python"]).parent.parent
        )
        env.update(
            UV_PROJECT_ENVIRONMENT=consumer,
            OPENSHELL_EXAMPLE_MODE="fabric_owned",
            CODE_REVIEW_MODEL_URL=plan["model_url"],
            CODE_REVIEW_SANDBOX_MODEL_URL=plan["sandbox_model_url"],
            CODE_REVIEW_MODEL=plan["model"],
            OPENAI_API_KEY=credential,
            CODE_REVIEW_POLICY=plan["policy"],
            CODE_REVIEW_HERMES_PYTHON=installation["hosts"]["hermes"]["python"],
            CODE_REVIEW_DEEPAGENTS_PYTHON=installation["hosts"]["deepagents"]["python"],
            CODE_REVIEW_DEADLINE_CASES=",".join(plan["cases"]),
            CODE_REVIEW_DEADLINE_LOCATIONS=",".join(plan["locations"]),
            CODE_REVIEW_EVIDENCE=str(output / "inference.json"),
        )
        for name, workload, port in (
            ("transport", "openshell-timeouts", args.port),
            ("inference", "code-review-deadlines", args.port + 1),
        ):
            report[name + "_junit"] = str(output / (name + ".xml"))
            env.update(
                OPENSHELL_EXAMPLE_PORT=str(port),
                OPENSHELL_EXAMPLE_RESULTS_DIR=str(output / name),
                PYTEST_ADDOPTS=f"--junitxml={report[name + '_junit']}",
            )
            print(
                f"Running {name} qualification; logs: {output / (name + '.log')}",
                flush=True,
            )
            command(
                [
                    "bash",
                    "examples/openshell/run-demo.sh",
                    f"tests/e2e/{workload}-workload.sh",
                ],
                env,
                output / (name + ".log"),
                timeout=3600,
            )
        report["inference"] = str(output / "inference.json")
        check_checkout(plan["revision"])
        require(
            digest(plan["policy"]) == plan["policy_sha256"],
            "network policy changed during qualification",
        )
        require(
            git("rev-parse", "HEAD", cwd=plan["hermes_source"])
            == plan["hermes_revision"]
            and not git("status", "--porcelain", cwd=plan["hermes_source"]),
            "Hermes source changed during qualification",
        )
        provider = output / "fabric-environment-openshell"
        shutil.copy2(ROOT / "target/debug/fabric-environment-openshell", provider)
        retained = [
            output / "plan.json",
            output / "installation.json",
            output / "inference.json",
            output / "transport.xml",
            output / "inference.xml",
            provider,
            output / "transport/image-id.txt",
            output / "inference/image-id.txt",
        ]
        report["artifacts"] = {str(path): digest(path) for path in retained}
        report["status"] = "passed"
        validate_report(report, plan["revision"], args.release)
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = str(error)
        raise
    finally:
        save(output / "report.json", report)
    print(
        f"{'Release gate' if full else 'Developer subset'} passed: {output / 'report.json'}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    for name in (
        "revision",
        "model-url",
        "model",
        "hermes-python",
        "deepagents-python",
        "hermes-source",
        "policy",
        "output",
    ):
        plan.add_argument("--" + name, required=True)
    plan.add_argument("--sandbox-model-url")
    plan.add_argument("--api-key-env", default="OPENAI_API_KEY")
    plan.add_argument("--cases", default=",".join(CASES))
    plan.add_argument("--locations", default=",".join(LOCATIONS))
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--port", type=int, default=18110)
    run.add_argument("--release", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--report", required=True)
    verify.add_argument("--revision", required=True)
    verify.add_argument("--release", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "plan":
            make_plan(args)
        elif args.command == "run":
            run_plan(args)
        else:
            report = json.loads(Path(args.report).read_text())
            full = validate_report(report, args.revision, args.release)
            print(
                "Release gate passed"
                if full
                else "Developer subset passed; not release-qualified"
            )
    except (
        ValueError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
        ET.ParseError,
    ) as error:
        parser.exit(1, f"Qualification failed: {error}\n")


if __name__ == "__main__":
    main()
