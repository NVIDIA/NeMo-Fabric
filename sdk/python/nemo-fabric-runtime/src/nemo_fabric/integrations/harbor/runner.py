# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Fabric SDK inside a Harbor task environment."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from nemo_fabric import Fabric
from nemo_fabric import FabricConfigError
from nemo_fabric import FabricError
from nemo_fabric import RunResult
from nemo_fabric.integrations.harbor.models import FabricRunPayload
from nemo_fabric.integrations.harbor.models import FabricRunnerError
from nemo_fabric.integrations.harbor.models import FabricRunnerFailure
from nemo_fabric.integrations.harbor.telemetry import publish_telemetry_evidence


async def run(payload: FabricRunPayload) -> RunResult:
    config = payload.config.model_copy(deep=True)
    for name in payload.environment_env_names:
        if name not in os.environ:
            raise ValueError(f"Harbor runner environment variable {name} is not set")
        config.environment.env[name] = os.environ[name]
    if payload.skills_dir is not None:
        base_dir = Path(payload.config_base_dir).resolve()
        root = Path(payload.skills_dir)
        if not root.is_absolute():
            root = base_dir / root
        if not root.is_dir():
            raise FabricConfigError(
                f"Harbor skills collection must be an existing task-side directory: {root}",
                stage="configuration",
                code="harbor_skills_invalid",
            )
        skills = sorted(root.iterdir())
        for skill in skills:
            skill_file = skill / "SKILL.md"
            if (
                not skill.is_dir()
                or skill_file.is_symlink()
                or not skill_file.is_file()
            ):
                raise FabricConfigError(
                    "Harbor skills collection entries must be directories "
                    f"containing a regular SKILL.md file: {skill}",
                    stage="configuration",
                    code="harbor_skills_invalid",
                )
        existing_paths = (
            {(base_dir / path).resolve() for path in config.skills.paths}
            if config.skills is not None
            else set()
        )
        for skill in skills:
            resolved = skill.resolve()
            if resolved not in existing_paths:
                config.add_skill_path(skill)
                existing_paths.add(resolved)
    compatibility = (
        {"expected_descriptor_sha256": payload.adapter_descriptor_sha256}
        if payload.adapter_descriptor_sha256 is not None
        else {}
    )
    result = await Fabric().run(
        config,
        base_dir=payload.config_base_dir,
        request=payload.request,
        **compatibility,
    )
    publish_telemetry_evidence(
        result,
        Path(payload.logs_dir),
        harbor_session_id=payload.request.context.get("harbor_session_id"),
        harbor_context_id=payload.request.context.get("harbor_context_id"),
        runtime_stopped=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    args.result.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = FabricRunPayload.model_validate_json(
            args.spec.read_text(encoding="utf-8")
        )
        result = asyncio.run(run(payload))
    except Exception as error:
        diagnostic = FabricRunnerError(
            stage=error.stage if isinstance(error, FabricError) else None,
            code=error.code if isinstance(error, FabricError) else None,
            message=(
                str(error)
                if isinstance(error, FabricError)
                else f"Fabric runner failed ({type(error).__name__})"
            ),
            retryable=error.retryable if isinstance(error, FabricError) else False,
        )
        args.result.write_text(
            FabricRunnerFailure(runner_error=diagnostic).model_dump_json(indent=2),
            encoding="utf-8",
        )
        raise SystemExit(1) from None
    args.result.write_text(json.dumps(result.to_mapping(), indent=2), encoding="utf-8")
    if result.status != "succeeded" or result.error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
