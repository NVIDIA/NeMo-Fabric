# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line telemetry quality gate for one Fabric Harbor run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nemo_fabric import RunResult
from nemo_fabric.integrations.harbor.telemetry import publish_telemetry_evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ATOF/ATIF and publish Harbor agent/trajectory.json",
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--harbor-session-id")
    parser.add_argument("--harbor-context-id")
    args = parser.parse_args()

    result = RunResult.from_mapping(json.loads(args.result.read_text(encoding="utf-8")))
    summary = publish_telemetry_evidence(
        result,
        args.logs_dir,
        strict=True,
        harbor_session_id=args.harbor_session_id,
        harbor_context_id=args.harbor_context_id,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
