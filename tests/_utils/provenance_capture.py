# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in acceptance artifacts and private comparison fixtures, never SDK APIs."""

import json
import os
import re
from collections import Counter
from pathlib import Path


def capture_result(name, result):
    destination = os.environ.get("FABRIC_PROVENANCE_ARTIFACTS")
    if destination:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        value = result.to_mapping() if hasattr(result, "to_mapping") else result
        (root / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")


def version_view(result):
    """Focused comparison projection: versions, topology and unknowns stay separate."""
    value = result.to_mapping() if hasattr(result, "to_mapping") else result
    execution = value.get("execution")
    if execution is None:
        return {
            "versions": [],
            "unknowns": [("execution", "missing")],
            "coverage": [],
            "topology": [],
        }
    versions, unknowns, coverage, topology = [], [], [], []
    contexts = Counter()
    components = Counter()
    participants_by_ref = {}
    for env in execution["software_environments"]:
        participants = tuple(sorted(env["participants"]))
        participants_by_ref[env["context_ref"]] = participants
        contexts[participants] += 1
        topology.append(
            (participants, env["status"], env["complete"], env.get("reason"))
        )
        if env["status"] != "available" or not env["complete"]:
            unknowns.append((participants, "inventory", env.get("reason")))
        for entry in env["entries"]:
            name = entry["name"]
            if entry["kind"] == "python_distribution":
                name = re.sub(r"[-_.]+", "-", name).lower()
            elif entry["kind"] == "npm_package":
                name = name.lower()
            key = (participants, entry["kind"], name)
            components[key] += 1
            observation = entry["version"]
            coverage.append(
                (
                    *key,
                    entry["role"],
                    observation["status"],
                    observation["origin"],
                    observation["source"],
                    observation.get("reason"),
                )
            )
            if observation["status"] == "available":
                versions.append((*key, observation["value"]))
            else:
                unknowns.append((*key, observation["status"], observation["reason"]))
    for category, report in (execution.get("infrastructure") or {}).items():
        served = tuple(
            sorted(participants_by_ref.get(ref, ()) for ref in report["context_refs"])
        )
        participants = tuple(
            sorted({participant for group in served for participant in group})
        )
        topology.append((participants, "infrastructure", category, served))
        if not served or any(not group for group in served):
            unknowns.append((participants, category, "unmatched_context"))
        fields = (
            ("provider", "service_version")
            if category == "sandbox"
            else ("engine", "engine_version")
        )
        for name in fields:
            observation = report[name]
            key = (participants, "infrastructure:" + category, name)
            coverage.append(
                (
                    *key,
                    name,
                    observation["status"],
                    observation["origin"],
                    observation["source"],
                    observation.get("reason"),
                )
            )
            if observation["status"] == "available":
                versions.append((*key, observation["value"]))
            else:
                unknowns.append((*key, observation["status"], observation["reason"]))
    unknowns.extend(
        (key, "ambiguous_participant") for key, count in contexts.items() if count > 1
    )
    unknowns.extend(
        (key, "ambiguous_component") for key, count in components.items() if count > 1
    )
    # No IDs, capture times, installation paths, or artifact roots enter this view.
    return {
        "versions": sorted(versions),
        "unknowns": sorted(unknowns, key=repr),
        "coverage": sorted(coverage, key=repr),
        "topology": sorted(topology, key=repr),
    }
