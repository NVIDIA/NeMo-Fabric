# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, selected local metadata lookups; never import or execute a target.

Package lookup and loaded-file attribution adapted from Maryam's NVIDIA/NeMo-
Fabric#458, commit 2bfd53321fa8d541d42abc195be5e9dd9db4350f, provenance.py.
The report representation, original times, budgets and full values differ.
"""

from __future__ import annotations

import importlib.metadata
import csv
import io
from email.parser import BytesHeaderParser
import json
import sys
import threading
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path, PurePosixPath

from nemo_fabric_adapter_contract.evidence import (
    ExecutionEvidence,
    MAX_REPORT_BYTES,
    SoftwareEntry,
    SoftwareEnvironment,
    VersionObservation,
)

MAX_METADATA_FILES = 4096
MAX_HEADER_BYTES = 64 * 1024
MAX_RECORD_BYTES = 512 * 1024
MAX_TARGETS = 32
COLLECTION_SECONDS = 0.25
# At most one outstanding metadata worker per process, including timed-out work.
# An unresponsive filesystem cannot accumulate workers or block invocations.
_worker = threading.Lock()


def observed_at() -> str:
    """Original capture time in UTC, independent of forwarding time."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def observation(
    value: str | None,
    source: str,
    captured: str,
    *,
    reason: str = "no_existing_version_report",
    origin: str = "observed",
) -> VersionObservation:
    """Build an observation without guessing or shortening a version."""
    return VersionObservation(
        status="available" if value else "unavailable",
        value=value or None,
        reason=None if value else reason,
        source=source,
        origin=origin,
        observed_at=captured,
    )


def package_version(
    distribution: str, module_name: str | None, source: str, captured: str
) -> VersionObservation:
    """Read selected installed metadata; attribute loaded modules when requested."""
    try:
        installed = importlib.metadata.distribution(distribution)
        # PathDistribution's read_text/files/version APIs load entire files.
        # Keep their path access in this one adapter and fail closed for other
        # metadata backends; never invoke an unbounded fallback reader.
        if not isinstance(installed, importlib.metadata.PathDistribution):
            return observation(
                None, source, captured, reason="metadata_backend_unsupported"
            )
        root = installed._path
        if module_name is not None:
            module_file = getattr(sys.modules.get(module_name), "__file__", None)
            if not isinstance(module_file, str):
                return observation(None, source, captured, reason="module_not_loaded")
            loaded = Path(module_file).resolve()
            try:
                with root.joinpath("RECORD").open("rb") as stream:
                    record = stream.read(MAX_RECORD_BYTES + 1)
            except FileNotFoundError:
                return observation(
                    None, source, captured, reason="metadata_file_list_unavailable"
                )
            if len(record) > MAX_RECORD_BYTES:
                return observation(None, source, captured, reason="metadata_byte_limit")
            owned = False
            for index, row in enumerate(
                csv.reader(io.StringIO(record.decode("utf-8")))
            ):
                if index >= MAX_METADATA_FILES:
                    return observation(
                        None, source, captured, reason="metadata_file_limit"
                    )
                if row:
                    file = PurePosixPath(row[0])
                    if (
                        file.name == loaded.name
                        and Path(str(installed.locate_file(file))).resolve() == loaded
                    ):
                        owned = True
                        break
            if not owned:
                return observation(
                    None, source, captured, reason="loaded_module_not_attributed"
                )
        headers = bytearray()
        try:
            stream = root.joinpath("METADATA").open("rb")
        except FileNotFoundError:
            stream = root.joinpath("PKG-INFO").open("rb")
        with stream:
            while True:
                line = stream.readline(MAX_HEADER_BYTES - len(headers) + 1)
                headers.extend(line)
                if len(headers) > MAX_HEADER_BYTES:
                    return observation(
                        None, source, captured, reason="metadata_byte_limit"
                    )
                if not line or line in (b"\n", b"\r\n"):
                    break
        version = BytesHeaderParser().parsebytes(headers)["Version"]
        if not isinstance(version, str) or not version.strip():
            return observation(None, source, captured, reason="empty_version")
        if len(version.encode("utf-8")) > MAX_REPORT_BYTES:
            return observation(None, source, captured, reason="version_over_budget")
        return observation(version, source, captured)
    except importlib.metadata.PackageNotFoundError:
        return observation(None, source, captured, reason="distribution_not_found")
    except Exception:
        return observation(None, source, captured, reason="metadata_read_failed")


def bounded_collect(
    participant: str, collect: Callable[[], ExecutionEvidence]
) -> ExecutionEvidence:
    """A stuck lookup leaves an explicit gap and at most one daemon worker."""
    if not _worker.acquire(blocking=False):
        return ExecutionEvidence.gap(participant, "collector_busy")
    done = threading.Event()
    output: list[ExecutionEvidence] = []

    def work():
        try:
            output.append(collect())
        except Exception:
            output.append(
                ExecutionEvidence.gap(participant, "collection_failed_or_over_budget")
            )
        finally:
            _worker.release()
            done.set()

    try:
        threading.Thread(target=work, name="fabric-software", daemon=True).start()
    except Exception:
        _worker.release()
        return ExecutionEvidence.gap(participant, "collector_unavailable")
    if not done.wait(COLLECTION_SECONDS):
        return ExecutionEvidence.gap(participant, "collection_time_budget")
    return (
        output[0] if output else ExecutionEvidence.gap(participant, "collection_failed")
    )


def collect_selected(
    participant: str, targets: Iterable[tuple[str, str, str | None]]
) -> ExecutionEvidence:
    """Capture selected packages and the executing interpreter exactly once."""
    # A bounded prefix also protects callers that accidentally supply an iterator.
    selected = list(islice(targets, MAX_TARGETS + 1))

    def collect():
        captured = observed_at()
        entries = [
            SoftwareEntry(
                role="interpreter",
                kind="executable",
                name="python",
                version=observation(
                    sys.version.split()[0], f"{participant}:sys.version", captured
                ),
            )
        ]
        complete = len(selected) <= MAX_TARGETS
        started = time.monotonic()
        for role, distribution, module_name in selected[:MAX_TARGETS]:
            # Leave time for the worker to publish a partial report before
            # bounded_collect's hard wait expires.
            if time.monotonic() - started >= COLLECTION_SECONDS * 0.8:
                complete = False
                break
            version = package_version(
                distribution, module_name, f"{participant}:importlib.metadata", captured
            )
            complete = complete and version.reason not in {
                "metadata_file_limit",
                "metadata_byte_limit",
                "version_over_budget",
            }
            entries.append(
                SoftwareEntry(
                    role=role,
                    kind="python_distribution",
                    name=distribution,
                    version=version,
                )
            )
        return ExecutionEvidence(
            software_environments=[
                SoftwareEnvironment(
                    context_ref="env-0",
                    participants=[participant],
                    status="available",
                    complete=complete,
                    reason=None if complete else "collection_truncated_or_work_budget",
                    entries=entries,
                )
            ]
        )

    return bounded_collect(participant, collect)


def combine(*reports: ExecutionEvidence) -> ExecutionEvidence:
    """Join distinct source contexts, remapping local refs without merging installations."""
    result: dict = {"software_environments": []}
    for report in reports:
        value = report.to_mapping()
        refs = {}
        for env in value["software_environments"]:
            refs[env["context_ref"]] = f"env-{len(result['software_environments'])}"
            env["context_ref"] = refs[env["context_ref"]]
            result["software_environments"].append(env)
        for kind, infrastructure in value.get("infrastructure", {}).items():
            infrastructure["context_refs"] = [
                refs[ref] for ref in infrastructure["context_refs"]
            ]
            if kind in result.get("infrastructure", {}):
                return ExecutionEvidence.gap(
                    "reporter", "conflicting_infrastructure_reports"
                )
            result.setdefault("infrastructure", {})[kind] = infrastructure
    environments = result["software_environments"]
    if len(environments) > 16:
        environments[15:] = ExecutionEvidence.gap(
            "additional_contexts", "report_context_limit"
        ).to_mapping()["software_environments"]
        environments[-1]["context_ref"] = "env-15"
    while (
        sum(len(env["entries"]) for env in environments) > 128
        or len(
            json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        > 64 * 1024
    ):
        env = environments[-1]
        if env["entries"]:
            env["entries"].pop()
            env.update(complete=False, reason="report_truncated_by_entry_or_byte_limit")
        elif len(environments) > 1:
            environments.pop()
            environments[-1].update(
                complete=False, reason="additional_contexts_over_budget"
            )
        else:
            return ExecutionEvidence.gap("reporter", "combined_report_over_budget")
    refs = {env["context_ref"] for env in environments}
    for kind, infrastructure in list(result.get("infrastructure", {}).items()):
        if not set(infrastructure["context_refs"]) <= refs:
            del result["infrastructure"][kind]
    return ExecutionEvidence.receive(result, "reporter")
