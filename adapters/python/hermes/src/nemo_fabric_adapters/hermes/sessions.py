# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge explicit NeMo Fabric session selection to Hermes's native session store."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common.lifecycle import LifecycleError


@dataclass
class NativeSession:
    """One NeMo Fabric-selected native conversation.  Hermes owns its representation."""

    session_id: str = field(repr=False)
    native_id: str = field(repr=False)
    database: Any = field(repr=False)
    marker: Path = field(repr=False)
    history: list[dict[str, Any]] | None = field(default=None, repr=False)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    recovery_required: bool = False

    def start_output(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "recovery_warnings": self.warnings}

    def begin_turn(self, context: RuntimeContext) -> None:
        rows = self.database.get_messages(self.native_id)
        self._write_marker(
            {
                "state": "running",
                "invocation_id": context.invocation_id,
                "request_id": context.request_id,
                "last_message_id": rows[-1]["id"] if rows else 0,
            }
        )

    def finish_turn(self, native_id: str, result: dict[str, Any]) -> bool:
        # Hermes may rotate a compression segment while preserving the logical session.
        self.native_id = native_id
        if not result.get("partial") and not _unfinished(
            self.database.get_messages(native_id)
        ):
            self._write_marker({"state": "complete"})
            return True
        return False

    def _write_marker(self, value: dict[str, Any]) -> None:
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        if self.marker.parent.is_symlink():
            raise LifecycleError(
                "session_incompatible", "Session completion directory is invalid"
            )
        temporary = self.marker.with_suffix(f".{uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "w") as output:
                json.dump(value, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.marker)
            if os.name == "posix":
                directory = os.open(
                    self.marker.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)


def _unfinished(rows: list[dict[str, Any]]) -> bool:
    messages = [row for row in rows if row["role"] in {"user", "assistant", "tool"}]
    return bool(
        messages
        and (messages[-1]["role"] != "assistant" or messages[-1].get("tool_calls"))
    )


def _read_marker(path: Path) -> dict[str, Any]:
    try:
        if path.parent.is_symlink():
            raise ValueError("invalid directory")
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            expected = path.lstat()
        except FileNotFoundError:
            return {}
        if not stat.S_ISREG(expected.st_mode):
            raise ValueError("invalid file")
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "r") as source:
            actual = os.fstat(source.fileno())
            if not stat.S_ISREG(actual.st_mode) or not os.path.samestat(
                expected, actual
            ):
                raise ValueError("invalid file")
            marker = json.loads(source.read(1024 * 1024))
        if (
            not isinstance(marker, dict)
            or not isinstance(marker.get("state"), str)
            or marker.get("state") not in {"running", "complete"}
            or type(marker.get("last_message_id", 0)) is not int
            or marker.get("last_message_id", 0) < 0
            or any(
                not isinstance(marker[name], str)
                for name in ("invocation_id", "request_id")
                if name in marker
            )
        ):
            raise ValueError("invalid turn marker")
        return marker
    except (OSError, ValueError) as error:
        raise LifecycleError(
            "session_incompatible", "Session completion record is unreadable"
        ) from error


def open_session(database: Any, context: RuntimeContext) -> NativeSession | None:
    """Use Hermes's own safe-resume check and native model-history projection."""
    reference = context.session_ref
    if not context.state_dir:
        if reference is not None:
            raise LifecycleError(
                "session_storage_required", "Durable sessions require retained storage"
            )
        return None
    if reference is not None:
        if reference.adapter_id != "nvidia.fabric.hermes":
            raise LifecycleError(
                "session_adapter_mismatch", "Session reference targets another adapter"
            )
        session_id = reference.session_id
        try:
            native_id = database.resolve_resume_session_id(session_id) or session_id
            if database.get_session(native_id) is None:
                raise LifecycleError(
                    "session_not_found",
                    "Native session was not found in retained storage",
                )
            database.assert_resume_safe(native_id, tip_only=True)
            rows = database.get_messages(native_id)
            history, _display = database.get_resume_conversations(native_id)
        except LifecycleError:
            raise
        except Exception as error:
            raise LifecycleError(
                "session_incompatible",
                "Native session cannot be restored by this Hermes version",
            ) from error
    else:
        session_id = str(uuid4())
        native_id = session_id
        database.create_session(native_id, "fabric")
        rows, history = [], None

    digest = hashlib.sha256(session_id.encode()).hexdigest()
    result = NativeSession(
        session_id,
        native_id,
        database,
        Path(context.state_dir) / ".fabric-turns" / f"{digest}.json",
        history,
    )
    marker = _read_marker(result.marker)

    # A persisted final native assistant response can establish completion even
    # when Fabric died before receiving that response or clearing its marker.
    completed_after_marker = bool(
        rows
        and rows[-1]["id"] > marker.get("last_message_id", 0)
        and rows[-1]["role"] == "assistant"
        and not rows[-1].get("tool_calls")
        and rows[-1].get("finish_reason") == "stop"
    )
    result.recovery_required = _unfinished(rows) or (
        marker.get("state") == "running" and not completed_after_marker
    )
    if result.recovery_required:
        pending: set[str] = set()
        for row in rows:
            calls = row.get("tool_calls") or []
            if isinstance(calls, str):
                try:
                    calls = json.loads(calls)
                except ValueError as error:
                    raise LifecycleError(
                        "session_incompatible", "Native tool-call history is invalid"
                    ) from error
            if not isinstance(calls, list):
                raise LifecycleError(
                    "session_incompatible", "Native tool-call history is invalid"
                )
            for call in calls:
                if isinstance(call, dict) and isinstance(call.get("id"), str):
                    pending.add(call["id"])
            if row["role"] == "tool":
                pending.discard(row.get("tool_call_id"))
        result.warnings = [
            {
                "stage": "start",
                "code": "session_recovery_required",
                "message": (
                    "Session restored. Earlier turn completion is uncertain. "
                    "Inspect native and external state before explicitly continuing. "
                    "The restored history retains unfinished requests. The model may "
                    "act on unfinished requests after acknowledgment."
                ),
                "retryable": False,
                "metadata": {
                    **{
                        name: marker[name]
                        for name in ("invocation_id", "request_id")
                        if name in marker
                    },
                    "pending_tool_call_ids": sorted(pending),
                    "external_effects": "unknown",
                },
            }
        ]
    if reference is not None:
        database.reopen_session(native_id)
    return result
