# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select native OpenClaw conversations through the Gateway session methods."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common.lifecycle import LifecycleError
from nemo_fabric_adapters.common.local_interfaces import write_private_file

SessionCall = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
HISTORY_LIMIT = 1000


def _native_id(value: Any) -> str:
    try:
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError("invalid native identifier")
    except ValueError as error:
        raise LifecycleError(
            "session_incompatible", "Native session identity is invalid"
        ) from error
    return value


def _read_marker(path: Path) -> dict[str, Any]:
    if path.parent.is_symlink():
        raise LifecycleError(
            "session_incompatible", "Session completion directory is invalid"
        )
    try:
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
            value = json.loads(source.read(1024 * 1024))
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("state"), str)
            or value.get("state") not in {"running", "complete"}
        ):
            raise ValueError("invalid completion record")
        if (
            type(value.get("message_count", 0)) is not int
            or value.get("message_count", 0) < 0
        ):
            raise ValueError("invalid completion boundary")
        for name in ("acknowledged_tool_calls", "pending_tool_calls"):
            calls = value.get(name, {})
            if not isinstance(calls, dict) or any(
                not isinstance(identifier, str) or not isinstance(fingerprint, str)
                for identifier, fingerprint in calls.items()
            ):
                raise ValueError("invalid recovery evidence")
        if type(value.get("history_incomplete", False)) is not bool or (
            value.get("history_anchor") is not None
            and not isinstance(value["history_anchor"], str)
        ):
            raise ValueError("invalid history boundary")
        count = value.get("history_count")
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("invalid history count")
        return value
    except (OSError, ValueError) as error:
        raise LifecycleError(
            "session_incompatible", "Session completion record is unreadable"
        ) from error


def _completed(messages: list[dict[str, Any]]) -> bool:
    if not messages:
        return False
    message = messages[-1].get("message", messages[-1])
    return (
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and message.get("stopReason") in {"stop", "end_turn"}
        and not message.get("tool_calls")
        and not any(
            isinstance(item, dict) and item.get("type") in {"toolCall", "tool_use"}
            for item in (
                message.get("content")
                if isinstance(message.get("content"), list)
                else []
            )
        )
    )


def _last_message_hash(messages: list[dict[str, Any]]) -> str | None:
    if not messages:
        return None
    return hashlib.sha256(json.dumps(messages[-1], sort_keys=True).encode()).hexdigest()


def _tool_call_hash(row: dict[str, Any]) -> str:
    value = dict(row)
    wrapped = isinstance(value.get("message"), dict)
    message = dict(value["message"] if wrapped else value)
    metadata = message.get("__openclaw")
    if isinstance(metadata, dict):
        message["__openclaw"] = {
            key: item
            for key, item in metadata.items()
            if key not in {"seq", "transcriptPosition"}
        }
    value = value | {"message": message} if wrapped else message
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _history_identity(row: dict[str, Any]) -> str:
    message = row.get("message", row)
    metadata = message.get("__openclaw") if isinstance(message, dict) else None
    identifier = metadata.get("id") if isinstance(metadata, dict) else None
    return (
        f"native:{identifier}"
        if isinstance(identifier, str) and identifier
        else _tool_call_hash(row)
    )


def _finished_after_marker(
    messages: list[dict[str, Any]], marker: dict[str, Any]
) -> bool:
    if "history_anchor" in marker:
        return (
            _completed(messages)
            and _history_identity(messages[-1]) != marker["history_anchor"]
        )
    return _completed(messages) and (
        _last_message_hash(messages) != marker["last_message_hash"]
        if "last_message_hash" in marker
        else len(messages) > marker.get("message_count", 0)
    )


def _pending_tool_calls(
    messages: list[dict[str, Any]], previous: dict[str, str] | None = None
) -> dict[str, str]:
    pending = dict(previous or {})
    for row in messages:
        message = row.get("message", row)
        if not isinstance(message, dict):
            continue
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                identifier = item.get("id")
                if item.get("type") in {"toolCall", "tool_use"} and isinstance(
                    identifier, str
                ):
                    pending[identifier] = _tool_call_hash(row)
                elif item.get("type") == "tool_result":
                    pending.pop(item.get("tool_use_id"), None)
        if message.get("role") in {"toolResult", "tool"}:
            pending.pop(message.get("toolCallId", message.get("tool_call_id")), None)
    return pending


def _unacknowledged_tool_calls(
    pending: dict[str, str], acknowledged: dict[str, str]
) -> list[str]:
    return sorted(
        identifier
        for identifier, fingerprint in pending.items()
        if acknowledged.get(identifier) != fingerprint
    )


@dataclass
class NativeSession:
    session_id: str = field(repr=False)
    key: str = field(repr=False)
    agent_id: str = field(repr=False)
    marker: Path = field(repr=False)
    call: SessionCall = field(repr=False)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    recovery_required: bool = False
    acknowledged_tool_calls: dict[str, str] = field(default_factory=dict, repr=False)
    pending_tool_calls: dict[str, str] = field(default_factory=dict, repr=False)
    history_anchor: str | None = field(default=None, repr=False)
    history_count: int | None = None
    history_incomplete: bool = False

    def start_output(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "recovery_warnings": self.warnings}

    async def confirm_identity(self) -> None:
        described = await self.call(
            "sessions.describe", {"key": self.key, "agentId": self.agent_id}
        )
        entry = described.get("session")
        if (
            not isinstance(entry, dict)
            or entry.get("sessionId") != self.session_id
            or entry.get("key") != self.key
        ):
            raise LifecycleError(
                "session_identity_changed", "Native session identity changed"
            )

    async def messages(self) -> list[dict[str, Any]]:
        before = await self._history_boundary()
        result = await self.call(
            "sessions.get",
            {"key": self.key, "agentId": self.agent_id, "limit": HISTORY_LIMIT},
        )
        messages = result.get("messages")
        if not isinstance(messages, list) or any(
            not isinstance(row, dict) for row in messages
        ):
            raise LifecycleError(
                "session_incompatible", "Native session history is invalid"
            )
        total = await self._history_boundary()
        if before != total:
            raise LifecycleError(
                "state_in_use", "Native session history changed while reading"
            )
        # A recent window can omit unresolved calls. Carry observed evidence
        # forward, and require inspection if the previous boundary fell out.
        covered = len(messages) == total or (
            self.history_count is not None
            and 0 <= total - self.history_count <= len(messages)
            and self.history_anchor is not None
            and any(_history_identity(row) == self.history_anchor for row in messages)
        )
        self.history_incomplete = self.history_incomplete or not covered
        self.pending_tool_calls = _pending_tool_calls(messages, self.pending_tool_calls)
        self.history_anchor = _history_identity(messages[-1]) if messages else None
        self.history_count = total
        return messages

    async def _history_boundary(self) -> int:
        result = await self.call(
            "chat.history", {"sessionKey": self.key, "offset": 0, "limit": 1}
        )
        if (
            result.get("sessionId") != self.session_id
            or result.get("sessionKey") != self.key
        ):
            raise LifecycleError(
                "session_identity_changed", "Native session identity changed"
            )
        count = result.get("totalMessages")
        if type(count) is not int or count < 0:
            raise LifecycleError(
                "session_incompatible", "Native history count is invalid"
            )
        return count

    def _write(self, value: dict[str, Any]) -> None:
        if self.marker.parent.is_symlink():
            raise LifecycleError(
                "session_incompatible", "Session completion directory is invalid"
            )
        value = value | {
            "acknowledged_tool_calls": self.acknowledged_tool_calls,
            "pending_tool_calls": self.pending_tool_calls,
            "history_anchor": self.history_anchor,
            "history_count": self.history_count,
            "history_incomplete": self.history_incomplete,
        }
        write_private_file(self.marker, json.dumps(value))
        if os.name == "posix":
            descriptor = os.open(
                self.marker.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    async def begin_turn(
        self, context: RuntimeContext, *, acknowledge_recovery: bool = False
    ) -> bool:
        messages = await self.messages()
        if acknowledge_recovery:
            self.acknowledged_tool_calls.update(self.pending_tool_calls)
            self.history_incomplete = False
            self.recovery_required = False
        elif (
            self.recovery_required
            or self.history_incomplete
            or _unacknowledged_tool_calls(
                self.pending_tool_calls, self.acknowledged_tool_calls
            )
        ):
            self.recovery_required = True
            self._write(_read_marker(self.marker) | {"state": "running"})
            return False
        self._write(
            {
                "state": "running",
                "message_count": len(messages),
                "last_message_hash": _last_message_hash(messages),
                "invocation_id": context.invocation_id,
                "request_id": context.request_id,
            }
        )
        await self.confirm_identity()
        return True

    async def finish_turn(self) -> bool:
        messages = await self.messages()
        marker = _read_marker(self.marker)
        completed = (
            not self.history_incomplete
            and _finished_after_marker(messages, marker)
            and not _unacknowledged_tool_calls(
                self.pending_tool_calls, self.acknowledged_tool_calls
            )
        )
        if completed:
            self._write({"state": "complete", "message_count": len(messages)})
        else:
            self._write(marker | {"state": "running"})
        return completed


async def open_session(
    context: RuntimeContext, agent_id: str, call: SessionCall
) -> NativeSession | None:
    reference = context.session_ref
    if not context.state_dir:
        if reference is not None:
            raise LifecycleError(
                "session_storage_required", "Durable sessions require retained storage"
            )
        return None
    agent_id = agent_id.lower()
    if reference is None:
        key = f"agent:{agent_id}:openai-user:{uuid4()}"
        created = await call("sessions.create", {"key": key, "agentId": agent_id})
        if (
            created.get("ok") is not True
            or created.get("key") != key
            or created.get("runStarted") is not False
        ):
            raise LifecycleError(
                "session_resume_unconfirmed",
                "Gateway did not confirm durable session creation",
            )
        session_id = _native_id(created.get("sessionId"))
    else:
        if reference.adapter_id != "nvidia.fabric.openclaw":
            raise LifecycleError(
                "session_adapter_mismatch", "Session reference targets another adapter"
            )
        session_id = _native_id(reference.session_id)
        resolved = await call(
            "sessions.resolve",
            {"sessionId": session_id, "agentId": agent_id, "allowMissing": True},
        )
        if resolved.get("ok") is not True:
            raise LifecycleError(
                "session_not_found", "Native session was not found in retained storage"
            )
        key = resolved.get("key")
        if not isinstance(key, str) or not key.startswith(f"agent:{agent_id}:"):
            raise LifecycleError(
                "session_incompatible", "Native session belongs to another agent"
            )
    described = await call("sessions.describe", {"key": key, "agentId": agent_id})
    entry = described.get("session")
    if (
        not isinstance(entry, dict)
        or entry.get("sessionId") != session_id
        or entry.get("key") != key
    ):
        raise LifecycleError(
            "session_resume_unconfirmed",
            "Gateway did not confirm the selected native session",
        )
    if entry.get("activeRunIds") or entry.get("status") in {"running", "queued"}:
        raise LifecycleError("state_in_use", "Native session still has active work")
    result = NativeSession(
        session_id,
        key,
        agent_id,
        Path(context.state_dir) / ".fabric-turns" / f"{session_id}.json",
        call,
    )
    marker = _read_marker(result.marker)
    result.acknowledged_tool_calls = dict(marker.get("acknowledged_tool_calls", {}))
    result.pending_tool_calls = dict(marker.get("pending_tool_calls", {}))
    result.history_anchor = marker.get("history_anchor")
    result.history_count = marker.get("history_count")
    result.history_incomplete = marker.get("history_incomplete", False)
    messages = await result.messages()
    finished_after_marker = _finished_after_marker(messages, marker)
    pending_tool_calls = _unacknowledged_tool_calls(
        result.pending_tool_calls, result.acknowledged_tool_calls
    )
    result.recovery_required = bool(
        pending_tool_calls
        or result.history_incomplete
        or (marker.get("state") == "running" and not finished_after_marker)
        or (messages and not _completed(messages))
        or (entry.get("abortedLastRun") and not finished_after_marker)
    )
    if result.recovery_required:
        result.warnings = [
            {
                "stage": "start",
                "code": "session_recovery_required",
                "message": "Session restored. Earlier turn completion is uncertain. Inspect native and external state before explicitly continuing. No request was replayed.",
                "retryable": False,
                "metadata": {
                    name: marker[name]
                    for name in ("invocation_id", "request_id")
                    if isinstance(marker.get(name), str)
                }
                | {
                    "external_effects": "unknown",
                    "pending_tool_call_ids": pending_tool_calls,
                    "history_incomplete": result.history_incomplete,
                },
            }
        ]
        result._write(marker | {"state": "running"})
    elif reference is None:
        result._write({"state": "complete", "message_count": 0})
    return result
