# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recovery records supplement native history without replaying requests."""

import hashlib
import json
import os
from unittest.mock import MagicMock

import pytest

from nemo_fabric_adapter_contract.models import RuntimeContext, SessionRef
from nemo_fabric_adapters.common.lifecycle import LifecycleError
from nemo_fabric_adapters.hermes.sessions import NativeSession, open_session


@pytest.fixture(name="session_store")
def session_store_fixture(tmp_path):
    context = MagicMock(spec=RuntimeContext)
    context.state_dir = str(tmp_path)
    context.session_ref = SessionRef(
        schema_version="fabric.session_ref/v1alpha1",
        adapter_id="nvidia.fabric.hermes",
        session_id="original",
    )
    database = MagicMock(
        spec=[
            "resolve_resume_session_id",
            "get_session",
            "assert_resume_safe",
            "get_messages",
            "get_resume_conversations",
            "reopen_session",
        ]
    )
    database.resolve_resume_session_id.return_value = "native-tip"
    database.get_session.return_value = {"id": "native-tip"}
    database.get_messages.return_value = []
    database.get_resume_conversations.return_value = ([], [])
    marker = (
        tmp_path / ".fabric-turns" / (hashlib.sha256(b"original").hexdigest() + ".json")
    )
    marker.parent.mkdir()
    return database, context, marker


def test_durable_completed_response_clears_uncertainty_after_lost_reply(session_store):
    database, context, marker = session_store
    marker.write_text(
        json.dumps(
            {"state": "running", "last_message_id": 2, "request_id": "lost-reply"}
        )
    )
    database.get_messages.return_value = [
        {"id": 3, "role": "assistant", "finish_reason": "stop"}
    ]
    restored = open_session(database, context)
    assert not restored.recovery_required
    assert restored.warnings == []
    database.reopen_session.assert_called_once_with("native-tip")


def test_recorded_tool_failure_followed_by_final_response_is_complete(session_store):
    database, context, marker = session_store
    database.get_messages.return_value = [
        {"id": 1, "role": "assistant", "tool_calls": [{"id": "failed-call"}]},
        {
            "id": 2,
            "role": "tool",
            "tool_call_id": "failed-call",
            "content": "tool failed",
        },
        {
            "id": 3,
            "role": "assistant",
            "finish_reason": "stop",
            "content": "The tool failed.",
        },
    ]
    assert not open_session(database, context).recovery_required


@pytest.mark.parametrize(
    "value",
    [
        {"state": "running", "last_message_id": "invalid"},
        {"state": "running", "request_id": {"secret": "private"}},
        ["invalid"],
    ],
)
def test_invalid_recovery_record_is_a_structured_failure(session_store, value):
    database, context, marker = session_store
    database.get_messages.return_value = [
        {"id": 3, "role": "assistant", "finish_reason": "stop"}
    ]
    marker.write_text(json.dumps(value))
    with pytest.raises(LifecycleError) as caught:
        open_session(database, context)
    assert caught.value.code == "session_incompatible"
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("nofollow_available", [True, False])
def test_recovery_record_cannot_read_a_symlink(
    session_store, tmp_path, monkeypatch, nofollow_available
):
    if not nofollow_available:
        monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    database, context, marker = session_store
    outside = tmp_path / "outside.json"
    outside.write_text('{"state":"complete"}')
    marker.symlink_to(outside)
    with pytest.raises(LifecycleError) as caught:
        open_session(database, context)
    assert caught.value.code == "session_incompatible"


def test_recovery_record_replaced_during_open_is_rejected(session_store, monkeypatch):
    database, context, marker = session_store
    marker.write_text('{"state":"complete"}')
    replacement = marker.with_suffix(".replacement")
    replacement.write_text('{"state":"running"}')
    original_open = os.open

    def replace_before_open(path, flags, *args, **kwargs):
        if path == marker:
            os.replace(replacement, marker)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_before_open)
    with pytest.raises(LifecycleError) as caught:
        open_session(database, context)
    assert caught.value.code == "session_incompatible"
    database.reopen_session.assert_not_called()


def test_turn_completion_ignores_and_preserves_an_unowned_stale_temporary(
    session_store,
):
    database, context, marker = session_store
    stale = marker.with_suffix(".tmp")
    stale.write_text("unowned")
    native = NativeSession("original", "native-tip", database, marker)
    native._write_marker({"state": "complete"})
    assert json.loads(marker.read_text()) == {"state": "complete"}
    assert stale.read_text() == "unowned"


def test_retained_storage_without_a_reference_creates_a_fresh_conversation(
    session_store,
):
    database, context, _marker = session_store
    context.session_ref = None
    database.create_session = MagicMock()
    created = open_session(database, context)
    assert created.history is None
    assert created.session_id == created.native_id
    database.create_session.assert_called_once_with(created.native_id, "fabric")
    database.get_resume_conversations.assert_not_called()
    database.reopen_session.assert_not_called()


def test_no_retention_or_reference_keeps_runtime_scoped_behavior(session_store):
    database, context, _marker = session_store
    context.state_dir = None
    context.session_ref = None
    assert open_session(database, context) is None
    database.get_resume_conversations.assert_not_called()


def test_reference_requires_retained_storage(session_store):
    database, context, _marker = session_store
    context.state_dir = None
    with pytest.raises(LifecycleError) as caught:
        open_session(database, context)
    assert caught.value.code == "session_storage_required"
    database.get_resume_conversations.assert_not_called()
