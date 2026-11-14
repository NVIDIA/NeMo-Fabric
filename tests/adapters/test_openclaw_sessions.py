# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate native session identity and safe recovery without invoking the model."""

import hashlib
import json
import os
import stat
from unittest.mock import AsyncMock

import pytest

from nemo_fabric_adapter_contract.models import RuntimeContext, SessionRef
from nemo_fabric_adapters.common.lifecycle import LifecycleError
from nemo_fabric_adapters.openclaw import sessions

NATIVE_ID = "0e28956c-4938-488f-bfef-953599c4e223"
NATIVE_KEY = "agent:default:openai-user:retained-conversation"


@pytest.fixture(name="session_context")
def session_context_fixture(tmp_path):
    return RuntimeContext.from_mapping(
        {
            "runtime_id": "replacement-runtime",
            "invocation_id": "next-turn",
            "request_id": "next-request",
            "environment": {
                "environment_id": "local",
                "provider": "local",
                "control_location": "in_env_control",
                "ownership": "caller_owned",
            },
            "artifacts": {},
            "state_dir": str(tmp_path),
            "session_ref": {
                "schema_version": "fabric.session_ref/v1alpha1",
                "adapter_id": "nvidia.fabric.openclaw",
                "session_id": NATIVE_ID,
            },
        }
    )


@pytest.fixture(name="native_session")
def native_session_fixture():
    return {"key": NATIVE_KEY, "sessionId": NATIVE_ID, "messages": []}


@pytest.fixture(name="session_call")
def session_call_fixture(native_session):
    def respond(method, params):
        if (
            method == "sessions.resolve"
            and params["sessionId"] != native_session["sessionId"]
        ):
            if params.get("allowMissing") is True:
                return {"ok": False}
            raise LifecycleError("native_rpc_error", "No matching native session")
        if method == "sessions.get":
            return {"messages": native_session["messages"][-params.get("limit", 200) :]}
        if method == "chat.history":
            return {
                "sessionId": native_session["sessionId"],
                "sessionKey": native_session["key"],
                "totalMessages": len(native_session["messages"]),
            }
        return {
            "sessions.resolve": {
                "ok": True,
                "key": native_session["key"],
                "agentId": "default",
            },
            "sessions.describe": {"session": native_session},
        }[method]

    return AsyncMock(side_effect=respond)


@pytest.mark.parametrize(
    "identifier", ["../other", "/private/file", "opaque-not-native"]
)
async def test_invalid_native_identity_never_calls_the_gateway(
    session_context, session_call, identifier
):
    session_context.session_ref = SessionRef(
        schema_version="fabric.session_ref/v1alpha1",
        adapter_id="nvidia.fabric.openclaw",
        session_id=identifier,
    )
    with pytest.raises(LifecycleError, match="identity is invalid"):
        await sessions.open_session(session_context, "default", session_call)
    session_call.assert_not_called()


async def test_wrong_adapter_is_rejected_without_native_access(
    session_context, session_call
):
    session_context.session_ref = SessionRef(
        schema_version="fabric.session_ref/v1alpha1",
        adapter_id="nvidia.fabric.hermes",
        session_id=NATIVE_ID,
    )
    with pytest.raises(LifecycleError) as caught:
        await sessions.open_session(session_context, "default", session_call)
    assert caught.value.code == "session_adapter_mismatch"
    session_call.assert_not_called()


@pytest.mark.parametrize(
    ("change", "error_code"),
    [
        ({"sessionId": "another-id"}, "session_not_found"),
        ({"key": "agent:reviewer:other"}, "session_incompatible"),
        ({"activeRunIds": ["still-running"]}, "state_in_use"),
    ],
)
async def test_native_identity_and_active_work_are_validated(
    session_context, session_call, native_session, change, error_code
):
    native_session.update(change)
    with pytest.raises(LifecycleError) as caught:
        await sessions.open_session(session_context, "default", session_call)
    assert caught.value.code == error_code


async def test_recorded_native_completion_resolves_a_lost_response(
    session_context, session_call, native_session, tmp_path
):
    directory = tmp_path / ".fabric-turns"
    directory.mkdir()
    (directory / f"{NATIVE_ID}.json").write_text(
        json.dumps(
            {"state": "running", "message_count": 0, "request_id": "lost-response"}
        )
    )
    native_session["messages"].append(
        {
            "role": "assistant",
            "content": "The operation completed.",
            "stopReason": "stop",
        }
    )
    result = await sessions.open_session(session_context, "default", session_call)
    assert result.session_id == NATIVE_ID
    assert result.warnings == []
    assert not result.recovery_required
    assert all(
        call.args[0]
        in {"sessions.resolve", "sessions.describe", "sessions.get", "chat.history"}
        for call in session_call.call_args_list
    )


async def test_pending_native_tool_calls_are_reported_without_replay(
    session_context, session_call, native_session
):
    native_session["messages"].append(
        {
            "role": "assistant",
            "stopReason": "tool_use",
            "content": [{"type": "toolCall", "id": "external-operation"}],
        }
    )
    result = await sessions.open_session(session_context, "default", session_call)
    assert result.recovery_required
    assert result.warnings[0]["metadata"]["pending_tool_call_ids"] == [
        "external-operation"
    ]
    assert all(
        call.args[0] != "sessions.create" for call in session_call.call_args_list
    )


@pytest.mark.skipif(os.name == "nt", reason="Uses POSIX symbolic links")
async def test_completion_marker_cannot_escape_through_a_symbolic_link(
    session_context, session_call, tmp_path
):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".fabric-turns").symlink_to(outside, target_is_directory=True)
    with pytest.raises(LifecycleError) as caught:
        await sessions.open_session(session_context, "default", session_call)
    assert caught.value.code == "session_incompatible"
    assert list(outside.iterdir()) == []


async def test_native_completion_boundary_works_with_a_full_recent_history_page(
    session_context, session_call, native_session, tmp_path
):
    previous = {"role": "assistant", "content": "Earlier answer.", "stopReason": "stop"}
    directory = tmp_path / ".fabric-turns"
    directory.mkdir()
    (directory / f"{NATIVE_ID}.json").write_text(
        json.dumps(
            {
                "state": "running",
                "message_count": 1000,
                "last_message_hash": hashlib.sha256(
                    json.dumps(previous, sort_keys=True).encode()
                ).hexdigest(),
            }
        )
    )
    native_session["messages"] = [
        {"role": "user", "content": "Earlier input."}
    ] * 999 + [
        {"role": "assistant", "content": "New final answer.", "stopReason": "stop"}
    ]
    result = await sessions.open_session(session_context, "default", session_call)
    assert not result.recovery_required


async def test_native_tool_result_resolves_only_the_matching_tool_call(
    session_context, session_call, native_session
):
    native_session["messages"] = [
        {
            "role": "assistant",
            "stopReason": "tool_use",
            "content": [
                {"type": "toolCall", "id": "completed-operation"},
                {"type": "toolCall", "id": "unknown-operation"},
            ],
        },
        {
            "role": "toolResult",
            "toolCallId": "completed-operation",
            "content": "Completed.",
        },
    ]
    result = await sessions.open_session(session_context, "default", session_call)
    assert result.warnings[0]["metadata"]["pending_tool_call_ids"] == [
        "unknown-operation"
    ]


@pytest.mark.parametrize("matched", [False, True])
async def test_final_assistant_message_does_not_hide_an_unresolved_tool_call(
    session_context, session_call, native_session, matched
):
    native_session["messages"] = [
        {
            "role": "assistant",
            "stopReason": "tool_use",
            "content": [{"type": "toolCall", "id": "create-ticket"}],
        }
    ]
    if matched:
        native_session["messages"].append(
            {
                "role": "toolResult",
                "toolCallId": "create-ticket",
                "isError": True,
                "content": "The ticket service rejected the request.",
            }
        )
    native_session["messages"].append(
        {
            "role": "assistant",
            "stopReason": "stop",
            "content": "The request failed.",
        }
    )
    result = await sessions.open_session(session_context, "default", session_call)
    assert result.recovery_required is not matched
    if not matched:
        assert result.warnings[0]["metadata"]["pending_tool_call_ids"] == [
            "create-ticket"
        ]
    else:
        assert result.warnings == []


@pytest.mark.parametrize("new_tool_call", [False, True])
async def test_current_turn_completion_requires_new_native_evidence(
    session_context, session_call, native_session, new_tool_call
):
    native_session["messages"] = [
        {
            "role": "assistant",
            "content": "Earlier completed answer.",
            "stopReason": "stop",
        }
    ]
    result = await sessions.open_session(session_context, "default", session_call)
    await result.begin_turn(session_context)
    if new_tool_call:
        native_session["messages"].extend(
            [
                {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "id": "current-operation"}],
                    "stopReason": "tool_use",
                },
                {
                    "role": "assistant",
                    "content": "A new answer with an unresolved operation.",
                    "stopReason": "stop",
                },
            ]
        )
    assert await result.finish_turn() is False
    assert json.loads(result.marker.read_text())["state"] == "running"


async def test_acknowledged_tool_uncertainty_does_not_block_completed_new_work(
    session_context, session_call, native_session
):
    native_session["messages"] = [
        {
            "role": "assistant",
            "content": [{"type": "toolCall", "id": "inspected-operation"}],
            "stopReason": "tool_use",
        }
    ]
    result = await sessions.open_session(session_context, "default", session_call)
    assert result.recovery_required
    await result.begin_turn(session_context, acknowledge_recovery=True)
    native_session["messages"].append(
        {
            "role": "assistant",
            "content": "New work completed after inspection.",
            "stopReason": "stop",
        }
    )
    assert await result.finish_turn() is True
    reopened = await sessions.open_session(session_context, "default", session_call)
    assert not reopened.recovery_required
    assert reopened.warnings == []


async def test_reused_tool_call_identifier_requires_a_new_recovery_acknowledgement(
    session_context, session_call, native_session
):
    native_session["messages"] = [
        {
            "role": "assistant",
            "timestamp": 1000,
            "stopReason": "tool_use",
            "content": [{"type": "toolCall", "id": "reused-call"}],
        }
    ]
    result = await sessions.open_session(session_context, "default", session_call)
    await result.begin_turn(session_context, acknowledge_recovery=True)
    native_session["messages"].append(
        {
            "role": "assistant",
            "timestamp": 2000,
            "stopReason": "stop",
            "content": "Completed after inspection.",
        }
    )
    assert await result.finish_turn() is True
    await result.begin_turn(session_context)
    native_session["messages"].extend(
        [
            {
                "role": "assistant",
                "timestamp": 3000,
                "stopReason": "tool_use",
                "content": [{"type": "toolCall", "id": "reused-call"}],
            },
            {
                "role": "assistant",
                "timestamp": 4000,
                "stopReason": "stop",
                "content": "A later answer with a new unresolved operation.",
            },
        ]
    )
    assert await result.finish_turn() is False
    reopened = await sessions.open_session(session_context, "default", session_call)
    assert reopened.recovery_required
    assert reopened.warnings[0]["metadata"]["pending_tool_call_ids"] == ["reused-call"]


@pytest.mark.parametrize("nofollow_available", [True, False])
async def test_completion_record_rejects_a_symbolic_link_without_platform_flags(
    session_context, session_call, tmp_path, monkeypatch, nofollow_available
):
    if not nofollow_available:
        monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    directory = tmp_path / ".fabric-turns"
    directory.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"state":"complete"}')
    (directory / f"{NATIVE_ID}.json").symlink_to(outside)
    with pytest.raises(LifecycleError) as caught:
        await sessions.open_session(session_context, "default", session_call)
    assert caught.value.code == "session_incompatible"


async def test_completion_record_replaced_during_open_is_rejected(
    session_context, session_call, tmp_path, monkeypatch
):
    directory = tmp_path / ".fabric-turns"
    directory.mkdir()
    marker = directory / f"{NATIVE_ID}.json"
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
        await sessions.open_session(session_context, "default", session_call)
    assert caught.value.code == "session_incompatible"


@pytest.mark.parametrize("agent_id", ["Reviewer", "agent_1-A"])
@pytest.mark.parametrize("resume", [False, True])
async def test_native_session_selection_uses_canonical_agent_identity(
    session_context, session_call, native_session, agent_id, resume
):
    native_session["key"] = (
        f"agent:{agent_id.lower()}:openai-user:retained-conversation"
    )
    original_respond = session_call.side_effect

    def respond(method, params):
        if method == "sessions.create":
            native_session["key"] = params["key"].lower()
            return {
                "ok": True,
                "key": native_session["key"],
                "sessionId": NATIVE_ID,
                "runStarted": False,
            }
        return original_respond(method, params)

    session_call.side_effect = respond
    if not resume:
        session_context.session_ref = None
    selected = await sessions.open_session(session_context, agent_id, session_call)
    assert selected.session_id == NATIVE_ID
    assert selected.key == native_session["key"]
    assert selected.agent_id == agent_id.lower()
    for call in session_call.call_args_list:
        if call.args[0] == "chat.history":
            assert call.args[1]["sessionKey"].startswith(f"agent:{agent_id.lower()}:")
        else:
            assert call.args[1]["agentId"] == agent_id.lower()


@pytest.mark.parametrize("sync_fails", [False, True])
async def test_completion_data_is_synced_before_publishing_the_marker(
    session_context, session_call, monkeypatch, sync_fails
):
    native = await sessions.open_session(session_context, "default", session_call)
    native.marker.parent.mkdir(exist_ok=True)
    native.marker.write_text('{"state":"complete"}')
    original = native.marker.read_bytes()
    synced = []
    replace = os.replace
    fsync = os.fsync

    def sync_before_publish(descriptor):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            if sync_fails:
                raise OSError("test sync failed")
            temporary = list(native.marker.parent.glob(f".{native.marker.name}.*"))
            assert len(temporary) == 1
            assert json.loads(temporary[0].read_text())["state"] == "running"
            synced.append(descriptor)
        fsync(descriptor)

    def publish(source, destination):
        assert synced, "marker contents must reach storage before replacement"
        assert native.marker.read_bytes() == original
        return replace(source, destination)

    monkeypatch.setattr(os, "fsync", sync_before_publish)
    monkeypatch.setattr(os, "replace", publish)
    if sync_fails:
        with pytest.raises(OSError):
            native._write({"state": "running"})
        assert native.marker.read_bytes() == original
        assert list(native.marker.parent.glob(f".{native.marker.name}.*")) == []
    else:
        native._write({"state": "running"})
        assert json.loads(native.marker.read_text())["state"] == "running"


@pytest.mark.parametrize("state", [[], {}])
async def test_malformed_completion_state_has_a_structured_resume_error(
    session_context, session_call, tmp_path, state
):
    directory = tmp_path / ".fabric-turns"
    directory.mkdir()
    (directory / f"{NATIVE_ID}.json").write_text(json.dumps({"state": state}))
    with pytest.raises(LifecycleError) as caught:
        await sessions.open_session(session_context, "default", session_call)
    assert caught.value.code == "session_incompatible"


async def test_missing_native_session_has_the_specific_not_found_error(
    session_context, session_call
):
    session_context.session_ref.session_id = "9ae11b64-c7cb-46e6-9c98-11d5f442cdd4"
    with pytest.raises(LifecycleError) as caught:
        await sessions.open_session(session_context, "default", session_call)
    assert caught.value.code == "session_not_found"


@pytest.mark.parametrize("phase", ["open", "begin", "finish"])
async def test_native_reset_during_history_read_is_rejected(
    session_context, session_call, native_session, phase
):
    if phase != "open":
        selected = await sessions.open_session(session_context, "default", session_call)
        if phase == "finish":
            await selected.begin_turn(session_context)
    original = session_call.side_effect

    def reset_during_get(method, params):
        if method == "sessions.get":
            native_session["sessionId"] = "9ae11b64-c7cb-46e6-9c98-11d5f442cdd4"
            return {"messages": [{"role": "user", "content": "replacement history"}]}
        return original(method, params)

    session_call.side_effect = reset_during_get
    with pytest.raises(LifecycleError) as caught:
        if phase == "open":
            await sessions.open_session(session_context, "default", session_call)
        elif phase == "begin":
            await selected.begin_turn(session_context)
        else:
            await selected.finish_turn()
    assert caught.value.code == "session_identity_changed"


@pytest.mark.parametrize("observed_call", [False, True])
async def test_history_window_cannot_hide_an_unresolved_tool_call(
    session_context, session_call, native_session, observed_call
):
    selected = await sessions.open_session(session_context, "default", session_call)
    await selected.begin_turn(session_context)
    native_session["messages"].append(
        {
            "role": "assistant",
            "stopReason": "tool_use",
            "content": [{"type": "toolCall", "id": "old-unresolved-operation"}],
        }
    )
    if observed_call:
        await selected.messages()
    native_session["messages"].extend(
        {"role": "assistant", "stopReason": "stop", "content": f"Later answer {index}"}
        for index in range(1001)
    )
    assert await selected.finish_turn() is False
    assert json.loads(selected.marker.read_text())["state"] == "running"
    restored = await sessions.open_session(session_context, "default", session_call)
    assert restored.recovery_required
    assert restored.warnings[0]["metadata"]["history_incomplete"]
    if observed_call:
        assert restored.warnings[0]["metadata"]["pending_tool_call_ids"] == [
            "old-unresolved-operation"
        ]


async def test_acknowledged_history_gap_allows_confirmed_new_work(
    session_context, session_call, native_session
):
    native_session["messages"] = [
        {
            "role": "assistant",
            "stopReason": "stop",
            "content": f"Earlier answer {index}",
        }
        for index in range(1001)
    ]
    selected = await sessions.open_session(session_context, "default", session_call)
    assert selected.recovery_required
    assert await selected.begin_turn(session_context) is False
    assert await selected.begin_turn(session_context, acknowledge_recovery=True) is True
    native_session["messages"].append(
        {
            "role": "assistant",
            "stopReason": "stop",
            "content": "New completed answer",
        }
    )
    assert await selected.finish_turn() is True
    restored = await sessions.open_session(session_context, "default", session_call)
    assert not restored.recovery_required


async def test_a_new_history_gap_blocks_dispatch_until_acknowledged(
    session_context, session_call, native_session
):
    native_session["messages"] = [
        {"role": "assistant", "stopReason": "stop", "content": "Earlier answer"}
    ]
    selected = await sessions.open_session(session_context, "default", session_call)
    await selected.begin_turn(session_context)
    native_session["messages"].append(
        {"role": "assistant", "stopReason": "stop", "content": "Completed turn"}
    )
    assert await selected.finish_turn() is True
    native_session["messages"].extend(
        {"role": "assistant", "stopReason": "stop", "content": f"Outside turn {index}"}
        for index in range(1001)
    )
    assert await selected.begin_turn(session_context) is False
    assert selected.recovery_required


async def test_native_reset_during_marker_write_is_rejected_before_dispatch(
    session_context, session_call, native_session, monkeypatch
):
    selected = await sessions.open_session(session_context, "default", session_call)
    write = selected._write

    def reset_after_write(value):
        write(value)
        native_session["sessionId"] = "9ae11b64-c7cb-46e6-9c98-11d5f442cdd4"

    monkeypatch.setattr(selected, "_write", reset_after_write)
    with pytest.raises(LifecycleError) as caught:
        await selected.begin_turn(session_context)
    assert caught.value.code == "session_identity_changed"
    assert json.loads(selected.marker.read_text())["state"] == "running"


@pytest.mark.parametrize(
    "bad_record",
    [
        {"pending_tool_calls": []},
        {"pending_tool_calls": {"call": False}},
        {"history_incomplete": "false"},
        {"history_anchor": []},
    ],
)
async def test_invalid_retained_history_evidence_is_rejected(
    session_context, session_call, tmp_path, bad_record
):
    directory = tmp_path / ".fabric-turns"
    directory.mkdir()
    (directory / f"{NATIVE_ID}.json").write_text(
        json.dumps({"state": "complete", **bad_record})
    )
    with pytest.raises(LifecycleError) as caught:
        await sessions.open_session(session_context, "default", session_call)
    assert caught.value.code == "session_incompatible"


async def test_native_byte_limit_cannot_hide_older_tool_calls(
    session_context, session_call, native_session
):
    native_session["messages"] = [
        {
            "role": "assistant",
            "content": [{"type": "toolCall", "id": "omitted-call"}],
            "stopReason": "tool_use",
        },
        {"role": "user", "content": "A later request"},
        {"role": "assistant", "content": "A later answer", "stopReason": "stop"},
    ]
    original = session_call.side_effect

    def omit_old_messages(method, params):
        if method == "sessions.get":
            return {"messages": native_session["messages"][-2:]}
        return original(method, params)

    session_call.side_effect = omit_old_messages
    selected = await sessions.open_session(session_context, "default", session_call)
    assert selected.recovery_required
    assert selected.warnings[0]["metadata"]["history_incomplete"]


async def test_retained_native_message_identity_survives_projection_changes(
    session_context, session_call, native_session
):
    native_session["messages"] = [
        {
            "role": "assistant",
            "content": [{"type": "toolCall", "id": "inspected-call"}],
            "stopReason": "tool_use",
            "__openclaw": {"id": "native-message-1", "seq": 1, "transcriptPosition": 1},
        }
    ]
    selected = await sessions.open_session(session_context, "default", session_call)
    assert selected.recovery_required
    await selected.begin_turn(session_context, acknowledge_recovery=True)
    native_session["messages"].append(
        {
            "role": "assistant",
            "content": "Completed after inspection",
            "stopReason": "stop",
            "__openclaw": {"id": "native-message-2", "seq": 2, "transcriptPosition": 2},
        }
    )
    assert await selected.finish_turn() is True
    for row in native_session["messages"]:
        row["__openclaw"]["seq"] += 10
        row["__openclaw"]["transcriptPosition"] += 10
    restored = await sessions.open_session(session_context, "default", session_call)
    assert not restored.recovery_required


@pytest.mark.parametrize("invalid_count", [None, -1, True, "1"])
async def test_invalid_native_history_count_is_rejected(
    session_context, session_call, invalid_count
):
    original = session_call.side_effect

    def invalid_boundary(method, params):
        result = original(method, params)
        if method == "chat.history":
            result["totalMessages"] = invalid_count
        return result

    session_call.side_effect = invalid_boundary
    with pytest.raises(LifecycleError) as caught:
        await sessions.open_session(session_context, "default", session_call)
    assert caught.value.code == "session_incompatible"


async def test_concurrent_history_append_is_rejected_before_consuming_the_page(
    session_context, session_call, native_session
):
    original = session_call.side_effect

    def append_during_get(method, params):
        result = original(method, params)
        if method == "sessions.get":
            native_session["messages"].append(
                {"role": "user", "content": "Concurrent work"}
            )
        return result

    session_call.side_effect = append_during_get
    with pytest.raises(LifecycleError) as caught:
        await sessions.open_session(session_context, "default", session_call)
    assert caught.value.code == "state_in_use"


async def test_projection_changes_do_not_confirm_an_unfinished_turn(
    session_context, session_call, native_session
):
    native_session["messages"] = [
        {
            "role": "assistant",
            "stopReason": "stop",
            "content": "Earlier answer",
            "__openclaw": {
                "id": "unchanged-native-message",
                "seq": 1,
                "transcriptPosition": 1,
            },
        }
    ]
    selected = await sessions.open_session(session_context, "default", session_call)
    await selected.begin_turn(session_context)
    native_session["messages"][0]["__openclaw"]["seq"] = 20
    native_session["messages"][0]["__openclaw"]["transcriptPosition"] = 20
    assert await selected.finish_turn() is False
    restored = await sessions.open_session(session_context, "default", session_call)
    assert restored.recovery_required
