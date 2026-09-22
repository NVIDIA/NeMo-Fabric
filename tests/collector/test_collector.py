# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging

import pytest

from nemo_fabric_collector.app import (
    AtofCollector,
    RequestId,
    ScopeUuid,
    _AtofQueueClosed,
    _AtofQueueFull,
    _AtofRecordQueue,
    _MAX_CANCELLED_REGISTRATION_TOKENS,
    _MAX_PI_COMPLETION_TOMBSTONES,
    _RecordTooLarge,
    _StreamAlreadyAttached,
    _SubscriptionPhase,
    _TerminationReason,
)


async def test_queue_drains_records_before_signaling_close():
    queue = _AtofRecordQueue(maxsize=2, max_bytes=100)
    first = {"uuid": "first"}
    second = {"uuid": "second"}
    await queue.put(first)
    await queue.put(second)

    queue.close(drain=True, reason=_TerminationReason.COMPLETED)

    assert await queue.get() == first
    assert await queue.get() == second
    with pytest.raises(_AtofQueueClosed) as exc_info:
        await queue.get()
    assert exc_info.value.reason is _TerminationReason.COMPLETED


async def test_queue_discard_close_clears_records():
    queue = _AtofRecordQueue(maxsize=1, max_bytes=100)
    await queue.put({"uuid": "discarded"})

    queue.close(drain=False, reason=_TerminationReason.DEREGISTERED)

    assert queue.empty()
    with pytest.raises(_AtofQueueClosed) as exc_info:
        await queue.get()
    assert exc_info.value.reason is _TerminationReason.DEREGISTERED


async def test_queue_close_wakes_blocked_producer():
    queue = _AtofRecordQueue(maxsize=1, max_bytes=100, put_timeout=1)
    await queue.put({"uuid": "first"})
    blocked_put = asyncio.create_task(queue.put({"uuid": "second"}))
    await asyncio.sleep(0)

    queue.close(drain=False, reason=_TerminationReason.DEREGISTERED)

    with pytest.raises(_AtofQueueClosed):
        await blocked_put


async def test_queue_rejects_record_larger_than_byte_limit():
    queue = _AtofRecordQueue(maxsize=1, max_bytes=4)

    with pytest.raises(_RecordTooLarge):
        await queue.put({"uuid": "large"})


async def test_queue_applies_byte_budget_backpressure():
    record = {"uuid": "record", "payload": "x" * 16}
    record_size = len(json.dumps(record).encode())
    queue = _AtofRecordQueue(maxsize=10, max_bytes=record_size, put_timeout=1)
    await queue.put(record)

    blocked_put = asyncio.create_task(queue.put(record))
    await asyncio.sleep(0)
    assert not blocked_put.done()

    assert await queue.get() == record
    await blocked_put
    assert await queue.get() == record


async def test_queue_raises_when_full_after_timeout():
    queue = _AtofRecordQueue(maxsize=1, max_bytes=100, put_timeout=0)
    first = {"uuid": "first"}
    await queue.put(first)

    with pytest.raises(_AtofQueueFull):
        await queue.put({"uuid": "second"})

    assert await queue.get() == first


async def test_collector_logs_queue_timeout_drops(
    caplog: pytest.LogCaptureFixture,
):
    collector = AtofCollector()
    request_id = RequestId("request-1")
    root = {
        "kind": "scope",
        "scope_category": "start",
        "uuid": "root-1",
        "metadata": {"nemo_fabric_request_id": request_id},
    }
    child = {"kind": "mark", "uuid": "mark-1", "parent_uuid": "root-1"}
    await collector.register(request_id)
    collector.request_messages[request_id] = _AtofRecordQueue(
        maxsize=1,
        max_bytes=100,
        put_timeout=0,
    )

    with caplog.at_level(logging.WARNING, logger="nemo_fabric_collector.app"):
        await collector.route(root, byte_size=1)
        await collector.route(child, byte_size=1)
    await collector.close()

    assert "Dropping ATOF record after queue backpressure timeout" in caplog.text
    assert caplog.records[-1].request_id == request_id
    assert caplog.records[-1].byte_size == 1


async def test_collector_routes_only_scope_descendants():
    collector = AtofCollector()
    request_id = RequestId("request-1")
    root = {
        "kind": "scope",
        "scope_category": "start",
        "uuid": "root-1",
        "metadata": {"nemo_fabric_request_id": request_id},
    }
    mark = {"kind": "mark", "uuid": "mark-1", "parent_uuid": "root-1"}
    nested_scope = {
        "kind": "scope",
        "scope_category": "start",
        "uuid": "scope-2",
        "parent_uuid": "root-1",
    }
    nested_mark = {
        "kind": "mark",
        "uuid": "mark-2",
        "parent_uuid": "scope-2",
    }
    non_scope_child = {
        "kind": "mark",
        "uuid": "not-routed",
        "parent_uuid": "mark-1",
    }
    await collector.register(request_id)

    for record in (root, mark, nested_scope, nested_mark, non_scope_child):
        await collector.route(record, byte_size=1)

    queue = collector.request_messages[request_id]
    assert await queue.get() == root
    assert await queue.get() == mark
    assert await queue.get() == nested_scope
    assert await queue.get() == nested_mark
    assert queue.empty()
    assert collector.request_uuids[request_id] == {
        ScopeUuid("root-1"),
        ScopeUuid("scope-2"),
    }


async def test_standalone_collector_routes_records_without_request_metadata():
    collector = AtofCollector(standalone=True)
    request_id = RequestId("request-1")
    record = {"kind": "scope", "scope_category": "start", "uuid": "root-1"}
    await collector.register(request_id)

    await collector.route(record, byte_size=1)

    queue = collector.request_messages[request_id]
    assert await queue.get() == record


async def test_scope_end_routes_by_its_own_uuid():
    collector = AtofCollector()
    request_id = RequestId("request-1")
    root = {
        "kind": "scope",
        "scope_category": "start",
        "uuid": "root-1",
        "metadata": {"nemo_fabric_request_id": request_id},
    }
    scope_end = {
        "kind": "scope",
        "scope_category": "end",
        "uuid": "root-1",
    }
    await collector.register(request_id)
    await collector.route(root, byte_size=1)
    await collector.route(scope_end, byte_size=1)

    queue = collector.request_messages[request_id]
    assert await queue.get() == root
    assert await queue.get() == scope_end


async def test_collector_isolates_concurrent_request_trees():
    collector = AtofCollector()
    request_a = RequestId("request-a")
    request_b = RequestId("request-b")
    root_a = {
        "kind": "scope",
        "scope_category": "start",
        "uuid": "root-a",
        "metadata": {"nemo_fabric_request_id": request_a},
    }
    root_b = {
        "kind": "scope",
        "scope_category": "start",
        "uuid": "root-b",
        "metadata": {"nemo_fabric_request_id": request_b},
    }
    child_a = {"kind": "mark", "uuid": "mark-a", "parent_uuid": "root-a"}
    child_b = {"kind": "mark", "uuid": "mark-b", "parent_uuid": "root-b"}
    await collector.register(request_a)
    await collector.register(request_b)

    for record in (root_a, root_b, child_b, child_a):
        await collector.route(record, byte_size=1)

    queue_a = collector.request_messages[request_a]
    queue_b = collector.request_messages[request_b]
    assert await queue_a.get() == root_a
    assert await queue_a.get() == child_a
    assert await queue_b.get() == root_b
    assert await queue_b.get() == child_b
    assert queue_a.empty()
    assert queue_b.empty()


async def test_deregister_cleans_routing_and_drains_queue():
    collector = AtofCollector()
    request_id = RequestId("request-1")
    root = {
        "kind": "scope",
        "scope_category": "start",
        "uuid": "root-1",
        "metadata": {"nemo_fabric_request_id": request_id},
    }
    await collector.register(request_id)
    await collector.route(root, byte_size=1)
    queue = collector.request_messages[request_id]

    await collector.deregister(request_id, remove_queue=False)

    assert request_id not in collector.request_uuids
    assert ScopeUuid("root-1") not in collector.uuid_to_request
    assert collector.request_states[request_id].phase is _SubscriptionPhase.DRAINING
    assert await queue.get() == root
    with pytest.raises(_AtofQueueClosed):
        await queue.get()


async def test_collector_rejects_duplicate_registration_and_stream():
    collector = AtofCollector()
    request_id = RequestId("request-1")

    await collector.register(request_id)
    with pytest.raises(RuntimeError, match="already registered"):
        await collector.register(request_id)
    attached = await collector.attach_stream(request_id)
    assert attached is not None
    with pytest.raises(_StreamAlreadyAttached):
        await collector.attach_stream(request_id)


async def test_standalone_collector_rejects_second_registration():
    collector = AtofCollector(standalone=True)
    await collector.register(RequestId("request-1"))

    with pytest.raises(RuntimeError, match="standalone collector"):
        await collector.register(RequestId("request-2"))


async def test_pi_registration_tombstones_evict_the_oldest_entry_at_the_bound():
    collector = AtofCollector(standalone=True)

    for index in range(_MAX_CANCELLED_REGISTRATION_TOKENS + 1):
        await collector.deregister(
            RequestId(f"request-{index}"),
            remove_queue=True,
            pi_boundary="release",
            registration_token=f"attempt-{index}",
        )

    assert len(collector._cancelled_registration_tokens) == (
        _MAX_CANCELLED_REGISTRATION_TOKENS
    )
    assert (
        RequestId("request-0"),
        "attempt-0",
    ) not in collector._cancelled_registration_tokens
    assert (
        RequestId(f"request-{_MAX_CANCELLED_REGISTRATION_TOKENS}"),
        f"attempt-{_MAX_CANCELLED_REGISTRATION_TOKENS}",
    ) in collector._cancelled_registration_tokens
    with pytest.raises(RuntimeError, match="registration attempt was cancelled"):
        await collector.register(
            RequestId(f"request-{_MAX_CANCELLED_REGISTRATION_TOKENS}"),
            correlation_mode="pi_turn_window",
            registration_token=f"attempt-{_MAX_CANCELLED_REGISTRATION_TOKENS}",
        )
    await collector.register(
        RequestId("request-0"),
        correlation_mode="pi_turn_window",
        registration_token="attempt-0",
    )
    assert RequestId("request-0") in collector.request_states


def test_pi_completion_tombstones_evict_the_oldest_entry_at_the_bound():
    collector = AtofCollector(standalone=True)

    for index in range(_MAX_PI_COMPLETION_TOMBSTONES + 1):
        collector._remember_pi_completion(0, ScopeUuid(f"completion-{index}"))

    assert len(collector._pi_completion_tombstones) == _MAX_PI_COMPLETION_TOMBSTONES
    assert (0, ScopeUuid("completion-0")) not in collector._pi_completion_tombstones
    assert (
        0,
        ScopeUuid(f"completion-{_MAX_PI_COMPLETION_TOMBSTONES}"),
    ) in collector._pi_completion_tombstones


async def test_pi_registration_tombstone_cancels_waiting_late_commit():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    first_id = RequestId("request-1")
    second_id = RequestId("request-2")
    await collector.register(
        first_id,
        correlation_mode="pi_turn_window",
        registration_token="first-attempt",
    )
    late_registration = asyncio.create_task(
        collector.register(
            second_id,
            correlation_mode="pi_turn_window",
            registration_token="second-attempt",
        )
    )
    await asyncio.sleep(0)
    assert not late_registration.done()

    await collector.deregister(
        second_id,
        remove_queue=True,
        pi_boundary="release",
        registration_token="second-attempt",
    )
    await collector.deregister(
        first_id,
        remove_queue=True,
        pi_boundary="release",
        registration_token="first-attempt",
    )

    with pytest.raises(RuntimeError, match="registration attempt was cancelled"):
        await late_registration
    assert second_id not in collector.request_states
    await collector.register(
        second_id,
        correlation_mode="pi_turn_window",
        registration_token="replacement-attempt",
    )
    assert second_id in collector.request_states


async def test_stale_pi_registration_token_cannot_release_current_lease():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    request_id = RequestId("request-1")
    await collector.register(
        request_id,
        correlation_mode="pi_turn_window",
        registration_token="current-attempt",
    )

    await collector.deregister(
        request_id,
        remove_queue=True,
        pi_boundary="release",
        registration_token="stale-attempt",
    )

    assert request_id in collector.request_uuids
    assert not collector._pi_boundary_ready.is_set()


async def test_stale_pi_registration_token_cannot_attach_to_reused_request_id():
    collector = AtofCollector(standalone=True)
    request_id = RequestId("request-1")
    await collector.register(
        request_id,
        correlation_mode="pi_turn_window",
        registration_token="old-attempt",
    )
    await collector.deregister(
        request_id,
        remove_queue=True,
        pi_boundary="release",
        registration_token="old-attempt",
    )
    await collector.register(
        request_id,
        correlation_mode="pi_turn_window",
        registration_token="new-attempt",
    )

    assert (
        await collector.attach_stream(
            request_id,
            registration_token="old-attempt",
        )
        is None
    )
    assert (
        await collector.attach_stream(
            request_id,
            registration_token="new-attempt",
        )
        is not None
    )


def _pi_record(
    hook_event_name: str,
    *,
    kind: str = "mark",
    uuid: str,
    turn_seq: int,
) -> dict:
    record = {
        "kind": kind,
        "uuid": uuid,
        "metadata": {
            "agent_kind": "pi",
            "hook_event_name": hook_event_name,
            "turn_seq": turn_seq,
        },
    }
    if hook_event_name == "turn_start":
        record.update({"scope_category": "start", "name": "pi-turn"})
        record["metadata"].update(
            {
                "nemo_relay_scope_role": "turn",
                "turn_source": "turn_start",
            }
        )
    return record


async def test_pi_turn_window_routes_each_turn_through_agent_settled():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    request_id = RequestId("request-1")
    await collector.register(request_id, correlation_mode="pi_turn_window")
    queue = collector.request_messages[request_id]
    stale = _pi_record("agent_end", uuid="stale", turn_seq=0)
    turn = _pi_record("turn_start", kind="scope", uuid="turn-1", turn_seq=1)
    model = {"kind": "scope", "uuid": "model-1", "parent_uuid": "turn-1"}
    settled = _pi_record("agent_settled", uuid="settled-1", turn_seq=1)

    await collector.route(stale, byte_size=1)
    assert queue.empty()
    await collector.route(turn, byte_size=1)
    await collector.route(model, byte_size=1)
    deregistration = asyncio.create_task(
        collector.deregister(
            request_id,
            remove_queue=False,
            pi_boundary="wait",
        )
    )
    await asyncio.sleep(0)
    assert not deregistration.done()

    await collector.route(settled, byte_size=1)
    await deregistration

    assert await queue.get() == turn
    assert await queue.get() == model
    assert await queue.get() == settled
    with pytest.raises(_AtofQueueClosed):
        await queue.get()


async def test_pi_turn_window_ignores_previous_boundary_for_next_registration():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    first_id = RequestId("request-1")
    first_turn = _pi_record(
        "turn_start",
        kind="scope",
        uuid="turn-1",
        turn_seq=0,
    )
    first_settled = _pi_record("agent_settled", uuid="settled-1", turn_seq=0)
    await collector.register(first_id, correlation_mode="pi_turn_window")
    await collector.route(first_turn, byte_size=1)
    await collector.route(first_settled, byte_size=1)
    await collector.deregister(
        first_id,
        remove_queue=True,
        pi_boundary="wait",
    )

    second_id = RequestId("request-2")
    await collector.register(second_id, correlation_mode="pi_turn_window")
    queue = collector.request_messages[second_id]
    await collector.route(first_turn, byte_size=1)
    await collector.route(first_settled, byte_size=1)

    state = collector.request_states[second_id]
    assert queue.empty()
    assert state.routing_ready is False
    assert not state.completion_seen.is_set()

    second_turn = _pi_record(
        "turn_start",
        kind="scope",
        uuid="turn-2",
        turn_seq=1,
    )
    second_settled = _pi_record("agent_settled", uuid="settled-2", turn_seq=1)
    mismatched_settled = _pi_record(
        "agent_settled",
        uuid="settled-future",
        turn_seq=2,
    )
    await collector.route(second_turn, byte_size=1)
    await collector.route(first_turn, byte_size=1)
    await collector.route(second_turn, byte_size=1)
    await collector.route(first_settled, byte_size=1)
    await collector.route(mismatched_settled, byte_size=1)

    assert not state.completion_seen.is_set()
    completion = asyncio.create_task(
        collector.deregister(
            second_id,
            remove_queue=False,
            pi_boundary="wait",
        )
    )
    await asyncio.sleep(0)
    assert not completion.done()

    await collector.route(second_settled, byte_size=1)
    await completion

    assert await queue.get() == second_turn
    assert await queue.get() == second_settled
    with pytest.raises(_AtofQueueClosed):
        await queue.get()


async def test_pi_turn_window_accepts_new_zero_turn_completion_at_prior_sequence():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    first_id = RequestId("request-1")
    await collector.register(first_id, correlation_mode="pi_turn_window")
    await collector.route(
        _pi_record("turn_start", kind="scope", uuid="turn-1", turn_seq=0),
        byte_size=1,
    )
    first_settled = _pi_record("agent_settled", uuid="settled-1", turn_seq=0)
    await collector.route(first_settled, byte_size=1)
    await collector.deregister(
        first_id,
        remove_queue=True,
        pi_boundary="wait",
    )

    second_id = RequestId("request-2")
    await collector.register(second_id, correlation_mode="pi_turn_window")
    queue = collector.request_messages[second_id]
    second_settled = _pi_record("agent_settled", uuid="settled-2", turn_seq=0)
    await collector.route(second_settled, byte_size=1)
    await collector.deregister(
        second_id,
        remove_queue=False,
        pi_boundary="wait",
    )

    assert await queue.get() == second_settled

    third_id = RequestId("request-3")
    await collector.register(third_id, correlation_mode="pi_turn_window")
    third_queue = collector.request_messages[third_id]
    await collector.route(first_settled, byte_size=1)
    await collector.route(second_settled, byte_size=1)

    state = collector.request_states[third_id]
    assert third_queue.empty()
    assert not state.completion_seen.is_set()

    third_settled = _pi_record("agent_settled", uuid="settled-3", turn_seq=0)
    await collector.route(third_settled, byte_size=1)
    await collector.deregister(
        third_id,
        remove_queue=False,
        pi_boundary="wait",
    )

    assert await third_queue.get() == third_settled


async def test_pi_turn_window_tombstones_late_released_completion():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    first_id = RequestId("request-1")
    await collector.register(first_id, correlation_mode="pi_turn_window")
    await collector.route(
        _pi_record("turn_start", kind="scope", uuid="turn-1", turn_seq=0),
        byte_size=1,
    )
    await collector.route(
        _pi_record("agent_settled", uuid="settled-1", turn_seq=0),
        byte_size=1,
    )
    await collector.deregister(
        first_id,
        remove_queue=True,
        pi_boundary="wait",
    )

    second_id = RequestId("request-2")
    await collector.register(second_id, correlation_mode="pi_turn_window")
    await collector.deregister(
        second_id,
        remove_queue=True,
        pi_boundary="release",
    )
    late_settled = _pi_record("agent_settled", uuid="settled-2", turn_seq=0)
    await collector.route(late_settled, byte_size=1)

    third_id = RequestId("request-3")
    await collector.register(third_id, correlation_mode="pi_turn_window")
    queue = collector.request_messages[third_id]
    await collector.route(late_settled, byte_size=1)

    state = collector.request_states[third_id]
    assert queue.empty()
    assert not state.completion_seen.is_set()
    await collector.deregister(
        third_id,
        remove_queue=True,
        pi_boundary="release",
    )


async def test_pi_turn_window_zero_turn_does_not_advance_turn_sequence():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    first_id = RequestId("request-1")
    await collector.register(first_id, correlation_mode="pi_turn_window")
    first_settled = _pi_record("agent_settled", uuid="settled-1", turn_seq=0)
    await collector.route(first_settled, byte_size=1)
    await collector.deregister(
        first_id,
        remove_queue=True,
        pi_boundary="wait",
    )

    second_id = RequestId("request-2")
    await collector.register(second_id, correlation_mode="pi_turn_window")
    queue = collector.request_messages[second_id]
    turn = _pi_record("turn_start", kind="scope", uuid="turn-1", turn_seq=0)
    settled = _pi_record("agent_settled", uuid="settled-2", turn_seq=0)
    await collector.route(turn, byte_size=1)
    await collector.route(settled, byte_size=1)
    await collector.deregister(
        second_id,
        remove_queue=False,
        pi_boundary="wait",
    )

    assert await queue.get() == turn
    assert await queue.get() == settled

    third_id = RequestId("request-3")
    await collector.register(third_id, correlation_mode="pi_turn_window")
    third_queue = collector.request_messages[third_id]
    await collector.route(first_settled, byte_size=1)

    assert third_queue.empty()
    assert not collector.request_states[third_id].completion_seen.is_set()
    await collector.deregister(
        third_id,
        remove_queue=True,
        pi_boundary="release",
    )


async def test_pi_turn_window_uses_count_after_lost_turn_start():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    first_id = RequestId("request-1")
    await collector.register(first_id, correlation_mode="pi_turn_window")
    await collector.route(
        _pi_record("agent_settled", uuid="settled-1", turn_seq=0),
        byte_size=1,
    )
    await collector.deregister(
        first_id,
        remove_queue=True,
        pi_boundary="wait",
        pi_turn_count=1,
    )

    second_id = RequestId("request-2")
    await collector.register(second_id, correlation_mode="pi_turn_window")
    queue = collector.request_messages[second_id]
    await collector.route(
        _pi_record("turn_start", kind="scope", uuid="late-turn-1", turn_seq=0),
        byte_size=1,
    )
    assert queue.empty()

    turn = _pi_record("turn_start", kind="scope", uuid="turn-2", turn_seq=1)
    settled = _pi_record("agent_settled", uuid="settled-2", turn_seq=1)
    await collector.route(turn, byte_size=1)
    await collector.route(settled, byte_size=1)
    await collector.deregister(
        second_id,
        remove_queue=False,
        pi_boundary="wait",
        pi_turn_count=2,
    )

    assert await queue.get() == turn
    assert await queue.get() == settled


async def test_pi_turn_window_logs_dropped_records_before_zero_turn_completion(
    caplog: pytest.LogCaptureFixture,
):
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    request_id = RequestId("request-1")
    await collector.register(request_id, correlation_mode="pi_turn_window")
    unmatched = _pi_record("renamed_turn_start", uuid="turn-1", turn_seq=0)
    settled = _pi_record("agent_settled", uuid="settled-1", turn_seq=0)

    with caplog.at_level(logging.WARNING, logger="nemo_fabric_collector.app"):
        await collector.route(unmatched, byte_size=1)
        await collector.route(unmatched, byte_size=1)
        await collector.route(settled, byte_size=1)
        await collector.deregister(
            request_id,
            remove_queue=True,
            pi_boundary="wait",
        )

    marker_warnings = [
        record
        for record in caplog.records
        if "no turn_start marker was observed" in record.getMessage()
    ]
    assert len(marker_warnings) == 1
    assert marker_warnings[0].request_id == request_id
    assert marker_warnings[0].dropped_record_count == 2


async def test_pi_turn_window_timeout_identifies_missing_completion_marker(
    caplog: pytest.LogCaptureFixture,
):
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.001)
    request_id = RequestId("request-1")
    await collector.register(request_id, correlation_mode="pi_turn_window")
    await collector.route(
        _pi_record("turn_start", kind="scope", uuid="turn-1", turn_seq=0),
        byte_size=1,
    )
    await collector.route(
        _pi_record("renamed_agent_settled", uuid="settled-1", turn_seq=0),
        byte_size=1,
    )

    with caplog.at_level(logging.WARNING, logger="nemo_fabric_collector.app"):
        await collector.deregister(
            request_id,
            remove_queue=True,
            pi_boundary="wait",
        )

    marker_warnings = [
        record
        for record in caplog.records
        if "expected agent_settled marker" in record.getMessage()
    ]
    assert len(marker_warnings) == 1
    assert marker_warnings[0].request_id == request_id
    assert marker_warnings[0].expected_marker == "agent_settled"
    assert marker_warnings[0].observed_record_count == 2


async def test_pi_turn_window_drops_late_tail_before_next_turn(
    caplog: pytest.LogCaptureFixture,
):
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    first_id = RequestId("request-1")
    await collector.register(first_id, correlation_mode="pi_turn_window")
    with caplog.at_level(logging.WARNING):
        await collector.deregister(
            first_id,
            remove_queue=False,
            pi_boundary="wait",
            pi_turn_count=2,
        )
    assert "Timed out waiting for the Pi ATOF invocation boundary" in caplog.text

    second_id = RequestId("request-2")
    await collector.register(second_id, correlation_mode="pi_turn_window")
    queue = collector.request_messages[second_id]
    for turn_seq in (0, 1):
        late_turn = _pi_record(
            "turn_start",
            kind="scope",
            uuid=f"late-turn-{turn_seq}",
            turn_seq=turn_seq,
        )
        late_model = {
            "kind": "scope",
            "uuid": f"late-model-{turn_seq}",
            "parent_uuid": f"late-turn-{turn_seq}",
        }
        late_settled = _pi_record(
            "agent_settled",
            uuid=f"late-settled-{turn_seq}",
            turn_seq=turn_seq,
        )
        for record in (late_turn, late_model, late_settled):
            await collector.route(record, byte_size=1)
    assert queue.empty()

    second_turn = _pi_record(
        "turn_start",
        kind="scope",
        uuid="turn-2",
        turn_seq=2,
    )
    second_model = {
        "kind": "scope",
        "uuid": "model-2",
        "parent_uuid": "turn-2",
    }
    second_settled = _pi_record("agent_settled", uuid="settled-2", turn_seq=2)
    for record in (second_turn, second_model, second_settled):
        await collector.route(record, byte_size=1)
    await collector.deregister(
        second_id,
        remove_queue=False,
        pi_boundary="wait",
        pi_turn_count=3,
    )

    assert await queue.get() == second_turn
    assert await queue.get() == second_model
    assert await queue.get() == second_settled
    with pytest.raises(_AtofQueueClosed):
        await queue.get()


async def test_pi_turn_window_unknown_boundary_recovers_from_later_turn_count():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.001)
    first_id = RequestId("request-1")
    await collector.register(first_id, correlation_mode="pi_turn_window")
    await collector.route(
        _pi_record("turn_start", kind="scope", uuid="turn-1", turn_seq=0),
        byte_size=1,
    )
    await collector.route(
        _pi_record("agent_settled", uuid="settled-1", turn_seq=0),
        byte_size=1,
    )
    await collector.deregister(
        first_id,
        remove_queue=True,
        pi_boundary="wait",
    )

    second_id = RequestId("request-2")
    await collector.register(second_id, correlation_mode="pi_turn_window")
    await collector.deregister(
        second_id,
        remove_queue=True,
        pi_boundary="wait",
    )

    third_id = RequestId("request-3")
    await collector.register(third_id, correlation_mode="pi_turn_window")
    third_queue = collector.request_messages[third_id]
    for record in (
        _pi_record("turn_start", kind="scope", uuid="turn-2", turn_seq=1),
        _pi_record("agent_settled", uuid="settled-2", turn_seq=1),
    ):
        await collector.route(record, byte_size=1)
    assert third_queue.empty()
    await collector.deregister(
        third_id,
        remove_queue=True,
        pi_boundary="wait",
        pi_turn_count=2,
    )

    fourth_id = RequestId("request-4")
    await collector.register(fourth_id, correlation_mode="pi_turn_window")
    fourth_queue = collector.request_messages[fourth_id]
    await collector.route(
        _pi_record("agent_settled", uuid="late-settled-2", turn_seq=1),
        byte_size=1,
    )
    turn = _pi_record("turn_start", kind="scope", uuid="turn-3", turn_seq=2)
    settled = _pi_record("agent_settled", uuid="settled-3", turn_seq=2)
    await collector.route(turn, byte_size=1)
    await collector.route(settled, byte_size=1)
    await collector.deregister(
        fourth_id,
        remove_queue=False,
        pi_boundary="wait",
        pi_turn_count=3,
    )

    assert await fourth_queue.get() == turn
    assert await fourth_queue.get() == settled


async def test_pi_turn_window_admits_reuse_after_missing_completion():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.001)
    await collector.register(
        RequestId("request-1"),
        correlation_mode="pi_turn_window",
    )
    await collector.deregister(
        RequestId("request-1"),
        remove_queue=True,
        pi_boundary="wait",
        pi_turn_count=1,
    )

    await collector.register(
        RequestId("request-2"),
        correlation_mode="pi_turn_window",
    )
    assert RequestId("request-2") in collector.request_states


async def test_pi_quarantine_uses_observed_completion_boundary():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    first_id = RequestId("request-1")
    await collector.register(first_id, correlation_mode="pi_turn_window")
    await collector.route(
        _pi_record("turn_start", kind="scope", uuid="turn-1", turn_seq=0),
        byte_size=1,
    )
    await collector.route(
        _pi_record("agent_settled", uuid="settled-1", turn_seq=0),
        byte_size=1,
    )
    await collector.deregister(
        first_id,
        remove_queue=True,
        pi_boundary="quarantine",
    )

    second_id = RequestId("request-2")
    await collector.register(second_id, correlation_mode="pi_turn_window")
    second_queue = collector.request_messages[second_id]
    turn = _pi_record("turn_start", kind="scope", uuid="turn-2", turn_seq=1)
    await collector.route(turn, byte_size=1)

    assert await second_queue.get() == turn
    await collector.deregister(
        second_id,
        remove_queue=True,
        pi_boundary="release",
    )


async def test_stale_recovery_cannot_poison_newer_pi_lease():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.001)
    first_id = RequestId("request-1")
    await collector.register(first_id, correlation_mode="pi_turn_window")
    first_stream = await collector.attach_stream(first_id)
    assert first_stream is not None
    first_queue, first_stream_token = first_stream
    await collector.deregister(
        first_id,
        remove_queue=False,
        pi_boundary="wait",
        pi_turn_count=1,
    )

    second_id = RequestId("request-2")
    await collector.register(second_id, correlation_mode="pi_turn_window")
    second_owner = collector._pi_boundary_owner
    recovery_mode = collector._pi_recovery_mode
    turn_seq = collector._pi_boundary_turn_seq

    await collector.deregister(
        first_id,
        remove_queue=False,
        pi_boundary="wait",
        pi_turn_count=100,
    )

    assert collector._pi_boundary_owner == second_owner
    assert collector._pi_recovery_mode is recovery_mode
    assert collector._pi_boundary_turn_seq == turn_seq
    assert not collector._pi_boundary_ready.is_set()

    await collector.deregister(
        second_id,
        remove_queue=True,
        pi_boundary="release",
    )
    await collector.detach_stream(first_id, first_queue, first_stream_token)


async def test_pi_turn_window_discards_plain_invoke_records():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    request_id = RequestId("request-1")
    await collector.register(
        request_id,
        correlation_mode="pi_turn_window",
        capture_records=False,
    )
    queue = collector.request_messages[request_id]
    turn = _pi_record("turn_start", kind="scope", uuid="turn-1", turn_seq=0)
    model = {"kind": "mark", "uuid": "model-1", "parent_uuid": "turn-1"}
    settled = _pi_record("agent_settled", uuid="settled-1", turn_seq=0)

    deregistration = asyncio.create_task(
        collector.deregister(
            request_id,
            remove_queue=True,
            pi_boundary="wait",
        )
    )
    for record in (turn, model, settled):
        await collector.route(record, byte_size=1)
    await deregistration

    assert queue.empty()
    assert request_id not in collector.request_states


async def test_pi_turn_window_preserves_boundary_after_consumer_disconnect():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    request_id = RequestId("request-1")
    await collector.register(request_id, correlation_mode="pi_turn_window")
    queue = collector.request_messages[request_id]
    await collector.deregister(
        request_id,
        remove_queue=True,
        pi_boundary="preserve",
    )

    assert queue.closed
    assert request_id in collector.request_states
    assert request_id in collector.request_uuids

    completion = asyncio.create_task(
        collector.deregister(
            request_id,
            remove_queue=True,
            pi_boundary="wait",
        )
    )
    await collector.route(
        _pi_record("turn_start", kind="scope", uuid="turn-1", turn_seq=0),
        byte_size=1,
    )
    await collector.route(
        _pi_record("agent_settled", uuid="settled-1", turn_seq=0),
        byte_size=1,
    )
    await completion

    assert request_id not in collector.request_states


async def test_pi_turn_window_outcome_cleans_preserved_attached_stream():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    request_id = RequestId("request-1")
    await collector.register(request_id, correlation_mode="pi_turn_window")
    attached = await collector.attach_stream(request_id)
    assert attached is not None
    queue, token = attached
    await collector.deregister(
        request_id,
        remove_queue=True,
        pi_boundary="preserve",
    )
    await collector.route(
        _pi_record("turn_start", kind="scope", uuid="turn-1", turn_seq=0),
        byte_size=1,
    )
    await collector.route(
        _pi_record("agent_settled", uuid="settled-1", turn_seq=0),
        byte_size=1,
    )
    await collector.deregister(
        request_id,
        remove_queue=False,
        pi_boundary="wait",
    )

    assert request_id in collector.request_states
    await collector.detach_stream(request_id, queue, token)
    assert request_id not in collector.request_states


async def test_pi_turn_window_late_preserve_does_not_rearm_completed_boundary():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    request_id = RequestId("request-1")
    await collector.register(request_id, correlation_mode="pi_turn_window")
    attached = await collector.attach_stream(request_id)
    assert attached is not None
    queue, token = attached
    await collector.route(
        _pi_record("turn_start", kind="scope", uuid="turn-1", turn_seq=0),
        byte_size=1,
    )
    await collector.route(
        _pi_record("agent_settled", uuid="settled-1", turn_seq=0),
        byte_size=1,
    )
    await collector.deregister(
        request_id,
        remove_queue=False,
        pi_boundary="wait",
    )

    await collector.deregister(
        request_id,
        remove_queue=True,
        pi_boundary="preserve",
    )
    await collector.detach_stream(request_id, queue, token)

    assert request_id not in collector.request_states


async def test_pi_completion_retry_after_cancelled_queue_put():
    collector = AtofCollector(
        standalone=True,
        queue_maxsize=1,
        completion_wait_timeout=0.1,
    )
    request_id = RequestId("request-1")
    await collector.register(request_id, correlation_mode="pi_turn_window")
    queue = collector.request_messages[request_id]
    turn = _pi_record("turn_start", kind="scope", uuid="turn-1", turn_seq=0)
    settled = _pi_record("agent_settled", uuid="settled-1", turn_seq=0)
    await collector.route(turn, byte_size=1)

    blocked_completion = asyncio.create_task(collector.route(settled, byte_size=1))
    await asyncio.sleep(0)
    assert not blocked_completion.done()

    blocked_completion.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked_completion
    state = collector.request_states[request_id]
    assert state.completion_key is None
    assert not state.completion_seen.is_set()

    assert await queue.get() == turn
    await collector.route(settled, byte_size=1)
    await collector.deregister(
        request_id,
        remove_queue=False,
        pi_boundary="wait",
    )

    assert await queue.get() == settled


async def test_pi_timeout_warns_when_observed_completion_is_undelivered(
    caplog: pytest.LogCaptureFixture,
):
    collector = AtofCollector(
        standalone=True,
        queue_maxsize=1,
        completion_wait_timeout=0.001,
    )
    first_id = RequestId("request-1")
    await collector.register(first_id, correlation_mode="pi_turn_window")
    await collector.route(
        _pi_record("turn_start", kind="scope", uuid="turn-1", turn_seq=0),
        byte_size=1,
    )
    completion = asyncio.create_task(
        collector.route(
            _pi_record("agent_settled", uuid="settled-1", turn_seq=0),
            byte_size=1,
        )
    )
    await asyncio.sleep(0)
    assert not completion.done()

    with caplog.at_level(logging.WARNING, logger="nemo_fabric_collector.app"):
        await collector.deregister(
            first_id,
            remove_queue=False,
            pi_boundary="wait",
        )
    await completion
    await collector.register(
        RequestId("request-2"),
        correlation_mode="pi_turn_window",
    )
    boundary_warnings = [
        record
        for record in caplog.records
        if "agent_settled marker was observed" in record.getMessage()
    ]
    assert len(boundary_warnings) == 1
    assert boundary_warnings[0].request_id == first_id
    assert boundary_warnings[0].expected_marker == "agent_settled"


async def test_pi_turn_window_observes_completion_dropped_by_backpressure():
    collector = AtofCollector(standalone=True, completion_wait_timeout=0.1)
    first_id = RequestId("request-1")
    await collector.register(first_id, correlation_mode="pi_turn_window")
    collector.request_messages[first_id] = _AtofRecordQueue(
        maxsize=1,
        max_bytes=1024,
        put_timeout=0,
    )
    await collector.route(
        _pi_record("turn_start", kind="scope", uuid="turn-1", turn_seq=0),
        byte_size=1,
    )
    await collector.route(
        _pi_record("agent_settled", uuid="settled-1", turn_seq=0),
        byte_size=1,
    )

    await collector.deregister(
        first_id,
        remove_queue=False,
        pi_boundary="wait",
    )
    await collector.register(
        RequestId("request-2"),
        correlation_mode="pi_turn_window",
    )


async def test_collector_close_wakes_waiting_consumer():
    collector = AtofCollector()
    request_id = RequestId("request-1")
    await collector.register(request_id)
    queue = collector.request_messages[request_id]
    blocked_get = asyncio.create_task(queue.get())
    await asyncio.sleep(0)

    await collector.close()

    with pytest.raises(_AtofQueueClosed) as exc_info:
        await blocked_get
    assert exc_info.value.reason is _TerminationReason.COLLECTOR_SHUTDOWN
    assert not collector.request_messages
