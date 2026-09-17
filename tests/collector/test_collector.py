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
