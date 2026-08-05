# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn


@contextmanager
def mock_api_server(port: int) -> Iterator[str]:
    """
    Context manager for a mock API server.

    Use the /_requests endpoint to inspect captured chat-completion payloads after a test action.
    Use the /_scenario endpoint to configure the server to return a specific status code for subsequent requests.

    Args:
        port (int): The port on which the server will listen.

    Yields:
        str: The base URL of the mock API server.
    """

    app = FastAPI()
    app.state.requests = []
    app.state.status_code = 200
    app.state.tool_call = None
    app.state.tool_call_sent = False

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "fabric-echo",
                    "object": "model",
                    "created": 0,
                    "owned_by": "fabric-test",
                }
            ],
        }

    @app.get("/_requests")
    def requests() -> list[dict[str, object]]:
        """GET this after a test action to inspect captured chat-completion payloads."""

        return list(app.state.requests)

    @app.post("/_scenario")
    async def scenario(request: Request) -> dict[str, object]:
        """Configure the status code or a single tool call for subsequent requests."""

        payload = await request.json()
        app.state.status_code = int(payload.get("status_code", 200))
        app.state.tool_call = payload.get("tool_call")
        app.state.tool_call_sent = False
        return {
            "status_code": app.state.status_code,
            "tool_call": app.state.tool_call,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        app.state.requests.append(payload)
        if app.state.status_code != 200:
            return JSONResponse(
                status_code=app.state.status_code,
                content={
                    "error": {
                        "message": f"configured status {app.state.status_code}",
                        "type": "api_error",
                    }
                },
            )

        tool_call = app.state.tool_call
        if (
            tool_call is not None
            and not app.state.tool_call_sent
            and _payload_has_tool(payload, tool_call["name"])
        ):
            app.state.tool_call_sent = True
            if payload.get("stream") is True:
                return StreamingResponse(
                    _stream_tool_call_completion(payload, tool_call),
                    media_type="text/event-stream",
                )
            return JSONResponse(_tool_call_completion(payload, tool_call))

        messages = payload.get("messages") or []
        user_messages = [
            message
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        latest = user_messages[-1].get("content", "") if user_messages else ""
        content = f"echo user_count={len(user_messages)} latest={latest}"
        if payload.get("stream") is True:
            return StreamingResponse(
                _stream_chat_completion(payload, content),
                media_type="text/event-stream",
            )

        return JSONResponse(
            {
                "id": "chatcmpl-fabric-test",
                "object": "chat.completion",
                "created": 0,
                "model": payload.get("model", "fabric-echo"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

    @app.post("/v1/responses")
    async def responses(request: Request):
        payload = await request.json()
        app.state.requests.append(payload)
        if app.state.status_code != 200:
            return JSONResponse(
                status_code=app.state.status_code,
                content={
                    "error": {
                        "message": f"configured status {app.state.status_code}",
                        "type": "api_error",
                    }
                },
            )

        tool_call = app.state.tool_call
        if (
            tool_call is not None
            and not app.state.tool_call_sent
            and _payload_has_tool(payload, tool_call["name"])
        ):
            app.state.tool_call_sent = True
            events = _responses_tool_call_events(payload, tool_call)
        else:
            events = _responses_text_events(payload, "echo response")
        return StreamingResponse(events, media_type="text/event-stream")

    @app.post("/v1/messages")
    async def messages(request: Request):
        payload = await request.json()
        app.state.requests.append(payload)
        if app.state.status_code != 200:
            return JSONResponse(
                status_code=app.state.status_code,
                content={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"configured status {app.state.status_code}",
                    },
                },
            )

        tool_call = app.state.tool_call
        if (
            tool_call is not None
            and not app.state.tool_call_sent
            and _payload_has_tool(payload, tool_call["name"])
        ):
            app.state.tool_call_sent = True
            events = _messages_tool_call_events(payload, tool_call)
        else:
            events = _messages_text_events(payload, "echo response")
        return StreamingResponse(events, media_type="text/event-stream")

    base_url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="off",
        ws="none",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 5
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("mock API server failed to start")
        if time.monotonic() > deadline:
            raise RuntimeError("mock API server did not start within 5 seconds")
        time.sleep(0.01)

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _stream_chat_completion(payload: dict[str, object], content: str) -> Iterator[str]:
    model = payload.get("model", "fabric-echo")
    chunks = [
        {
            "id": "chatcmpl-fabric-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-fabric-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-fabric-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        },
    ]

    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _tool_call_completion(
    payload: dict[str, object], tool_call: dict[str, object]
) -> dict[str, object]:
    return {
        "id": "chatcmpl-fabric-tool-test",
        "object": "chat.completion",
        "created": 0,
        "model": payload.get("model", "fabric-echo"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_tool_call_payload(tool_call)],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _stream_tool_call_completion(
    payload: dict[str, object], tool_call: dict[str, object]
) -> Iterator[str]:
    model = payload.get("model", "fabric-echo")
    chunks = [
        {
            "id": "chatcmpl-fabric-tool-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                **_tool_call_payload(tool_call),
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-fabric-tool-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]

    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _tool_call_payload(tool_call: dict[str, object]) -> dict[str, object]:
    return {
        "id": "call-fabric-test",
        "type": "function",
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["arguments"]),
        },
    }


def _responses_tool_call_events(
    payload: dict[str, object], tool_call: dict[str, object]
) -> Iterator[str]:
    arguments = json.dumps(tool_call["arguments"])
    item = {
        "id": "fc_fabric_test",
        "type": "function_call",
        "call_id": "call_fabric_test",
        "name": tool_call["name"],
        "arguments": arguments,
        "status": "completed",
    }
    if namespace := tool_call.get("namespace"):
        item["namespace"] = namespace
    response = _responses_response(payload, [item])
    events = [
        {"type": "response.created", "sequence_number": 0, "response": response},
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {**item, "arguments": "", "status": "in_progress"},
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 2,
            "item_id": item["id"],
            "output_index": 0,
            "delta": arguments,
        },
        {
            "type": "response.function_call_arguments.done",
            "sequence_number": 3,
            "item_id": item["id"],
            "output_index": 0,
            "arguments": arguments,
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 4,
            "output_index": 0,
            "item": item,
        },
        {"type": "response.completed", "sequence_number": 5, "response": response},
    ]
    yield from _responses_sse(events)


def _responses_text_events(payload: dict[str, object], content: str) -> Iterator[str]:
    part = {"type": "output_text", "text": content, "annotations": []}
    item = {
        "id": "msg_fabric_test",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [part],
    }
    response = _responses_response(payload, [item])
    events = [
        {"type": "response.created", "sequence_number": 0, "response": response},
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {**item, "status": "in_progress", "content": []},
        },
        {
            "type": "response.content_part.added",
            "sequence_number": 2,
            "item_id": item["id"],
            "output_index": 0,
            "content_index": 0,
            "part": {**part, "text": ""},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 3,
            "item_id": item["id"],
            "output_index": 0,
            "content_index": 0,
            "delta": content,
        },
        {
            "type": "response.output_text.done",
            "sequence_number": 4,
            "item_id": item["id"],
            "output_index": 0,
            "content_index": 0,
            "text": content,
        },
        {
            "type": "response.content_part.done",
            "sequence_number": 5,
            "item_id": item["id"],
            "output_index": 0,
            "content_index": 0,
            "part": part,
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 6,
            "output_index": 0,
            "item": item,
        },
        {"type": "response.completed", "sequence_number": 7, "response": response},
    ]
    yield from _responses_sse(events)


def _responses_response(
    payload: dict[str, object], output: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "id": "resp_fabric_test",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": payload.get("model", "fabric-echo"),
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": 0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
        "metadata": {},
    }


def _responses_sse(events: list[dict[str, object]]) -> Iterator[str]:
    for event in events:
        yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


def _messages_tool_call_events(
    payload: dict[str, object], tool_call: dict[str, object]
) -> Iterator[str]:
    arguments = json.dumps(tool_call["arguments"])
    events = [
        _messages_start(payload),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_fabric_test",
                "name": tool_call["name"],
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": arguments},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 0},
        },
        {"type": "message_stop"},
    ]
    yield from _messages_sse(events)


def _messages_text_events(payload: dict[str, object], content: str) -> Iterator[str]:
    events = [
        _messages_start(payload),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": content},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 0},
        },
        {"type": "message_stop"},
    ]
    yield from _messages_sse(events)


def _messages_start(payload: dict[str, object]) -> dict[str, object]:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_fabric_test",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": payload.get("model", "fabric-echo"),
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }


def _messages_sse(events: list[dict[str, object]]) -> Iterator[str]:
    for event in events:
        yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


def _payload_has_tool(payload: dict[str, object], tool_name: object) -> bool:
    def contains(value: object) -> bool:
        if isinstance(value, dict):
            if value.get("name") == tool_name:
                return True
            return any(contains(item) for item in value.values())
        if isinstance(value, list):
            return any(contains(item) for item in value)
        return False

    return contains(payload.get("tools", []))
