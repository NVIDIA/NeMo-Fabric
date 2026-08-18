{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# Optional Native OpenAI Streaming

Implement native streaming only when consumers need the Adapter Target's
progressive OpenAI Chat Completions output. Relay-backed
`Runtime.invoke_stream()` remains independent and continues to call ordinary
adapter `invoke`.

## Declare the Capability

Set `capabilities.streaming: true` only when the adapter implements
`invoke_openai_stream` through its selected runtime binding:

```json
"capabilities": {
  "streaming": true
}
```

The v1alpha2 profile is exactly `openai.chat_completions.chunk/v1`. OpenAI
Responses API events, target-native event objects, Server-Sent Events framing,
and terminal results are outside the progressive stream.

## Implement the Python Method

The common Python host calls
`async invoke_openai_stream(request, context, emit)`. The method executes the
target exactly once, awaits `emit(chunk)` for each valid OpenAI Chat
Completions chunk mapping, and returns one separate `AgentRunResult`:

```python
class TargetRuntime:
    async def invoke_openai_stream(self, request, context, emit):
        async for chunk in self.target.stream(request.input):
            await emit(chunk)
        return AgentRunResult(
            status=AgentRunStatus.SUCCEEDED,
            output={"response": self.target.final_text},
        )
```

Each chunk contains:

- Nonempty `id` and `model` strings
- A nonnegative integer `created` value
- The exact `chat.completion.chunk` object discriminator
- Structurally valid `choices`

An invocation can emit zero chunks. Its terminal value remains separate and
authoritative. Ending the iteration early does not cancel the target invocation.
Instead, the SDK drains the invocation so that the runtime can safely accept its
next turn.

`runtime.timeout_seconds` limits the complete streamed invocation. Receiving a
chunk does not reset that deadline. A timeout invalidates the local adapter
host under the same lifecycle timeout rules as ordinary invocation.

## Follow the NVIDIA NeMo Fabric Transport

NeMo Fabric owns the authenticated loopback HTTP transport, chunked
newline-delimited JSON (NDJSON) framing, correlation, validation, buffering,
and consumer lifecycle. The common host removes transport credentials before
it calls the typed adapter method. Do not persist or log the bearer token, emit
chunks on stdout, or add SSE framing.

Bindings that do not use the common Python host read the sink from
[`openai-stream-invocation.schema.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/openai-stream-invocation.schema.json)
and send records that satisfy
[`openai-stream-record.schema.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/openai-stream-record.schema.json).

The `fabric.openai_stream/v1alpha1` wire sequence is:

1. Open one connection to the supplied loopback host and port.
2. Send `POST /openai-stream` with the supplied bearer token,
   `Content-Type: application/x-ndjson`, `Transfer-Encoding: chunked`, and
   `Expect: 100-continue`.
3. Wait for `100 Continue` before invoking the target.
4. Send zero or more monotonic `chunk` records followed by exactly one `end`
   record. Encode every record as one newline-terminated JSON value.
5. Finish the HTTP chunked body, wait for `200 OK`, then write the one terminal
   lifecycle response to stdout.

The token is single-use for one invocation. A rejection, incomplete sequence,
or transport loss is a lifecycle failure. Do not retry or replay the target
after that failure. Each encoded record is limited to 1 MiB.

Verify empty and multi-chunk streams, invalid and oversized records, early
consumer close, the explicit end record, a separate terminal value, and
exactly one target invocation. Then continue with
[terminal outcomes](results.md).
