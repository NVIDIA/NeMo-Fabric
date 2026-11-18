<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Adapter Utilities

`nemo-fabric-adapters-common` provides shared Python helpers for NeMo Fabric
adapter implementations. Adapter packages normally install
this package as a dependency.

Install the package directly when developing an adapter:

```bash
pip install nemo-fabric-adapters-common
```

Use the maintained
[adapter contract documentation](https://github.com/NVIDIA/NeMo-Fabric/tree/main/docs/adapter-contract)
for adapter and configuration guidance. Source code is available in the
[NVIDIA NeMo Fabric repository](https://github.com/NVIDIA/NeMo-Fabric/).

## Persistent Local Hosts

Every local Process or Python adapter implements the ordered persistent-host
wire protocol. Python adapters can use
`nemo_fabric_adapters.common.lifecycle`. Supply a factory that creates one
adapter-owned runtime with asynchronous `start`, `invoke`, and `stop` methods:

```python
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle


class AdapterRuntime:
    async def start(self, payload):
        self.client = await connect_client(payload)

    async def invoke(
        self,
        request: AgentRunRequest,
        context: RuntimeContext,
    ) -> AgentRunResult:
        native = await self.client.run(request.input)
        return AgentRunResult(
            status=AgentRunStatus.SUCCEEDED,
            output={"response": native.text},
        )

    async def stop(self):
        await self.client.close()


lifecycle.serve(AdapterRuntime, config_loader=AgentConfig.from_mapping)
```

If the adapter descriptor declares `capabilities.streaming`, the runtime must
also implement native OpenAI Chat Completions streaming:

```python
class AdapterRuntime:
    async def invoke_openai_stream(self, request, context, emit):
        async for chunk in self.client.stream(request.input):
            await emit(chunk)
        return AgentRunResult(
            status=AgentRunStatus.SUCCEEDED,
            output={"response": self.client.final_text},
        )
```

The optional method must have the signature
`async invoke_openai_stream(request, context, emit)`. Execute the target exactly once,
await `emit(chunk)` only for mappings in the
`openai.chat_completions.chunk/v1` profile, and return one `AgentRunResult`.
Each chunk requires non-empty `id` and `model` strings, a
nonnegative integer `created`, the exact `chat.completion.chunk` object
discriminator, and structurally valid `choices`. An empty chunk stream is
valid.

The SDK owns the authenticated loopback HTTP transport with chunked NDJSON
framing. The common host validates its credentials and framing, removes the
transport from the adapter payload, and supplies `emit`. Do not write chunks to
stdout or log stream credentials. This method is not used for Relay-backed
`Runtime.invoke_stream()`, which continues to execute ordinary `invoke`.
Process bindings that implement the wire protocol directly must follow the
generated
[`openai-stream-record.schema.json`](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/adapter-contract/openai-stream-record.schema.json)
chunk and explicit-end envelopes.

Adapters ask the host to validate the southbound config before `start`:

```python
from nemo_fabric_adapter_contract.models import AgentConfig

lifecycle.serve(AdapterRuntime, config_loader=AgentConfig.from_mapping)
```

The runtime then receives an `AgentConfig` instance in `payload["config"]`.
New adapters provide this loader; `FabricConfig` never crosses the supported
southbound boundary.

Adapters that call an OpenAI-compatible Chat Completions endpoint can use
`nemo_fabric_adapters.common.openai_chat`. The helper requests an SSE stream,
waits for its terminal event, and normalizes response text and token usage while
the adapter continues to own its HTTP client, authentication, lifecycle, and
error mapping.

NeMo Fabric calls the factory once per local host to create one runtime instance and
serializes invocations through that instance. The host keeps one event loop
alive for the complete lifecycle so SDK clients, compiled graphs,
checkpointers, and harness databases can remain live safely. NeMo Fabric sends the
resolved configuration and capability plan during `start`. Each subsequent
`invoke` wire payload contains only `runtime_context` and `request`, and the
helper validates those values and calls `AdapterRuntime.invoke(request,
context)` with typed models. It also requires a typed `AgentRunResult`. An
adapter that needs configuration during invocation retains it as runtime-owned
state during `start`. Adapter stdout is reserved for the protocol; diagnostics
are redirected to stderr. A host crash or protocol timeout terminates that
runtime.

## Relay Gateway Readiness

The Claude and Codex adapters wait up to 30 seconds for the supervised Relay
gateway's HTTP health endpoint after launch, including required worker initialization.
The Relay CLI check and plugin configuration writing happen before launch and
are outside this readiness wait. Process exit is checked before each health
probe and again before accepting a successful response. An exit during an
unsuccessful probe is detected at the next poll. A gateway that never becomes
healthy fails when the budget expires and is stopped.
The adapters retain its log for diagnosis. Local Python hosts separately enforce
a 90-second adapter startup limit.

Relay 0.11 includes the
[upstream startup-memory fix](https://github.com/NVIDIA/NeMo-Relay/pull/1375)
introduced in Relay 0.10.3. NeMo Fabric supports the `>=0.9,<0.12` CLI range.

## Runtime Health

An optional `AdapterRuntime.check(level)` method returns native health evidence
through the shared `AdapterCheckResult` contract. The host serves checks through
an authenticated loopback channel and applies the remaining request deadline.
Refer to the [SDK health transport diagram](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/sdk/health-checks.mdx#runtime-health) for lifecycle and health request routing.

## Relay Request Correlation

In-process Relay SDK adapters that own their Agent scope can use
`nemo_fabric_adapters.common.utils.relay_request_context()` around each Agent
scope. A UUID `RuntimeContext.request_id` becomes the propagated Relay root;
other request IDs remain available as `nemo_fabric_request_id` metadata. The
helper always preserves that metadata so Relay-backed streaming can identify
the active turn.

Pass `request.relay_session_root` as the helper's `session_root` to let a
caller group invocations into one Relay session. SDK callers set
`RunRequest.relay_session_root`; core forwards it as the typed
`AgentRunRequest.relay_session_root` field. Context keys do not control Relay
propagation.
When that value is a UUID string Relay accepts, it becomes the propagated root
and is recorded as `nemo_fabric_session_root` metadata. A UUID request ID stays
the parent; otherwise the session root is also the parent. Relay derives ATIF
`session_id` from the root, so invocations that share a session root export
separate trajectories under one session. A missing value, a non-string, a
string that is not a UUID, the nil UUID, or a UUID whose final eight bytes are
zero falls back to the request root without an error.

This helper does not apply to adapters that send telemetry through an external
Relay gateway or whose upstream integration creates an isolated scope context.
Those adapters retain their native session correlation until their Relay
boundary accepts a per-turn propagation context.

### Correlation Metadata Keys

Adapters that own their invocation-level Agent scope attach NeMo Fabric IDs to
that root scope as Relay metadata. The following metadata keys are reserved for
this purpose, and telemetry consumers match them literally:

| Relay metadata key | `RuntimeContext` field | Meaning |
| --- | --- | --- |
| `nemo_fabric_request_id` | `request_id` | Correlates the caller's logical request. A caller can reuse the value across retries, so several invocations can share it. |
| `nemo_fabric_invocation_id` | `invocation_id` | Identifies one invocation. NeMo Fabric assigns a new value to every invocation, so this key joins Relay telemetry to exactly one `RunResult`. |
| `nemo_fabric_runtime_id` | `runtime_id` | Identifies the runtime that processed the invocation. One runtime can process multiple invocations. |

Build the keys with
`nemo_fabric_adapters.common.utils.relay_correlation_metadata()` and merge them
into the metadata that `relay_request_context()` returns, as in the following
example. Merging rather than replacing keeps any key that
`relay_request_context()` adds, such as `nemo_fabric_session_root`.

```python
import nemo_fabric_adapters.common.utils as common_utils
from nemo_relay import ScopeType
from nemo_relay import scope

request_context, metadata = common_utils.relay_request_context(
    context.request_id, request.relay_session_root
)
metadata.update(common_utils.relay_correlation_metadata(context))
with (
    request_context,
    scope.scope("my-adapter.request", ScopeType.Agent, metadata=metadata),
):
    ...
```

Nested scopes do not repeat the keys because the Relay scope hierarchy already
links them to the root scope. The LangChain Deep Agents, mini-SWE-agent, and
NOOA adapters attach all three keys. NOOA calls only
`relay_correlation_metadata()`, so its request ID does not become the
propagated Relay root.

Claude, Codex, and Pi send telemetry through an external Relay gateway and do
not yet carry per-invocation NeMo Fabric IDs into Relay. Hermes Agent emits
Relay telemetry through its own plugin, so its adapter does not attach these
keys either. Remote Agent forwards only `nemo_fabric_request_id`, in the remote
API request body.

ATOF records carry these keys in the root scope's event metadata. To export
them as OpenTelemetry span attributes, add the `nemo_fabric_` prefix to
`promote_metadata_prefixes` on the trace endpoint. NeMo Fabric passes the field
to Relay unchanged:

```python
from nemo_fabric import RelayOpenTelemetryEndpointConfig

endpoint = RelayOpenTelemetryEndpointConfig(
    type="gen_ai",
    endpoint="http://localhost:4318/v1/traces",
    promote_metadata_prefixes=["nemo_fabric_"],
)
```

Relay promotes the keys onto the invocation's root Agent span; child spans
share that span's trace. Refer to "Event Metadata Promotion" in the
[NeMo Relay OpenTelemetry documentation](https://github.com/NVIDIA/NeMo-Relay/blob/0.9.3/docs/configure-plugins/observability/opentelemetry.mdx)
for prefix syntax and collision rules.

The lifecycle host carries optional typed software evidence beside its outcome, leaving `AgentRunResult` unchanged. It collects selected metadata after successful startup and reuses that snapshot across invocations. The start payload's `provenance.enabled=false` skips collectors and passive version capture. Missing reports from older hosts remain explicit gaps at the core receiver. Metadata helpers adapt selected code from NVIDIA/NeMo-Fabric#458 at `2bfd53321fa8d541d42abc195be5e9dd9db4350f`; the current report model, bounds and timestamps are independent of that PR's versioned extension envelope.

## MCP Client Credentials

Adapters whose MCP client accepts an `httpx.Auth` can implement
`service_account` MCP authentication with
`nemo_fabric_adapters.common.mcp_service_account`. Build a
`ServiceAccountTokenClient` from the server's authentication settings, using
`validate_token_url()` and `resolve_client_secret()` to check `token_url` and
read the secret named by `client_secret_env`, and pass
`service_account_auth(client)` to the MCP client:

```python
from nemo_fabric_adapters.common import mcp_service_account

auth = server.authentication
token_client = mcp_service_account.ServiceAccountTokenClient(
    client_id=auth.client_id,
    client_secret=mcp_service_account.resolve_client_secret(name, auth.client_secret_env),
    token_url=mcp_service_account.validate_token_url(name, auth.token_url),
    scopes=auth.scopes,
    token_endpoint_auth_method=auth.token_endpoint_auth_method,
    token_cache_buffer_seconds=auth.token_cache_buffer_seconds,
)
http_auth = mcp_service_account.service_account_auth(token_client)
```

The token is cached in memory and renewed before it expires. On a `401` from the
MCP server the auth requests a new token and retries the request once. A
`ServiceAccountTokenError` with `rejected=True` means the token endpoint refused
the client credentials. The module imports `httpx` when it needs it, so an
adapter that uses it must depend on `httpx`. Use `secret_env_names()` to find the
secret variables an adapter must keep out of agent-run commands.

## MCP Browser Sign-In

Adapters whose MCP client accepts an `httpx.Auth` can implement `oauth2` MCP
authentication with `nemo_fabric_adapters.common.mcp_oauth`.
`browser_sign_in_auth()` returns the MCP SDK's OAuth client provider
(authorization code with PKCE) with tokens stored in owner-only files in the
given directory and the redirect received on a loopback callback:

```python
from nemo_fabric_adapters.common import mcp_oauth

auth = mcp_oauth.browser_sign_in_auth(
    name,
    url,
    state_folder / mcp_oauth.TOKEN_DIRECTORY_NAME,
    client_id=config.client_id,
    scopes=config.scopes,
    redirect_uri=config.redirect_uri,
    authorization_timeout_seconds=config.authorization_timeout_seconds,
)
auth.sign_in.interactive = True  # allow a browser sign-in now
```

Sign-in starts off: until the adapter sets `auth.sign_in.interactive`, a server
that needs a new sign-in fails with `McpSignInRequired` instead of opening a
browser, so an adapter can start without waiting for one. Stored tokens are used
and refreshed either way. `is_sign_in_error()` recognizes the errors that mean a
server needs a sign-in. The module imports `mcp` and `httpx` when it needs them,
so an adapter that uses it must depend on both.
