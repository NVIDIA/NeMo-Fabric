# OpenClaw adapter

The adapter owns a persistent OpenClaw gateway through its RPC interface.
Install the Node OpenClaw runtime separately, then install this package and use
`nvidia.fabric.openclaw`. `harness.settings.cli` locates its `openclaw.mjs`.

Fabric model roles become isolated OpenClaw provider routes; each model requires
`base_url` and native `settings.api`. `settings.model_metadata` carries native
model fields. `harness.settings.native_config` supplies native gateway, plugin,
OTLP, tool search, heartbeat, and other advertised settings without a consumer
translation layer. Keep secret values in environment variables and use native
OpenClaw environment references.

The adapter owns configured models and agent entries. It refuses conflicting
persisted native configuration and locks retained state against concurrent
runtimes. It never retries an uncertain invocation.

Native runtime availability and configuration loading require the installed
OpenClaw executable; descriptor planning does not establish runtime readiness.
