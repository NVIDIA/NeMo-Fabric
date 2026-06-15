# Examples

This directory holds sample Fabric agent packages and single-file configs used
by smoke tests and demos.

The first example focuses on the shared Fabric contract:

- validating `agent.yaml`;
- resolving named profiles from configured profile directories;
- resolving direct profile YAML paths into run plans;
- resolving an environment context without requiring Fabric to provision it;
- resolving maintained Hermes adapters from the repository adapter registry.

Start with:

```bash
cargo run -p fabric-cli -- validate examples/code-review-agent
cargo run -p fabric-cli -- inspect examples/code-review-agent
cargo run -p fabric-cli -- plan examples/code-review-agent
cargo run -p fabric-cli -- plan examples/code-review-agent --profile env_local --profile mcp_github
cargo run -p fabric-cli -- plan examples/code-review-agent --profile hermes_cli
```

The dependency-free Hermes shim used by smoke tests lives under
`tests/fixtures/hermes-shim-agent`; it is not a maintained adapter.
