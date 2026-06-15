# Fabric JSON Schemas

This directory contains committed JSON Schema snapshots for the public Fabric
contract. The files are generated from the Rust core types, not edited by hand.

Use the Fabric CLI to regenerate them after intentional contract changes:

```bash
cargo run -p fabric-cli -- schema --output-dir schemas
```

Use the CLI to inspect one schema:

```bash
cargo run -p fabric-cli -- schema --name agent
```

Run `cargo test` after regenerating schemas. The snapshot tests compare the
committed files against the schemas generated from the current Rust types and
fail on accidental drift.
