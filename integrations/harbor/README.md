# Harbor Integration

Fabric exposes a Harbor consumer wrapper at
`nemo_fabric.integrations.harbor:FabricAgent`.

Harbor remains responsible for task materialization, environment lifecycle,
artifact transfer, verifier execution, and reward calculation. Fabric is used
inside the Harbor agent phase to invoke the selected Fabric agent package and
profile stack.

Install with the Harbor extra when using the wrapper from an environment that
does not already provide Harbor:

```bash
pip install "nemo-fabric[harbor]"
```

Local checkout smoke:

```bash
python3 -m pip install -e ../harbor
python3 python/tests/smoke_harbor_integration.py
```

The Docker-backed SWE-Bench smoke remains opt-in:

```bash
RUN_FABRIC_HARBOR_SWEBENCH_DOCKER=1 python3 tests/smoke_harbor_swebench_task.py
```
