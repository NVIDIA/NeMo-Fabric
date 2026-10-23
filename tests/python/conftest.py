# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest
from nemo_fabric import Fabric, FabricConfig, PricingConfig


@pytest.fixture(name="accounting_catalog")
def accounting_catalog_fixture():
    return None


@pytest.fixture(name="accounting_runtime")
async def accounting_runtime_fixture(tmp_path, accounting_catalog):
    """Normalize controlled adapter evidence through the native public runtime."""
    root = Path(__file__).resolve().parents[2]
    accounting_schema = json.loads(
        (root / "schemas/extensions/usage-accounting.schema.json").read_text()
    )
    definitions = accounting_schema.pop("$defs")
    descriptor = {
        "contract_version": "fabric.adapter/v1alpha2",
        "adapter_id": "test.accounting",
        "adapter_kind": "process",
        "runner": {"command": sys.executable, "script": "adapter.py"},
        "settings_schema": {"type": "object", "additionalProperties": False},
        "extension_schemas": {
            "usage": {
                "type": "object",
                "properties": {
                    "nemo_fabric.usage_accounting.v1": accounting_schema,
                    "nemo_fabric.detailed_token_usage.v1": {
                        "$ref": "#/$defs/ReportedUsage"
                    },
                },
                "additionalProperties": False,
                "$defs": definitions,
            }
        },
    }
    (tmp_path / "accounting.fabric-adapter.json").write_text(json.dumps(descriptor))
    (tmp_path / "adapter.py").write_text("""
import json
import sys

for line in sys.stdin:
    message = json.loads(line)
    operation = message["operation"]
    output = None
    if operation == "invoke":
        output = {
            "status": "succeeded",
            "output": None,
            "usage": message["payload"]["request"]["input"],
        }
    print(json.dumps({
        "operation": operation,
        "outcome": {"status": "succeeded", "output": output},
    }), flush=True)
    if operation == "stop":
        break
""")
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "accounting"},
            "harness": {"adapter_id": "test.accounting"},
            "discovery": {"local_paths": ["accounting.fabric-adapter.json"]},
            "environment": {"provider": "local", "workspace": "."},
            "runtime": {"artifacts": "./artifacts"},
        }
    )
    if accounting_catalog is not None:
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps(accounting_catalog))
        config.pricing = PricingConfig(catalog_path=catalog_path)
    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        yield runtime


@pytest.fixture(name="blocking_adapter_config")
def blocking_adapter_config_fixture(tmp_path):
    descriptor = {
        "contract_version": "fabric.adapter/v1alpha2",
        "adapter_id": "test.blocking",
        "adapter_kind": "process",
        "runner": {"command": sys.executable, "script": "adapter.py"},
        "settings_schema": {"type": "object", "additionalProperties": False},
    }
    (tmp_path / "blocking.fabric-adapter.json").write_text(json.dumps(descriptor))
    (tmp_path / "adapter.py").write_text("""
import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path.cwd()
(root/'adapter.pid').write_text(str(os.getpid()))
harness=None
if (root/'spawn-harness').exists():
    harness=subprocess.Popen([sys.executable,'-c','''
import os,signal,time
from pathlib import Path
signal.signal(signal.SIGTERM,signal.SIG_IGN)
signal.signal(signal.SIGINT,signal.SIG_IGN)
Path('harness.pid').write_text(str(os.getpid()))
while True: time.sleep(0.01)
'''],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    while not (root/'harness.pid').exists(): time.sleep(0.01)
try:
    for line in sys.stdin:
        message=json.loads(line); operation=message['operation']; output=None
        if operation=='invoke':
            (root/'entered').write_text('entered')
            while not (root/'release').exists(): time.sleep(0.01)
            output={'status':'succeeded','output':{'response':'released'}}
        print(json.dumps({'operation':operation,'outcome':{'status':'succeeded','output':output}}),flush=True)
        if operation=='stop': break
finally:
    if harness is not None:
        (root/'harness-cleanup-entered').write_text('entered')
        harness.terminate()
        harness.wait()
    (root/'stopped').write_text('cleanup')
""")
    return FabricConfig.model_validate(
        {
            "metadata": {"name": "blocking"},
            "harness": {"adapter_id": "test.blocking"},
            "discovery": {"local_paths": ["blocking.fabric-adapter.json"]},
            "environment": {"provider": "local", "workspace": "."},
            "runtime": {"artifacts": "./artifacts"},
        }
    )
