# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The packaged descriptor accepts the collector's shared usage evidence."""

import json
import importlib.util
from pathlib import Path

from jsonschema import Draft202012Validator
from nemo_fabric_adapter_contract.models import ReportedUsage
from nemo_fabric_adapter_contract.usage import UsageObservation, usage_from_observations


def test_deepagents_usage_declaration_matches_packaged_copy():
    root = Path(__file__).resolve().parents[2]
    canonical = root / "adapters/python/deepagents/deepagents.fabric-adapter.json"
    packaged = (
        root
        / "crates/fabric-cli/assets/adapters/deepagents/deepagents.fabric-adapter.json"
    )
    assert canonical.read_bytes() == packaged.read_bytes()
    schema = json.loads(canonical.read_text())["extension_schemas"]["usage"]
    canonical_schema = json.loads(
        (root / "schemas/extensions/usage-accounting.schema.json").read_text()
    )
    definitions = canonical_schema.pop("$defs")
    canonical_schema.pop("$schema")
    assert schema["properties"]["nemo_fabric.usage_accounting.v1"] == canonical_schema
    assert schema["$defs"] == definitions
    Draft202012Validator.check_schema(schema)
    usage = usage_from_observations(
        [
            UsageObservation(
                reported=ReportedUsage(
                    input_tokens=100,
                    input_tokens_include_cache=True,
                    cached_input_tokens=0,
                    output_tokens=2,
                    output_tokens_include_reasoning=True,
                    total_tokens=102,
                ),
                provider="openai",
                model="reported-model",
                tokens_aligned=True,
                is_model_call=True,
            )
        ]
    )
    Draft202012Validator(schema).validate(usage.extensions)


def test_usage_schema_generator_removes_obsolete_definitions(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    source = root / "scripts/contracts/generate_usage_accounting.py"
    spec = importlib.util.spec_from_file_location("generate_usage_accounting", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)

    schema = json.loads(
        (root / "schemas/extensions/usage-accounting.schema.json").read_text()
    )
    canonical = tmp_path / "schemas/extensions/usage-accounting.schema.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(json.dumps(schema))

    descriptor = tmp_path / "adapters/python/example/example.fabric-adapter.json"
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text(
        json.dumps(
            {
                "extension_schemas": {
                    "usage": {
                        "$defs": {**schema["$defs"], "Obsolete": {"type": "string"}},
                        "properties": {module.KEY: {}, "other": {"type": "object"}},
                    }
                }
            },
            indent=2,
        )
    )

    module.synchronize(check=False)
    usage = json.loads(descriptor.read_text())["extension_schemas"]["usage"]
    assert usage["$defs"] == schema["$defs"]
    assert usage["properties"]["other"] == {"type": "object"}
