# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK admission of explicit HTTP model hosts across restricted harnesses."""

from pathlib import Path

import pytest
from nemo_fabric import DiscoveryConfig
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("adapter", ["opencode", "qwen", "kilo"])
@pytest.mark.parametrize(
    "host",
    [
        "host.docker.internal",
        "release-nhx-api",
        "gateway.ns.svc.cluster.local",
        "10.86.19.10",
    ],
)
def test_http_model_hosts_survive_sdk_planning(adapter: str, host: str):
    settings = {"insecure_http_model_hosts": [host]}
    if adapter == "qwen":
        settings["permission_mode"] = "plan"
    if adapter == "kilo":
        settings["allow_insecure_http_model_endpoint"] = True
    config = FabricConfig(
        metadata=MetadataConfig(name="http-model-hosts"),
        discovery=DiscoveryConfig(
            local_paths=[
                ROOT / f"adapters/typescript/{adapter}/{adapter}.fabric-adapter.json"
            ]
        ),
        harness=HarnessConfig(adapter_id=f"nvidia.fabric.{adapter}", settings=settings),
        models={
            "default": ModelConfig(
                provider="openai",
                model="test-model",
                api_key_env="MODEL_KEY",
                base_url=f"http://{host}:8080/apis/inference-gateway/v1",
            )
        },
    )
    plan = Fabric().plan(config, base_dir=ROOT)
    assert plan.agent_config["harness"]["settings"] == settings
    assert (
        plan.agent_config["models"]["default"]["base_url"]
        == config.models["default"].base_url
    )


@pytest.mark.parametrize("adapter", ["opencode", "qwen", "kilo"])
@pytest.mark.parametrize(
    "hosts",
    [
        None,
        "gateway",
        ["http://gateway"],
        ["gateway:8080"],
        ["gateway/v1"],
        ["*.internal"],
        [""],
        ["gateway..internal"],
    ],
)
def test_sdk_rejects_malformed_http_model_hosts(adapter: str, hosts):
    config = FabricConfig(
        metadata=MetadataConfig(name="http-model-hosts"),
        discovery=DiscoveryConfig(
            local_paths=[
                ROOT / f"adapters/typescript/{adapter}/{adapter}.fabric-adapter.json"
            ]
        ),
        harness=HarnessConfig(
            adapter_id=f"nvidia.fabric.{adapter}",
            settings={"insecure_http_model_hosts": hosts},
        ),
        models={
            "default": ModelConfig(
                provider="openai", model="test-model", api_key_env="MODEL_KEY"
            )
        },
    )
    with pytest.raises(FabricConfigError, match="insecure_http_model_hosts"):
        Fabric().plan(config, base_dir=ROOT)
