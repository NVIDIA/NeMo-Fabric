# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free smoke for the importable public SDK contract."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

from nemo_fabric import (  # noqa: E402
    Fabric,
    FabricConfig,
    FabricNativeUnavailableError,
    HarnessConfig,
    MetadataConfig,
    RunRequest,
    RuntimeConfig,
)
from nemo_fabric import client as client_mod  # noqa: E402


def main() -> None:
    config = FabricConfig(
        metadata=MetadataConfig(name="demo"),
        harness=HarnessConfig(
            adapter_id="test.fabric.shim",
            settings={"future_adapter_option": True},
        ),
        runtime=RuntimeConfig(),
        extra_fields={"future_config": {"enabled": True}},
    )
    request = RunRequest(
        input="hello",
        request_id="request-1",
        context={"job_id": "job-1"},
    )

    assert config.metadata.name == "demo"
    assert config.harness.settings["future_adapter_option"] is True
    assert config.to_mapping()["future_config"] == {"enabled": True}
    assert request.to_mapping()["context"] == {"job_id": "job-1"}

    client_mod._native = None
    try:
        Fabric().plan(config)
    except FabricNativeUnavailableError:
        pass
    else:
        raise AssertionError("native-only Fabric must reject a missing extension")


if __name__ == "__main__":
    main()
