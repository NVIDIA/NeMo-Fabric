# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch a native Hermes child with Fabric's agent-construction settings.

Hermes v0.22.5's API path does not read all behavioral config from config.yaml.
Install the mapping before loading the CLI; execution and transport stay native.
The replacement is local to this adapter-owned child process.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapters.hermes import configuration


def configured_kwargs(
    config: AgentConfig, native_kwargs: dict[str, Any], native_config: dict[str, Any]
) -> dict[str, Any]:
    """Apply Fabric-owned behavior while retaining native routing and identity."""
    enabled = configuration.resolve_hermes_toolsets(config, native_config)
    behavior = configuration.agent_behavior_kwargs(config, enabled)
    overrides = {
        **(native_kwargs.get("request_overrides") or {}),
        **(behavior["request_overrides"] or {}),
    }
    result = {**native_kwargs, **behavior, "request_overrides": overrides or None}
    if config.instructions and config.instructions.system:
        result["ephemeral_system_prompt"] = config.instructions.system.content
    return result


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fabric-config", type=Path, required=True)
    options, arguments = parser.parse_known_args()
    config = AgentConfig.from_mapping(json.loads(options.fabric_config.read_text()))

    import run_agent
    from hermes_cli.config import load_config

    native_agent = run_agent.AIAgent
    signature = inspect.signature(native_agent)

    class ConfiguredAIAgent(native_agent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            bound = signature.bind(*args, **kwargs)
            super().__init__(
                **configured_kwargs(config, bound.arguments, load_config())
            )

    run_agent.AIAgent = ConfiguredAIAgent
    from hermes_cli.main import main as hermes_main

    sys.argv = ["hermes", *arguments]
    hermes_main()


if __name__ == "__main__":
    main()
