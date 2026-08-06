# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an application-owned email-phishing analysis graph."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import TypedDict

from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import HarnessConfig
from nemo_fabric import InstructionConfig
from nemo_fabric import InstructionsConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ModelConfig
from nemo_fabric import RuntimeConfig
from nemo_fabric import WorkflowConfig
from nemo_fabric import WorkflowEntrypointConfig


class EmailState(TypedDict):
    """State for one email analysis."""

    input: str
    assessment: str


class TextInputGraph:
    """Adapt a raw text invocation to the graph's application state."""

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def compile(self) -> "CompiledTextInputGraph":
        """Compile the application graph and retain its text-input adapter."""

        return CompiledTextInputGraph(self._graph.compile())


class CompiledTextInputGraph:
    """Invoke a compiled graph with raw text supplied by the application."""

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    async def ainvoke(self, input_value: str) -> dict[str, str]:
        """Map a text input into the state expected by the compiled graph."""

        return await self._graph.ainvoke({"input": input_value})


def build_graph(
    model_config: Mapping[str, Any], system_instruction: str | None = None
) -> TextInputGraph:
    """Return an uncompiled graph that uses an OpenAI-compatible model endpoint."""

    from langchain_openai import ChatOpenAI
    from langgraph.graph import END
    from langgraph.graph import START
    from langgraph.graph import StateGraph

    provider = model_config.get("provider")
    if provider not in {"nim", "nvidia"}:
        raise ValueError("The email example supports the nim and nvidia model providers")
    api_key_env = str(model_config.get("api_key_env") or "NVIDIA_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable {api_key_env!r} is required")
    llm = ChatOpenAI(
        model=str(model_config["model"]),
        base_url=str(
            model_config.get("base_url") or "https://integrate.api.nvidia.com/v1"
        ),
        api_key=api_key,
        temperature=float(model_config.get("temperature") or 0),
    )

    async def classify(state: EmailState) -> dict[str, str]:
        response = await llm.ainvoke(
            [
                (
                    "system",
                    system_instruction
                    or "Classify the email as phishing or benign. Explain the evidence briefly.",
                ),
                ("user", state["input"]),
            ]
        )
        return {"assessment": str(response.content)}

    graph = StateGraph(EmailState)
    graph.add_node("classify", classify)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", END)
    return TextInputGraph(graph)


def build_config() -> FabricConfig:
    """Build the email-phishing graph configuration."""

    return FabricConfig(
        metadata=MetadataConfig(
            name="langgraph-email-phishing",
            description="Classifies suspicious email supplied by an application user.",
        ),
        harness=HarnessConfig(
            adapter_id="example.fabric.langgraph",
            resolution="preinstalled",
        ),
        workflow=WorkflowConfig(
            entrypoint=WorkflowEntrypointConfig(
                kind="langgraph_factory",
                ref="email_phishing:build_graph",
            ),
            settings={"llm_name": "nim_llm"},
        ),
        models={
            "nim_llm": ModelConfig(
                provider="nim",
                model="meta/llama-3.1-70b-instruct",
                api_key_env="NVIDIA_API_KEY",
            )
        },
        instructions=InstructionsConfig(
            system=InstructionConfig(
                content="Classify the email as phishing or benign. Explain the evidence briefly."
            )
        ),
        runtime=RuntimeConfig(input_schema="text", output_schema="json"),
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--input",
        default="Urgent: confirm your password at http://example.invalid today.",
    )
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()

    fabric = Fabric()
    config = build_config()
    output = (
        fabric.plan(config, base_dir=args.base_dir)
        if args.plan
        else await fabric.run(config, base_dir=args.base_dir, input=args.input)
    )
    print(json.dumps(output.to_mapping(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
