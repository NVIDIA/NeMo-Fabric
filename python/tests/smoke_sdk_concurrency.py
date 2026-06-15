"""Smoke test for independent concurrent Fabric SDK runs."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from shutil import copytree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

from nemo_fabric import FabricClient


async def run_copy(client: FabricClient, fixture_agent: Path, root: Path, name: str) -> dict:
    agent = root / name
    copytree(fixture_agent, agent)
    return await client.run(agent, profile="env_local", input_text=f"hello from {name}")


async def main() -> None:
    fixture_agent = ROOT / "tests" / "fixtures" / "hermes-shim-agent"

    async with FabricClient(
        command=("cargo", "run", "-q", "-p", "fabric-cli", "--"),
        cwd=ROOT,
    ) as client:
        with tempfile.TemporaryDirectory(prefix="fabric-sdk-concurrency-") as tmpdir:
            temp_root = Path(tmpdir)
            first, second = await asyncio.gather(
                run_copy(client, fixture_agent, temp_root, "agent-one"),
                run_copy(client, fixture_agent, temp_root, "agent-two"),
            )

    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert first["runtime_id"] != second["runtime_id"]
    assert first["invocation_id"] != second["invocation_id"]
    assert first["output"]["received"] == "hello from agent-one"
    assert second["output"]["received"] == "hello from agent-two"
    assert first["artifacts"]["root"] != second["artifacts"]["root"]


if __name__ == "__main__":
    asyncio.run(main())
