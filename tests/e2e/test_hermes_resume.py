# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cold resume uses the real Hermes store and captured native model requests."""

import json
import os
import sys
import shutil
import re
from uuid import uuid4

import pytest
import requests

from nemo_fabric import Fabric, FabricConfig, RunRequest, SessionRef
from nemo_fabric.errors import FabricRuntimeError

pytestmark = pytest.mark.usefixtures("requires_hermes_agent")


@pytest.fixture(name="resume_config")
def resume_config_fixture(tmp_path, api_server):
    os.environ["ADAPTER_PYTHON"] = sys.executable
    os.environ["RESUME_FIXTURE_KEY"] = "local-test-fixture"
    return FabricConfig.from_mapping(
        {
            "metadata": {"name": "assitant"},
            "harness": {"adapter_id": "nvidia.fabric.hermes"},
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "fabric-echo",
                    "base_url": api_server + "/v1",
                    "api_key_env": "RESUME_FIXTURE_KEY",
                }
            },
            "tools": {"enabled": []},
            "runtime": {"keep_state": True, "artifacts": str(tmp_path / "artifacts")},
            "environment": {"provider": "local", "workspace": str(tmp_path)},
        }
    )


async def test_explicit_resume_restores_context_after_agent_name_correction(
    resume_config, tmp_path, api_server
):
    nonce = "conversation-only-" + uuid4().hex
    async with await Fabric().start_runtime(resume_config, base_dir=tmp_path) as first:
        reference = first.session_ref
        assert isinstance(reference, SessionRef)
        first_id = first.runtime_id
        assert (await first.invoke(input=nonce)).session_ref == reference

    corrected = resume_config.model_copy(deep=True)
    corrected.metadata.name = "assistant"
    async with await Fabric().start_runtime(
        corrected, base_dir=tmp_path, session_ref=reference
    ) as resumed:
        assert resumed.runtime_id != first_id
        assert resumed.session_ref == reference
        assert resumed.recovery_warnings == ()
        result = await resumed.invoke(input="Use our earlier conversation.")
        assert result.status == "succeeded"
        assert result.session_ref == reference

    captured = requests.get(api_server + "/_requests", timeout=5).json()
    assert nonce in json.dumps(captured[-1]["messages"])


async def test_retention_does_not_select_the_latest_conversation(
    resume_config, tmp_path, api_server
):
    nonce = "first-session-" + uuid4().hex
    async with await Fabric().start_runtime(resume_config, base_dir=tmp_path) as first:
        reference = first.session_ref
        assert (await first.invoke(input=nonce)).status == "succeeded"

    async with await Fabric().start_runtime(resume_config, base_dir=tmp_path) as fresh:
        assert fresh.session_ref != reference
        assert (await fresh.invoke(input="A new conversation.")).status == "succeeded"
    captured = requests.get(api_server + "/_requests", timeout=5).json()
    assert nonce not in json.dumps(captured[-1]["messages"])

    async with await Fabric().start_runtime(
        resume_config, base_dir=tmp_path, session_ref=reference
    ) as original:
        assert (
            await original.invoke(input="Return to the first session.")
        ).status == "succeeded"
    captured = requests.get(api_server + "/_requests", timeout=5).json()
    assert nonce in json.dumps(captured[-1]["messages"])


async def test_resume_after_storage_move_preserves_native_prompt_and_current_sampling(
    resume_config, tmp_path, api_server
):
    nonce = "retained-context-" + uuid4().hex
    async with await Fabric().start_runtime(
        resume_config, base_dir=tmp_path
    ) as original:
        reference = original.session_ref
        await original.invoke(input=nonce)
    moved = tmp_path / "moved-artifacts"
    shutil.move(str(tmp_path / "artifacts"), moved)
    current = resume_config.to_mapping()
    current["runtime"]["artifacts"] = str(moved)
    current["models"]["default"]["temperature"] = 0.21
    current["instructions"] = {
        "system": {
            "content": "Current startup instruction: cedar-428.",
            "mode": "replace",
        }
    }
    async with await Fabric().start_runtime(
        FabricConfig.from_mapping(current),
        base_dir=tmp_path,
        session_ref=reference,
    ) as resumed:
        assert resumed.session_ref == reference
        assert (
            await resumed.invoke(input="Continue with the current instruction.")
        ).status == "succeeded"
    captured = requests.get(api_server + "/_requests", timeout=5).json()[-1]
    messages = captured["messages"]
    assert nonce in json.dumps(messages)
    # Hermes pins the saved prompt on native resume. Fabric still supplies the
    # current instructions; Fabric must not rewrite that native resume policy.
    assert "cedar-428" not in json.dumps(messages)
    assert captured["temperature"] == 0.21


async def test_other_adapter_reference_is_rejected_before_native_start(
    resume_config, tmp_path, api_server
):
    reference = SessionRef.from_mapping(
        {
            "schema_version": "fabric.session_ref/v1alpha1",
            "adapter_id": "nvidia.fabric.openclaw",
            "session_id": "opaque",
        }
    )
    with pytest.raises(FabricRuntimeError) as caught:
        await Fabric().start_runtime(
            resume_config, base_dir=tmp_path, session_ref=reference
        )
    assert caught.value.code == "session_adapter_mismatch"
    assert requests.get(api_server + "/_requests", timeout=5).json() == []


async def test_resume_missing_session_fails_without_starting_fresh(
    resume_config, tmp_path
):
    reference = SessionRef.from_mapping(
        {
            "schema_version": "fabric.session_ref/v1alpha1",
            "adapter_id": "nvidia.fabric.hermes",
            "session_id": "missing-" + uuid4().hex,
        }
    )
    with pytest.raises(FabricRuntimeError) as caught:
        await Fabric().start_runtime(
            resume_config, base_dir=tmp_path, session_ref=reference
        )
    assert caught.value.code == "session_not_found"


async def test_missing_native_database_is_not_recreated_on_resume(
    resume_config, tmp_path
):
    async with await Fabric().start_runtime(resume_config, base_dir=tmp_path) as first:
        reference = first.session_ref
    database = (
        tmp_path / "artifacts/.fabric/state/nvidia.fabric.hermes/assitant/state.db"
    )
    database.unlink()
    with pytest.raises(FabricRuntimeError) as caught:
        await Fabric().start_runtime(
            resume_config, base_dir=tmp_path, session_ref=reference
        )
    assert caught.value.code == "session_not_found"
    assert not database.exists()


async def test_resume_uses_the_original_lock_after_rename(resume_config, tmp_path):
    async with await Fabric().start_runtime(resume_config, base_dir=tmp_path) as first:
        corrected = resume_config.model_copy(deep=True)
        corrected.metadata.name = "assistant"
        with pytest.raises(FabricRuntimeError) as caught:
            await Fabric().start_runtime(
                corrected,
                base_dir=tmp_path,
                session_ref=first.session_ref,
            )
        assert caught.value.code == "state_in_use"


@pytest.mark.skipif(os.name != "posix", reason="Symbolic-link retained storage")
async def test_a_home_alias_cannot_acquire_a_second_writer_lock(
    resume_config, tmp_path
):
    namespace = tmp_path / "artifacts/.fabric/state/nvidia.fabric.hermes"
    (namespace / "real").mkdir(parents=True)
    (namespace / "assitant").symlink_to(namespace / "real", target_is_directory=True)
    async with await Fabric().start_runtime(resume_config, base_dir=tmp_path) as first:
        with pytest.raises(FabricRuntimeError) as caught:
            async with await Fabric().start_runtime(
                resume_config,
                base_dir=tmp_path,
                session_ref=first.session_ref,
            ):
                pass
        assert caught.value.code == "state_in_use"


@pytest.mark.skipif(os.name != "posix", reason="Symbolic-link retained storage")
async def test_a_home_alias_outside_storage_is_rejected_before_native_writes(
    resume_config, tmp_path
):
    namespace = tmp_path / "artifacts/.fabric/state/nvidia.fabric.hermes"
    namespace.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (namespace / "assitant").symlink_to(outside, target_is_directory=True)
    with pytest.raises(FabricRuntimeError) as caught:
        await Fabric().start_runtime(resume_config, base_dir=tmp_path)
    assert caught.value.code == "session_incompatible"
    assert list(outside.iterdir()) == []


async def test_uncertain_native_turn_resumes_with_warning_without_replay(
    resume_config, tmp_path, api_server
):
    from hermes_state import SessionDB

    async with await Fabric().start_runtime(resume_config, base_dir=tmp_path) as first:
        reference = first.session_ref
        assert (await first.invoke(input="A completed turn.")).status == "succeeded"

    database = (
        tmp_path / "artifacts/.fabric/state/nvidia.fabric.hermes/assitant/state.db"
    )
    native = SessionDB(database)
    try:
        native.append_message(
            reference.session_id, "user", "Create the release ticket."
        )
        native.append_message(
            reference.session_id,
            "assistant",
            tool_calls=[
                {
                    "id": "ticket-call",
                    "type": "function",
                    "function": {"name": "create_ticket", "arguments": "{}"},
                }
            ],
        )
        before = native.get_messages(reference.session_id)
    finally:
        native.close()
    calls_before = len(requests.get(api_server + "/_requests", timeout=5).json())

    async with await Fabric().start_runtime(
        resume_config, base_dir=tmp_path, session_ref=reference
    ) as resumed:
        assert resumed.session_ref == reference
        assert resumed.recovery_warnings[0].code == "session_recovery_required"
        assert (
            len(requests.get(api_server + "/_requests", timeout=5).json())
            == calls_before
        )
        blocked = await resumed.invoke(input="Do not acknowledge recovery yet.")
        assert blocked.error.code == "session_recovery_required"
        native = SessionDB(database)
        try:
            assert native.get_messages(reference.session_id) == before
        finally:
            native.close()
        assert (
            len(requests.get(api_server + "/_requests", timeout=5).json())
            == calls_before
        )


async def test_acknowledged_recovery_allows_a_new_explicit_turn(
    resume_config, tmp_path, api_server
):
    from hermes_state import SessionDB

    async with await Fabric().start_runtime(resume_config, base_dir=tmp_path) as first:
        reference = first.session_ref
    database = (
        tmp_path / "artifacts/.fabric/state/nvidia.fabric.hermes/assitant/state.db"
    )
    native = SessionDB(database)
    try:
        native.append_message(reference.session_id, "user", "An interrupted request.")
    finally:
        native.close()

    async with await Fabric().start_runtime(
        resume_config, base_dir=tmp_path, session_ref=reference
    ) as resumed:
        assert (
            "model may act on unfinished requests"
            in resumed.recovery_warnings[0].message
        )
        result = await resumed.invoke(
            request=RunRequest(
                input="I inspected the state; answer this new request.",
                acknowledge_recovery=True,
            )
        )
        assert result.status == "succeeded"
    captured = requests.get(api_server + "/_requests", timeout=5).json()
    messages = json.dumps(captured[-1]["messages"])
    assert "An interrupted request." in messages
    assert "I inspected the state; answer this new request." in messages


async def test_managed_server_explicitly_resumes_a_stopped_runtime_session(
    resume_config, tmp_path, api_server
):
    from nemo_fabric.runtime_server.state import RuntimeServer

    server = RuntimeServer(directory=tmp_path)
    base = {"interface_version": 2, "agent": "assitant"}
    start = {**base, "operation": "start", "config": resume_config.to_mapping()}
    first = await server.handle(start)
    assert first["status"] == "succeeded"
    reference = first["result"]["session_ref"]
    owner = {**base, "runtime_id": first["result"]["runtime_id"]}
    nonce = "managed-session-" + uuid4().hex
    try:
        result = await server.handle({**owner, "operation": "invoke", "input": nonce})
        assert result["result"]["status"] == "succeeded"
        assert (await server.handle({**owner, "operation": "stop"}))[
            "status"
        ] == "succeeded"
        resumed = await server.handle({**start, "session_ref": reference})
        assert resumed["status"] == "succeeded"
        assert resumed["result"]["session_ref"] == reference
        assert resumed["result"]["runtime_id"] != owner["runtime_id"]
        result = await server.handle(
            {
                **base,
                "operation": "invoke",
                "runtime_id": resumed["result"]["runtime_id"],
                "input": "Continue the recorded conversation.",
            }
        )
        assert result["result"]["status"] == "succeeded"
        messages = requests.get(api_server + "/_requests", timeout=5).json()[-1][
            "messages"
        ]
        assert nonce in json.dumps(messages)
    finally:
        await server.shutdown()


@pytest.mark.skipif(os.name != "posix", reason="Hermes dashboard requires POSIX")
async def test_native_dashboard_can_read_the_retained_conversation(
    resume_config, tmp_path, unused_tcp_port
):
    nonce = "native-dashboard-history-" + uuid4().hex
    async with await Fabric().start_runtime(resume_config, base_dir=tmp_path) as first:
        reference = first.session_ref
        await first.invoke(input=nonce)
    web_dist = tmp_path / "web-dist"
    web_dist.mkdir()
    (web_dist / "index.html").write_text(
        "<html><head></head><body>Hermes native dashboard</body></html>"
    )
    os.environ["HERMES_WEB_DIST"] = str(web_dist)
    current = resume_config.to_mapping()
    current["harness"]["settings"] = {
        "interfaces": {"dashboard": {"port": unused_tcp_port}}
    }
    async with await Fabric().start_runtime(
        FabricConfig.from_mapping(current),
        base_dir=tmp_path,
        session_ref=reference,
    ):
        dashboard = f"http://127.0.0.1:{unused_tcp_port}"
        page = requests.get(dashboard + "/", timeout=10)
        page.raise_for_status()
        token_match = re.search(r'__HERMES_SESSION_TOKEN__\s*=\s*("[^"]*")', page.text)
        assert token_match is not None
        token = json.loads(token_match[1])
        response = requests.get(
            dashboard + "/api/sessions/" + reference.session_id + "/messages",
            headers={"Authorization": "Bearer " + token},
            timeout=10,
        )
        response.raise_for_status()
        assert nonce in json.dumps(response.json()["messages"])


@pytest.mark.parametrize("entrypoint", ["run", "start_runtime_in"])
async def test_sdk_lifecycle_methods_resume_a_direct_reference(
    resume_config, tmp_path, api_server, entrypoint
):
    fabric = Fabric()
    nonce = "sdk-entrypoint-" + uuid4().hex
    async with await fabric.start_runtime(resume_config, base_dir=tmp_path) as first:
        reference = first.session_ref
        assert (await first.invoke(input=nonce)).status == "succeeded"

    if entrypoint == "run":
        result = await fabric.run(
            resume_config,
            base_dir=tmp_path,
            session_ref=reference,
            input="Continue our earlier conversation.",
        )
    else:
        environment = await fabric.prepare_environment(resume_config, base_dir=tmp_path)
        try:
            async with await fabric.start_runtime_in(
                resume_config, environment, base_dir=tmp_path, session_ref=reference
            ) as resumed:
                assert resumed.runtime_id != first.runtime_id
                result = await resumed.invoke(
                    input="Continue our earlier conversation."
                )
        finally:
            await fabric.release_environment(environment)
    assert result.status == "succeeded"
    assert result.session_ref == reference
    captured = requests.get(api_server + "/_requests", timeout=5).json()
    assert nonce in json.dumps(captured[-1]["messages"])


@pytest.mark.skipif(os.name != "posix", reason="Hermes interfaces require a POSIX host")
async def test_native_only_retention_preserves_startup_without_sdk_session_selection(
    resume_config, tmp_path, api_server, unused_tcp_port
):
    mapping = resume_config.to_mapping()
    mapping["harness"]["settings"] = {
        "sdk": {"enabled": False},
        "interfaces": {"api": {"port": unused_tcp_port}},
    }
    config = FabricConfig.from_mapping(mapping)
    fabric = Fabric()
    assert fabric.plan(config, base_dir=tmp_path).capabilities.session_resume is True
    async with await fabric.start_runtime(resume_config, base_dir=tmp_path) as seeded:
        reference = seeded.session_ref
    runtime_ids = []
    for _ in range(2):
        async with await fabric.start_runtime(config, base_dir=tmp_path) as runtime:
            assert runtime.session_ref is None
            runtime_ids.append(runtime.runtime_id)
    assert runtime_ids[0] != runtime_ids[1]
    with pytest.raises(FabricRuntimeError) as caught:
        await fabric.start_runtime(config, base_dir=tmp_path, session_ref=reference)
    assert caught.value.code == "session_resume_unsupported"
    assert requests.get(api_server + "/_requests", timeout=5).json() == []
