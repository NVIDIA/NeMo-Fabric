#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run OO Agents ``BenchAgent`` through the NeMo Fabric harness contract."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentRunError
from nemo_fabric_adapter_contract.models import AgentRunRequest
from nemo_fabric_adapter_contract.models import AgentRunResult
from nemo_fabric_adapter_contract.models import AgentRunStatus
from nemo_fabric_adapter_contract.models import AgentUsage
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle
import nemo_fabric_adapters.common.utils as common_utils
from nemo_fabric_adapters.nooa.model_support import build_models
from nemo_fabric_adapters.nooa.model_support import close_models
from nemo_fabric_adapters.nooa.telemetry import RelayReport
from nemo_fabric_adapters.nooa.telemetry import RelayTelemetry

ADAPTER = "python"
HARNESS = "nooa-bench"
MODE = "bench_agent"
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""

    lifecycle.serve(BenchRuntime, config_loader=AgentConfig.from_mapping)


def _config_error(code: str, message: str, **metadata: Any) -> lifecycle.LifecycleError:
    return lifecycle.LifecycleError(code, message, metadata=metadata or None)


def _agent_config(payload: dict[str, Any]) -> AgentConfig:
    config = payload.get("config")
    if not isinstance(config, AgentConfig):
        raise _config_error(
            "nooa_bench_invalid_agent_config",
            "OO Agents BenchAgent requires a validated AgentConfig start payload",
        )
    return config


def _selected_model_role(config: AgentConfig) -> str:
    if "default" in config.models:
        if len(config.models) != 1:
            raise _config_error(
                "nooa_bench_invalid_models",
                "OO Agents BenchAgent accepts exactly one model role",
                field="models",
            )
        return "default"
    if len(config.models) == 1:
        return next(iter(config.models))
    raise _config_error(
        "nooa_bench_invalid_models",
        "OO Agents BenchAgent requires exactly one model",
        field="models",
    )


def _system_instruction(config: AgentConfig) -> str | None:
    if config.instructions is None or config.instructions.system is None:
        return None
    return config.instructions.system.content


def _workspace(payload: dict[str, Any]) -> Path:
    try:
        base_dir = Path(common_utils.base_dir(payload))
    except ValueError as error:
        raise _config_error(
            "nooa_bench_invalid_runtime_context",
            "OO Agents BenchAgent lifecycle payload is missing its base directory",
        ) from error
    value = common_utils.environment_payload(payload).get("workspace")
    if value is None:
        return base_dir
    if not isinstance(value, (str, Path)):
        raise _config_error(
            "nooa_bench_invalid_runtime_context",
            "OO Agents BenchAgent received an invalid workspace path",
        )
    workspace = Path(value)
    if not workspace.is_dir():
        raise _config_error(
            "nooa_bench_invalid_workspace",
            "OO Agents BenchAgent workspace must be an existing directory",
        )
    return workspace


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _close_shell(agent: Any) -> None:
    shell = getattr(agent, "shell", None)
    close = getattr(shell, "close", None)
    if callable(close):
        await _await_if_needed(close())


def _failure_output(code: str, message: str) -> AgentRunResult:
    return AgentRunResult(
        status=AgentRunStatus.FAILED,
        output={
            "harness": HARNESS,
            "adapter": ADAPTER,
            "mode": MODE,
            "response": None,
            "completed": False,
        },
        error=AgentRunError(code=code, message=message),
    )


def _usage(tokens: Any) -> AgentUsage | None:
    if not isinstance(tokens, dict):
        return None
    input_tokens = tokens.get("n_input_tokens")
    output_tokens = tokens.get("n_output_tokens")
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
    ):
        return None
    if input_tokens < 0 or output_tokens < 0:
        return None
    return AgentUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _success_output(native: dict[str, Any], usage: AgentUsage | None) -> AgentRunResult:
    response = native.get("response")
    structured = native.get("result")
    if not isinstance(response, str):
        raise ValueError("BenchAgent response must be a string")
    return AgentRunResult(
        status=AgentRunStatus.SUCCEEDED,
        output={
            "harness": HARNESS,
            "adapter": ADAPTER,
            "mode": MODE,
            "response": response,
            "completed": True,
            "result": structured,
        },
        usage=usage,
    )


def _with_telemetry(
    result: AgentRunResult,
    report: RelayReport | None,
) -> AgentRunResult:
    if report is None:
        return result
    output = dict(result.output)
    telemetry: dict[str, Any] = {
        "enabled": report.enabled,
        "provider": "relay",
        "emitter": "nooa.nemo_relay_middleware",
    }
    if report.error is not None:
        telemetry["degraded"] = True
        telemetry["error"] = report.error
    if report.quarantine_cause is not None:
        telemetry["quarantine_cause"] = report.quarantine_cause
    output["telemetry"] = telemetry
    output["relay_artifacts"] = list(report.artifacts)
    result.output = output
    return result


class BenchRuntime:
    """Runtime-owned model resources that create one BenchAgent per invocation."""

    def __init__(self) -> None:
        self._runtime_id: str | None = None
        self._bench_agent_factory: Callable[..., Any] | None = None
        self._model_role: str | None = None
        self._models: dict[str, Any] = {}
        self._workspace: Path | None = None
        self._instruction: str | None = None
        self._telemetry: RelayTelemetry | None = None

    async def start(self, payload: dict[str, Any]) -> None:
        if self._bench_agent_factory is not None:
            raise lifecycle.LifecycleError(
                "nooa_bench_runtime_already_started",
                "OO Agents BenchAgent runtime is already started",
            )

        config = _agent_config(payload)
        role = _selected_model_role(config)
        models: dict[str, Any] = {}
        bench_agent_factory: Callable[..., Any] | None = None
        telemetry: RelayTelemetry | None = None
        try:
            runtime_id = common_utils.runtime_id(payload)
            base_dir = Path(common_utils.base_dir(payload))
            workspace = _workspace(payload)
            models = await build_models(config)
            from nooa_bench.bench_agent import BenchAgent

            bench_agent_factory = BenchAgent
            telemetry = RelayTelemetry(
                agent_name=common_utils.agent_name(payload),
                base_dir=base_dir,
                config=config,
                scope_name="nooa-bench-agent-request",
            )
        except asyncio.CancelledError:
            await close_models(models)
            if telemetry is not None:
                await telemetry.close()
            raise
        except lifecycle.LifecycleError:
            await close_models(models)
            if telemetry is not None:
                await telemetry.close()
            raise
        except Exception as error:
            await close_models(models)
            if telemetry is not None:
                await telemetry.close()
            raise lifecycle.LifecycleError(
                "nooa_bench_start_failed",
                "OO Agents BenchAgent failed to start",
            ) from error

        assert bench_agent_factory is not None
        self._runtime_id = runtime_id
        self._bench_agent_factory = bench_agent_factory
        self._model_role = role
        self._models = models
        self._workspace = workspace
        self._instruction = _system_instruction(config)
        self._telemetry = telemetry

    async def invoke(
        self,
        request: AgentRunRequest,
        runtime_context: RuntimeContext,
    ) -> AgentRunResult:
        if (
            self._bench_agent_factory is None
            or self._model_role is None
            or self._runtime_id is None
            or self._workspace is None
            or self._telemetry is None
        ):
            raise lifecycle.LifecycleError(
                "nooa_bench_runtime_not_started",
                "OO Agents BenchAgent runtime is not started",
            )
        if runtime_context.runtime_id != self._runtime_id:
            raise lifecycle.LifecycleError(
                "nooa_bench_runtime_mismatch",
                "OO Agents BenchAgent invocation does not match the active runtime",
            )
        if not isinstance(request.input, str):
            return _failure_output(
                "nooa_bench_invalid_request",
                "OO Agents BenchAgent input must be a string",
            )

        try:
            agent = self._bench_agent_factory(llm=self._models[self._model_role])
        except asyncio.CancelledError:
            raise
        except Exception as error:
            LOGGER.error(
                "OO Agents BenchAgent construction failed (error_type=%s)",
                type(error).__name__,
            )
            return _failure_output(
                "nooa_bench_invoke_failed",
                "OO Agents BenchAgent invocation failed; inspect adapter stderr for details",
            )

        async def invoke_agent() -> AgentRunResult:
            task_input: dict[str, Any] = {
                "user_message": request.input,
                "working_dir": str(self._workspace),
            }
            if self._instruction is not None:
                task_input["instructions"] = self._instruction
            try:
                from nooa.runtime.token_usage import get_task_tokens
                from nooa.runtime.token_usage import start_task_tokens

                # BenchAgent creates a replacement shell for each task. Close the
                # previous one first so persistent Fabric runtimes do not leak it.
                await _close_shell(agent)
                start_task_tokens()
                native = await agent._run_evaluation(task_input)
                usage = _usage(get_task_tokens())
                if not isinstance(native, dict):
                    raise ValueError("BenchAgent result must be an object")
                if native.get("success") is not True:
                    return _failure_output(
                        "nooa_bench_task_failed",
                        "OO Agents BenchAgent reported task failure; inspect adapter stderr for details",
                    )
                return _success_output(native, usage)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                LOGGER.error(
                    "OO Agents BenchAgent invocation failed (error_type=%s)",
                    type(error).__name__,
                )
                return _failure_output(
                    "nooa_bench_invoke_failed",
                    "OO Agents BenchAgent invocation failed; inspect adapter stderr for details",
                )

        try:
            relay_invocation = await self._telemetry.invoke(
                agent=agent,
                runtime_context=runtime_context,
                call=invoke_agent,
            )
        except BaseException:
            try:
                await _close_shell(agent)
            except BaseException as cleanup_error:
                LOGGER.error(
                    "OO Agents BenchAgent cleanup failed while preserving invocation failure "
                    "(error_type=%s)",
                    type(cleanup_error).__name__,
                )
            raise
        try:
            await _close_shell(agent)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise lifecycle.LifecycleError(
                "nooa_bench_invoke_cleanup_failed",
                "OO Agents BenchAgent invocation failed to clean up",
            ) from error
        if not relay_invocation.called:
            result = _failure_output(
                "nooa_bench_telemetry_setup_failed",
                "OO Agents BenchAgent Relay telemetry setup failed before task execution",
            )
        else:
            assert isinstance(relay_invocation.result, AgentRunResult)
            result = relay_invocation.result
        return _with_telemetry(result, relay_invocation.report)

    async def stop(self) -> None:
        models = self._models
        telemetry = self._telemetry
        self._runtime_id = None
        self._bench_agent_factory = None
        self._model_role = None
        self._models = {}
        self._workspace = None
        self._instruction = None
        self._telemetry = None
        primary: BaseException | None = None
        if telemetry is not None:
            try:
                await telemetry.close()
            except BaseException as error:
                primary = error
        try:
            await close_models(models)
        except BaseException as error:
            if primary is None:
                primary = error
        if primary is not None:
            if isinstance(primary, asyncio.CancelledError):
                raise primary
            raise lifecycle.LifecycleError(
                "nooa_bench_runtime_stop_failed",
                "OO Agents BenchAgent runtime failed to stop cleanly",
            ) from primary


if __name__ == "__main__":
    main()
