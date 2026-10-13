# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned local requests and safe command responses."""

import json
from typing import Any

from nemo_fabric import FabricConfigError, RunResult, RuntimeHandle

INTERFACE_VERSION = 2
REQUEST_LIMIT = 512 * 1024
RESPONSE_LIMIT = 4 * 1024 * 1024
CALL_SECONDS = 320.0
READ_SECONDS = 10.0
SHUTDOWN_SECONDS = 4.0
SOCKET_PATH = "/sandbox/fabric.sock"
OPERATIONS = ("start", "invoke", "stop", "serve")
CONFIG_OPERATIONS = ("start",)
LIFECYCLE_OPERATIONS = ("start", "stop")

MESSAGES = {
    "platform_unsupported": "The command bridge requires a Unix platform.",
    "cleanup_unconfirmed": "Previous runtime cleanup is unconfirmed; replace the sandbox before restarting the server.",
    "invalid_request": "The command request is invalid.",
    "interface_unsupported": "The requested local interface version is unsupported.",
    "wrong_agent": "The request does not identify this agent.",
    "validation_failed": "Fabric rejected the configuration.",
    "validation_unavailable": "Fabric could not establish configuration compatibility.",
    "start_failed": "Fabric could not establish an active runtime.",
    "start_uncertain": "A previous runtime start has an uncertain outcome; replace the sandbox before starting another.",
    "runtime_id_conflict": "Fabric returned a runtime ID that is already in use.",
    "stop_failed": "Fabric could not complete the stop operation.",
    "invoke_failed": "Fabric did not report a successful invocation.",
    "host_stopping": "The Fabric host is shutting down.",
    "runtime_unavailable": "No active Fabric runtime is available.",
    # Keep the v2 response code recognizable for callers connected to an older
    # runtime-server image that still enforces one start per sandbox.
    "runtime_exists": "This sandbox has used its runtime lifecycle; replace the sandbox before starting another.",
    "streaming_unsupported": "Streaming invocation is unsupported by this command interface.",
    "host_unavailable": "The Fabric host is unavailable.",
    "invalid_response": "The host response is missing, invalid, or truncated.",
    "request_too_large": "The request exceeds the transport limit.",
    "response_too_large": "The response exceeds the transport limit; the outcome is unknown.",
    "invalid_input": "Input must contain one bounded UTF-8 JSON value.",
}


class ProtocolError(ValueError):
    """A fixed diagnostic that never includes request values or raw exceptions."""

    def __init__(
        self,
        code: str = "invalid_request",
        stage: str = "request",
        *,
        unsupported: bool = False,
    ):
        self.code = code
        self.stage = stage
        self.unsupported = unsupported
        super().__init__(MESSAGES[code])


def envelope(
    operation: str,
    result: dict[str, Any] | None = None,
    *,
    changed: bool | None = False,
    error: ProtocolError | None = None,
    effects: str = "none",
) -> dict[str, Any]:
    return {
        "operation": operation,
        "status": "unsupported"
        if error and error.unsupported
        else "failed"
        if error
        else "succeeded",
        "changed": changed,
        "result": result,
        "error": {
            "code": error.code,
            "stage": error.stage,
            "message": str(error),
            "effects": effects,
        }
        if error
        else None,
    }


def validate_request(request: Any) -> str:
    if not isinstance(request, dict) or request.get("operation") not in OPERATIONS:
        raise ProtocolError()
    operation = request["operation"]
    if (
        type(request.get("interface_version")) is not int
        or request["interface_version"] != INTERFACE_VERSION
    ):
        raise ProtocolError("interface_unsupported", unsupported=True)
    required = {"interface_version", "operation"}
    if operation in ("start", "invoke", "stop"):
        required.add("agent")
    if operation in ("invoke", "stop"):
        required.add("runtime_id")
    if operation == "start":
        required.add("config")
    if operation == "invoke":
        payload_fields = request.keys() & {"input", "request"}
        if len(payload_fields) != 1:
            raise ProtocolError()
        required.update(payload_fields)
    if not required <= request.keys() or not request.keys() <= required:
        raise ProtocolError()
    for field in ("agent", "runtime_id"):
        if field in request and (
            not isinstance(request[field], str)
            or not request[field].strip()
            or "\x00" in request[field]
        ):
            raise ProtocolError()
    if operation == "start":
        config = request["config"]
        if not isinstance(config, dict):
            raise ProtocolError()
        metadata = config.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("name") != request["agent"]:
            raise ProtocolError("wrong_agent")
    if (
        operation == "invoke"
        and "request" in request
        and not isinstance(request["request"], dict)
    ):
        raise ProtocolError()
    return operation


def decode_json(encoded: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError("non-finite JSON number")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    return json.loads(
        encoded.decode("utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def decode_object(encoded: bytes) -> dict[str, Any]:
    value = decode_json(encoded)
    if not isinstance(value, dict):
        raise ValueError("expected one JSON object")
    return value


def _encode(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )


def encode_request(request: dict[str, Any]) -> bytes:
    validate_request(request)
    try:
        encoded = _encode(request)
    except (ValueError, TypeError, RecursionError) as error:
        raise ProtocolError() from error
    if len(encoded) > REQUEST_LIMIT:
        raise ProtocolError("request_too_large")
    return encoded


def encode_response(response: dict[str, Any]) -> bytes:
    try:
        encoded = _encode(response)
        if len(encoded) <= RESPONSE_LIMIT:
            return encoded
    except (ValueError, TypeError, RecursionError):
        pass
    return _encode(
        envelope(
            response["operation"],
            changed=None,
            error=ProtocolError("response_too_large", "transport"),
            effects="unknown",
        )
    )


def validate_response(response: dict[str, Any], operation: str) -> None:
    invalid = ProtocolError("invalid_response", "transport")
    if (
        operation not in OPERATIONS
        or set(response) != {"operation", "status", "changed", "result", "error"}
        or response["operation"] != operation
        or response["status"] not in ("succeeded", "failed", "unsupported")
        or not (response["changed"] is None or type(response["changed"]) is bool)
        or not (response["result"] is None or isinstance(response["result"], dict))
        or (response["status"] == "succeeded") != (response["error"] is None)
    ):
        raise invalid
    error = response["error"]
    if error is not None:
        if (
            not isinstance(error, dict)
            or set(error) != {"code", "stage", "message", "effects"}
            or not isinstance(error["code"], str)
            or error["code"] not in MESSAGES
            or error["stage"]
            not in (
                "request",
                "validation",
                "start",
                "stop",
                "invoke",
                "transport",
            )
            or error["effects"] not in ("none", "applied", "unknown")
        ):
            raise invalid
        error["message"] = MESSAGES[error["code"]]
        return
    result = response["result"]
    try:
        if operation == "invoke":
            RunResult.from_mapping(result)
            if response["changed"] is not None:
                raise invalid
        elif operation == "start":
            RuntimeHandle.from_mapping(result)
            if type(response["changed"]) is not bool:
                raise invalid
        elif operation == "stop":
            if result is not None or type(response["changed"]) is not bool:
                raise invalid
        else:
            raise invalid
    except (FabricConfigError, TypeError, ValueError, KeyError) as error:
        raise invalid from error
