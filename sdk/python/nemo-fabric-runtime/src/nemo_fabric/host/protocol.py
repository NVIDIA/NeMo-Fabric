# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned local requests and safe command responses."""

import json
from typing import Any

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
    "stop_failed": "Fabric could not complete the stop operation.",
    "invoke_failed": "Fabric did not report a successful invocation.",
    "host_stopping": "The Fabric host is shutting down.",
    "runtime_unavailable": "No active Fabric runtime is available.",
    "runtime_exists": "This sandbox has used its runtime lifecycle; replace the sandbox before starting another.",
    "streaming_unsupported": "Streaming invocation is unsupported by this command interface.",
    "host_unavailable": "The Fabric host is unavailable.",
    "invalid_response": "The host response is missing, invalid, or truncated.",
    "request_too_large": "The request exceeds the transport limit.",
    "response_too_large": "The response exceeds the transport limit; the outcome is unknown.",
    "invalid_input": "Input must contain one bounded UTF-8 JSON object.",
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
        required.add("input")
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
    if operation == "invoke":
        if not isinstance(request["input"], dict):
            raise ProtocolError()
        if request["input"].get("stream") or request["input"].get("streaming"):
            raise ProtocolError("streaming_unsupported", "invoke", unsupported=True)
    return operation


def decode_object(encoded: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError("non-finite JSON number")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    value = json.loads(
        encoded.decode("utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
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
    if not isinstance(result, dict):
        raise invalid
    if operation == "invoke":
        native = result.get("fabric_result")
        if (
            not isinstance(result.get("runtime_id"), str)
            or not result["runtime_id"]
            or not isinstance(native, dict)
            or native.get("status") != "succeeded"
            or response["changed"] is not None
        ):
            raise invalid
        return
    if operation not in ("start", "stop") or set(result) != {"runtime_id"}:
        raise invalid
    if not isinstance(result["runtime_id"], str) or not result["runtime_id"]:
        raise invalid
    if type(response["changed"]) is not bool:
        raise invalid
