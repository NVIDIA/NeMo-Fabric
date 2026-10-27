// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reports an adapter's own package version in the `nemo_fabric.versions.v1`
// result extension. NeMo Fabric core reads the key when the adapter descriptor
// names it under `extension_schemas.run_result`, and moves it into
// `RunResult.what_ran`.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import type { AgentRunRequest, AgentRunResult, JsonObject, RuntimeContext } from "nemo-fabric-adapter-contract";

import type { AdapterRuntime, AdapterStartInput } from "./lifecycle.js";

export const VERSIONS_REPORT_KEY = "nemo_fabric.versions.v1";

const MAX_VALUE_CHARS = 256;
const TRUNCATED_PREFIX_CHARS = 240;
const TRUNCATED_DIGEST_HEX_DIGITS = 15;
const MAX_MODEL_ROLES = 16;
const MAX_KEY_CHARS = 128;

/**
 * Apply NeMo Fabric core's string length rule, counting Unicode scalar values.
 * A lone surrogate becomes `?` first, because core rejects a result that holds
 * one.
 */
export function boundedText(text: string): string {
  const chars = Array.from(text, (char) => {
    const code = char.codePointAt(0) ?? 0;
    return code >= 0xd800 && code <= 0xdfff ? "?" : char;
  });
  if (chars.length <= MAX_VALUE_CHARS) return chars.join("");
  const digest = createHash("sha256").update(chars.join(""), "utf8").digest("hex");
  return `${chars.slice(0, TRUNCATED_PREFIX_CHARS).join("")}~${digest.slice(0, TRUNCATED_DIGEST_HEX_DIGITS)}`;
}

/** A reported fact without a value. */
export function missingFact(state: string, reason: string, method = "adapter_report"): JsonObject {
  return {state, reason, source: "reported", collected_by: "adapter", method};
}

/** The served-model fact an adapter reports for one model role. */
export interface ServedModelFact {
  role: string;
  fact: JsonObject;
}

/**
 * The `version` field of the adapter's own `package.json`, read synchronously.
 * Missing metadata reads `unavailable/not_collected`; unreadable or invalid
 * metadata reads `collection_failed/no_result`. An empty version reads
 * `unavailable/empty_response`.
 */
export function packageVersionFact(packageJson: URL): JsonObject {
  let version: unknown;
  try {
    version = (JSON.parse(readFileSync(packageJson, "utf8")) as {version?: unknown}).version;
  } catch (error) {
    const absent = error instanceof Error && "code" in error && error.code === "ENOENT";
    return missingFact(absent ? "unavailable" : "collection_failed", absent ? "not_collected" : "no_result", "package_metadata");
  }
  if (typeof version !== "string" || version.length === 0) {
    return missingFact("unavailable", "empty_response", "package_metadata");
  }
  return {value: boundedText(version), source: "reported", collected_by: "adapter", method: "package_metadata"};
}

/**
 * Order two strings by Unicode code point, which is the UTF-8 byte order that
 * NeMo Fabric core and the Python helper use. `Array.prototype.sort` compares
 * UTF-16 code units, which puts a character above U+FFFF before one in
 * U+E000 to U+FFFF.
 */
function compareCodePoints(left: string, right: string): number {
  const leftPoints = Array.from(left, (char) => char.codePointAt(0) ?? 0);
  const rightPoints = Array.from(right, (char) => char.codePointAt(0) ?? 0);
  for (const [index, point] of leftPoints.entries()) {
    const other = rightPoints[index];
    if (other === undefined) return 1;
    if (point !== other) return point - other;
  }
  return leftPoints.length - rightPoints.length;
}

/**
 * The report of an adapter that has a source only for its package version.
 * The harness name and version read `unsupported` with reason `no_source`, and
 * so does each role's served model, except the role that `servedModel` names,
 * which gets its fact. Like core, it keeps at most 16 roles in sorted order
 * and skips a role longer than 128 characters.
 */
export function packageVersionReport(
  packageVersion: JsonObject,
  modelRoles: readonly string[],
  servedModel?: ServedModelFact,
): JsonObject {
  const noSource = () => missingFact("unsupported", "no_source");
  const roles = modelRoles
    .filter((role) => Array.from(role).length <= MAX_KEY_CHARS)
    .sort(compareCodePoints)
    .slice(0, MAX_MODEL_ROLES);
  return {
    adapter: {package_version: packageVersion},
    harness: {name: noSource(), version: noSource()},
    models: Object.fromEntries(
      roles.map((role) => [role, {served: role === servedModel?.role ? {...servedModel.fact} : noSource()}]),
    ),
  };
}

/**
 * Wrap `runtime` so each result reports the adapter package version read from
 * `packageJson` when the runtime starts. `servedModel`, when given, picks the
 * served-model fact of one role from the start input. Results and errors are
 * otherwise unchanged.
 */
export function withPackageVersion(
  runtime: AdapterRuntime,
  packageJson: URL,
  servedModel?: (input: AdapterStartInput) => ServedModelFact | undefined,
): AdapterRuntime {
  let report: JsonObject | undefined;
  const wrapped: AdapterRuntime = {
    async start(input: AdapterStartInput): Promise<void> {
      report = packageVersionReport(
        packageVersionFact(packageJson),
        Object.keys(input.config.models ?? {}),
        servedModel?.(input),
      );
      await runtime.start(input);
    },
    async invoke(request: AgentRunRequest, context: RuntimeContext): Promise<AgentRunResult> {
      const result = await runtime.invoke(request, context);
      if (report === undefined) return result;
      return {...result, extensions: {...(result.extensions ?? {}), [VERSIONS_REPORT_KEY]: structuredClone(report)}};
    },
    stop(): Promise<void> {
      return runtime.stop();
    },
  };
  if (runtime.check !== undefined) {
    wrapped.check = runtime.check.bind(runtime);
  }
  return wrapped;
}
