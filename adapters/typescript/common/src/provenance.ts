// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Selected package lookup adapted from NVIDIA/NeMo-Fabric#458,
// 2bfd53321fa8d541d42abc195be5e9dd9db4350f (Maryam). Preserve full values,
// original UTC capture times and explicit gaps; never probe a harness.
import { open, stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExecutionEvidence, SoftwareEntry, VersionObservation } from "nemo-fabric-adapter-contract";

export interface SoftwareSelection {
  packageJson: URL;
  packages?: readonly { role: string; name: string }[];
  harness: string;
}
const MAX_BYTES = 64 * 1024;
const MAX_TARGETS = 32;
const COLLECTION_MS = 250;

function wellFormed(value: string): boolean {
  for (const character of value) {
    const code = character.codePointAt(0)!;
    if (code >= 0xd800 && code <= 0xdfff) return false;
  }
  return true;
}

function version(value: unknown, source: string, observedAt: string, reason: string): VersionObservation {
  if (typeof value === "string" && value.trim().length > 0 && wellFormed(value)) {
    return { status: "available", value, source, origin: "observed", observed_at: observedAt };
  }
  return { status: "unavailable", reason: typeof value === "string" && !wellFormed(value) ? "invalid_version_encoding" : reason,
    source, origin: "observed", observed_at: observedAt };
}

async function manifest(path: string): Promise<{ name?: unknown; version?: unknown }> {
  const file = await open(path, "r");
  try {
    const buffer = Buffer.alloc(MAX_BYTES + 1);
    const { bytesRead } = await file.read(buffer, 0, buffer.length, 0);
    if (bytesRead > MAX_BYTES) throw new Error("metadata_over_budget");
    return JSON.parse(buffer.toString("utf8", 0, bytesRead)) as { name?: unknown; version?: unknown };
  } finally {
    await file.close();
  }
}

/** Read one selected package from the installation that resolves the adapter. */
export async function packageVersion(path: string, captured: string, expectedName?: string): Promise<VersionObservation> {
  try {
    const value = await manifest(path);
    if (expectedName !== undefined && value.name !== expectedName) {
      return version(undefined, "adapter:package.json", captured, "metadata_name_mismatch");
    }
    return version(value.version, "adapter:package.json", captured, "empty_version");
  }
  catch (error) {
    const reason = error instanceof Error && error.message === "metadata_over_budget" ? "metadata_byte_limit"
      : error instanceof SyntaxError ? "metadata_malformed"
      : error instanceof Error && "code" in error && error.code === "ENOENT" ? "metadata_not_found" : "metadata_read_failed";
    return version(undefined, "adapter:package.json", captured, reason);
  }
}

async function selectedVersion(name: string, packageJson: URL, captured: string, active: () => boolean): Promise<VersionObservation> {
  if (!/^(?:@[a-z0-9._-]+\/)?[a-z0-9._-]+$/.test(name) || [".", ".."].includes(name.split("/").at(-1)!)) {
    return version(undefined, "adapter:package.json", captured, "selected_package_name_invalid");
  }
  // Async, bounded node_modules lookup also supports import-only exports.
  // require.resolve reads package metadata synchronously before our byte/time
  // limits can apply, so it cannot be used inside this collector.
  let directory = dirname(fileURLToPath(packageJson));
  for (let depth = 0; depth < 8 && active(); depth++) {
    const packageDirectory = join(directory, "node_modules", name);
    const observation = await packageVersion(join(packageDirectory, "package.json"), captured, name);
    if (observation.reason !== "metadata_not_found") return observation;
    try {
      await stat(packageDirectory);
      // An existing source tree without metadata can shadow an outer install.
      return observation;
    } catch (error) {
      if (!(error instanceof Error && "code" in error && ["ENOENT", "ENOTDIR"].includes(String(error.code)))) {
        return version(undefined, "adapter:package.json", captured, "metadata_read_failed");
      }
    }
    const parent = dirname(directory);
    if (parent === directory) break;
    directory = parent;
  }
  return version(undefined, "adapter:package.json", captured, "selected_package_not_resolved");
}

export async function collectSoftware(selection?: SoftwareSelection): Promise<ExecutionEvidence> {
  const captured = new Date().toISOString();
  const source = "adapter:process.versions";
  const interpreter = process.versions.bun === undefined ? "node" : "bun";
  const entries: SoftwareEntry[] = [{ role: "interpreter", kind: "executable", name: interpreter,
    version: version(process.versions[interpreter], source, captured, "runtime_version_missing") }];
  let complete = true;
  let active = true;
  const collect = async (): Promise<ExecutionEvidence> => {
    if (selection !== undefined) {
      const own = await manifest(fileURLToPath(selection.packageJson));
      if (typeof own.name !== "string" || !own.name.trim() || !wellFormed(own.name)) return gap("adapter_package_name_invalid");
      entries.push({ role: "adapter", kind: "npm_package", name: typeof own.name === "string" ? own.name : "unknown-adapter",
        version: version(own.version, "adapter:package.json", captured, "empty_version") });
      const commonPackage = new URL("../package.json", import.meta.url);
      const targets = [{ role: "contract", name: "nemo-fabric-adapter-contract" },
        { role: "shared_library", name: "nemo-fabric-adapters-common" }, ...(selection.packages ?? [])];
      if (targets.length > MAX_TARGETS) complete = false;
      for (const target of targets.slice(0, MAX_TARGETS)) {
        if (!active) break;
        if (!wellFormed(target.name) || !wellFormed(target.role)) { complete = false; continue; }
        const observation = target.name === "nemo-fabric-adapters-common"
          ? await packageVersion(fileURLToPath(commonPackage), captured, target.name)
          : await selectedVersion(target.name, target.name === "nemo-fabric-adapter-contract" ? commonPackage : selection.packageJson,
            captured, () => active);
        if (observation.reason === "metadata_byte_limit") complete = false;
        entries.push({ ...target, kind: "npm_package", version: observation });
      }
      if (wellFormed(selection.harness)) entries.push({ role: "harness", kind: "executable", name: selection.harness,
        version: version(undefined, "adapter:startup", captured, "no_existing_version_report") });
      else complete = false;
    } else {
      complete = false;
    }
    const participants: [string] = ["adapter"];
    const inventory = { context_ref: "env-0", participants, status: "available" as const, entries };
    const result: ExecutionEvidence = { software_environments: [complete
      ? { ...inventory, complete: true }
      : { ...inventory, complete: false, reason: "selected_package_report_missing_or_truncated" }] };
    if (Buffer.byteLength(JSON.stringify(result), "utf8") > MAX_BYTES) return gap("report_over_budget");
    return result;
  };
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([collect().catch(() => gap("collection_failed")), new Promise<ExecutionEvidence>((resolve) => {
      timer = setTimeout(() => { active = false; resolve(gap("collection_time_budget")); }, COLLECTION_MS);
    })]);
  } finally { if (timer !== undefined) clearTimeout(timer); active = false; }
}

function gap(reason: string): ExecutionEvidence {
  return { software_environments: [{ context_ref: "env-0", participants: ["adapter"],
    status: "unavailable", reason, complete: false, entries: [] }] };
}

// Existing startup checks can supply observations without any new probe. The
// async context belongs to one lifecycle host, including concurrently tested hosts.
import { AsyncLocalStorage } from "node:async_hooks";
const passive = new AsyncLocalStorage<Map<string, SoftwareEntry> | undefined>();

export function withSoftwareObservations<T>(observations: Map<string, SoftwareEntry> | undefined, action: () => Promise<T>): Promise<T> {
  return passive.run(observations, action);
}

/** Forward a value from an existing startup exchange; never initiate collection. */
export function recordSoftwareVersion(name: string, value: string, source: string, role = "harness", origin: "observed" | "reported" = "reported"): void {
  const observations = passive.getStore();
  if (observations === undefined || observations.size >= 8 || typeof value !== "string" || !value.trim()
      || !wellFormed(name) || !wellFormed(source) || !wellFormed(role)) return;
  const key = `${role}:${name}`;
  if (!observations.has(key)) observations.set(key, { role, name, kind: "executable",
    version: { ...version(value, source, new Date().toISOString(), "invalid_version"), origin } });
}

export function mergeSoftwareObservations(report: ExecutionEvidence, observations?: Map<string, SoftwareEntry>): ExecutionEvidence {
  const result = structuredClone(report);
  const env = result.software_environments[0];
  if (env?.status !== "available") return result;
  for (const observation of observations?.values() ?? []) {
    const index = env.entries.findIndex(entry => entry.role === observation.role && entry.kind === observation.kind && entry.name === observation.name);
    const previous = env.entries[index];
    if (index < 0) env.entries.push(observation); else env.entries[index] = observation;
    if (Buffer.byteLength(JSON.stringify(result), "utf8") > MAX_BYTES) {
      if (index < 0) env.entries.pop(); else env.entries[index] = previous!;
      env.complete = false;
      env.reason = "passive_report_over_budget";
    }
  }
  return result;
}
