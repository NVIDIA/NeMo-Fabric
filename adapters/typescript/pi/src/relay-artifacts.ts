// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFile, stat } from "node:fs/promises";
import { setTimeout as delay } from "node:timers/promises";

import { collectRelayArtifacts, type RelayPluginConfig } from "./relay-config.js";

export const ATIF_FINALIZATION_TIMEOUT_MS = 5_000;
export const ATIF_POLL_INTERVAL_MS = 50;

export type AtifSnapshot = Map<string, string>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function pluginComponents(pluginConfig: RelayPluginConfig): unknown[] {
  return Array.isArray(pluginConfig.components) ? pluginConfig.components : [];
}

export function expectsLocalAtif(pluginConfig: RelayPluginConfig): boolean {
  for (const component of pluginComponents(pluginConfig)) {
    if (
      !isRecord(component) ||
      component.kind !== "observability" ||
      component.enabled === false ||
      !isRecord(component.config)
    ) {
      continue;
    }
    const atif = component.config.atif;
    if (isRecord(atif) && atif.enabled === true && (!Array.isArray(atif.storage) || atif.storage.length === 0)) {
      return true;
    }
  }
  return false;
}

async function fingerprint(path: string): Promise<string | undefined> {
  try {
    const value = await stat(path, { bigint: true });
    return `${value.dev}:${value.ino}:${value.size}:${value.mtimeNs}`;
  } catch {
    return undefined;
  }
}

export async function snapshotAtifFiles(pluginConfig: RelayPluginConfig): Promise<AtifSnapshot> {
  const snapshot: AtifSnapshot = new Map();
  for (const artifact of await collectRelayArtifacts(pluginConfig)) {
    if (artifact.kind !== "atif") {
      continue;
    }
    const value = await fingerprint(artifact.path);
    if (value !== undefined) {
      snapshot.set(artifact.path, value);
    }
  }
  return snapshot;
}

async function finalizedAtifPath(pluginConfig: RelayPluginConfig, before: AtifSnapshot): Promise<string | undefined> {
  const current = await snapshotAtifFiles(pluginConfig);
  for (const path of [...current.keys()].sort()) {
    if (before.get(path) === current.get(path)) {
      continue;
    }
    try {
      const document: unknown = JSON.parse(await readFile(path, "utf8"));
      if (isRecord(document)) {
        return path;
      }
    } catch {
      // Relay writes directly to the final path, so retry partial JSON.
    }
  }
  return undefined;
}

export async function waitForFinalizedAtif(
  pluginConfig: RelayPluginConfig,
  before: AtifSnapshot,
  options: { timeoutMs?: number; pollIntervalMs?: number } = {},
): Promise<string | undefined> {
  const timeoutMs = options.timeoutMs ?? ATIF_FINALIZATION_TIMEOUT_MS;
  const pollIntervalMs = options.pollIntervalMs ?? ATIF_POLL_INTERVAL_MS;
  const deadline = performance.now() + timeoutMs;
  while (true) {
    const path = await finalizedAtifPath(pluginConfig, before);
    if (path !== undefined) {
      return path;
    }
    const remaining = deadline - performance.now();
    if (remaining <= 0) {
      return undefined;
    }
    await delay(Math.min(pollIntervalMs, remaining));
  }
}
