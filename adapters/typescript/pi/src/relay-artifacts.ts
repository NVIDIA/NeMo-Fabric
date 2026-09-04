// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { watch, type FSWatcher } from "node:fs";
import { realpath, stat } from "node:fs/promises";
import { resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";

import {
  collectAtifArtifacts,
  matchesRelayAtifPath,
  prepareRelayAtifMatchers,
  type RelayAtifMatcher,
  type RelayPluginConfig,
} from "./relay-config.js";

export const ATIF_FINALIZATION_TIMEOUT_MS = 5_000;
export const ATIF_FALLBACK_POLL_INTERVAL_MS = 50;

export type AtifSnapshot = Map<string, string>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function pluginComponents(pluginConfig: RelayPluginConfig): unknown[] {
  return Array.isArray(pluginConfig.components) ? pluginConfig.components : [];
}

export function expectsLocalAtif(pluginConfig: RelayPluginConfig, matchers?: RelayAtifMatcher[]): boolean {
  if (matchers !== undefined) {
    return matchers.some((matcher) => matcher.local);
  }
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
  } catch (error) {
    if (isRecord(error) && error.code === "ENOENT") {
      return undefined;
    }
    throw error;
  }
}

export async function snapshotAtifFiles(
  pluginConfig: RelayPluginConfig,
  matchers?: RelayAtifMatcher[],
): Promise<AtifSnapshot> {
  const snapshot: AtifSnapshot = new Map();
  const prepared = matchers ?? (await prepareRelayAtifMatchers(pluginConfig));
  for (const artifact of await collectAtifArtifacts(prepared, { localOnly: true, strict: true })) {
    const value = await fingerprint(artifact.path);
    if (value !== undefined) {
      snapshot.set(artifact.path, value);
    }
  }
  return snapshot;
}

async function finalizedAtifPath(
  pluginConfig: RelayPluginConfig,
  before: AtifSnapshot,
  matchers?: RelayAtifMatcher[],
): Promise<string | undefined> {
  const current = await snapshotAtifFiles(pluginConfig, matchers);
  for (const path of [...current.keys()].sort()) {
    if (before.get(path) === current.get(path)) {
      continue;
    }
    return path;
  }
  return undefined;
}

async function changedAtifPath(
  path: string,
  matchers: RelayAtifMatcher[],
  before: AtifSnapshot,
): Promise<string | undefined> {
  let candidate: string;
  try {
    candidate = await realpath(path);
  } catch (error) {
    if (isRecord(error) && error.code === "ENOENT") {
      return undefined;
    }
    throw error;
  }
  if (!matchers.some((matcher) => matcher.local && matchesRelayAtifPath(matcher, candidate))) {
    return undefined;
  }
  const current = await fingerprint(candidate);
  return current !== undefined && before.get(candidate) !== current ? candidate : undefined;
}

class AtifWatchUnavailableError extends Error {
  constructor(cause: unknown) {
    super("NeMo Relay ATIF artifact watching is unavailable", { cause });
  }
}

async function pollForFinalizedAtif(
  pluginConfig: RelayPluginConfig,
  before: AtifSnapshot,
  matchers: RelayAtifMatcher[],
  timeoutMs: number,
): Promise<string | undefined> {
  const deadline = performance.now() + timeoutMs;
  while (true) {
    const path = await finalizedAtifPath(pluginConfig, before, matchers);
    if (path !== undefined) {
      return path;
    }
    const remaining = deadline - performance.now();
    if (remaining <= 0) {
      return undefined;
    }
    await delay(Math.min(ATIF_FALLBACK_POLL_INTERVAL_MS, remaining));
  }
}

export async function waitForFinalizedAtif(
  pluginConfig: RelayPluginConfig,
  before: AtifSnapshot,
  options: { matchers?: RelayAtifMatcher[]; timeoutMs?: number } = {},
): Promise<string | undefined> {
  const timeoutMs = options.timeoutMs ?? ATIF_FINALIZATION_TIMEOUT_MS;
  const startedAt = performance.now();
  const matchers = options.matchers ?? (await prepareRelayAtifMatchers(pluginConfig));
  const localMatchers = matchers.filter((matcher) => matcher.local);
  const directories = new Map<string, boolean>();
  for (const matcher of localMatchers) {
    directories.set(matcher.directory, (directories.get(matcher.directory) ?? false) || matcher.recursive);
  }
  const watchers: FSWatcher[] = [];
  try {
    return await new Promise((resolvePromise, reject) => {
      let complete = false;
      const finish = (error?: unknown, path?: string): void => {
        if (complete) {
          return;
        }
        complete = true;
        clearTimeout(timer);
        for (const watcher of watchers) {
          watcher.close();
        }
        if (error !== undefined) {
          reject(error);
        } else {
          resolvePromise(path);
        }
      };
      const check = (path: string): void => {
        void changedAtifPath(path, localMatchers, before).then(
          (changed) => {
            if (changed !== undefined) {
              finish(undefined, changed);
            }
          },
          (error: unknown) => finish(error),
        );
      };
      const timer = setTimeout(() => finish(), timeoutMs);
      try {
        for (const [directory, recursive] of directories) {
          const watcher = watch(directory, { persistent: false, recursive }, (_event, filename) => {
            if (filename === null) {
              void finalizedAtifPath(pluginConfig, before, matchers).then(
                (changed) => {
                  if (changed !== undefined) {
                    finish(undefined, changed);
                  }
                },
                (error: unknown) => finish(error),
              );
            } else {
              check(resolve(directory, filename));
            }
          });
          watcher.once("error", (error) => finish(new AtifWatchUnavailableError(error)));
          watchers.push(watcher);
        }
      } catch (error) {
        finish(new AtifWatchUnavailableError(error));
        return;
      }
      void finalizedAtifPath(pluginConfig, before, matchers).then(
        (changed) => {
          if (changed !== undefined) {
            finish(undefined, changed);
          }
        },
        (error: unknown) => finish(error),
      );
    });
  } catch (error) {
    if (!(error instanceof AtifWatchUnavailableError)) {
      throw error;
    }
    return pollForFinalizedAtif(
      pluginConfig,
      before,
      matchers,
      Math.max(0, timeoutMs - (performance.now() - startedAt)),
    );
  }
}
