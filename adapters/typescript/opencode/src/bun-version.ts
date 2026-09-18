// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LifecycleError } from "nemo-fabric-adapters-common";

type Version = Readonly<{
  major: number;
  minor: number;
  patch: number;
  prerelease: boolean;
}>;

function parseVersion(value: string): Version | undefined {
  const match = /^(\d+)\.(\d+)\.(\d+)(-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/.exec(value);
  if (match === null) {
    return undefined;
  }
  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: Number(match[3]),
    prerelease: match[4] !== undefined,
  };
}

function isOlderThan(actual: Version, minimum: Version): boolean {
  for (const part of ["major", "minor", "patch"] as const) {
    if (actual[part] !== minimum[part]) {
      return actual[part] < minimum[part];
    }
  }
  return false;
}

export function assertSupportedBunVersion(current: string | undefined, requirement: unknown): void {
  const minimumMatch = typeof requirement === "string" ? /^>=(\d+\.\d+\.\d+)$/.exec(requirement) : null;
  const actual = typeof current === "string" ? parseVersion(current) : undefined;
  const minimum = minimumMatch?.[1] === undefined ? undefined : parseVersion(minimumMatch[1]);
  if (
    minimum === undefined ||
    actual === undefined ||
    actual.prerelease ||
    isOlderThan(actual, minimum)
  ) {
    throw new LifecycleError(
      "opencode_bun_version_unsupported",
      `The OpenCode adapter requires Bun ${typeof requirement === "string" ? requirement : ">=1.4.2"}; current runtime is ${current ?? "unknown"}`,
    );
  }
}
