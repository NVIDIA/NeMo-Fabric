// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Enforce the package's minimum Node.js version at the process boundary before
// the runner loads Pi SDK modules. The accepted requirement shape matches the
// package.json engines.node declaration owned by this adapter.

type Version = Readonly<{
  major: number;
  minor: number;
  patch: number;
  prerelease: boolean;
}>;

function parseVersion(value: string, label: string): Version {
  const match = /^(\d+)\.(\d+)\.(\d+)(-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/.exec(value);
  if (match === null) {
    throw new Error(`${label} must be a semantic version`);
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
    const actualPart = actual[part];
    const minimumPart = minimum[part];
    if (actualPart !== minimumPart) {
      return actualPart < minimumPart;
    }
  }
  return false;
}

export function assertSupportedNodeVersion(current: string, requirement: unknown): void {
  if (typeof requirement !== "string") {
    throw new Error("The Pi package must declare a Node.js engine requirement");
  }
  const minimumMatch = /^>=(\d+\.\d+\.\d+)$/.exec(requirement);
  if (minimumMatch?.[1] === undefined) {
    throw new Error(`The Pi adapter cannot enforce Node.js engine requirement ${requirement}`);
  }
  const actual = parseVersion(current, "The current Node.js version");
  const minimum = parseVersion(minimumMatch[1], "The minimum Node.js version");
  if (actual.prerelease || isOlderThan(actual, minimum)) {
    throw new Error(`The Pi adapter requires Node.js ${requirement}; current runtime is ${current}`);
  }
}
