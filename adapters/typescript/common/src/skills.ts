// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Resolve skills.paths with the adapter contract's rule and report failures as
// lifecycle errors with each adapter's own error codes.

import { resolveSkills, SkillPathError } from "nemo-fabric-adapter-contract/skills";
import type { SkillPathFailure } from "nemo-fabric-adapter-contract/skills";

import { LifecycleError } from "./lifecycle.js";

/** Lifecycle error code that an adapter reports for each `skills.paths` failure. */
export type SkillPathErrorCodes = Readonly<Record<SkillPathFailure, string>>;

/**
 * Return the folder of each skill that `paths` names, resolving relative
 * entries against `baseDir`. An entry that contains `SKILL.md` is one skill;
 * otherwise each folder directly inside it that contains `SKILL.md` is.
 *
 * @throws {LifecycleError} With the code from `codes` and the failing
 *   `skills.paths[N]` field when an entry is missing, is not a readable folder,
 *   or two skills share a name.
 */
export async function resolveSkillFolders(
  paths: readonly string[],
  baseDir: string,
  codes: SkillPathErrorCodes,
): Promise<string[]> {
  try {
    return (await resolveSkills(paths, baseDir)).map((skill) => skill.path);
  } catch (error) {
    if (error instanceof SkillPathError) {
      throw new LifecycleError(codes[error.reason], error.message, {
        metadata: { field: error.field },
      });
    }
    throw error;
  }
}
