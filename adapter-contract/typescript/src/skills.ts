// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Resolve `skills.paths` entries in the task environment.
 *
 * Each entry names one skill or a folder of skills:
 *
 * 1. An entry that contains `SKILL.md` is one skill.
 * 2. Otherwise each folder directly inside the entry that contains `SKILL.md`
 *    is a skill. Other files and folders, including folders that cannot be
 *    read, are ignored.
 * 3. An entry that exists but holds no skills loads nothing.
 * 4. An entry that does not exist, is not a folder, or cannot be read fails.
 * 5. Two skills with the same name fail. The same folder reached through
 *    several entries loads once.
 *
 * A skill's name is the `name` field of its `SKILL.md` front matter, or its
 * folder name when the front matter has none.
 *
 * This module uses Node.js file system APIs; import it from
 * `nemo-fabric-adapter-contract/skills`.
 *
 * @module
 */

import type { FileHandle } from "node:fs/promises";
import { open, readdir, realpath, stat } from "node:fs/promises";
import { basename, isAbsolute, join, sep } from "node:path";

/** File that marks a folder as a skill. */
export const SKILL_FILE = "SKILL.md";

/** Why a `skills.paths` entry cannot be loaded. */
export type SkillPathFailure = "not_found" | "invalid" | "duplicate";

/** One skill found through `skills.paths`. */
export interface Skill {
  /** Name from the skill's front matter, or its folder name. */
  readonly name: string;
  /** Resolved folder that contains the skill's `SKILL.md`. */
  readonly path: string;
  /** The `skills.paths` entry that supplied the skill, such as `skills.paths[0]`. */
  readonly field: string;
}

/** A `skills.paths` entry that cannot be loaded. */
export class SkillPathError extends Error {
  override readonly name = "SkillPathError";
  readonly reason: SkillPathFailure;
  readonly field: string;
  readonly paths: readonly string[];

  constructor(
    reason: SkillPathFailure,
    message: string,
    options: { field: string; paths: readonly string[] },
  ) {
    super(message);
    this.reason = reason;
    this.field = options.field;
    this.paths = options.paths;
  }
}

const NAME = /^name:(?:[ \t]+(.*?))?[ \t]*$/;
const COMMENT = /[ \t]+#/;

/**
 * Return the skills named by `paths`, resolving relative entries against
 * `baseDir`.
 *
 * @throws {SkillPathError} If an entry does not exist, is not a folder, or
 *   cannot be read, if a skill's `SKILL.md` cannot be read, or if two skills
 *   share a name.
 */
export async function resolveSkills(
  paths: Iterable<string>,
  baseDir: string,
): Promise<Skill[]> {
  const skills: Skill[] = [];
  const byName = new Map<string, Skill>();
  const folders = new Set<string>();
  let index = 0;
  for (const value of paths) {
    const field = `skills.paths[${index}]`;
    index += 1;
    // Do not normalize ".." here: realpath follows symbolic links before "..".
    const configured = isAbsolute(value) ? value : `${baseDir}${sep}${value}`;
    let entry: string;
    try {
      entry = await realpath(configured);
    } catch {
      throw new SkillPathError(
        "not_found",
        `NeMo Fabric skill path ${field} does not exist: ${configured}`,
        { field, paths: [configured] },
      );
    }
    if (!(await isDirectory(entry))) {
      throw new SkillPathError(
        "invalid",
        `NeMo Fabric skill path ${field} is not a folder: ${configured}`,
        { field, paths: [configured] },
      );
    }
    for (const directory of await skillDirectories(entry, field, configured)) {
      const folder = await folderIdentity(directory);
      if (folders.has(folder)) {
        continue;
      }
      folders.add(folder);
      const skill: Skill = {
        name: await skillName(directory, field),
        path: directory,
        field,
      };
      const existing = byName.get(skill.name);
      if (existing !== undefined) {
        throw new SkillPathError(
          "duplicate",
          `NeMo Fabric skill '${skill.name}' is configured twice: ` +
            `${existing.path} (${existing.field}) and ${skill.path} (${field})`,
          { field, paths: [existing.path, skill.path] },
        );
      }
      byName.set(skill.name, skill);
      skills.push(skill);
    }
  }
  return skills;
}

async function skillDirectories(
  entry: string,
  field: string,
  configured: string,
): Promise<string[]> {
  let children: string[];
  try {
    if (await isFile(join(entry, SKILL_FILE))) {
      return [entry];
    }
    children = (await readdir(entry)).sort(compareNames);
  } catch {
    throw new SkillPathError(
      "invalid",
      `NeMo Fabric skill path ${field} cannot be read: ${configured}`,
      { field, paths: [configured] },
    );
  }
  const directories: string[] = [];
  for (const child of children) {
    const path = join(entry, child);
    // A folder that cannot be read, such as a volume's lost+found, is not a skill.
    try {
      if ((await isDirectory(path)) && (await isFile(join(path, SKILL_FILE)))) {
        directories.push(await realpath(path));
      }
    } catch {
      continue;
    }
  }
  return directories;
}

function compareNames(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

const MISSING = new Set(["ENOENT", "ENOTDIR", "ELOOP"]);

async function isDirectory(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isDirectory();
  } catch (error) {
    if (MISSING.has(errorCode(error))) {
      return false;
    }
    throw error;
  }
}

async function isFile(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isFile();
  } catch (error) {
    if (MISSING.has(errorCode(error))) {
      return false;
    }
    throw error;
  }
}

function errorCode(error: unknown): string {
  return typeof error === "object" && error !== null && "code" in error
    ? String(error.code)
    : "";
}

async function folderIdentity(directory: string): Promise<string> {
  // Compare folders by identity so that spellings realpath keeps apart, such
  // as a different case on a case-insensitive file system, still match. An
  // inode of 0 identifies nothing, so fall back to the resolved path.
  const info = await stat(directory, { bigint: true });
  return info.ino === 0n ? `path:${directory}` : `inode:${info.dev}:${info.ino}`;
}

async function skillName(directory: string, field: string): Promise<string> {
  const skillFile = join(directory, SKILL_FILE);
  let handle: FileHandle | undefined;
  let name: string | undefined;
  try {
    handle = await open(skillFile, "r");
    name = await frontMatterName(handle);
  } catch {
    throw new SkillPathError(
      "invalid",
      `NeMo Fabric skill path ${field} has a ${SKILL_FILE} that cannot be read: ${skillFile}`,
      { field, paths: [skillFile] },
    );
  } finally {
    await handle?.close();
  }
  return name || basename(directory);
}

/** Return the `name` scalar of a `---` front matter block, if any. */
async function frontMatterName(handle: FileHandle): Promise<string | undefined> {
  let first = true;
  let name: string | undefined;
  for await (let line of handle.readLines({ encoding: "utf8", autoClose: false })) {
    line = line.trimEnd();
    if (first) {
      first = false;
      if (line.replace(/^\uFEFF/, "") !== "---") {
        return undefined;
      }
      continue;
    }
    if (line === "---") {
      return name;
    }
    const match = NAME.exec(line);
    if (match !== null && !name) {
      name = scalar(match[1] ?? "");
    }
  }
  // Without its closing line the block is not front matter.
  return undefined;
}

function scalar(value: string): string {
  const quote = value[0];
  if (quote === "'" || quote === '"') {
    const end = value.indexOf(quote, 1);
    const rest = end > 0 ? value.slice(end + 1).trim() : "";
    if (end > 0 && (rest === "" || rest.startsWith("#"))) {
      return value.slice(1, end);
    }
    return value;
  }
  if (value.startsWith("#")) {
    return "";
  }
  return value.split(COMMENT, 1)[0] ?? "";
}
