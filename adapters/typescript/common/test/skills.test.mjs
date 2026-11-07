// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdir, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { LifecycleError, resolveSkillFolders } from "../dist/index.js";

const codes = { not_found: "x_skill_not_found", invalid: "x_skill_invalid", duplicate: "x_skill_duplicate" };

async function skill(directory, name) {
  await mkdir(directory, { recursive: true });
  await writeFile(join(directory, "SKILL.md"), `---\nname: ${name}\n---\n`, "utf8");
  return directory;
}

test("resolves a skill and each skill inside a folder of skills", async (t) => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "fabric-common-skills-")));
  t.after(() => rm(root, { recursive: true, force: true }));
  const one = await skill(join(root, "shared", "one"), "one");
  const two = await skill(join(root, "shared", "two"), "two");
  const three = await skill(join(root, "three"), "three");

  assert.deepEqual(await resolveSkillFolders(["shared", three], root, codes), [one, two, three]);
});

for (const [reason, layout] of [
  ["not_found", async () => ["missing"]],
  ["invalid", async (root) => {
    await writeFile(join(root, "file"), "not a folder\n", "utf8");
    return ["file"];
  }],
  ["duplicate", async (root) => {
    await skill(join(root, "a", "review"), "review");
    await skill(join(root, "b", "review"), "review");
    return ["a", "b"];
  }],
]) {
  test(`reports ${reason} with the adapter's code and the failing field`, async (t) => {
    const root = await realpath(await mkdtemp(join(tmpdir(), "fabric-common-skills-")));
    t.after(() => rm(root, { recursive: true, force: true }));
    const paths = await layout(root);

    await assert.rejects(resolveSkillFolders(paths, root, codes), (error) => {
      assert.ok(error instanceof LifecycleError);
      assert.equal(error.code, codes[reason]);
      assert.deepEqual(error.metadata, { field: `skills.paths[${paths.length - 1}]` });
      assert.match(error.message, /skills\.paths\[\d\]/);
      return true;
    });
  });
}
