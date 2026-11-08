// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Resolve skills.paths entries that name a skill or a folder of skills. These
// run against the build in dist/, so test:skills builds first.
// cases mirror tests/adapter_contract/test_skills.py for the Python contract.

import assert from "node:assert/strict";
import { chmod, mkdir, mkdtemp, realpath, rm, symlink, writeFile } from "node:fs/promises";
import fs, { existsSync } from "node:fs";
import { syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { resolveSkills, SkillPathError } from "../../dist/skills.js";

const posixPermissions =
  typeof process.geteuid === "function" && process.geteuid() !== 0
    ? {}
    : { skip: "needs POSIX permissions enforced for a non-root user" };

async function temporary(t) {
  const root = await realpath(await mkdtemp(join(tmpdir(), "fabric-skills-")));
  const locked = [];
  t.after(async () => {
    for (const path of locked.reverse()) {
      await chmod(path, 0o700);
    }
    await rm(root, { recursive: true, force: true });
  });
  return {
    root,
    async lock(path) {
      await chmod(path, 0);
      locked.push(path);
      return path;
    },
  };
}

async function skill(directory, name) {
  await mkdir(directory, { recursive: true });
  const frontMatter = name ? `---\nname: ${name}\ndescription: Test skill.\n---\n` : "";
  await writeFile(join(directory, "SKILL.md"), `${frontMatter}# Test skill\n`, "utf8");
  return directory;
}

async function rejection(promise) {
  try {
    await promise;
  } catch (error) {
    assert.ok(error instanceof SkillPathError, String(error));
    return error;
  }
  assert.fail("expected a SkillPathError");
}

test("entries name skills or folders of skills", async (t) => {
  const { root } = await temporary(t);
  const shared = join(root, "shared");
  const one = await skill(join(shared, "one"), "one");
  const two = await skill(join(shared, "two"), "two");
  await writeFile(join(shared, "notes.md"), "Not a skill.\n", "utf8");
  await mkdir(join(shared, "SKILL.md"));
  await skill(join(shared, "group", "nested"), "nested");
  const three = await skill(join(root, "mounted", "three"), "three");
  await mkdir(join(root, "empty"));

  assert.deepEqual(await resolveSkills([shared, "mounted/three", "empty"], root), [
    { name: "one", path: one, field: "skills.paths[0]" },
    { name: "two", path: two, field: "skills.paths[0]" },
    { name: "three", path: three, field: "skills.paths[1]" },
  ]);
});

const failures = [
  ["not_found", {}, async (root) => [[await skill(join(root, "review")), join(root, "missing")], [join(root, "missing")]]],
  ["invalid", {}, async (root) => {
    await writeFile(join(root, "SKILL.md"), "# Not a folder\n", "utf8");
    return [[await skill(join(root, "review")), join(root, "SKILL.md")], [join(root, "SKILL.md")]];
  }],
  ["duplicate", {}, async (root) => {
    const first = await skill(join(root, "shared", "review"), "review");
    const second = await skill(join(root, "mounted", "review-copy"), "review");
    return [[join(root, "shared"), second], [first, second]];
  }],
  ["invalid", posixPermissions, async (root, lock) => {
    await mkdir(join(root, "locked"));
    return [[await skill(join(root, "review")), await lock(join(root, "locked"))], [join(root, "locked")]];
  }],
  ["invalid", posixPermissions, async (root, lock) => {
    await skill(join(root, "review"));
    const other = await skill(join(root, "shared", "other"));
    await lock(join(other, "SKILL.md"));
    return [[join(root, "review"), join(root, "shared")], [join(other, "SKILL.md")]];
  }],
];

for (const [index, [reason, options, layout]] of failures.entries()) {
  test(`failure ${index} reports ${reason}, naming the entry and paths`, options, async (t) => {
    const { root, lock } = await temporary(t);
    const [entries, paths] = await layout(root, lock);

    const error = await rejection(resolveSkills(entries, root));

    assert.equal(error.reason, reason);
    assert.equal(error.field, "skills.paths[1]");
    assert.deepEqual(error.paths, paths);
    assert.match(error.message, /skills\.paths\[1\]/);
    assert.ok(paths.every((path) => error.message.includes(path)), error.message);
  });
}

test("an unreadable folder inside an entry is ignored", posixPermissions, async (t) => {
  const { root, lock } = await temporary(t);
  const shared = join(root, "shared");
  const one = await skill(join(shared, "one"), "one");
  await mkdir(join(shared, "lost+found"));
  await lock(join(shared, "lost+found"));

  assert.deepEqual(await resolveSkills([shared], root), [
    { name: "one", path: one, field: "skills.paths[0]" },
  ]);
});

for (const [text, name] of [
  ['---\r\nname: "code-review"\r\n---\r\n', "code-review"],
  ["---\nname: code-review # main skill\n---\n", "code-review"],
  ["---\nname: 'code-review' # quoted\n---\n", "code-review"],
  ["---\nname: code#review\n---\n", "code#review"],
  ["--- \nname: code-review\n---\n", "code-review"],
  ["\uFEFF---\nname: code-review\n---\n", "code-review"],
  ["---\nname:\n---\n", "review"],
  ["---\ndescription: Front matter without an end.\nname: code-review\n", "review"],
  ["# Review\n\nname: code-review\n", "review"],
]) {
  test(`a skill's name follows its front matter: ${JSON.stringify(text)}`, async (t) => {
    const { root } = await temporary(t);
    const review = join(root, "review");
    await mkdir(review);
    await writeFile(join(review, "SKILL.md"), text, "utf8");

    assert.deepEqual((await resolveSkills([review], root)).map((s) => s.name), [name]);
  });
}

for (const spelling of ["shared", "shared/review/", "./shared/review", "other/../shared/review", "link"]) {
  test(`the same skill reached as ${spelling} loads once`, async (t) => {
    const { root } = await temporary(t);
    const review = await skill(join(root, "shared", "review"), "review");
    await mkdir(join(root, "other"));
    await symlink(review, join(root, "link"), "dir");

    assert.deepEqual(await resolveSkills(["shared/review", spelling], root), [
      { name: "review", path: review, field: "skills.paths[0]" },
    ]);
  });
}

test("the same skill in another case loads once", async (t) => {
  const { root } = await temporary(t);
  const review = await skill(join(root, "Review"));
  if (!existsSync(join(root, "review"))) {
    t.skip("needs a case-insensitive file system");
    return;
  }

  assert.deepEqual(await resolveSkills(["Review", "review"], root), [
    { name: "Review", path: review, field: "skills.paths[0]" },
  ]);
});

test("folders without inode numbers compare by path", async (t) => {
  const { root } = await temporary(t);
  const shared = join(root, "shared");
  const one = await skill(join(shared, "one"), "one");
  const two = await skill(join(shared, "two"), "two");
  // Some file systems report inode 0 for every folder.
  const stat = fs.promises.stat;
  fs.promises.stat = async (path, options) => {
    const info = await stat(path, options);
    info.ino = typeof info.ino === "bigint" ? 0n : 0;
    return info;
  };
  syncBuiltinESMExports();
  t.after(() => {
    fs.promises.stat = stat;
    syncBuiltinESMExports();
  });

  assert.deepEqual(await resolveSkills([shared, one], root), [
    { name: "one", path: one, field: "skills.paths[0]" },
    { name: "two", path: two, field: "skills.paths[0]" },
  ]);
});
