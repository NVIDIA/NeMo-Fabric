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

test("an entry with SKILL.md is one skill", async (t) => {
  const { root } = await temporary(t);
  const review = await skill(join(root, "review"), "code-review");

  assert.deepEqual(await resolveSkills([review], root), [
    { name: "code-review", path: review, field: "skills.paths[0]" },
  ]);
});

test("a skill's name falls back to its folder", async (t) => {
  const { root } = await temporary(t);
  const review = await skill(join(root, "review"));

  assert.deepEqual((await resolveSkills([review], root)).map((s) => s.name), ["review"]);
});

test("a quoted front matter name is unquoted", async (t) => {
  const { root } = await temporary(t);
  const review = join(root, "review");
  await mkdir(review);
  await writeFile(join(review, "SKILL.md"), '---\r\nname: "code-review"\r\n---\r\n', "utf8");

  assert.deepEqual((await resolveSkills([review], root)).map((s) => s.name), ["code-review"]);
});

test("a folder loads the skills directly inside it", async (t) => {
  const { root } = await temporary(t);
  const shared = join(root, "shared");
  const one = await skill(join(shared, "one"), "one");
  const two = await skill(join(shared, "two"), "two");
  await writeFile(join(shared, "notes.md"), "Not a skill.\n", "utf8");
  await mkdir(join(shared, "docs"));
  await writeFile(join(shared, "docs", "README.md"), "Not a skill.\n", "utf8");
  await skill(join(shared, "group", "nested"), "nested");

  assert.deepEqual(await resolveSkills([shared], root), [
    { name: "one", path: one, field: "skills.paths[0]" },
    { name: "two", path: two, field: "skills.paths[0]" },
  ]);
});

test("a folder and a separate skill mix", async (t) => {
  const { root } = await temporary(t);
  const shared = join(root, "shared");
  await skill(join(shared, "one"), "one");
  await skill(join(shared, "two"), "two");
  const separate = await skill(join(root, "mounted", "three"), "three");

  const skills = await resolveSkills([shared, separate], root);

  assert.deepEqual(skills.map((s) => [s.name, s.field]), [
    ["one", "skills.paths[0]"],
    ["two", "skills.paths[0]"],
    ["three", "skills.paths[1]"],
  ]);
});

test("an existing entry without skills loads nothing", async (t) => {
  const { root } = await temporary(t);
  await mkdir(join(root, "empty"));

  assert.deepEqual(await resolveSkills([join(root, "empty")], root), []);
});

test("a relative entry resolves against the base folder", async (t) => {
  const { root } = await temporary(t);
  const review = await skill(join(root, "skills", "review"), "review");

  assert.deepEqual(await resolveSkills(["skills/review"], root), [
    { name: "review", path: review, field: "skills.paths[0]" },
  ]);
});

test("a missing entry fails naming it", async (t) => {
  const { root } = await temporary(t);
  const review = await skill(join(root, "review"), "review");
  const missing = join(root, "missing");

  const error = await rejection(resolveSkills([review, missing], root));

  assert.equal(error.reason, "not_found");
  assert.equal(error.field, "skills.paths[1]");
  assert.deepEqual(error.paths, [missing]);
  assert.match(error.message, /skills\.paths\[1\]/);
  assert.ok(error.message.includes(missing), error.message);
});

test("a file entry fails", async (t) => {
  const { root } = await temporary(t);
  const entry = join(root, "SKILL.md");
  await writeFile(entry, "# Not a folder\n", "utf8");

  const error = await rejection(resolveSkills([entry], root));

  assert.equal(error.reason, "invalid");
  assert.equal(error.field, "skills.paths[0]");
  assert.ok(error.message.includes(entry), error.message);
});

test("duplicate skill names fail naming both folders", async (t) => {
  const { root } = await temporary(t);
  const shared = join(root, "shared");
  const first = await skill(join(shared, "review"), "review");
  const second = await skill(join(root, "mounted", "review-copy"), "review");

  const error = await rejection(resolveSkills([shared, second], root));

  assert.equal(error.reason, "duplicate");
  assert.equal(error.field, "skills.paths[1]");
  assert.deepEqual(error.paths, [first, second]);
  assert.ok(error.message.includes(first) && error.message.includes(second), error.message);
});

test("the same skill listed twice loads once", async (t) => {
  const { root } = await temporary(t);
  const shared = join(root, "shared");
  const review = await skill(join(shared, "review"), "review");

  assert.deepEqual(await resolveSkills([shared, review], root), [
    { name: "review", path: review, field: "skills.paths[0]" },
  ]);
});

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

test("an unreadable entry fails naming it", posixPermissions, async (t) => {
  const { root, lock } = await temporary(t);
  const review = await skill(join(root, "review"), "review");
  const entry = join(root, "locked");
  await mkdir(entry);
  await lock(entry);

  const error = await rejection(resolveSkills([review, entry], root));

  assert.equal(error.reason, "invalid");
  assert.equal(error.field, "skills.paths[1]");
  assert.deepEqual(error.paths, [entry]);
  assert.ok(error.message.includes(entry), error.message);
});

test("an unreadable SKILL.md fails naming it", posixPermissions, async (t) => {
  const { root, lock } = await temporary(t);
  const shared = join(root, "shared");
  const review = await skill(join(shared, "review"), "review");
  await lock(join(review, "SKILL.md"));

  const error = await rejection(resolveSkills([shared], root));

  assert.equal(error.reason, "invalid");
  assert.equal(error.field, "skills.paths[0]");
  assert.deepEqual(error.paths, [join(review, "SKILL.md")]);
  assert.ok(error.message.includes(join(review, "SKILL.md")), error.message);
});

for (const [text, name] of [
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

test("a SKILL.md folder does not make a skill", async (t) => {
  const { root } = await temporary(t);
  const shared = join(root, "shared");
  await mkdir(join(shared, "SKILL.md"), { recursive: true });
  const one = await skill(join(shared, "one"), "one");

  assert.deepEqual(await resolveSkills([shared], root), [
    { name: "one", path: one, field: "skills.paths[0]" },
  ]);
});

for (const spelling of ["review/", "./review", "other/../review", "link"]) {
  test(`the same skill spelled ${spelling} loads once`, async (t) => {
    const { root } = await temporary(t);
    const review = await skill(join(root, "review"), "review");
    await mkdir(join(root, "other"));
    if (spelling === "link") {
      await symlink(review, join(root, "link"), "dir");
    }

    assert.deepEqual(await resolveSkills(["review", spelling], root), [
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
