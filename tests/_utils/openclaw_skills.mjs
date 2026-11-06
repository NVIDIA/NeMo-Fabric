// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

// The pinned release exposes its native loader through a hashed bundle facade.
// Do not replace discovery or file reading with a test implementation.
const dist = join(process.argv[2], "dist");
const loaders = [];
for (const file of readdirSync(dist).filter((name) =>
  /^workspace-skill-loader-.*\.mjs$/.test(name),
)) {
  const module = await import(pathToFileURL(join(dist, file)).href);
  if (typeof module.loadWorkspaceSkills === "function") {
    loaders.push(module.loadWorkspaceSkills);
  }
}
assert.equal(
  loaders.length,
  1,
  "the pinned package must expose one native workspace loader",
);
const config = JSON.parse(readFileSync(process.argv[3], "utf8"));
const entries = loaders[0](config.agents.defaults.workspace, { config });
process.stdout.write(
  JSON.stringify(
    entries.map(({ skill }) => ({
      name: skill.name,
      source: skill.source,
      path: skill.filePath,
      contentHash: skill.contentHash,
    })),
  ),
);
