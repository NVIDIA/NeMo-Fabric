// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";
import { collectSoftware, packageVersion } from "../dist/provenance.js";

test("preserves full selected versions and original times without executable probes", async () => {
  const root = await mkdtemp(join(tmpdir(), "fabric-software-"));
  try {
    const path = join(root, "package.json");
    const value = "1.2.3-rc.1+" + "build".repeat(100);
    await writeFile(path, JSON.stringify({ name: "selected-adapter", version: value }));
    const captured = "2026-10-27T12:00:00Z";
    const observation = await packageVersion(path, captured);
    assert.equal(observation.value, value);
    assert.equal(observation.observed_at, captured);
    assert.equal(observation.origin, "observed");
    const report = await collectSoftware({ packageJson: pathToFileURL(path), harness: "selected-harness" });
    const env = report.software_environments[0];
    assert.equal(env.entries.find(entry => entry.role === "adapter").version.value, value);
    assert.equal(env.entries.find(entry => entry.role === "harness").version.status, "unavailable");
    assert.equal(env.complete, true);
  } finally { await rm(root, { recursive: true }); }
});

test("missing, invalid and over-budget metadata remain explicit unknowns", async () => {
  const root = await mkdtemp(join(tmpdir(), "fabric-software-"));
  try {
    const path = join(root, "package.json");
    for (const text of ["{", JSON.stringify({ version: "x".repeat(70_000) })]) {
      await writeFile(path, text);
      assert.equal((await packageVersion(path, "2026-10-27T12:00:00Z")).status, "unavailable");
    }
    const missing = await packageVersion(join(root, "missing"), "2026-10-27T12:00:00Z");
    assert.equal(missing.status, "unavailable");
    assert.ok(missing.reason);
  } finally { await rm(root, { recursive: true }); }
});

test("selected contract and common metadata resolve despite import-only root exports", async () => {
  const report = await collectSoftware({ packageJson: new URL("../../pi/package.json", import.meta.url), harness: "pi" });
  const entries = report.software_environments[0].entries;
  for (const name of ["nemo-fabric-adapter-contract", "nemo-fabric-adapters-common"]) {
    const entry = entries.find(entry => entry.name === name);
    assert.equal(entry.version.status, "available");
    assert.ok(entry.version.value);
  }
});

test("invalid UTF-16 metadata cannot make an otherwise valid lifecycle frame undecodable", async () => {
  const root = await mkdtemp(join(tmpdir(), "fabric-software-"));
  try {
    const path = join(root, "package.json");
    await writeFile(path, JSON.stringify({ name: "selected-adapter", version: "bad\ud800" }));
    const observation = await packageVersion(path, "2026-10-27T12:00:00Z");
    assert.equal(observation.status, "unavailable");
    assert.equal(observation.reason, "invalid_version_encoding");
    assert.equal("value" in observation, false);
  } finally { await rm(root, { recursive: true }); }
});

test("selected dependency metadata is bounded before parsing and never bypasses a shadowing package", async () => {
  const root = await mkdtemp(join(tmpdir(), "fabric-software-"));
  try {
    const adapter = join(root, "adapter");
    await mkdir(adapter);
    const own = join(adapter, "package.json");
    await writeFile(own, JSON.stringify({ name: "selected-adapter", version: "1.0+full" }));
    const dependency = join(adapter, "node_modules", "selected-sdk");
    await mkdir(dependency, { recursive: true });
    const outer = join(root, "node_modules", "selected-sdk");
    await mkdir(outer, { recursive: true });
    await writeFile(join(outer, "package.json"), JSON.stringify({ name: "selected-sdk", version: "outer-install" }));
    for (const [text, reason] of [
      [JSON.stringify({ name: "selected-sdk", version: "2.0", padding: "x".repeat(70_000) }), "metadata_byte_limit"],
      ["{", "metadata_malformed"],
      [JSON.stringify({ name: "other-package", version: "wrong" }), "metadata_name_mismatch"],
    ]) {
      await writeFile(join(dependency, "package.json"), text);
      const report = await collectSoftware({ packageJson: pathToFileURL(own), harness: "fixture",
        packages: [{ role: "harness_sdk", name: "selected-sdk" }] });
      const environment = report.software_environments[0];
      const entry = environment.entries.find(entry => entry.name === "selected-sdk");
      assert.equal(entry.version.status, "unavailable");
      assert.equal(entry.version.reason, reason);
      if (reason === "metadata_byte_limit") assert.equal(environment.complete, false);
    }
    await rm(join(dependency, "package.json"));
    await writeFile(join(dependency, "index.js"), "module.exports = {};\n");
    const report = await collectSoftware({ packageJson: pathToFileURL(own), harness: "fixture",
      packages: [{ role: "harness_sdk", name: "selected-sdk" }] });
    assert.equal(report.software_environments[0].entries.find(entry => entry.name === "selected-sdk").version.reason, "metadata_not_found");
  } finally { await rm(root, { recursive: true }); }
});
