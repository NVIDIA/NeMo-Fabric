// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";

import {
  VERSIONS_REPORT_KEY,
  boundedText,
  missingFact,
  packageVersionFact,
  packageVersionReport,
  withPackageVersion,
} from "../dist/index.js";

const noSource = {state: "unsupported", reason: "no_source", source: "reported", collected_by: "adapter", method: "adapter_report"};

// The same vectors as core's string length test in `what_ran.rs`.
test("bounded text matches core's string length rule", () => {
  assert.equal(boundedText("é".repeat(256)), "é".repeat(256));
  assert.equal(boundedText(`${"m".repeat(300)}a`), `${"m".repeat(240)}~96858eedeb3c8e6`);
  assert.equal(boundedText("é".repeat(300)), `${"é".repeat(240)}~7250b66610f8b7d`);
  assert.equal(boundedText("v\ud800"), "v?");
});

test("each TypeScript adapter reads its own package version", async () => {
  for (const adapter of ["cline", "droid", "kilo", "opencode", "pi", "qwen"]) {
    const packageJson = new URL(`../../${adapter}/package.json`, import.meta.url);
    const {version} = JSON.parse(await readFile(packageJson, "utf8"));
    assert.deepEqual(packageVersionFact(packageJson), {
      value: version, source: "reported", collected_by: "adapter", method: "package_metadata",
    }, adapter);
  }
});

test("an unreadable package.json reads not_collected", () => {
  assert.deepEqual(packageVersionFact(new URL("file:///nonexistent/package.json")), {
    state: "unavailable", reason: "not_collected", source: "reported", collected_by: "adapter", method: "package_metadata",
  });
});

test("the wrapper adds the versions report and keeps the result", async () => {
  const directory = await mkdtemp(join(tmpdir(), "fabric-provenance-"));
  try {
    const packageJson = pathToFileURL(join(directory, "package.json"));
    await writeFile(packageJson, JSON.stringify({name: "adapter", version: "1.2.3"}));
    const calls = [];
    const runtime = withPackageVersion({
      async start(input) { calls.push(["start", input.agentName]); },
      async invoke() { return {status: "succeeded", output: "done", extensions: {session_id: "s-1"}}; },
      async stop() { calls.push(["stop"]); },
    }, packageJson);

    await runtime.start({agentName: "agent", baseDir: directory, config: {models: {default: {}, judge: {}}}, runtimeContext: {}});
    const result = await runtime.invoke({input: "hi"}, {});
    await runtime.stop();

    assert.deepEqual(calls, [["start", "agent"], ["stop"]]);
    assert.equal(result.status, "succeeded");
    assert.equal(result.output, "done");
    assert.equal(result.extensions.session_id, "s-1");
    assert.deepEqual(result.extensions[VERSIONS_REPORT_KEY], {
      adapter: {package_version: {value: "1.2.3", source: "reported", collected_by: "adapter", method: "package_metadata"}},
      harness: {name: noSource, version: noSource},
      models: {default: {served: noSource}, judge: {served: noSource}},
    });
  } finally {
    await rm(directory, {recursive: true, force: true});
  }
});

test("a served-model override reaches only the role it names", () => {
  const packageVersion = {value: "1.2.3", source: "reported", collected_by: "adapter", method: "package_metadata"};
  const notAttributed = missingFact("unavailable", "not_attributed", "harness_api");

  assert.deepEqual(packageVersionReport(packageVersion, ["judge", "default"], {role: "default", fact: notAttributed}).models, {
    default: {served: notAttributed},
    judge: {served: noSource},
  });
  // A role that is not configured adds no entry.
  assert.deepEqual(packageVersionReport(packageVersion, ["default"], {role: "absent", fact: notAttributed}).models, {
    default: {served: noSource},
  });
});

test("the wrapper picks the served-model role from the start input", async () => {
  const directory = await mkdtemp(join(tmpdir(), "fabric-provenance-"));
  try {
    const packageJson = pathToFileURL(join(directory, "package.json"));
    await writeFile(packageJson, JSON.stringify({name: "adapter", version: "1.2.3"}));
    const notAttributed = missingFact("unavailable", "not_attributed", "harness_api");
    const runtime = withPackageVersion({
      async start() {},
      async invoke() { return {status: "succeeded", output: "done"}; },
      async stop() {},
    }, packageJson, (input) => ({role: Object.keys(input.config.models)[0], fact: notAttributed}));

    await runtime.start({agentName: "agent", baseDir: directory, config: {models: {primary: {}}}, runtimeContext: {}});
    const result = await runtime.invoke({input: "hi"}, {});

    assert.deepEqual(result.extensions[VERSIONS_REPORT_KEY].models, {primary: {served: notAttributed}});
  } finally {
    await rm(directory, {recursive: true, force: true});
  }
});

test("the report keeps the roles core keeps, in code point order", async () => {
  // The vectors Rust core and the Python helper also read.
  const vectors = JSON.parse(await readFile(new URL("../../../../tests/fixtures/what_ran/role-order.json", import.meta.url), "utf8"));
  const packageVersion = {value: "1.2.3", source: "reported", collected_by: "adapter", method: "package_metadata"};
  const notAttributed = missingFact("unavailable", "not_attributed", "harness_api");

  const report = packageVersionReport(packageVersion, vectors.roles, {role: "\u{E000}", fact: notAttributed});

  assert.deepEqual(Object.keys(report.models), vectors.kept);
  assert.deepEqual(report.models["\u{E000}"], {served: notAttributed});
});
