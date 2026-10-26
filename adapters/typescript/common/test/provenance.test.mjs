// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
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

const fixtures = new URL("../../../../tests/fixtures/what_ran/", import.meta.url);

test("bounded text matches core's string length rule", async () => {
  // The vectors Rust core and the Python helper also read.
  const {cases} = JSON.parse(await readFile(new URL("string-length.json", fixtures), "utf8"));
  assert.equal(cases.length, 3);
  for (const {text, expected} of cases) assert.equal(boundedText(text), expected);
  // JSON fixtures can't carry a lone surrogate into Rust, so this case stays here.
  assert.equal(boundedText("v\ud800"), "v?");
});

test("each TypeScript adapter reads its own package version", async () => {
  for (const adapter of ["cline", "droid", "kilo", "kiro", "opencode", "pi", "qwen"]) {
    const packageJson = new URL(`../../${adapter}/package.json`, import.meta.url);
    const {version} = JSON.parse(await readFile(packageJson, "utf8"));
    assert.deepEqual(packageVersionFact(packageJson), {
      value: version, source: "reported", collected_by: "adapter", method: "package_metadata",
    }, adapter);
  }
});

test("a missing package.json reads not_collected", () => {
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
    // Retain this uniquely named temporary directory for test diagnostics.
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
    // Retain this uniquely named temporary directory for test diagnostics.
  }
});

test("the report keeps the roles core keeps, in code point order", async () => {
  // The vectors Rust core and the Python helper also read.
  const vectors = JSON.parse(await readFile(new URL("role-order.json", fixtures), "utf8"));
  const packageVersion = {value: "1.2.3", source: "reported", collected_by: "adapter", method: "package_metadata"};
  const notAttributed = missingFact("unavailable", "not_attributed", "harness_api");

  const report = packageVersionReport(packageVersion, vectors.roles, {role: "\u{E000}", fact: notAttributed});

  assert.deepEqual(Object.keys(report.models), vectors.kept);
  assert.deepEqual(report.models["\u{E000}"], {served: notAttributed});
});


test("returned version facts are detached from later invocations", async () => {
  const runtime = withPackageVersion({
    async start() {}, async stop() {},
    async invoke() { return {status: "succeeded", output: "done"}; },
  }, new URL("../package.json", import.meta.url));
  await runtime.start({agentName: "test", baseDir: ".", config: {models: {default: {}}}, runtimeContext: {}});
  const first = await runtime.invoke({input: "one"}, {});
  first.extensions[VERSIONS_REPORT_KEY].harness.version.reason = "changed";
  const second = await runtime.invoke({input: "two"}, {});
  assert.equal(second.extensions[VERSIONS_REPORT_KEY].harness.version.reason, "no_source");
});


test("invalid package metadata is a collection failure", async () => {
  const directory = await mkdtemp(join(tmpdir(), "fabric-provenance-invalid-"));
  try {
    const packageJson = pathToFileURL(join(directory, "package.json"));
    await writeFile(packageJson, "invalid json");
    assert.deepEqual(packageVersionFact(packageJson), {
      state: "collection_failed", reason: "no_result", source: "reported", collected_by: "adapter", method: "package_metadata",
    });
  } finally {
    // Retain this uniquely named temporary directory for test diagnostics.
  }
});
