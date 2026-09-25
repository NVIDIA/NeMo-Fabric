// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import {KiloAdapterRuntime} from "../dist/runtime.js";

const context = {runtime_id: "r", invocation_id: "i", request_id: "q", artifacts: {}, environment: {environment_id: "e", provider: "local", ownership: "caller_owned", control_location: "external_control", env: {}}};

test("keeps one warm Kilo Code session across invocations", async () => {
  const prompts = [];
  let stopped = 0;
  const runtime = new KiloAdapterRuntime({create: async () => ({id: "s", prompt: async (text) => { prompts.push(text); return {text: `answer:${text}`, usage: {input_tokens: 1, output_tokens: 2, total_tokens: 3}}; }, stop: async () => { stopped += 1; }})});
  await runtime.start({agentName: "agent", baseDir: ".", config: {}, runtimeContext: context});
  assert.deepEqual(await runtime.invoke({input: "one"}, context), {status: "succeeded", output: {response: "answer:one"}, usage: {input_tokens: 1, output_tokens: 2, total_tokens: 3}});
  assert.deepEqual(await runtime.invoke({input: "two"}, context), {status: "succeeded", output: {response: "answer:two"}, usage: {input_tokens: 1, output_tokens: 2, total_tokens: 3}});
  await runtime.stop();
  assert.deepEqual(prompts, ["one", "two"]);
  assert.equal(stopped, 1);
});

test("returns normalized model and empty-response failures", async () => {
  const outcomes = [{error: true}, {}];
  const runtime = new KiloAdapterRuntime({create: async () => ({id: "s", prompt: async () => outcomes.shift(), stop: async () => {}})});
  await runtime.start({agentName: "agent", baseDir: ".", config: {}, runtimeContext: context});
  assert.equal((await runtime.invoke({input: "one"}, context)).error.code, "kilo_model_error");
  assert.equal((await runtime.invoke({input: "two"}, context)).error.code, "kilo_no_assistant_response");
});

test("invalidates the runtime after transport failure", async () => {
  let stopped = 0;
  const runtime = new KiloAdapterRuntime({create: async () => ({id: "s", prompt: async () => { throw new Error("offline"); }, stop: async () => { stopped += 1; }})});
  await runtime.start({agentName: "agent", baseDir: ".", config: {}, runtimeContext: context});
  await assert.rejects(runtime.invoke({input: "one"}, context), (error) => error.code === "kilo_session_failed");
  await assert.rejects(runtime.invoke({input: "two"}, context), (error) => error.code === "kilo_not_started");
  assert.equal(stopped, 1);
});
