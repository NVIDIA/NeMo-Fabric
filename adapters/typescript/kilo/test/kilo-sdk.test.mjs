// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import {EventEmitter} from "node:events";
import test from "node:test";
import {mkdtemp, mkdir, rm, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {PassThrough} from "node:stream";

import {classifySpawnError, closeChild, extractInvocationUsage, extractPromptOutcome, findPrompt, KiloSdkSessionFactory, startKiloServer} from "../dist/kilo-sdk.js";

let clock = 1000;

function step(tokens, cost = 0) {
  return {type: "step-finish", reason: "stop", cost, tokens};
}

// Kilo Code overwrites a message's tokens with each finished step, so the message mirrors its last step.
// Kilo Code sets each assistant message's parentID to the user message of its turn and records when it finished.
function assistant(id, parts, parentID = "user-1") {
  clock += 1;
  const tokens = parts.findLast((part) => part.type === "step-finish")?.tokens ?? {input: 0, output: 0, reasoning: 0, cache: {read: 0, write: 0}};
  return {info: {id, role: "assistant", parentID, cost: 0, tokens, time: {created: clock, completed: clock}}, parts};
}

function user(id, parts = [{type: "text", text: "do it"}]) {
  clock += 1;
  return {info: {id, role: "user", time: {created: clock}}, parts};
}

function taskCall(sessionId, start, background = false) {
  return {type: "tool", tool: "task", state: {status: "completed", input: {background}, metadata: {sessionId, ...(background ? {background: true} : {})}, time: {start}}};
}

// Kilo Code's task tool prompts the parent session with a background subagent's result as synthetic text.
function backgroundResult(id, sessionId) {
  return user(id, [{type: "text", synthetic: true, metadata: {background: true}, text: `<task id="${sessionId}" state="completed">done</task>`}]);
}

const call = (input, output, read = 0, write = 0, reasoning = 0) => ({input, output, reasoning, cache: {read, write}});
const userPrompt = user("user-1");
const usageOf = (messages, prompt = "user-1", prompts = [prompt], children = new Map()) => extractInvocationUsage(messages, prompt, new Set(prompts), children);

test("extracts Kilo Code terminal text without taking usage from the last message", () => {
  assert.deepEqual(extractPromptOutcome({
    info: {tokens: {input: 4, output: 6, total: 10}, cost: 0.05},
    parts: [{type: "text", text: "hello"}, {type: "tool", name: "read"}, {type: "text", text: " world"}],
  }), {text: "hello world"});
});

test("sums every Kilo Code model call of the prompt with cache and reasoning", () => {
  const messages = [
    user("old-user"),
    assistant("old-assistant", [step(call(999, 999))], "old-user"),
    userPrompt,
    assistant("read-file", [step(call(1200, 80, 0, 1000))]),
    assistant("edit-file", [step(call(150, 40, 2200, 0, 20))]),
    assistant("answer", [step(call(90, 200, 2350)), {type: "text", text: "done"}]),
  ];
  // The first call's cache read is 0, which Kilo Code also stores for omitted cache details.
  assert.deepEqual(usageOf(messages, "user-1", ["old-user", "user-1"]), {
    input_tokens: 6990,
    input_tokens_include_cache: true,
    output_tokens: 340,
    total_tokens: 7330,
  });
});

test("sums every finished step that Kilo Code recorded in one message", () => {
  const messages = [userPrompt, assistant("a", [step(call(4, 6)), step(call(8, 12))])];
  assert.deepEqual(usageOf(messages), {input_tokens: 12, input_tokens_include_cache: true, output_tokens: 18, total_tokens: 30});
});

test("counts Kilo Code compaction calls inside the prompt's turn", () => {
  const compaction = [
    userPrompt,
    assistant("a", [step(call(100, 10, 50))]),
    user("compaction", [{type: "compaction", auto: true}]),
    assistant("summary", [step(call(300, 40))], "compaction"),
    user("continue", [{type: "text", synthetic: true, metadata: {compaction_continue: true}, text: "Continue"}]),
    assistant("b", [step(call(60, 5, 20))], "continue"),
  ];
  assert.deepEqual(usageOf(compaction), {input_tokens: 530, input_tokens_include_cache: true, output_tokens: 55, total_tokens: 585});
  // Compaction before the prompt's first call follows the previous prompt's last call.
  const preflight = [
    user("user-0"),
    assistant("earlier", [step(call(10, 1))], "user-0"),
    userPrompt,
    user("compaction", [{type: "compaction", auto: true}]),
    assistant("summary", [step(call(300, 40))], "compaction"),
    assistant("answer", [step(call(20, 2))]),
  ];
  assert.deepEqual(usageOf(preflight, "user-1", ["user-0", "user-1"]), {input_tokens: 320, input_tokens_include_cache: true, output_tokens: 42, total_tokens: 362});
});

test("adds the calls of a Kilo Code foreground subagent from its child session", () => {
  const messages = [userPrompt, assistant("delegate", [taskCall("child-1", 2000), step(call(1500, 15))])];
  const child = [
    {info: {id: "c-user", role: "user", time: {created: 2001}}, parts: [{type: "text", text: "sub task"}]},
    {info: {id: "c-a", role: "assistant", parentID: "c-user", time: {created: 2002, completed: 2003}}, parts: [step(call(5000, 50))]},
  ];
  assert.deepEqual(usageOf(messages, "user-1", ["user-1"], new Map([["child-1", child]])), {input_tokens: 6500, input_tokens_include_cache: true, output_tokens: 65, total_tokens: 6565});
  assert.equal(usageOf(messages), undefined, "an unread child session leaves usage unknown");
});

test("adds only the calls of the resumed Kilo Code subagent window", () => {
  const messages = [
    user("user-0"),
    assistant("first-delegate", [taskCall("child-1", 2000), step(call(10, 1))], "user-0"),
    userPrompt,
    assistant("resume", [taskCall("child-1", 3000), step(call(20, 2))]),
  ];
  const child = [
    {info: {id: "c-user-0", role: "user", time: {created: 2001}}, parts: []},
    {info: {id: "c-a-0", role: "assistant", parentID: "c-user-0", time: {created: 2002, completed: 2003}}, parts: [step(call(700, 7))]},
    {info: {id: "c-user-1", role: "user", time: {created: 3001}}, parts: []},
    {info: {id: "c-a-1", role: "assistant", parentID: "c-user-1", time: {created: 3002, completed: 3003}}, parts: [step(call(400, 4))]},
  ];
  assert.deepEqual(usageOf(messages, "user-1", ["user-0", "user-1"], new Map([["child-1", child]])), {input_tokens: 420, input_tokens_include_cache: true, output_tokens: 6, total_tokens: 426});
});

test("reports no Kilo Code usage when a background turn may own a compaction after a newer prompt", () => {
  // P0 starts a background subagent and returns. Its result arrives, and P1 is stored while that background turn runs.
  // The background turn compacts before its first assistant message, then P1's own call runs.
  const messages = [
    user("user-0"),
    assistant("delegate", [taskCall("child-1", 2000, true), step(call(10, 1))], "user-0"),
    backgroundResult("result", "child-1"),
    userPrompt,
    user("compaction", [{type: "compaction", auto: true}]),
    assistant("summary", [step(call(800, 17))], "compaction"),
    user("continue", [{type: "text", synthetic: true, metadata: {compaction_continue: true}, text: "Continue"}]),
    assistant("answer", [step(call(3, 1))]),
  ];
  assert.equal(usageOf(messages, "user-1", ["user-0", "user-1"]), undefined);
  // The same when the background result was stored while P0's turn was still running.
  const queued = [
    user("user-x"),
    assistant("delegate", [taskCall("child-1", 2000, true), step(call(10, 1))], "user-x"),
    user("user-0"),
    backgroundResult("result", "child-1"),
    assistant("p0-answer", [step(call(5, 1))], "user-0"),
    userPrompt,
    user("compaction", [{type: "compaction", auto: true}]),
    assistant("summary", [step(call(800, 17))], "compaction"),
    assistant("answer", [step(call(3, 1))]),
  ];
  assert.equal(usageOf(queued, "user-1", ["user-x", "user-0", "user-1"]), undefined);
});

test("reports no Kilo Code usage when two subagent task windows start at the same time", () => {
  const messages = [userPrompt, assistant("delegate", [taskCall("child-1", 2000), taskCall("child-1", 2000), step(call(10, 1))])];
  const child = [
    {info: {id: "c-user", role: "user", time: {created: 2001}}, parts: []},
    {info: {id: "c-a", role: "assistant", parentID: "c-user", time: {created: 2002, completed: 2003}}, parts: [step(call(500, 5))]},
  ];
  assert.equal(usageOf(messages, "user-1", ["user-1"], new Map([["child-1", child]])), undefined);
});

test("reports no Kilo Code usage when a subagent starts its own subagent", () => {
  const messages = [userPrompt, assistant("delegate", [taskCall("child-1", 2000), step(call(10, 1))])];
  const child = [{info: {id: "c-a", role: "assistant", parentID: "c-user", time: {created: 2002, completed: 2003}}, parts: [taskCall("grandchild", 2002), step(call(5, 1))]}];
  assert.equal(usageOf(messages, "user-1", ["user-1"], new Map([["child-1", child]])), undefined);
});

test("reports no Kilo Code usage when the prompt starts a background subagent", () => {
  const messages = [userPrompt, assistant("delegate", [taskCall("child-1", 2000, true), step(call(10, 1))]), assistant("answer", [step(call(20, 2))])];
  assert.equal(usageOf(messages), undefined);
});

test("leaves a Kilo Code reply to an earlier background subagent out of the prompt that it ran during", () => {
  const messages = [
    user("user-0"),
    assistant("delegate", [taskCall("child-1", 2000, true), step(call(10, 1))], "user-0"),
    userPrompt,
    backgroundResult("result", "child-1"),
    assistant("background-reply", [step(call(7000, 70))], "result"),
    assistant("answer", [step(call(300, 3))]),
  ];
  assert.deepEqual(usageOf(messages, "user-1", ["user-0", "user-1"]), {input_tokens: 300, input_tokens_include_cache: true, output_tokens: 3, total_tokens: 303});
  assert.equal(usageOf(messages, "user-0", ["user-0", "user-1"]), undefined);
});

test("reports no Kilo Code usage when a new call cannot be attributed", () => {
  const unknownResult = [user("user-0"), userPrompt, backgroundResult("result", "missing"), assistant("reply", [step(call(4, 6))], "result"), assistant("answer", [step(call(4, 6))])];
  assert.equal(usageOf(unknownResult, "user-1", ["user-0", "user-1"]), undefined);
  const orphan = assistant("orphan", [step(call(4, 6))]);
  delete orphan.info.parentID;
  assert.equal(usageOf([userPrompt, orphan]), undefined);
  assert.equal(usageOf([assistant("a", [step(call(4, 6))])]), undefined, "a missing prompt message leaves usage unknown");
});

test("finds the Kilo Code prompt after the newest message before it", () => {
  const messages = [user("user-0"), assistant("a", [step(call(4, 6))], "user-0"), backgroundResult("result", "child-1"), user("user-1")];
  assert.equal(findPrompt(messages, "a"), "user-1");
  assert.equal(findPrompt(messages, undefined), "user-0");
  assert.equal(findPrompt(messages, "deleted"), undefined);
  assert.equal(findPrompt(undefined, undefined), undefined);
});

test("reports no Kilo Code usage when a model call never finished", () => {
  const running = assistant("running", [{type: "text", text: "partial"}]);
  delete running.info.time.completed;
  assert.equal(usageOf([userPrompt, assistant("a", [step(call(4, 6))]), running]), undefined);
  assert.equal(usageOf([userPrompt, assistant("a", [step(call(4, 6))]), assistant("aborted", [{type: "text", text: "partial"}])]), undefined);
  assert.equal(usageOf([], "user-1"), undefined);
  assert.equal(usageOf(undefined, "user-1"), undefined);
});

test("derives the Kilo Code total from input and output", () => {
  const messages = [userPrompt, assistant("a", [step({total: 28, ...call(10, 15, 3)})]), assistant("b", [step(call(7, 3, 1, 0, 2))])];
  assert.deepEqual(usageOf(messages), {input_tokens: 21, cached_input_tokens: 4, input_tokens_include_cache: true, output_tokens: 20, total_tokens: 41});
});

test("leaves Kilo Code counts unknown when any call omitted them", () => {
  const missingOutput = [userPrompt, assistant("a", [step({total: 10, ...call(4, 6)})]), assistant("b", [step({input: 4, cache: {read: 1, write: 0}})])];
  assert.deepEqual(usageOf(missingOutput), {input_tokens: 9, input_tokens_include_cache: true});
  assert.deepEqual(usageOf([userPrompt, assistant("a", [step({input: 4, output: 6, reasoning: 0})])]), {output_tokens: 6});
});

// Kilo Code replaces every count a provider omitted with 0 and keeps an omitted total undefined.
const zeroFilled = call(0, 0);

test("reports no Kilo Code usage when a finished call reported no counts", () => {
  assert.equal(usageOf([userPrompt, assistant("a", [step(zeroFilled)])]), undefined);
  assert.equal(usageOf([userPrompt, assistant("a", [step({...zeroFilled, total: 0})])]), undefined);
  assert.equal(usageOf([userPrompt, assistant("a", [step(call(4, 6, 2))]), assistant("b", [step(zeroFilled)])]), undefined);
});

test("leaves a Kilo Code count unknown when a call zero-filled it", () => {
  const reported = assistant("a", [step(call(10, 15, 0, 0, 5))]);
  const zeroOutput = call(7, 0, 2);
  assert.deepEqual(usageOf([userPrompt, reported, assistant("b", [step(zeroOutput)])]), {input_tokens: 19, input_tokens_include_cache: true});
  // Kilo Code's total is the AI SDK's input plus output with the omitted output counted as 0.
  assert.deepEqual(usageOf([userPrompt, reported, assistant("b", [step({...zeroOutput, total: 9})])]), {input_tokens: 19, input_tokens_include_cache: true});
});

test("leaves Kilo Code cached input unknown when a call's cache read was zero-filled", () => {
  // An earlier call read 20 tokens from the cache, then a call omitted its cache details, which Kilo Code stores as 0.
  const earlier = [user("user-0"), assistant("earlier", [step(call(100, 10, 20))], "user-0")];
  assert.deepEqual(usageOf([...earlier, userPrompt, assistant("a", [step(call(10, 20))])], "user-1", ["user-0", "user-1"]), {input_tokens: 10, input_tokens_include_cache: true, output_tokens: 20, total_tokens: 30});
  // A cache write does not show that the call reported its cache reads.
  assert.deepEqual(usageOf([userPrompt, assistant("a", [step(call(10, 20, 0, 500))])]), {input_tokens: 510, input_tokens_include_cache: true, output_tokens: 20, total_tokens: 530});
  // Every call read from the cache, so the cached input is known.
  assert.deepEqual(usageOf([userPrompt, assistant("a", [step(call(10, 20, 30))]), assistant("b", [step(call(5, 5, 40))])]), {input_tokens: 85, cached_input_tokens: 70, input_tokens_include_cache: true, output_tokens: 25, total_tokens: 110});
});

// Step tokens recorded by a real Kilo Code 7.7.12 server for a provider usage report with one side missing.
test("does not take Kilo Code's total when the provider omitted one side", () => {
  assert.deepEqual(usageOf([userPrompt, assistant("a", [step({total: 100, ...call(100, 0)})])]), {input_tokens: 100, input_tokens_include_cache: true});
  assert.deepEqual(usageOf([userPrompt, assistant("a", [step({total: 7, ...call(0, 7)})])]), {output_tokens: 7});
  assert.equal(usageOf([userPrompt, assistant("a", [step({total: 0, ...call(0, 0)})])]), undefined);
});

test("leaves a Kilo Code sum unknown when it passes the safe integer range", () => {
  const half = 2 ** 52;
  const big = {total: half + 1, ...call(half, 1)};
  assert.deepEqual(usageOf([userPrompt, assistant("a", [step(big)]), assistant("b", [step(big)])]), {output_tokens: 2});
});

const sessionInput = (workspace) => ({
  agentName: "reviewer",
  baseDir: workspace,
  config: {models: {default: {provider: "nvidia", model: "nemotron", api_key_env: "MODEL_KEY"}}},
  runtimeContext: {runtime_id: "r", invocation_id: "i", request_id: "q", artifacts: {}, environment: {environment_id: "e", provider: "local", ownership: "caller_owned", control_location: "external_control", workspace, env: {MODEL_KEY: "secret"}}},
});

// A Kilo Code session fake that lists messages oldest first, or only the newest `limit` of them.
function sessionClient(history, respond, listings = [], children = new Map()) {
  return {
    app: {skills: async () => ({data: []})},
    mcp: {status: async () => ({data: {}})},
    session: {
      create: async () => ({data: {id: "session-1"}}),
      messages: async (parameters) => {
        const result = listings.shift();
        if (result instanceof Error) throw result;
        const source = parameters.sessionID === "session-1" ? history : children.get(parameters.sessionID) ?? [];
        const listed = parameters.limit === undefined ? source : source.slice(-parameters.limit);
        return {data: listed.map((message) => structuredClone(message))};
      },
      prompt: async () => ({data: respond()}),
      delete: async () => {},
    },
  };
}

async function withSession(client, check) {
  const workspace = await mkdtemp(join(tmpdir(), "kilo-adapter-test-"));
  const factory = new KiloSdkSessionFactory(async () => () => client, async () => ({url: "http://127.0.0.1:4123", close: async () => {}}));
  try {
    const handle = await factory.create(sessionInput(workspace));
    await check(handle);
    await handle.stop();
  } finally {
    await rm(workspace, {recursive: true, force: true});
  }
}

test("reports each Kilo Code prompt's own usage on a reused session", async () => {
  const history = [];
  let turn = 0;
  const client = sessionClient(history, () => {
    turn += 1;
    history.push(user(`user-${turn}`));
    history.push(assistant(`tool-${turn}`, [step({total: 12, ...call(10, 2)})], `user-${turn}`));
    const final = assistant(`final-${turn}`, [step({total: 4, ...call(3, 1)}), {type: "text", text: `answer ${turn}`}], `user-${turn}`);
    history.push(final);
    return final;
  });
  await withSession(client, async (handle) => {
    const expected = {input_tokens: 13, input_tokens_include_cache: true, output_tokens: 3, total_tokens: 16};
    assert.deepEqual(await handle.prompt("one"), {text: "answer 1", usage: expected});
    assert.deepEqual(await handle.prompt("two"), {text: "answer 2", usage: expected});
  });
});

test("reads a Kilo Code foreground subagent's child session for the prompt's usage", async () => {
  const history = [];
  const children = new Map([["child-1", [
    {info: {id: "c-user", role: "user", time: {created: 5001}}, parts: [{type: "text", text: "sub task"}]},
    {info: {id: "c-a", role: "assistant", parentID: "c-user", time: {created: 5002, completed: 5003}}, parts: [step(call(5000, 50))]},
  ]]]);
  const client = sessionClient(history, () => {
    history.push(user("user-1"));
    const final = assistant("final", [taskCall("child-1", 5000), step(call(1500, 15)), {type: "text", text: "answer"}]);
    history.push(final);
    return final;
  }, [], children);
  await withSession(client, async (handle) => {
    assert.deepEqual(await handle.prompt("one"), {text: "answer", usage: {input_tokens: 6500, input_tokens_include_cache: true, output_tokens: 65, total_tokens: 6565}});
  });
});

test("keeps a late Kilo Code background reply out of the next prompt's usage", async () => {
  const history = [];
  let turn = 0;
  const client = sessionClient(history, () => {
    turn += 1;
    history.push(user(`user-${turn}`));
    if (turn === 2) {
      // The background subagent from the first prompt finishes, and Kilo Code prompts the session with its result.
      history.push(backgroundResult("result", "child-1"));
      history.push(assistant("background-reply", [step(call(100, 10, 5))], "result"));
    }
    const parts = turn === 1 ? [taskCall("child-1", 5000, true)] : [];
    const final = assistant(`final-${turn}`, [...parts, step(call(3, 1, 1)), {type: "text", text: `answer ${turn}`}], `user-${turn}`);
    history.push(final);
    return final;
  });
  await withSession(client, async (handle) => {
    assert.deepEqual(await handle.prompt("one"), {text: "answer 1"});
    assert.deepEqual(await handle.prompt("two"), {
      text: "answer 2",
      usage: {input_tokens: 4, cached_input_tokens: 1, input_tokens_include_cache: true, output_tokens: 1, total_tokens: 5},
    });
  });
});

test("returns the Kilo Code answer without usage when listing messages fails after the prompt", async () => {
  const history = [];
  const client = sessionClient(history, () => {
    history.push(user("user-1"));
    const final = assistant("final", [step(call(3, 1)), {type: "text", text: "answer"}]);
    history.push(final);
    return final;
  }, [undefined, new Error("listing failed")]);
  await withSession(client, async (handle) => {
    assert.deepEqual(await handle.prompt("one"), {text: "answer"});
  });
});

test("reports a Kilo Code model error without usage when the call never finished", async () => {
  const history = [];
  const client = sessionClient(history, () => {
    history.push(user("user-1"));
    const failed = {info: {id: "failed", role: "assistant", parentID: "user-1", cost: 0, tokens: call(0, 0), time: {created: 1, completed: 2}, error: {name: "APIError", data: {message: "secret"}}}, parts: []};
    history.push(failed);
    return failed;
  });
  await withSession(client, async (handle) => {
    assert.deepEqual(await handle.prompt("one"), {error: true});
  });
});

test("marks upstream assistant errors without exposing their content", () => {
  assert.deepEqual(extractPromptOutcome({info: {error: {name: "ProviderError", data: {message: "secret"}}}, parts: []}), {error: true});
});

test("distinguishes a missing Kilo executable from other launch failures", () => {
  assert.equal(classifySpawnError(Object.assign(new Error("missing"), {code: "ENOENT"})).code, "kilo_harness_unavailable");
  assert.equal(classifySpawnError(Object.assign(new Error("denied"), {code: "EACCES"})).code, "kilo_start_failed");
  assert.equal(classifySpawnError(new Error("unknown")).code, "kilo_start_failed");
});

test("waits for process exit after escalating shutdown", async () => {
  const child = new EventEmitter();
  child.pid = 42;
  child.exitCode = null;
  child.signalCode = null;
  child.kill = (signal) => {
    child.signals.push(signal);
    if (signal === "SIGKILL") {
      setImmediate(() => {
        child.signalCode = signal;
        child.emit("exit", null, signal);
      });
    }
    return true;
  };
  child.signals = [];

  await closeChild(child, 1);
  assert.deepEqual(child.signals, ["SIGTERM", "SIGKILL"]);
  assert.equal(child.signalCode, "SIGKILL");
});

test("waits for server cleanup before rejecting startup", async () => {
  const child = new EventEmitter();
  child.pid = 42;
  child.exitCode = null;
  child.signalCode = null;
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  child.signals = [];
  child.kill = (signal) => {
    child.signals.push(signal);
    return true;
  };
  let spawned;
  const didSpawn = new Promise((resolveSpawn) => { spawned = resolveSpawn; });
  let rejected = false;
  const result = startKiloServer(process.cwd(), process.env, "kilo", () => {
    spawned();
    return child;
  }, async () => 4123).catch((error) => {
    rejected = true;
    return error;
  });

  await didSpawn;
  child.emit("error", Object.assign(new Error("denied"), {code: "EACCES"}));
  await new Promise((resolveImmediate) => setImmediate(resolveImmediate));
  assert.equal(rejected, false);
  assert.deepEqual(child.signals, ["SIGTERM"]);

  child.signalCode = "SIGTERM";
  child.emit("exit", null, "SIGTERM");
  const error = await result;
  assert.equal(rejected, true);
  assert.equal(error.code, "kilo_start_failed");
});

test("projects normalized configuration into an isolated Kilo Code server", async () => {
  const workspace = await mkdtemp(join(tmpdir(), "kilo-adapter-test-"));
  const skill = join(workspace, "review-skill");
  await mkdir(skill);
  await writeFile(join(skill, "SKILL.md"), "# Review\n", "utf8");
  let serverEnvironment;
  let createParameters;
  let deleteParameters;
  let deleteSignal;
  const client = {
    app: {skills: async () => ({data: [{name: "review", location: join(skill, "SKILL.md"), content: "# Review"}]})},
    mcp: {status: async () => ({data: {local: {status: "connected"}}})},
    session: {
      create: async (parameters) => { createParameters = parameters; return {data: {id: "session-1"}}; },
      prompt: async () => ({data: {info: {}, parts: [{type: "text", text: "done"}]}}),
      delete: async (parameters, options) => { deleteParameters = parameters; deleteSignal = options.signal; },
    },
  };
  const factory = new KiloSdkSessionFactory(
    async () => () => client,
    async (_workspace, environment) => { serverEnvironment = environment; return {url: "http://127.0.0.1:4123", close: async () => {}}; },
  );
  try {
    const handle = await factory.create({
      agentName: "reviewer",
      baseDir: workspace,
      config: {
        models: {default: {provider: "nvidia", model: "nemotron", api_key_env: "MODEL_KEY", base_url: "https://provider.example/v1", temperature: 0.1, top_p: 0.8}},
        instructions: {system: {content: "Review carefully.", mode: "replace"}},
        runtime: {max_turns: 9},
        tools: {enabled: ["read", "grep"], blocked: ["bash"]},
        skills: {paths: ["review-skill"]},
        mcp: {servers: {local: {transport: "stdio", url: "mcp-server", args: ["--stdio"], env: {MODE: "test"}}}},
      },
      runtimeContext: {runtime_id: "r", invocation_id: "i", request_id: "q", artifacts: {}, environment: {environment_id: "e", provider: "local", ownership: "caller_owned", control_location: "external_control", workspace, env: {MODEL_KEY: "secret"}}},
    });
    const projected = JSON.parse(serverEnvironment.KILO_CONFIG_CONTENT);
    assert.deepEqual(projected.provider.nvidia, {npm: "@ai-sdk/openai-compatible", options: {apiKey: "{env:MODEL_KEY}", baseURL: "https://provider.example/v1"}, models: {nemotron: {}}});
    assert.deepEqual(projected.permission, {question: "deny", external_directory: "deny", doom_loop: "deny", read: {"*": "allow", "*.env": "deny", "*.env.*": "deny", "*.env.example": "allow"}, "*": "deny", grep: "allow", bash: "deny"});
    assert.deepEqual(projected.agent, {build: {model: "nvidia/nemotron", prompt: "Review carefully.", steps: 9, temperature: 0.1, top_p: 0.8}});
    assert.deepEqual(projected.skills.paths, [skill]);
    assert.deepEqual(projected.mcp.local, {type: "local", command: ["mcp-server", "--stdio"], environment: {MODE: "test"}});
    assert.equal(serverEnvironment.KILO_DISABLE_PROJECT_CONFIG, "1");
    assert.equal(serverEnvironment.MODEL_KEY, "secret");
    assert.equal(serverEnvironment.NVIDIA_API_KEY, undefined);
    assert.deepEqual(createParameters, {directory: workspace, title: "NeMo Fabric", agent: "build", model: {providerID: "nvidia", id: "nemotron"}});
    await handle.stop();
    assert.deepEqual(deleteParameters, {sessionID: "session-1", directory: workspace});
    assert.ok(deleteSignal instanceof AbortSignal);
  } finally {
    await rm(workspace, {recursive: true, force: true});
  }
});

test("denies interactive permissions when normalized tools are omitted", async () => {
  const workspace = await mkdtemp(join(tmpdir(), "kilo-adapter-test-"));
  let serverEnvironment;
  const client = {
    app: {skills: async () => ({data: []})},
    mcp: {status: async () => ({data: {}})},
    session: {
      create: async () => ({data: {id: "session-1"}}),
      prompt: async () => ({data: {info: {}, parts: []}}),
      delete: async () => {},
    },
  };
  const factory = new KiloSdkSessionFactory(
    async () => () => client,
    async (_workspace, environment) => {
      serverEnvironment = environment;
      return {url: "http://127.0.0.1:4123", close: async () => {}};
    },
  );
  try {
    const handle = await factory.create({
      agentName: "reviewer",
      baseDir: workspace,
      config: {
        models: {default: {provider: "nvidia", model: "nemotron", api_key_env: "MODEL_KEY"}},
      },
      runtimeContext: {
        runtime_id: "r",
        invocation_id: "i",
        request_id: "q",
        artifacts: {},
        environment: {
          environment_id: "e",
          provider: "local",
          ownership: "caller_owned",
          control_location: "external_control",
          workspace,
          env: {MODEL_KEY: "secret"},
        },
      },
    });
    const projected = JSON.parse(serverEnvironment.KILO_CONFIG_CONTENT);
    assert.deepEqual(projected.permission, {
      question: "deny",
      external_directory: "deny",
      doom_loop: "deny",
      read: {"*": "allow", "*.env": "deny", "*.env.*": "deny", "*.env.example": "allow"},
    });
    assert.deepEqual(projected.agent, {build: {model: "nvidia/nemotron"}});
    await handle.stop();
  } finally {
    await rm(workspace, {recursive: true, force: true});
  }
});

test("aborts the Kilo session when a prompt reaches its deadline", async () => {
  const workspace = await mkdtemp(join(tmpdir(), "kilo-adapter-test-"));
  let promptSignal;
  let abortParameters;
  let abortSignal;
  const client = {
    app: {skills: async () => ({data: []})},
    mcp: {status: async () => ({data: {}})},
    session: {
      create: async () => ({data: {id: "session-timeout"}}),
      prompt: async (_parameters, options) => {
        promptSignal = options.signal;
        return await new Promise(() => {});
      },
      abort: async (parameters, options) => {
        abortParameters = parameters;
        abortSignal = options.signal;
      },
      delete: async () => {},
    },
  };
  const factory = new KiloSdkSessionFactory(
    async () => () => client,
    async () => ({url: "http://127.0.0.1:4123", close: async () => {}}),
    0,
  );
  try {
    const handle = await factory.create({
      agentName: "reviewer",
      baseDir: workspace,
      config: {models: {default: {provider: "nvidia", model: "nemotron", api_key_env: "MODEL_KEY"}}},
      runtimeContext: {
        runtime_id: "r",
        invocation_id: "i",
        request_id: "q",
        artifacts: {},
        environment: {
          environment_id: "e",
          provider: "local",
          ownership: "caller_owned",
          control_location: "external_control",
          workspace,
          env: {MODEL_KEY: "secret"},
        },
      },
    });
    await assert.rejects(handle.prompt("wait forever"), (error) => error.code === "kilo_prompt_timeout");
    assert.equal(promptSignal.aborted, true);
    assert.deepEqual(abortParameters, {sessionID: "session-timeout", directory: workspace});
    assert.ok(abortSignal instanceof AbortSignal);
    await handle.stop();
  } finally {
    await rm(workspace, {recursive: true, force: true});
  }
});
