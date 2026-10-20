// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, realpath, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { extractOpenCodePromptOutcome, loadOpenCodeSdk, OpenCodeSdkSessionFactory } from "../dist/opencode-sdk.js";

function startInput() {
  return {
    agentName: "opencode-sdk-test",
    baseDir: "/fallback",
    config: {
      models: {
        default: { provider: "nvidia", model: "nvidia/nemotron-3.5-lightning-30b-a3b", api_key_env: "OPENCODE_TEST_KEY" },
      },
    },
    runtimeContext: {
      artifacts: {},
      environment: {
        control_location: "external_control",
        env: { OPENCODE_TEST_KEY: "test-secret" },
        environment_id: "environment-1",
        ownership: "caller_owned",
        provider: "local",
        workspace: "/workspace",
      },
      invocation_id: "start",
      request_id: "request-start",
      runtime_id: "runtime-1",
    },
  };
}

test("reports an unresolvable OpenCode peer as unavailable before attempting to load it", async () => {
  const resolved = [];
  await assert.rejects(
    loadOpenCodeSdk(
      (specifier) => {
        resolved.push(specifier);
        throw new Error("package is not installed");
      },
      async () => {
        throw new Error("should not load an unresolved package");
      },
    ),
    (error) => error.code === "opencode_harness_unavailable",
  );
  assert.deepEqual(resolved, ["@opencode/core/config"]);
});

test("reports an installed OpenCode peer with a missing transitive module as a load failure", async () => {
  const loaded = [];
  await assert.rejects(
    loadOpenCodeSdk(
      (specifier) => `resolved:${specifier}`,
      async (specifier) => {
        loaded.push(specifier);
        if (specifier === "resolved:@opencode/sdk") {
          const error = new Error("Cannot find @effect/platform-node");
          Object.assign(error, { code: "ERR_MODULE_NOT_FOUND" });
          throw error;
        }
        return {};
      },
    ),
    (error) => error.code === "opencode_harness_load_failed",
  );
  assert.deepEqual(loaded, ["resolved:@opencode/core/config", "resolved:@opencode/sdk"]);
});

test("rejects a missing configured credential before loading the OpenCode SDK", async () => {
  const input = startInput();
  input.runtimeContext.environment.env = {};
  const previous = process.env.OPENCODE_TEST_KEY;
  delete process.env.OPENCODE_TEST_KEY;
  try {
    const factory = new OpenCodeSdkSessionFactory(async () => {
      throw new Error("the SDK must not load without a credential");
    });

    await assert.rejects(factory.create(input), (error) => error.code === "opencode_credential_missing");
  } finally {
    if (previous === undefined) {
      delete process.env.OPENCODE_TEST_KEY;
    } else {
      process.env.OPENCODE_TEST_KEY = previous;
    }
  }
});

test("does not resolve inherited names as OpenCode credentials", async () => {
  const input = startInput();
  input.config.models.default.api_key_env = "constructor";
  input.runtimeContext.environment.env = {};
  const factory = new OpenCodeSdkSessionFactory(async () => {
    throw new Error("the SDK must not load without a credential");
  });

  await assert.rejects(factory.create(input), (error) => error.code === "opencode_credential_missing");
});

test("retains an explicitly configured __proto__ credential", async () => {
  const input = startInput();
  input.config.models.default.api_key_env = "__proto__";
  input.runtimeContext.environment.env = JSON.parse('{"__proto__":"proto-secret"}');
  const hadPrevious = Object.hasOwn(process.env, "__proto__");
  const previous = process.env.__proto__;
  delete process.env.__proto__;
  try {
    let observedCredential;
    const factory = new OpenCodeSdkSessionFactory(async () => ({
      OpenCode: {
        async create() {
          observedCredential = process.env.__proto__;
          return {
            sessions: {
              async create() { return { id: "session-prototype-credential" }; },
              async remove() {},
            },
            async close() {},
          };
        },
      },
    }));

    const handle = await factory.create(input);
    assert.equal(observedCredential, "proto-secret");
    await handle.stop();
    assert.equal(Object.hasOwn(process.env, "__proto__"), false);
  } finally {
    if (!hadPrevious) {
      delete process.env.__proto__;
    } else {
      process.env.__proto__ = previous;
    }
  }
});

test("rejects append system instructions before loading the OpenCode SDK", async () => {
  const input = startInput();
  input.config.instructions = {
    system: { content: "Append to the OpenCode defaults", mode: "append" },
  };
  const factory = new OpenCodeSdkSessionFactory(async () => {
    throw new Error("the SDK must not load for an unsupported instruction mode");
  });

  await assert.rejects(
    factory.create(input),
    (error) =>
      error.code === "unsupported_system_instruction_mode" &&
      error.metadata.field === "instructions.system.mode",
  );
});

test("passes replace system instructions through OpenCode's build agent", async () => {
  let capturedConfig;
  const factory = new OpenCodeSdkSessionFactory(
    async () => ({
      OpenCode: {
        async create(options) {
          capturedConfig = options.config.content;
          return {
            sessions: {
              async create() { return { id: "session-system-instruction" }; },
              async remove() {},
            },
            async close() {},
          };
        },
      },
    }),
  );
  const input = startInput();
  input.config.instructions = {
    system: { content: "Follow Fabric's system policy.", mode: "replace" },
  };

  const handle = await factory.create(input);

  assert.deepEqual(JSON.parse(capturedConfig), {
    providers: {
      nvidia: {
        settings: { apiKey: "{env:OPENCODE_TEST_KEY}" },
      },
    },
    agents: {
      build: { system: "Follow Fabric's system policy." },
    },
  });
  await handle.stop();
});

test("encodes system instructions literally before OpenCode parses configuration variables", async () => {
  let capturedConfig;
  const factory = new OpenCodeSdkSessionFactory(
    async () => ({
      OpenCode: {
        async create(options) {
          capturedConfig = options.config.content;
          return {
            sessions: {
              async create() { return { id: "session-literal-system-instruction" }; },
              async remove() {},
            },
            async close() {},
          };
        },
      },
    }),
  );
  const input = startInput();
  const instruction = "Preserve {env:OPENCODE_TEST_KEY}, {file:/tmp/fabric-secret}, $&, $`, and $' literally.";
  input.config.instructions = { system: { content: instruction, mode: "replace" } };

  const handle = await factory.create(input);

  assert.ok(capturedConfig.includes("\\u007benv:OPENCODE_TEST_KEY}"));
  assert.ok(capturedConfig.includes("\\u007bfile:/tmp/fabric-secret}"));
  assert.equal(JSON.parse(capturedConfig).agents.build.system, instruction);
  await handle.stop();
});

test("configures validated Fabric skill directories for OpenCode", async () => {
  const baseDir = await mkdtemp(join(tmpdir(), "fabric-opencode-skills-"));
  const skillDirectory = join(baseDir, "skills", "review");
  let capturedConfig;
  let skillListCalls = 0;
  try {
    await mkdir(skillDirectory, { recursive: true });
    await writeFile(
      join(skillDirectory, "SKILL.md"),
      "---\nname: review\ndescription: FABRIC_SKILL_SENTINEL\n---\nReview the change.\n",
      "utf8",
    );
    const factory = new OpenCodeSdkSessionFactory(
      async () => ({
        OpenCode: {
          async create(options) {
            capturedConfig = options.config.content;
            const [stagedSkillDirectory] = JSON.parse(capturedConfig).skills;
            return {
              skill: {
                async list() {
                  skillListCalls += 1;
                  return {
                    data: skillListCalls === 1
                      ? []
                      : [{ name: "review", path: await realpath(join(stagedSkillDirectory, "SKILL.md")) }],
                  };
                },
              },
              sessions: {
                async create() { return { id: "session-skills" }; },
                async remove() {},
              },
              async close() {},
            };
          },
        },
      }),
    );
    const input = startInput();
    input.baseDir = baseDir;
    input.config.skills = { paths: ["skills/review"] };

    const handle = await factory.create(input);

    const parsedConfig = JSON.parse(capturedConfig);
    assert.deepEqual(parsedConfig.providers, {
      nvidia: {
        settings: { apiKey: "{env:OPENCODE_TEST_KEY}" },
      },
    });
    assert.equal(parsedConfig.skills.length, 1);
    assert.notEqual(parsedConfig.skills[0], await realpath(skillDirectory));
    assert.equal(
      await readFile(join(parsedConfig.skills[0], "SKILL.md"), "utf8"),
      "---\nname: review\ndescription: FABRIC_SKILL_SENTINEL\n---\nReview the change.\n",
    );
    assert.equal(skillListCalls, 2);
    await handle.stop();
    await assert.rejects(stat(parsedConfig.skills[0]));
  } finally {
    await rm(baseDir, { recursive: true, force: true });
  }
});

test("aborts a stalled skill discovery request at the discovery deadline", async () => {
  const originalTimeout = AbortSignal.timeout;
  AbortSignal.timeout = () => {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 0);
    return controller.signal;
  };
  const baseDir = await mkdtemp(join(tmpdir(), "fabric-opencode-stalled-skill-"));
  const skillDirectory = join(baseDir, "skills", "review");
  let stagedSkillDirectory;
  let closed = false;
  try {
    await mkdir(skillDirectory, { recursive: true });
    await writeFile(join(skillDirectory, "SKILL.md"), "---\nname: review\n---\nReview the change.\n", "utf8");
    const factory = new OpenCodeSdkSessionFactory(async () => ({
      OpenCode: {
        async create(options) {
          [stagedSkillDirectory] = JSON.parse(options.config.content).skills;
          return {
            skill: {
              async list(_input, requestOptions) {
                if (requestOptions?.signal === undefined) {
                  throw new Error("OpenCode request options did not include an abort signal");
                }
                return new Promise((_resolve, reject) => {
                  requestOptions.signal.addEventListener("abort", () => reject(requestOptions.signal.reason), {
                    once: true,
                  });
                });
              },
            },
            sessions: {
              async create() { throw new Error("a stalled skill request must fail before session creation"); },
              async remove() {},
            },
            async close() { closed = true; },
          };
        },
      },
    }));
    const input = startInput();
    input.baseDir = baseDir;
    input.config.skills = { paths: ["skills/review"] };

    await assert.rejects(factory.create(input), (error) => error.code === "opencode_skill_load_failed");
    assert.equal(closed, true);
    await assert.rejects(stat(stagedSkillDirectory));
  } finally {
    AbortSignal.timeout = originalTimeout;
    await rm(baseDir, { recursive: true, force: true });
  }
});

test("rejects a missing Fabric skill directory before loading the OpenCode SDK", async () => {
  const input = startInput();
  input.config.skills = { paths: ["skills/missing"] };
  const factory = new OpenCodeSdkSessionFactory(async () => {
    throw new Error("the SDK must not load for a missing skill");
  });

  await assert.rejects(factory.create(input), (error) => error.code === "opencode_skill_not_found");
});

test("rejects a configured skill that OpenCode did not load", async () => {
  const baseDir = await mkdtemp(join(tmpdir(), "fabric-opencode-malformed-skill-"));
  const skillDirectory = join(baseDir, "skills", "malformed");
  try {
    await mkdir(skillDirectory, { recursive: true });
    await writeFile(join(skillDirectory, "SKILL.md"), "---\nname: [not a string]\n---\n", "utf8");
    const factory = new OpenCodeSdkSessionFactory(async () => ({
      OpenCode: {
        async create() {
          return {
            skill: { async list() { return { data: [] }; } },
            sessions: {
              async create() { throw new Error("a malformed skill must fail before session creation"); },
              async remove() {},
            },
            async close() {},
          };
        },
      },
    }));
    const input = startInput();
    input.baseDir = baseDir;
    input.config.skills = { paths: ["skills/malformed"] };

    await assert.rejects(factory.create(input), (error) => error.code === "opencode_skill_load_failed");
  } finally {
    await rm(baseDir, { recursive: true, force: true });
  }
});

test("rejects additional OpenCode-discoverable files in a configured skill directory", async () => {
  const baseDir = await mkdtemp(join(tmpdir(), "fabric-opencode-unexpected-skills-"));
  const skillDirectory = join(baseDir, "skills", "review");
  const nestedDirectory = join(skillDirectory, "examples", "experimental");
  try {
    await mkdir(nestedDirectory, { recursive: true });
    await Promise.all([
      writeFile(join(skillDirectory, "SKILL.md"), "---\nname: review\n---\nReview the change.\n", "utf8"),
      writeFile(join(skillDirectory, "README.md"), "---\nname: readme\n---\nUnexpected skill.\n", "utf8"),
      writeFile(join(nestedDirectory, "SKILL.md"), "---\nname: experimental\n---\nUnexpected skill.\n", "utf8"),
    ]);
    const factory = new OpenCodeSdkSessionFactory(async () => {
      throw new Error("unexpected skills must fail before loading the SDK");
    });
    const input = startInput();
    input.baseDir = baseDir;
    input.config.skills = { paths: ["skills/review"] };

    await assert.rejects(factory.create(input), (error) => error.code === "opencode_skill_unexpected");
  } finally {
    await rm(baseDir, { recursive: true, force: true });
  }
});

test("isolates OpenCode discovery from skills added to the source after startup begins", async () => {
  const baseDir = await mkdtemp(join(tmpdir(), "fabric-opencode-snapshot-skills-"));
  const skillDirectory = join(baseDir, "skills", "review");
  const lateSkill = join(skillDirectory, "examples", "late", "SKILL.md");
  let stagedSkillDirectory;
  let skillListCalls = 0;
  try {
    await mkdir(skillDirectory, { recursive: true });
    await writeFile(join(skillDirectory, "SKILL.md"), "---\nname: review\n---\nReview the change.\n", "utf8");
    const factory = new OpenCodeSdkSessionFactory(async () => ({
      OpenCode: {
        async create(options) {
          [stagedSkillDirectory] = JSON.parse(options.config.content).skills;
          await mkdir(join(skillDirectory, "examples", "late"), { recursive: true });
          await writeFile(lateSkill, "---\nname: late\n---\nLate skill.\n", "utf8");
          return {
            skill: {
              async list() {
                skillListCalls += 1;
                return {
                  data: skillListCalls === 1
                    ? []
                    : [{ name: "review", path: await realpath(join(stagedSkillDirectory, "SKILL.md")) }],
                };
              },
            },
            sessions: {
              async create() { return { id: "session-snapshot-skill" }; },
              async remove() {},
            },
            async close() {},
          };
        },
      },
    }));
    const input = startInput();
    input.baseDir = baseDir;
    input.config.skills = { paths: ["skills/review"] };

    const handle = await factory.create(input);

    assert.equal(skillListCalls, 2);
    await assert.rejects(stat(join(stagedSkillDirectory, "examples", "late", "SKILL.md")));
    await handle.stop();
    await assert.rejects(stat(stagedSkillDirectory));
  } finally {
    await rm(baseDir, { recursive: true, force: true });
  }
});

test("allows OpenCode built-in skills outside configured Fabric skill directories", async () => {
  const baseDir = await mkdtemp(join(tmpdir(), "fabric-opencode-built-in-skills-"));
  const skillDirectory = join(baseDir, "skills", "review");
  try {
    await mkdir(skillDirectory, { recursive: true });
    await writeFile(join(skillDirectory, "SKILL.md"), "---\nname: review\n---\nReview the change.\n", "utf8");
    const factory = new OpenCodeSdkSessionFactory(async () => ({
      OpenCode: {
        async create(options) {
          const [stagedSkillDirectory] = JSON.parse(options.config.content).skills;
          return {
            skill: {
              async list() {
                return {
                  data: [
                    { name: "review", path: await realpath(join(stagedSkillDirectory, "SKILL.md")) },
                    { name: "opencode", path: "/opencode-core/skills/opencode/SKILL.md" },
                  ],
                };
              },
            },
            sessions: {
              async create() { return { id: "session-built-in-skill" }; },
              async remove() {},
            },
            async close() {},
          };
        },
      },
    }));
    const input = startInput();
    input.baseDir = baseDir;
    input.config.skills = { paths: ["skills/review"] };

    const handle = await factory.create(input);

    await handle.stop();
  } finally {
    await rm(baseDir, { recursive: true, force: true });
  }
});

test("rejects configured skills with duplicate loaded names", async () => {
  const baseDir = await mkdtemp(join(tmpdir(), "fabric-opencode-duplicate-skills-"));
  const first = join(baseDir, "skills", "first");
  const second = join(baseDir, "skills", "second");
  try {
    await Promise.all([mkdir(first, { recursive: true }), mkdir(second, { recursive: true })]);
    await Promise.all([
      writeFile(join(first, "SKILL.md"), "---\nname: duplicate\n---\nFirst skill.\n", "utf8"),
      writeFile(join(second, "SKILL.md"), "---\nname: duplicate\n---\nSecond skill.\n", "utf8"),
    ]);
    const factory = new OpenCodeSdkSessionFactory(async () => ({
      OpenCode: {
        async create(options) {
          const stagedSkillDirectories = JSON.parse(options.config.content).skills;
          return {
            skill: {
              async list() {
                return {
                  data: [
                    { name: "duplicate", path: await realpath(join(stagedSkillDirectories[0], "SKILL.md")) },
                    { name: "duplicate", path: await realpath(join(stagedSkillDirectories[1], "SKILL.md")) },
                  ],
                };
              },
            },
            sessions: {
              async create() { throw new Error("duplicate skills must fail before session creation"); },
              async remove() {},
            },
            async close() {},
          };
        },
      },
    }));
    const input = startInput();
    input.baseDir = baseDir;
    input.config.skills = { paths: ["skills/first", "skills/second"] };

    await assert.rejects(factory.create(input), (error) => error.code === "opencode_skill_name_duplicate");
  } finally {
    await rm(baseDir, { recursive: true, force: true });
  }
});

test("resolves configured skill paths literally before staging them for OpenCode", async () => {
  const baseDir = await mkdtemp(join(tmpdir(), "fabric-opencode-literal-skills-"));
  const skillDirectory = join(baseDir, "skills", "{env:FABRIC_SKILL_SEGMENT}-{file:secret}");
  let capturedConfig;
  try {
    await mkdir(skillDirectory, { recursive: true });
    await writeFile(join(skillDirectory, "SKILL.md"), "---\nname: literal\n---\nLiteral skill.\n", "utf8");
    const factory = new OpenCodeSdkSessionFactory(async () => ({
      OpenCode: {
        async create(options) {
          capturedConfig = options.config.content;
          const [stagedSkillDirectory] = JSON.parse(capturedConfig).skills;
          return {
            skill: {
              async list() {
                return { data: [{ name: "literal", path: await realpath(join(stagedSkillDirectory, "SKILL.md")) }] };
              },
            },
            sessions: {
              async create() { return { id: "session-literal-skill" }; },
              async remove() {},
            },
            async close() {},
          };
        },
      },
    }));
    const input = startInput();
    input.baseDir = baseDir;
    input.config.skills = { paths: ["skills/{env:FABRIC_SKILL_SEGMENT}-{file:secret}"] };

    const handle = await factory.create(input);

    const [stagedSkillDirectory] = JSON.parse(capturedConfig).skills;
    assert.notEqual(stagedSkillDirectory, await realpath(skillDirectory));
    assert.equal(await readFile(join(stagedSkillDirectory, "SKILL.md"), "utf8"), "---\nname: literal\n---\nLiteral skill.\n");
    await handle.stop();
  } finally {
    await rm(baseDir, { recursive: true, force: true });
  }
});

test("configures Fabric stdio and streamable-HTTP MCP servers for OpenCode", async () => {
  let capturedConfig;
  let readinessSignal;
  const factory = new OpenCodeSdkSessionFactory(
    async () => ({
      OpenCode: {
        async create(options) {
          capturedConfig = options.config.content;
          return {
            mcp: {
              async list(_input, requestOptions) {
                readinessSignal = requestOptions.signal;
                return {
                  data: [
                    { name: "local", status: { status: "connected" } },
                    { name: "remote", status: { status: "connected" } },
                  ],
                };
              },
            },
            sessions: {
              async create() { return { id: "session-mcp" }; },
              async remove() {},
            },
            async close() {},
          };
        },
      },
    }),
  );
  const input = startInput();
  input.runtimeContext.environment.env.MCP_ACCESS_TOKEN = "mcp-test-token";
  input.config.mcp = {
    servers: {
      local: {
        transport: "stdio",
        url: "node",
        args: ["mcp-server.mjs"],
        env: { MCP_TOKEN: "declared-token" },
      },
      remote: {
        transport: "streamable-http",
        url: "https://mcp.example.test",
        custom_headers: { Authorization: "Bearer ${MCP_ACCESS_TOKEN}" },
      },
    },
  };

  const handle = await factory.create(input);

  assert.deepEqual(JSON.parse(capturedConfig), {
    providers: {
      nvidia: {
        settings: { apiKey: "{env:OPENCODE_TEST_KEY}" },
      },
    },
    mcp: {
      servers: {
        local: {
          type: "local",
          command: ["node", "mcp-server.mjs"],
          environment: { MCP_TOKEN: "declared-token" },
        },
        remote: {
          type: "remote",
          url: "https://mcp.example.test",
          headers: { Authorization: "Bearer mcp-test-token" },
          oauth: false,
        },
      },
    },
  });
  assert.ok(readinessSignal instanceof AbortSignal);
  await handle.stop();
});

test("encodes every Fabric-supplied MCP value literally before OpenCode parses configuration variables", async () => {
  let capturedConfig;
  const factory = new OpenCodeSdkSessionFactory(
    async () => ({
      OpenCode: {
        async create(options) {
          capturedConfig = options.config.content;
          return {
            mcp: {
              async list() {
                return {
                  data: [
                    { name: "local", status: { status: "connected" } },
                    { name: "remote", status: { status: "connected" } },
                  ],
                };
              },
            },
            sessions: {
              async create() { return { id: "session-mcp-literals" }; },
              async remove() {},
            },
            async close() {},
          };
        },
      },
    }),
  );
  const input = startInput();
  input.config.mcp = {
    servers: {
      local: {
        transport: "stdio",
        url: "command-{env:OPENCODE_TEST_KEY}",
        args: ["argument-{file:/tmp/fabric-secret}"],
        env: { "ENV-{env:OPENCODE_TEST_KEY}": "value-{file:/tmp/fabric-secret}" },
      },
      remote: {
        transport: "streamable-http",
        url: "https://mcp.example.test/{file:/tmp/fabric-secret}",
        custom_headers: { "X-Literal": "value-{file:/tmp/fabric-secret}" },
      },
    },
  };

  const handle = await factory.create(input);

  assert.ok(capturedConfig.includes("\\u007benv:OPENCODE_TEST_KEY}"));
  assert.ok(capturedConfig.includes("\\u007bfile:/tmp/fabric-secret}"));
  assert.deepEqual(JSON.parse(capturedConfig).mcp.servers, {
    local: {
      type: "local",
      command: ["command-{env:OPENCODE_TEST_KEY}", "argument-{file:/tmp/fabric-secret}"],
      environment: { "ENV-{env:OPENCODE_TEST_KEY}": "value-{file:/tmp/fabric-secret}" },
    },
    remote: {
      type: "remote",
      url: "https://mcp.example.test/{file:/tmp/fabric-secret}",
      headers: { "X-Literal": "value-{file:/tmp/fabric-secret}" },
      oauth: false,
    },
  });
  await handle.stop();
});

test("resolves MCP header references from the parent environment without exposing them to OpenCode", async () => {
  const headerName = "MCP_AMBIENT_ACCESS_TOKEN";
  const previous = process.env[headerName];
  process.env[headerName] = "ambient-mcp-token";
  try {
    let capturedConfig;
    let observedHeaderEnvironment;
    const factory = new OpenCodeSdkSessionFactory(
      async () => ({
        OpenCode: {
          async create(options) {
            capturedConfig = options.config.content;
            observedHeaderEnvironment = process.env[headerName];
            return {
              mcp: {
                async list() {
                  return { data: [{ name: "remote", status: { status: "connected" } }] };
                },
              },
              sessions: {
                async create() { return { id: "session-mcp-ambient-header" }; },
                async remove() {},
              },
              async close() {},
            };
          },
        },
      }),
    );
    const input = startInput();
    input.config.mcp = {
      servers: {
        remote: {
          transport: "streamable-http",
          url: "https://mcp.example.test",
          custom_headers: { Authorization: `Bearer \${${headerName}}` },
        },
      },
    };

    const handle = await factory.create(input);

    assert.equal(observedHeaderEnvironment, undefined);
    assert.equal(
      JSON.parse(capturedConfig).mcp.servers.remote.headers.Authorization,
      "Bearer ambient-mcp-token",
    );
    await handle.stop();
  } finally {
    if (previous === undefined) {
      delete process.env[headerName];
    } else {
      process.env[headerName] = previous;
    }
  }
});

test("prefers explicitly configured values over parent environment values in MCP headers", async () => {
  const headerName = "MCP_PREFERRED_ACCESS_TOKEN";
  const previous = process.env[headerName];
  process.env[headerName] = "ambient-mcp-token";
  try {
    let capturedConfig;
    const factory = new OpenCodeSdkSessionFactory(
      async () => ({
        OpenCode: {
          async create(options) {
            capturedConfig = options.config.content;
            return {
              mcp: {
                async list() {
                  return { data: [{ name: "remote", status: { status: "connected" } }] };
                },
              },
              sessions: {
                async create() { return { id: "session-mcp-preferred-header" }; },
                async remove() {},
              },
              async close() {},
            };
          },
        },
      }),
    );
    const input = startInput();
    input.runtimeContext.environment.env[headerName] = "explicit-mcp-token";
    input.config.mcp = {
      servers: {
        remote: {
          transport: "streamable-http",
          url: "https://mcp.example.test",
          custom_headers: { Authorization: `Bearer \${${headerName}}` },
        },
      },
    };

    const handle = await factory.create(input);

    assert.equal(
      JSON.parse(capturedConfig).mcp.servers.remote.headers.Authorization,
      "Bearer explicit-mcp-token",
    );
    await handle.stop();
  } finally {
    if (previous === undefined) {
      delete process.env[headerName];
    } else {
      process.env[headerName] = previous;
    }
  }
});

test("rejects a missing MCP header reference before loading the OpenCode SDK", async () => {
  const input = startInput();
  input.config.mcp = {
    servers: {
      remote: {
        transport: "streamable-http",
        url: "https://mcp.example.test",
        custom_headers: { Authorization: "Bearer ${MISSING_MCP_ACCESS_TOKEN}" },
      },
    },
  };
  const factory = new OpenCodeSdkSessionFactory(async () => {
    throw new Error("the SDK must not load for a missing MCP header reference");
  });

  await assert.rejects(
    factory.create(input),
    (error) => error.code === "opencode_mcp_header_variable_missing",
  );
});

test("does not resolve inherited names in MCP header references", async () => {
  const input = startInput();
  input.config.mcp = {
    servers: {
      remote: {
        transport: "streamable-http",
        url: "https://mcp.example.test",
        custom_headers: { Authorization: "Bearer ${constructor}" },
      },
    },
  };
  const factory = new OpenCodeSdkSessionFactory(async () => {
    throw new Error("the SDK must not load for an inherited MCP header reference");
  });

  await assert.rejects(
    factory.create(input),
    (error) => error.code === "opencode_mcp_header_variable_missing",
  );
});

test("rejects invalid expanded MCP HTTP headers before loading the SDK", async () => {
  for (const [name, value] of [
    ["Bad Header", "value"],
    ["X-Test", ""],
    ["X-Test", " leading"],
    ["X-Test", "line\nbreak"],
    ["X-Test", "🚫"],
    ["X-Test", "${MCP_INVALID_HEADER_VALUE}"],
  ]) {
    const input = startInput();
    input.runtimeContext.environment.env.MCP_INVALID_HEADER_VALUE = "line\nbreak";
    input.config.mcp = {
      servers: {
        remote: {
          transport: "streamable-http",
          url: "https://mcp.example.test",
          custom_headers: { [name]: value },
        },
      },
    };
    const factory = new OpenCodeSdkSessionFactory(async () => {
      throw new Error("the SDK must not load for an invalid MCP header");
    });

    await assert.rejects(factory.create(input), (error) => error.code === "opencode_mcp_invalid_header");
  }
});

test("rejects command arguments on streamable-HTTP MCP servers before loading the SDK", async () => {
  const input = startInput();
  input.config.mcp = {
    servers: {
      remote: {
        transport: "streamable-http",
        url: "https://mcp.example.test",
        args: ["--unsupported"],
      },
    },
  };
  const factory = new OpenCodeSdkSessionFactory(async () => {
    throw new Error("the SDK must not load for network MCP command arguments");
  });

  await assert.rejects(factory.create(input), (error) => error.code === "opencode_mcp_invalid_server");
});

test("rejects environment variables on streamable-HTTP MCP servers before loading the SDK", async () => {
  const input = startInput();
  input.config.mcp = {
    servers: {
      remote: {
        transport: "streamable-http",
        url: "https://mcp.example.test",
        env: { MCP_TOKEN: "ignored-value" },
      },
    },
  };
  const factory = new OpenCodeSdkSessionFactory(async () => {
    throw new Error("the SDK must not load for network MCP environment variables");
  });

  await assert.rejects(factory.create(input), (error) => error.code === "opencode_mcp_invalid_server");
});

test("rejects non-loopback HTTP MCP servers before expanding headers or loading the SDK", async () => {
  const input = startInput();
  input.runtimeContext.environment.env.MCP_ACCESS_TOKEN = "mcp-test-token";
  input.config.mcp = {
    servers: {
      remote: {
        transport: "streamable-http",
        url: "http://mcp.example.test/mcp",
        custom_headers: { Authorization: "Bearer ${MCP_ACCESS_TOKEN}" },
      },
    },
  };
  const factory = new OpenCodeSdkSessionFactory(async () => {
    throw new Error("the SDK must not load for non-loopback HTTP MCP servers");
  });

  await assert.rejects(factory.create(input), (error) => error.code === "opencode_mcp_invalid_server");
});

test("accepts loopback HTTP and remote HTTPS MCP servers", async () => {
  let capturedConfig;
  const factory = new OpenCodeSdkSessionFactory(
    async () => ({
      OpenCode: {
        async create(options) {
          capturedConfig = options.config.content;
          return {
            mcp: {
              async list() {
                return {
                  data: [
                    { name: "loopback", status: { status: "connected" } },
                    { name: "secure", status: { status: "connected" } },
                  ],
                };
              },
            },
            sessions: {
              async create() { return { id: "session-mcp-transport-policy" }; },
              async remove() {},
            },
            async close() {},
          };
        },
      },
    }),
  );
  const input = startInput();
  input.config.mcp = {
    servers: {
      loopback: { transport: "streamable-http", url: "http://127.0.0.1:8080/mcp" },
      secure: { transport: "streamable-http", url: "https://mcp.example.test/mcp" },
    },
  };

  const handle = await factory.create(input);

  assert.deepEqual(JSON.parse(capturedConfig).mcp.servers, {
    loopback: { type: "remote", url: "http://127.0.0.1:8080/mcp", oauth: false },
    secure: { type: "remote", url: "https://mcp.example.test/mcp", oauth: false },
  });
  await handle.stop();
});

test("rejects custom HTTP headers on stdio MCP servers before loading the SDK", async () => {
  const input = startInput();
  input.config.mcp = {
    servers: {
      local: {
        transport: "stdio",
        url: "mcp-server",
        custom_headers: { Authorization: "Bearer token" },
      },
    },
  };
  const factory = new OpenCodeSdkSessionFactory(async () => {
    throw new Error("the SDK must not load for stdio MCP headers");
  });

  await assert.rejects(factory.create(input), (error) => error.code === "opencode_mcp_invalid_server");
});

test("retains an MCP server named __proto__", async () => {
  let capturedConfig;
  const factory = new OpenCodeSdkSessionFactory(
    async () => ({
      OpenCode: {
        async create(options) {
          capturedConfig = options.config.content;
          return {
            mcp: {
              async list() {
                return { data: [{ name: "__proto__", status: { status: "connected" } }] };
              },
            },
            sessions: {
              async create() { return { id: "session-mcp-proto" }; },
              async remove() {},
            },
            async close() {},
          };
        },
      },
    }),
  );
  const input = startInput();
  input.config.mcp = {
    servers: JSON.parse('{"__proto__":{"transport":"streamable-http","url":"https://mcp.example.test"}}'),
  };

  const handle = await factory.create(input);

  assert.equal(JSON.parse(capturedConfig).mcp.servers.__proto__.type, "remote");
  await handle.stop();
});

test("fails startup when an HTTP or stdio MCP server cannot connect", async () => {
  for (const [transport, server] of [
    ["streamable-http", { url: "http://127.0.0.1:1/mcp" }],
    ["stdio", { url: "not-a-real-mcp-command" }],
  ]) {
    const input = startInput();
    input.config.mcp = { servers: { unavailable: { transport, ...server } } };
    let removed = false;
    const factory = new OpenCodeSdkSessionFactory(
      async () => ({
        OpenCode: {
          async create() {
            return {
              mcp: {
                async list() {
                  return {
                    data: [{ name: "unavailable", status: { status: "failed", error: "private connection detail" } }],
                  };
                },
              },
              sessions: {
                async create() { return { id: "session-mcp-failure" }; },
                async remove() { removed = true; },
              },
              async close() {},
            };
          },
        },
      }),
    );

    await assert.rejects(
      factory.create(input),
      (error) =>
        error.code === "opencode_mcp_connection_failed" &&
        !error.message.includes("private connection detail"),
    );
    assert.equal(removed, true);
  }
});

test("allows an MCP server that connects and lists tools within OpenCode's combined defaults", async () => {
  const originalNow = Date.now;
  let now = 0;
  Date.now = () => now;
  try {
    const input = startInput();
    input.config.mcp = {
      servers: {
        slow: { transport: "streamable-http", url: "http://127.0.0.1:1/mcp" },
      },
    };
    let statusChecks = 0;
    const factory = new OpenCodeSdkSessionFactory(
      async () => ({
        OpenCode: {
          async create() {
            return {
              mcp: {
                async list() {
                  statusChecks += 1;
                  if (statusChecks === 1) {
                    now = 30_000;
                    return { data: [{ name: "slow", status: { status: "pending" } }] };
                  }
                  if (statusChecks === 2) {
                    now = 45_000;
                    return { data: [{ name: "slow", status: { status: "pending" } }] };
                  }
                  return { data: [{ name: "slow", status: { status: "connected" } }] };
                },
              },
              sessions: {
                async create() { return { id: "session-mcp-slow-startup" }; },
                async remove() {},
              },
              async close() {},
            };
          },
        },
      }),
    );

    const handle = await factory.create(input);

    assert.equal(statusChecks, 3);
    await handle.stop();
  } finally {
    Date.now = originalNow;
  }
});

test("aborts a stalled MCP readiness request at the connection deadline", async () => {
  const originalTimeout = AbortSignal.timeout;
  AbortSignal.timeout = () => {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 0);
    return controller.signal;
  };
  try {
    const input = startInput();
    input.config.mcp = {
      servers: {
        unavailable: { transport: "streamable-http", url: "http://127.0.0.1:1/mcp" },
      },
    };
    let removed = false;
    const factory = new OpenCodeSdkSessionFactory(
      async () => ({
        OpenCode: {
          async create() {
            return {
              mcp: {
                async list(_input, requestOptions) {
                  if (requestOptions?.signal === undefined) {
                    throw new Error("OpenCode request options did not include an abort signal");
                  }
                  return new Promise((_resolve, reject) => {
                    requestOptions.signal.addEventListener("abort", () => reject(requestOptions.signal.reason), {
                      once: true,
                    });
                  });
                },
              },
              sessions: {
                async create() { return { id: "session-mcp-timeout" }; },
                async remove() { removed = true; },
              },
              async close() {},
            };
          },
        },
      }),
    );

    await assert.rejects(factory.create(input), (error) => error.code === "opencode_mcp_connection_timeout");
    assert.equal(removed, true);
  } finally {
    AbortSignal.timeout = originalTimeout;
  }
});

test("rejects unsupported OpenCode MCP transports before loading the SDK", async () => {
  const input = startInput();
  input.config.mcp = {
    servers: {
      legacy: { transport: "sse", url: "https://mcp.example.test/sse" },
    },
  };
  const factory = new OpenCodeSdkSessionFactory(async () => {
    throw new Error("the SDK must not load for an unsupported MCP transport");
  });

  await assert.rejects(factory.create(input), (error) => error.code === "opencode_mcp_transport_unsupported");
});

test("rejects Fabric sampling settings for native OpenCode providers before loading the SDK", async () => {
  const factory = new OpenCodeSdkSessionFactory(async () => {
    throw new Error("the SDK must not load for unsupported sampling settings");
  });
  const input = startInput();
  Object.assign(input.config.models.default, { temperature: 0.25, top_p: 0.8 });

  await assert.rejects(factory.create(input), (error) => error.code === "opencode_sampling_requires_base_url");
});

test("rejects undeclared provider-specific model settings before loading the SDK", async () => {
  const factory = new OpenCodeSdkSessionFactory(async () => {
    throw new Error("the SDK must not load for unsupported model settings");
  });
  const input = startInput();
  input.config.models.default.settings = {};

  await assert.rejects(factory.create(input), (error) => error.code === "opencode_model_settings_unsupported");
});

test("extracts the final assistant text from OpenCode session history", () => {
  const outcome = extractOpenCodePromptOutcome([
    { type: "user", text: "first" },
    {
      id: "assistant-first",
      type: "assistant",
      content: [{ type: "text", text: "first answer" }],
      finish: "stop",
      tokens: { input: 2, output: 3, reasoning: 0, cache: { read: 0, write: 0 } },
      cost: 0.012,
    },
    { type: "user", text: "second" },
    {
      id: "assistant-second",
      type: "assistant",
      content: [{ type: "reasoning", text: "hidden" }, { type: "text", text: "second answer" }],
      finish: "stop",
      tokens: { input: 5, output: 7, reasoning: 2, cache: { read: 1, write: 0 } },
      cost: 0.023,
    },
  ], new Set(["assistant-first"]));

  assert.deepEqual(outcome, { text: "second answer" });
});

// OpenCode's Session.Info shape. `tokens.input` excludes cache reads and
// writes, and `tokens.output` excludes reasoning, which OpenCode counts
// separately.
function openCodeSession(tokens, id = "session-usage", parentID = undefined) {
  return {
    id,
    ...(parentID === undefined ? {} : { parentID }),
    projectID: "project-1",
    cost: 0.5,
    tokens,
    time: { created: 1784045900000, updated: 1784045904000 },
    location: { directory: "/workspace" },
  };
}

function openCodeTokens(input, output, reasoning, cacheRead, cacheWrite) {
  return { input, output, reasoning, cache: { read: cacheRead, write: cacheWrite } };
}

const NO_TOKENS = openCodeTokens(0, 0, 0, 0, 0);
const ROOT_SESSION = "session-usage";

const TERMINAL_ASSISTANT = {
  id: "assistant-terminal",
  type: "assistant",
  content: [{ type: "text", text: "done" }],
  finish: "stop",
  tokens: openCodeTokens(1_000, 500, 2_000, 20_000, 0),
  cost: 0.01,
};

function assistantCall(id, tokens, content = []) {
  return { id, type: "assistant", content, finish: "tool-calls", tokens };
}

function subagentCall(id, childSessionID, tokens) {
  return assistantCall(id, tokens, [
    {
      type: "tool",
      id: `call-${id}`,
      name: "subagent",
      state: { status: "completed", metadata: { sessionID: childSessionID, status: "completed" } },
    },
  ]);
}

// Each fake session is `{ parentID, before, after }`. `before` is omitted for a
// session the prompt created. Each state has `session` (an Error simulates a
// failed read), `history` (oldest first) and an optional `active` flag.
function rootSession(beforeTokens, afterTokens, newMessages = [TERMINAL_ASSISTANT], priorMessages = []) {
  return {
    before: { session: openCodeSession(beforeTokens), history: priorMessages },
    after: { session: openCodeSession(afterTokens), history: [...priorMessages, ...newMessages] },
  };
}

// A fake OpenCode client that serves the session tree and full message
// history the way OpenCode 2.0.23 pages them: newest message first, and at
// most `limit` items per page.
function usageClient(sessions, { context, diff = [], active, withoutCursors = false } = {}) {
  let prompted = false;
  const state = (sessionID) => (prompted ? sessions[sessionID].after : sessions[sessionID].before);
  const page = (items, offset, limit, cursorPrefix) => {
    const data = items.slice(offset, offset + limit);
    const next = data.length === 0 || withoutCursors ? undefined : `${cursorPrefix}${offset + data.length}`;
    return { data, cursor: { next } };
  };
  return {
    sessions: {
      async create() { return { id: ROOT_SESSION }; },
      async get({ sessionID }) {
        const session = state(sessionID).session;
        if (session instanceof Error) {
          throw session;
        }
        return session;
      },
      async list({ parentID, cursor, limit = 50 }) {
        const [parent, offset] = cursor === undefined ? [parentID, 0] : [cursor.split("@")[0], Number(cursor.split("@")[1])];
        const children = Object.keys(sessions)
          .filter((sessionID) => sessions[sessionID].parentID === parent && state(sessionID) !== undefined)
          .map((sessionID) => openCodeSession(NO_TOKENS, sessionID, parent));
        return page(children, offset, limit, `${parent}@`);
      },
      async active() {
        if (active !== undefined) {
          return active;
        }
        return Object.fromEntries(
          Object.keys(sessions)
            .filter((sessionID) => state(sessionID)?.active === true)
            .map((sessionID) => [sessionID, { type: "running" }]),
        );
      },
      async context() {
        return prompted ? (context ?? sessions[ROOT_SESSION].after.history) : sessions[ROOT_SESSION].before.history;
      },
      async prompt() { prompted = true; },
      async wait() {},
      async diff() { return diff; },
      async remove() {},
    },
    message: {
      async list({ sessionID, cursor, limit = 50, ...rest }) {
        assert.equal(cursor !== undefined && rest.order !== undefined, false);
        return page(state(sessionID).history.toReversed(), cursor === undefined ? 0 : Number(cursor), limit, "");
      },
    },
    async close() {},
  };
}

async function promptWithSessionUsage(sessions, options) {
  const factory = new OpenCodeSdkSessionFactory(async () => ({
    OpenCode: { async create() { return usageClient(sessions, options); } },
  }));
  const handle = await factory.create(startInput());
  try {
    return await handle.prompt("fix the failing test");
  } finally {
    await handle.stop();
  }
}

test("counts cached input and reasoning output in OpenCode usage", async () => {
  // Second invocation of a session: three model calls, each with 1k uncached
  // input, 500 visible output and 2k reasoning; the first call wrote 20k
  // tokens to the prompt cache and the next two read them.
  const priorTurn = [
    { id: "user-1", type: "user", text: "first" },
    { ...TERMINAL_ASSISTANT, id: "assistant-prior", tokens: openCodeTokens(1_200, 300, 800, 0, 18_000) },
  ];
  const outcome = await promptWithSessionUsage({
    [ROOT_SESSION]: rootSession(
      openCodeTokens(1_200, 300, 800, 0, 18_000),
      openCodeTokens(4_200, 1_800, 6_800, 40_000, 38_000),
      [
        { id: "user-2", type: "user", text: "second" },
        assistantCall("call-1", openCodeTokens(1_000, 500, 2_000, 0, 20_000)),
        assistantCall("call-2", openCodeTokens(1_000, 500, 2_000, 20_000, 0)),
        TERMINAL_ASSISTANT,
      ],
      priorTurn,
    ),
  });

  assert.deepEqual(outcome, {
    text: "done",
    usage: {
      input_tokens: 63_000,
      cached_input_tokens: 40_000,
      input_tokens_include_cache: true,
      output_tokens: 7_500,
      total_tokens: 70_500,
    },
  });
});

test("counts subagent sessions and model calls that a compaction hides from the session context", async () => {
  // After a compaction completes, the session context starts at it, so it no
  // longer lists the prompt's earlier subagent call. The full history does.
  const compaction = {
    id: "compaction-1",
    type: "compaction",
    status: "completed",
    reason: "auto",
    summary: "Earlier work summarized.",
    recent: "",
    tokens: openCodeTokens(100, 20, 0, 0, 0),
  };
  const terminal = { ...TERMINAL_ASSISTANT, tokens: openCodeTokens(50, 10, 0, 0, 0) };
  const outcome = await promptWithSessionUsage(
    {
      [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(180, 40, 0, 0, 0), [
        { id: "user-1", type: "user", text: "fix the failing test" },
        subagentCall("assistant-subagent", "session-child", openCodeTokens(30, 10, 0, 0, 0)),
        compaction,
        terminal,
      ]),
      "session-child": ({
        parentID: ROOT_SESSION,
        after: {
          session: openCodeSession(openCodeTokens(400, 60, 40, 0, 0), "session-child", ROOT_SESSION),
          history: [{ id: "child-user", type: "user", text: "look" }, assistantCall("child-call", openCodeTokens(400, 60, 40, 0, 0))],
        },
      }),
    },
    { context: [compaction, terminal] },
  );

  assert.deepEqual(outcome, {
    text: "done",
    usage: {
      input_tokens: 580,
      cached_input_tokens: 0,
      input_tokens_include_cache: true,
      output_tokens: 140,
      total_tokens: 720,
    },
  });
});

test("counts only the new usage of a reused subagent session", async () => {
  const priorChildCall = assistantCall("child-call-1", openCodeTokens(10, 5, 0, 0, 0));
  const outcome = await promptWithSessionUsage({
    [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(100, 20, 0, 0, 0), [
      subagentCall("assistant-subagent", "session-child", openCodeTokens(50, 10, 0, 0, 0)),
      { ...TERMINAL_ASSISTANT, tokens: openCodeTokens(50, 10, 0, 0, 0) },
    ]),
    "session-child": ({
      parentID: ROOT_SESSION,
      before: {
        session: openCodeSession(openCodeTokens(10, 5, 0, 0, 0), "session-child", ROOT_SESSION),
        history: [priorChildCall],
      },
      after: {
        session: openCodeSession(openCodeTokens(50, 15, 0, 0, 0), "session-child", ROOT_SESSION),
        history: [priorChildCall, assistantCall("child-call-2", openCodeTokens(40, 10, 0, 0, 0))],
      },
    }),
  });

  assert.deepEqual(outcome.usage, {
    input_tokens: 140,
    cached_input_tokens: 0,
    input_tokens_include_cache: true,
    output_tokens: 30,
    total_tokens: 170,
  });
});

test("leaves OpenCode usage unknown when a subagent session was running at either end or reported only zeros", async () => {
  const sessions = (child) => ({
    [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(100, 20, 0, 0, 0), [
      subagentCall("assistant-subagent", "session-child", openCodeTokens(50, 10, 0, 0, 0)),
      { ...TERMINAL_ASSISTANT, tokens: openCodeTokens(50, 10, 0, 0, 0) },
    ]),
    "session-child": ({ parentID: ROOT_SESSION, after: child }),
  });
  const childSession = openCodeSession(openCodeTokens(40, 10, 0, 0, 0), "session-child", ROOT_SESSION);
  const reported = assistantCall("child-call", openCodeTokens(40, 10, 0, 0, 0));

  assert.deepEqual(
    await promptWithSessionUsage(sessions({ session: childSession, history: [reported], active: true })),
    { text: "done" },
  );
  assert.deepEqual(
    await promptWithSessionUsage(
      sessions({ session: childSession, history: [reported, assistantCall("child-zero", NO_TOKENS)] }),
    ),
    { text: "done" },
  );
});

test("leaves out a background subagent that an earlier invocation started", async () => {
  // The child was still running when this invocation began, so its growth
  // belongs to the earlier invocation, which reported no usage.
  const earlierChild = {
    parentID: ROOT_SESSION,
    before: { session: openCodeSession(NO_TOKENS, "session-child", ROOT_SESSION), history: [], active: true },
    after: {
      session: openCodeSession(openCodeTokens(7, 3, 0, 0, 0), "session-child", ROOT_SESSION),
      history: [assistantCall("child-call", openCodeTokens(7, 3, 0, 0, 0))],
    },
  };
  const notice = {
    id: "notice",
    type: "synthetic",
    text: "<subagent ...>",
    metadata: { source: "subagent", childID: "session-child", state: "completed" },
  };
  const ownCall = { ...TERMINAL_ASSISTANT, tokens: openCodeTokens(50, 10, 0, 0, 0) };
  const prompt = (newMessages) =>
    promptWithSessionUsage({
      [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(50, 10, 0, 0, 0), newMessages),
      "session-child": earlierChild,
    });

  assert.deepEqual((await prompt([ownCall])).usage, {
    input_tokens: 50,
    cached_input_tokens: 0,
    input_tokens_include_cache: true,
    output_tokens: 10,
    total_tokens: 60,
  });
  assert.equal((await prompt([ownCall, notice])).usage?.total_tokens, 60);
  // A call after the completion notice may answer it, and a subagent call can
  // continue the earlier child, so neither can be separated.
  assert.equal((await prompt([notice, ownCall])).usage, undefined);
  assert.equal(
    (await prompt([
      { ...TERMINAL_ASSISTANT, content: [{ type: "tool", id: "t", name: "subagent", state: { status: "completed", input: { sessionID: "session-child" } } }], tokens: openCodeTokens(50, 10, 0, 0, 0) },
    ])).usage,
    undefined,
  );
});

test("leaves OpenCode usage unknown when a compaction hides a failed call without usage", async () => {
  const compaction = {
    id: "compaction-1",
    type: "compaction",
    status: "completed",
    reason: "auto",
    summary: "Earlier work summarized.",
    recent: "",
    tokens: openCodeTokens(100, 20, 0, 0, 0),
  };
  const terminal = { ...TERMINAL_ASSISTANT, tokens: openCodeTokens(50, 10, 0, 0, 0) };
  const failed = { id: "assistant-failed", type: "assistant", content: [], error: { type: "provider.error" } };
  const outcome = await promptWithSessionUsage(
    { [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(150, 30, 0, 0, 0), [failed, compaction, terminal]) },
    { context: [compaction, terminal] },
  );

  assert.deepEqual(outcome, { text: "done" });
});

test("checks every page of the session history", async () => {
  const calls = Array.from({ length: 150 }, (_, index) => assistantCall(`call-${index}`, openCodeTokens(1, 1, 0, 0, 0)));
  const failed = { id: "assistant-failed", type: "assistant", content: [], error: { type: "provider.error" } };
  const prior = [{ id: "prior-user", type: "user", text: "earlier" }];
  const counted = await promptWithSessionUsage({
    [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(150, 150, 0, 0, 0), calls, prior),
  });
  const withFailure = await promptWithSessionUsage({
    [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(150, 150, 0, 0, 0), [failed, ...calls], prior),
  });

  assert.deepEqual(counted.usage, {
    input_tokens: 150,
    cached_input_tokens: 0,
    input_tokens_include_cache: true,
    output_tokens: 150,
    total_tokens: 300,
  });
  assert.equal(withFailure.usage, undefined);
});

test("keeps OpenCode usage when the prompt produced no assistant message", async () => {
  const compactionCall = openCodeTokens(30_000, 2_000, 0, 0, 0);
  const outcome = await promptWithSessionUsage({
    [ROOT_SESSION]: rootSession(NO_TOKENS, compactionCall, [
      {
        id: "compaction-1",
        type: "compaction",
        status: "failed",
        reason: "auto",
        error: { type: "aborted" },
        tokens: compactionCall,
      },
    ]),
  });

  assert.deepEqual(outcome, {
    usage: {
      input_tokens: 30_000,
      cached_input_tokens: 0,
      input_tokens_include_cache: true,
      output_tokens: 2_000,
      total_tokens: 32_000,
    },
  });
});

test("leaves OpenCode usage unknown when model calls added only zeros", async () => {
  // OpenCode records a model call whose provider sent no usage report as
  // zeros, so zero growth after a model call is not a measured zero.
  const counters = openCodeTokens(1_000, 200, 50, 300, 0);
  const outcome = await promptWithSessionUsage({
    [ROOT_SESSION]: rootSession(counters, counters, [{ ...TERMINAL_ASSISTANT, tokens: NO_TOKENS }]),
  });

  assert.deepEqual(outcome, { text: "done" });
});

test("leaves OpenCode usage unknown when one of several model calls reported only zeros", async () => {
  const outcome = await promptWithSessionUsage({
    [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(100, 20, 0, 0, 0), [
      assistantCall("assistant-reported", openCodeTokens(100, 20, 0, 0, 0)),
      { ...TERMINAL_ASSISTANT, tokens: NO_TOKENS },
    ]),
  });

  assert.deepEqual(outcome, { text: "done" });
});

test("leaves OpenCode usage unknown when an idle earlier subagent's completion notice arrives", async () => {
  // The child finished before this invocation started, but OpenCode delivered
  // its completion notice during it, and the next call may answer the notice.
  const idleChild = {
    parentID: ROOT_SESSION,
    before: {
      session: openCodeSession(openCodeTokens(7, 3, 0, 0, 0), "session-child", ROOT_SESSION),
      history: [assistantCall("child-call", openCodeTokens(7, 3, 0, 0, 0))],
    },
    after: {
      session: openCodeSession(openCodeTokens(7, 3, 0, 0, 0), "session-child", ROOT_SESSION),
      history: [assistantCall("child-call", openCodeTokens(7, 3, 0, 0, 0))],
    },
  };
  const notice = {
    id: "notice",
    type: "synthetic",
    text: "<subagent ...>",
    metadata: { source: "subagent", childID: "session-child", state: "completed" },
  };
  const delayed = await promptWithSessionUsage({
    [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(150, 30, 0, 0, 0), [
      assistantCall("own-call", openCodeTokens(100, 20, 0, 0, 0)),
      notice,
      { ...TERMINAL_ASSISTANT, tokens: openCodeTokens(50, 10, 0, 0, 0) },
    ]),
    "session-child": idleChild,
  });
  // The parent answers the earlier notice, then resumes the same child. The
  // resume does not make the earlier notice this invocation's work.
  const resumeCall = {
    ...assistantCall("resume-child", openCodeTokens(100, 20, 0, 0, 0)),
    content: [
      {
        type: "tool",
        id: "t",
        name: "subagent",
        state: { status: "completed", input: { sessionID: "session-child" }, metadata: { sessionID: "session-child" } },
      },
    ],
  };
  const answeredThenResumed = await promptWithSessionUsage({
    [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(250, 50, 0, 0, 0), [
      notice,
      assistantCall("answer-notice", openCodeTokens(100, 20, 0, 0, 0)),
      resumeCall,
      { ...TERMINAL_ASSISTANT, tokens: openCodeTokens(50, 10, 0, 0, 0) },
    ]),
    "session-child": {
      ...idleChild,
      after: {
        session: openCodeSession(openCodeTokens(14, 6, 0, 0, 0), "session-child", ROOT_SESSION),
        history: [...idleChild.after.history, assistantCall("child-call-2", openCodeTokens(7, 3, 0, 0, 0))],
      },
    },
  });

  assert.equal(delayed.usage, undefined);
  assert.equal(answeredThenResumed.usage, undefined);
});

test("leaves out total_tokens when the sum is past the safe integer range", async () => {
  const large = Number.MAX_SAFE_INTEGER - 5;
  const outcome = await promptWithSessionUsage({
    [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(large, 10, 0, 0, 0), [
      { ...TERMINAL_ASSISTANT, tokens: openCodeTokens(large, 10, 0, 0, 0) },
    ]),
  });

  assert.deepEqual(outcome.usage, {
    input_tokens: large,
    cached_input_tokens: 0,
    input_tokens_include_cache: true,
    output_tokens: 10,
  });
});

test("leaves a side unknown when a model call reported none of it", async () => {
  // OpenCode stores a count the provider left out as 0, and a real call always
  // has input, so a call with no output tokens leaves output unknown.
  const promptOnly = await promptWithSessionUsage({
    [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(100, 0, 0, 0, 0), [
      { ...TERMINAL_ASSISTANT, tokens: openCodeTokens(100, 0, 0, 0, 0) },
    ]),
  });
  const mixed = await promptWithSessionUsage({
    [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(150, 20, 0, 0, 0), [
      assistantCall("call-reported", openCodeTokens(100, 20, 0, 0, 0)),
      { ...TERMINAL_ASSISTANT, tokens: openCodeTokens(50, 0, 0, 0, 0) },
    ]),
  });
  const cachedOnly = await promptWithSessionUsage({
    [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(0, 20, 0, 100, 0), [
      { ...TERMINAL_ASSISTANT, tokens: openCodeTokens(0, 20, 0, 100, 0) },
    ]),
  });

  const inputOnly = (input) => ({ input_tokens: input, cached_input_tokens: 0, input_tokens_include_cache: true });
  assert.deepEqual(promptOnly.usage, inputOnly(100));
  assert.deepEqual(mixed.usage, inputOnly(150));
  assert.deepEqual(cachedOnly.usage, {
    input_tokens: 100,
    cached_input_tokens: 100,
    input_tokens_include_cache: true,
    output_tokens: 20,
    total_tokens: 120,
  });
});

test("leaves OpenCode usage unknown when a failed compaction recorded no usage", async () => {
  const failedCompaction = {
    id: "compaction-failed",
    type: "compaction",
    status: "failed",
    reason: "auto",
    error: { type: "provider.error", message: "upstream connection reset" },
  };

  assert.deepEqual(
    await promptWithSessionUsage({ [ROOT_SESSION]: rootSession(NO_TOKENS, NO_TOKENS, [failedCompaction]) }),
    {},
  );
  assert.deepEqual(
    await promptWithSessionUsage({
      [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(100, 20, 0, 0, 0), [
        { ...TERMINAL_ASSISTANT, tokens: openCodeTokens(100, 20, 0, 0, 0) },
        failedCompaction,
      ]),
    }),
    { text: "done" },
  );
});

test("leaves OpenCode usage unknown when a failed model call recorded no usage", async () => {
  const outcome = await promptWithSessionUsage({
    [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(900, 300, 0, 0, 0), [
      assistantCall("assistant-tool-call", openCodeTokens(900, 300, 0, 0, 0)),
      {
        id: "assistant-failed",
        type: "assistant",
        content: [],
        error: { type: "provider.error", message: "upstream connection reset" },
      },
    ]),
  });

  assert.deepEqual(outcome, { errorMessage: "OpenCode model invocation failed" });
});

test("leaves OpenCode token counts unknown when a session counter is missing", async () => {
  // OpenCode 2.0.23 always returns every counter; these shapes check that a
  // malformed session never reads as a measured count.
  const before = openCodeTokens(100, 20, 5, 0, 0);
  const terminal = [{ ...TERMINAL_ASSISTANT, tokens: openCodeTokens(300, 50, 10, 0, 0) }];

  assert.deepEqual(
    (await promptWithSessionUsage({ [ROOT_SESSION]: rootSession(before, { input: 400, output: 70, reasoning: 15 }, terminal) }))
      .usage,
    { output_tokens: 60 },
  );
  assert.deepEqual(
    (await promptWithSessionUsage({
      [ROOT_SESSION]: rootSession(before, { input: 400, output: 70, cache: { read: 0, write: 0 } }, terminal),
    })).usage,
    { input_tokens: 300, cached_input_tokens: 0, input_tokens_include_cache: true },
  );
});

test("leaves OpenCode token counts unknown for malformed counters and records", async () => {
  // OpenCode 2.0.23 validates these shapes; the checks keep a malformed or
  // future shape from turning into a measured count.
  const terminal = [{ ...TERMINAL_ASSISTANT, tokens: openCodeTokens(100, 20, 0, 0, 0) }];
  const outputOnly = { output_tokens: 20 };
  for (const input of [null, "100", -1, 1.5, Number.MAX_SAFE_INTEGER + 1, true]) {
    const outcome = await promptWithSessionUsage({
      [ROOT_SESSION]: rootSession(NO_TOKENS, { ...openCodeTokens(100, 20, 0, 0, 0), input }, terminal),
    });
    assert.deepEqual(outcome.usage, outputOnly, `session input ${String(input)}`);
  }
  for (const tokens of [undefined, null, { ...openCodeTokens(100, 20, 0, 0, 0), output: "20" }, openCodeTokens(100, -20, 0, 0, 0)]) {
    const outcome = await promptWithSessionUsage({
      [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(100, 20, 0, 0, 0), [{ ...TERMINAL_ASSISTANT, tokens }]),
    });
    assert.equal(outcome.usage, undefined, `record tokens ${JSON.stringify(tokens)}`);
  }
});

test("leaves OpenCode usage unknown for unsafe sums and malformed listings", async () => {
  const large = Number.MAX_SAFE_INTEGER - 10;
  const outcome = await promptWithSessionUsage({
    [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(large, 20, 0, 0, 0), [
      subagentCall("assistant-subagent", "session-child", openCodeTokens(large, 10, 0, 0, 0)),
      { ...TERMINAL_ASSISTANT, tokens: openCodeTokens(1, 10, 0, 0, 0) },
    ]),
    "session-child": {
      parentID: ROOT_SESSION,
      after: {
        session: openCodeSession(openCodeTokens(100, 10, 0, 0, 0), "session-child", ROOT_SESSION),
        history: [assistantCall("child-call", openCodeTokens(100, 10, 0, 0, 0))],
      },
    },
  });
  assert.deepEqual(outcome.usage, { output_tokens: 30 });

  const normal = { [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(1_000, 500, 2_000, 20_000, 0)) };
  assert.equal((await promptWithSessionUsage(normal, { withoutCursors: true })).usage, undefined);
  assert.equal((await promptWithSessionUsage(normal, { active: null })).usage, undefined);
});

test("leaves OpenCode usage unknown when the session counters reset or cannot be read", async () => {
  const before = openCodeTokens(5_000, 900, 100, 2_000, 0);
  const unreadable = (state) => ({ ...state, session: new Error("session unavailable") });
  const root = rootSession(before, openCodeTokens(10, 5, 0, 0, 0));

  assert.deepEqual(await promptWithSessionUsage({ [ROOT_SESSION]: root }), { text: "done" });
  assert.deepEqual(
    await promptWithSessionUsage({ [ROOT_SESSION]: { ...root, before: unreadable(root.before) } }),
    { text: "done" },
  );
  assert.deepEqual(
    await promptWithSessionUsage({ [ROOT_SESSION]: { ...root, after: unreadable(root.after) } }),
    { text: "done" },
  );
});

test("preserves a redacted OpenCode terminal error and handles malformed assistant output", () => {
  assert.deepEqual(
    extractOpenCodePromptOutcome([
      {
        type: "assistant",
        content: [{ type: "text", text: "partial" }],
        finish: "error",
        error: { message: "provider returned HTTP 401: Authorization: Bearer test-secret" },
      },
    ]),
    {
      errorMessage: "OpenCode model invocation failed",
    },
  );
  assert.deepEqual(extractOpenCodePromptOutcome([{ type: "assistant", content: "invalid", finish: "stop" }]), {});
  assert.deepEqual(extractOpenCodePromptOutcome([]), {});
});

test("collects usage and a diff after a terminal model error", async () => {
  const outcome = await promptWithSessionUsage(
    {
      [ROOT_SESSION]: rootSession(NO_TOKENS, openCodeTokens(5, 7, 0, 0, 0), [
        {
          id: "assistant-error",
          type: "assistant",
          content: [],
          finish: "error",
          error: { message: "provider error" },
          tokens: openCodeTokens(5, 7, 0, 0, 0),
          cost: 0.125,
        },
      ]),
    },
    { diff: [{ file: "example.txt", patch: "diff --git a/example.txt b/example.txt\n+changed\n" }] },
  );

  assert.deepEqual(outcome, {
    errorMessage: "OpenCode model invocation failed",
    patch: "diff --git a/example.txt b/example.txt\n+changed\n",
    usage: {
      input_tokens: 5,
      cached_input_tokens: 0,
      input_tokens_include_cache: true,
      output_tokens: 7,
      total_tokens: 12,
    },
  });
});

test("does not return a prior assistant response for a new prompt", async () => {
  let contextCalls = 0;
  const factory = new OpenCodeSdkSessionFactory(async () => ({
    OpenCode: {
      async create() {
        return {
          sessions: {
            async create() {
              return { id: "session-stale-response" };
            },
            async prompt() {},
            async wait() {},
            async context() {
              contextCalls += 1;
              const priorTurn = [
                { id: "user-first", type: "user", text: "first" },
                {
                  id: "assistant-first",
                  type: "assistant",
                  content: [{ type: "text", text: "first answer" }],
                  finish: "stop",
                },
              ];
              return contextCalls === 1
                ? priorTurn
                : [...priorTurn, { id: "user-second", type: "user", text: "second" }];
            },
            async diff() {
              throw new Error("no diff available");
            },
            async remove() {},
          },
          async close() {},
        };
      },
    },
  }));

  const handle = await factory.create(startInput());
  assert.deepEqual(await handle.prompt("second"), {});
  assert.equal(contextCalls, 2);
  await handle.stop();
});

test("finds a new assistant response after OpenCode compacts session history", async () => {
  let contextCalls = 0;
  const factory = new OpenCodeSdkSessionFactory(async () => ({
    OpenCode: {
      async create() {
        return {
          sessions: {
            async create() {
              return { id: "session-compaction" };
            },
            async prompt() {},
            async wait() {},
            async context() {
              contextCalls += 1;
              if (contextCalls === 1) {
                return Array.from({ length: 8 }, (_, index) => ({
                  id: `prior-${index}`,
                  type: index % 2 === 0 ? "user" : "assistant",
                  content: [{ type: "text", text: `prior ${index}` }],
                  finish: "stop",
                }));
              }
              return [
                { id: "compaction-marker", type: "compaction" },
                {
                  id: "assistant-after-compaction",
                  type: "assistant",
                  content: [{ type: "text", text: "compacted answer" }],
                  finish: "stop",
                },
              ];
            },
            async diff() {
              throw new Error("no diff available");
            },
            async remove() {},
          },
          async close() {},
        };
      },
    },
  }));

  const handle = await factory.create(startInput());
  assert.deepEqual(await handle.prompt("continue"), { text: "compacted answer" });
  await handle.stop();
});

test("creates, invokes, and cleans up one embedded OpenCode session", async () => {
  const calls = [];
  let contextCalls = 0;
  let closed = false;
  const previous = process.env.OPENCODE_TEST_KEY;
  delete process.env.OPENCODE_TEST_KEY;
  try {
    const factory = new OpenCodeSdkSessionFactory(async () => ({
      OpenCode: {
        async create(options) {
          calls.push(["create", options, process.env.OPENCODE_TEST_KEY]);
          return {
            sessions: {
              async create(input) {
                calls.push(["session.create", input]);
                return { id: "session-1" };
              },
              async prompt(input) {
                calls.push(["session.prompt", input]);
              },
              async wait(input) {
                calls.push(["session.wait", input]);
              },
              async context(input) {
                calls.push(["session.context", input]);
                contextCalls += 1;
                return contextCalls === 1
                  ? []
                  : [{ id: "assistant-done", type: "assistant", content: [{ type: "text", text: "done" }], finish: "stop" }];
              },
              async diff(input) {
                calls.push(["session.diff", input]);
                return [{ file: "example.txt", patch: "diff --git a/example.txt b/example.txt\n+added\n" }];
              },
              async remove(input) {
                calls.push(["session.remove", input]);
              },
            },
            async close() {
              closed = true;
            },
          };
        },
      },
    }));

    const handle = await factory.create(startInput());
    assert.equal(handle.id, "session-1");
    assert.deepEqual(await handle.prompt("hello"), {
      text: "done",
      patch: "diff --git a/example.txt b/example.txt\n+added\n",
    });
    await handle.stop();

    assert.equal(closed, true);
    assert.equal(process.env.OPENCODE_TEST_KEY, undefined);
    assert.deepEqual(calls, [
      [
        "create",
        {
          config: {
            directory: "/workspace",
            project: false,
            content: JSON.stringify({
              providers: { nvidia: { settings: { apiKey: "{env:OPENCODE_TEST_KEY}" } } },
            }),
          },
          fs: { filewatcher: false },
          models: { fetch: false },
        },
        "test-secret",
      ],
      [
        "session.create",
        {
          title: "NeMo Fabric",
          location: { directory: "/workspace" },
          model: { providerID: "nvidia", id: "nvidia/nemotron-3.5-lightning-30b-a3b" },
          permissions: [
            { action: "external_directory", resource: "*", effect: "deny" },
            { action: "read", resource: "*.env", effect: "deny" },
            { action: "read", resource: "*.env.*", effect: "deny" },
            { action: "read", resource: "*.env.example", effect: "allow" },
            { action: "question", resource: "*", effect: "deny" },
          ],
        },
      ],
      ["session.context", { sessionID: "session-1" }],
      ["session.prompt", { sessionID: "session-1", text: "hello" }],
      ["session.wait", { sessionID: "session-1" }],
      ["session.context", { sessionID: "session-1" }],
      ["session.diff", { sessionID: "session-1" }],
      ["session.remove", { sessionID: "session-1" }],
    ]);
  } finally {
    if (previous === undefined) {
      delete process.env.OPENCODE_TEST_KEY;
    } else {
      process.env.OPENCODE_TEST_KEY = previous;
    }
  }
});

test("limits the embedded OpenCode environment to declared values and restores it after stop", async () => {
  const ambientName = "OPENCODE_AMBIENT_SECRET";
  const declaredName = "OPENCODE_DECLARED_VALUE";
  const previousAmbient = process.env[ambientName];
  const previousDeclared = process.env[declaredName];
  process.env[ambientName] = "ambient-secret";
  delete process.env[declaredName];
  try {
    const input = startInput();
    input.runtimeContext.environment.env[declaredName] = "declared-value";
    let observed;
    const factory = new OpenCodeSdkSessionFactory(async () => ({
      OpenCode: {
        async create() {
          observed = {
            ambient: process.env[ambientName],
            declared: process.env[declaredName],
            credential: process.env.OPENCODE_TEST_KEY,
          };
          return {
            sessions: {
              async create() { return { id: "session-environment" }; },
              async remove() {},
            },
            async close() {},
          };
        },
      },
    }));

    const handle = await factory.create(input);
    assert.deepEqual(observed, {
      ambient: undefined,
      declared: "declared-value",
      credential: "test-secret",
    });
    await handle.stop();
    assert.equal(process.env[ambientName], "ambient-secret");
    assert.equal(process.env[declaredName], undefined);
  } finally {
    if (previousAmbient === undefined) {
      delete process.env[ambientName];
    } else {
      process.env[ambientName] = previousAmbient;
    }
    if (previousDeclared === undefined) {
      delete process.env[declaredName];
    } else {
      process.env[declaredName] = previousDeclared;
    }
  }
});

test("uses a configured credential name for a native OpenCode provider", async () => {
  let createOptions;
  const factory = new OpenCodeSdkSessionFactory(async () => ({
    OpenCode: {
      async create(options) {
        createOptions = options;
        return {
          sessions: {
            async create() {
              return { id: "session-native-provider" };
            },
            async remove() {},
          },
          async close() {},
        };
      },
    },
  }));

  const handle = await factory.create(startInput());
  await handle.stop();

  assert.deepEqual(JSON.parse(createOptions.config.content), {
    providers: { nvidia: { settings: { apiKey: "{env:OPENCODE_TEST_KEY}" } } },
  });
});

test("turns SDK transport failures into a stable lifecycle error without request details", async () => {
  const factory = new OpenCodeSdkSessionFactory(async () => ({
    OpenCode: {
      async create() {
        return {
          sessions: {
            async create() {
              return { id: "session-transport-failure" };
            },
            async context() {
              throw new Error("request https://example.test/?token=supersecret failed for private prompt text");
            },
            async remove() {},
          },
          async close() {},
        };
      },
    },
  }));

  const handle = await factory.create(startInput());
  await assert.rejects(
    handle.prompt("private prompt text"),
    (error) =>
      error.code === "opencode_session_failed" &&
      error.message === "OpenCode session communication failed" &&
      error.retryable === true &&
      !error.message.includes("supersecret") &&
      !error.message.includes("private prompt"),
  );
  await handle.stop();
});

test("does not mark a failure after prompt submission as retryable", async () => {
  const factory = new OpenCodeSdkSessionFactory(async () => ({
    OpenCode: {
      async create() {
        return {
          sessions: {
            async create() {
              return { id: "session-wait-failure" };
            },
            async context() {
              return [];
            },
            async prompt() {},
            async wait() {
              throw new Error("connection lost after the prompt was submitted");
            },
            async remove() {},
          },
          async close() {},
        };
      },
    },
  }));

  const handle = await factory.create(startInput());
  await assert.rejects(
    handle.prompt("edit a file"),
    (error) =>
      error.code === "opencode_session_failed" &&
      error.message === "OpenCode session communication failed" &&
      error.retryable === false,
  );
  await handle.stop();
});

test("releases the client, endpoint proxy, and environment when session removal fails", async () => {
  let clientClosed = false;
  let proxyClosed = false;
  const previous = process.env.OPENCODE_TEST_KEY;
  delete process.env.OPENCODE_TEST_KEY;
  try {
    const input = startInput();
    input.config.models.default.base_url = "http://127.0.0.1:1/v1";
    const factory = new OpenCodeSdkSessionFactory(
      async () => ({
        OpenCode: {
          async create() {
            return {
              sessions: {
                async create() { return { id: "session-cleanup" }; },
                async remove() { throw new Error("remove failed"); },
              },
              async close() { clientClosed = true; },
            };
          },
        },
      }),
      async () => ({}),
      async () => ({ url: "http://127.0.0.1:12345", async close() { proxyClosed = true; } }),
    );

    const handle = await factory.create(input);
    await assert.rejects(handle.stop(), /remove failed/);
    assert.equal(clientClosed, true);
    assert.equal(proxyClosed, true);
    assert.equal(process.env.OPENCODE_TEST_KEY, undefined);
  } finally {
    if (previous === undefined) {
      delete process.env.OPENCODE_TEST_KEY;
    } else {
      process.env.OPENCODE_TEST_KEY = previous;
    }
  }
});

test("configures sampling for an OpenAI-compatible OpenCode provider endpoint", async () => {
  let createOptions;
  const factory = new OpenCodeSdkSessionFactory(async () => ({
    OpenCode: {
      async create(options) {
        createOptions = options;
        return {
          sessions: {
            async create() {
              return { id: "session-endpoint" };
            },
            async remove() {},
          },
          async close() {},
        };
      },
    },
  }), async () => ({}), async () => ({ url: "http://127.0.0.1:12345", async close() {} }));

  const input = startInput();
  input.config.models.default = {
    provider: "nvidia",
    model: "nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key_env: "OPENCODE_TEST_KEY",
    base_url: "http://127.0.0.1:8080/v1",
    temperature: 0.25,
    top_p: 0.8,
  };
  const handle = await factory.create(input);
  await handle.stop();

  const provider = JSON.parse(createOptions.config.content).providers.nvidia;
  assert.equal(provider.package, "aisdk:@ai-sdk/openai-compatible");
  assert.equal(provider.settings.apiKey, "{env:OPENCODE_TEST_KEY}");
  assert.equal(new URL(provider.settings.baseURL).hostname, "127.0.0.1");
  assert.deepEqual(provider.models, {
    "nvidia/nemotron-3.5-lightning-30b-a3b": { body: { temperature: 0.25, top_p: 0.8 } },
  });
});

test("restores the environment lease when startup cleanup also fails", async () => {
  const ambientName = "OPENCODE_AMBIENT_SECRET";
  const previous = process.env.OPENCODE_TEST_KEY;
  const previousAmbient = process.env[ambientName];
  delete process.env.OPENCODE_TEST_KEY;
  process.env[ambientName] = "ambient-secret";
  try {
    let observedAmbient;
    const factory = new OpenCodeSdkSessionFactory(async () => ({
      OpenCode: {
        async create() {
          observedAmbient = process.env[ambientName];
          return {
            sessions: {
              async create() {
                throw new Error("session creation failed");
              },
            },
            async close() {
              throw new Error("close failed");
            },
          };
        },
      },
    }));

    await assert.rejects(factory.create(startInput()), (error) => error.code === "opencode_start_failed");
    assert.equal(observedAmbient, undefined);
    assert.equal(process.env.OPENCODE_TEST_KEY, undefined);
    assert.equal(process.env[ambientName], "ambient-secret");
  } finally {
    if (previous === undefined) {
      delete process.env.OPENCODE_TEST_KEY;
    } else {
      process.env.OPENCODE_TEST_KEY = previous;
    }
    if (previousAmbient === undefined) {
      delete process.env[ambientName];
    } else {
      process.env[ambientName] = previousAmbient;
    }
  }
});
