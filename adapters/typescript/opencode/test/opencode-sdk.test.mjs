// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdir, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
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
        default: { provider: "openai", model: "gpt-4.1-mini", api_key_env: "OPENCODE_TEST_KEY" },
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
      openai: {
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
            return {
              plugin: { async awaitActivation() {} },
              skill: {
                async list() {
                  return {
                    data: [{ name: "review", location: await realpath(join(skillDirectory, "SKILL.md")) }],
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

    assert.deepEqual(JSON.parse(capturedConfig), {
      providers: {
        openai: {
          settings: { apiKey: "{env:OPENCODE_TEST_KEY}" },
        },
      },
      skills: [await realpath(skillDirectory)],
    });
    await handle.stop();
  } finally {
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
            plugin: { async awaitActivation() {} },
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
        async create() {
          return {
            plugin: { async awaitActivation() {} },
            skill: {
              async list() {
                return {
                  data: [
                    { name: "duplicate", location: await realpath(join(first, "SKILL.md")) },
                    { name: "duplicate", location: await realpath(join(second, "SKILL.md")) },
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

test("encodes configured skill paths literally before OpenCode parses configuration variables", async () => {
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
          return {
            plugin: { async awaitActivation() {} },
            skill: {
              async list() {
                return { data: [{ name: "literal", location: await realpath(join(skillDirectory, "SKILL.md")) }] };
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

    assert.ok(capturedConfig.includes("\\u007benv:FABRIC_SKILL_SEGMENT}"));
    assert.ok(capturedConfig.includes("\\u007bfile:secret}"));
    assert.deepEqual(JSON.parse(capturedConfig).skills, [await realpath(skillDirectory)]);
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
      openai: {
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

test("extracts the final assistant text and usage from OpenCode session history", () => {
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

  assert.deepEqual(outcome, {
    text: "second answer",
    usage: { input_tokens: 5, output_tokens: 7, total_tokens: 12, cost_usd: 0.023 },
  });
});

test("sums usage across every new assistant message in a tool-using prompt", () => {
  const outcome = extractOpenCodePromptOutcome(
    [
      {
        id: "assistant-tool-call",
        type: "assistant",
        content: [{ type: "tool", name: "read", input: { path: "example.txt" } }],
        finish: "tool-calls",
        tokens: { input: 11, output: 13 },
        cost: 0.01,
      },
      {
        id: "assistant-terminal",
        type: "assistant",
        content: [{ type: "text", text: "done" }],
        finish: "stop",
        tokens: { input: 17, output: 19 },
        cost: 0.02,
      },
    ],
    new Set(),
  );

  assert.deepEqual(outcome, {
    text: "done",
    usage: { input_tokens: 28, output_tokens: 32, total_tokens: 60, cost_usd: 0.03 },
  });
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
  let contextCalls = 0;
  const factory = new OpenCodeSdkSessionFactory(async () => ({
    OpenCode: {
      async create() {
        return {
          sessions: {
            async create() { return { id: "session-terminal-error" }; },
            async context() {
              contextCalls += 1;
              return contextCalls === 1
                ? []
                : [{
                  id: "assistant-error",
                  type: "assistant",
                  content: [],
                  finish: "error",
                  error: { message: "provider error" },
                  tokens: { input: 5, output: 7 },
                  cost: 0.012,
                }];
            },
            async prompt() {},
            async wait() {},
            async diff() { return [{ file: "example.txt", patch: "diff --git a/example.txt b/example.txt\n+changed\n" }]; },
            async remove() {},
          },
          async close() {},
        };
      },
    },
  }));

  const handle = await factory.create(startInput());
  assert.deepEqual(await handle.prompt("edit"), {
    errorMessage: "OpenCode model invocation failed",
    patch: "diff --git a/example.txt b/example.txt\n+changed\n",
    usage: { input_tokens: 5, output_tokens: 7, total_tokens: 12, cost_usd: 0.012 },
  });
  await handle.stop();
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
              providers: { openai: { settings: { apiKey: "{env:OPENCODE_TEST_KEY}" } } },
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
          location: { directory: "/workspace" },
          model: { providerID: "openai", id: "gpt-4.1-mini" },
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
    providers: { openai: { settings: { apiKey: "{env:OPENCODE_TEST_KEY}" } } },
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
    provider: "local-test",
    model: "test-model",
    api_key_env: "OPENCODE_TEST_KEY",
    base_url: "http://127.0.0.1:8080/v1",
    temperature: 0.25,
    top_p: 0.8,
  };
  const handle = await factory.create(input);
  await handle.stop();

  const provider = JSON.parse(createOptions.config.content).providers["local-test"];
  assert.equal(provider.package, "aisdk:@ai-sdk/openai-compatible");
  assert.equal(provider.settings.apiKey, "{env:OPENCODE_TEST_KEY}");
  assert.equal(new URL(provider.settings.baseURL).hostname, "127.0.0.1");
  assert.deepEqual(provider.models, { "test-model": { body: { temperature: 0.25, top_p: 0.8 } } });
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
