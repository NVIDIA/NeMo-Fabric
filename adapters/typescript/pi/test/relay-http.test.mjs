// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { relayHttpModelConfig } from "../dist/relay-http.js";
import { PiRelayFactory } from "../dist/relay.js";
import { writeRelayConfigs } from "../dist/relay-config.js";

const current = { version: [0, 11, 0] };
const old = { version: [0, 10, 4] };

test("preserves omitted trust, explicit denial, and normalized exact hosts", () => {
  assert.deepEqual(relayHttpModelConfig({}, old), {});
  assert.deepEqual(
    relayHttpModelConfig({ insecure_http_model_hosts: [] }, current),
    {
      upstream: { insecure_http_model_hosts: [] },
    },
  );
  assert.deepEqual(
    relayHttpModelConfig(
      {
        insecure_http_model_hosts: [
          "MODEL.TEST.",
          "model.test",
          "192.168.1.10",
          "2001:0DB8:0:0::1",
        ],
      },
      current,
    ),
    {
      upstream: {
        insecure_http_model_hosts: [
          "model.test",
          "192.168.1.10",
          "2001:db8::1",
        ],
      },
    },
  );
});

for (const hosts of [
  null,
  true,
  "model.test",
  [1],
  [""],
  ["http://model.test"],
  ["model.test:80"],
  ["model.test/v1"],
  ["*.test"],
  ["010.0.0.1"],
  ["10.1"],
  ["0x7f000001"],
  ["0x"],
  ["model.0x"],
  [" secret"],
  ["[2001:db8::1]"],
  ["fe80::1%eth0"],
  ["2001:db8::g"],
]) {
  test(`rejects malformed model-host trust ${JSON.stringify(hosts)}`, () => {
    assert.throws(
      () => relayHttpModelConfig({ insecure_http_model_hosts: hosts }, current),
      {
        code: "pi_relay_invalid_http_model_hosts",
      },
    );
  });
}

for (const hosts of [[], ["model.test"]]) {
  test(`requires Relay 0.11 for explicit trust ${JSON.stringify(hosts)}`, () => {
    assert.throws(
      () => relayHttpModelConfig({ insecure_http_model_hosts: hosts }, old),
      /requires NeMo Relay CLI >=0\.11\.0/,
    );
  });
}

test("factory writes explicit hosts into gateway TOML and rejects old Relay before launch", async () => {
  const root = await mkdtemp(join(tmpdir(), "fabric-pi-http-"));
  const previous = process.env.FABRIC_RELAY_CONFIG_PATH;
  process.env.FABRIC_RELAY_CONFIG_PATH = join(root, "relay.json");
  await writeFile(
    join(root, "extension.js"),
    "export default function () {}\n",
  );
  let version = current;
  let launches = 0;
  const factory = new PiRelayFactory({
    async resolveCommand() {
      return "/opt/bin/nemo-relay";
    },
    async checkContract() {
      return version;
    },
    async loadPluginConfig() {
      return { version: 1, components: [] };
    },
    writeConfigs: writeRelayConfigs,
    async findPort() {
      return 41001;
    },
    async startGateway() {
      launches += 1;
      return {};
    },
    async stopGateway() {},
  });
  const input = {
    agentName: "pi-http",
    baseDir: root,
    config: {
      harness: {
        settings: {
          relay_extension_path: join(root, "extension.js"),
          insecure_http_model_hosts: [],
        },
      },
    },
    runtimeContext: {
      environment: { workspace: root },
      runtime_id: "pi-http",
      telemetry: { relay_enabled: true },
    },
  };
  try {
    for (const hosts of [[], ["MODEL.TEST."]]) {
      input.config.harness.settings.insecure_http_model_hosts = hosts;
      const runtime = await factory.start(input, {
        api: "openai-responses",
        baseUrl: "http://model.test:8000/v1",
      });
      try {
        const toml = await readFile(
          join(root, "relay-config", "config.toml"),
          "utf8",
        );
        assert.match(toml, /\[upstream\]/);
        assert.ok(
          toml.includes(
            `insecure_http_model_hosts = ${JSON.stringify(hosts.length === 0 ? [] : ["model.test"])}`,
          ),
        );
      } finally {
        await runtime.stop();
      }
    }
    version = old;
    await assert.rejects(
      factory.start(input, {
        api: "openai-responses",
        baseUrl: "http://model.test:8000/v1",
      }),
      { code: "pi_relay_invalid_http_model_hosts" },
    );
    assert.equal(launches, 2);
    version = current;
    const restarted = await factory.start(input, {
      api: "openai-responses",
      baseUrl: "http://model.test:8000/v1",
    });
    await restarted.stop();
    assert.equal(launches, 3);
  } finally {
    if (previous === undefined) delete process.env.FABRIC_RELAY_CONFIG_PATH;
    else process.env.FABRIC_RELAY_CONFIG_PATH = previous;
    await rm(root, { recursive: true, force: true });
  }
});
