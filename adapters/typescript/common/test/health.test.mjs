// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createHmac, randomBytes } from "node:crypto";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { connect } from "node:net";
import test from "node:test";
import { serveHealth } from "../dist/health.js";

const passed = () => ({
  outcome: "passed",
  reason_code: null,
  activity: "idle",
  source: "fabric.adapter.pi.session",
  observed_at: new Date().toISOString(),
});

async function setup(t, probe) {
  const directory = await mkdtemp(join(tmpdir(), "fabric-health-"));
  const channel = await serveHealth("runtime-1", directory, probe);
  t.after(async () => {
    await channel.close();
    await rm(directory, { recursive: true, force: true });
  });
  const path = join(directory, "health.endpoint.json");
  const endpoint = JSON.parse(await readFile(path, "utf8"));
  return { endpoint, path };
}

async function call(endpoint, fields = {}) {
  const { token = endpoint.token, ...requestFields } = fields;
  const nonce = randomBytes(32).toString("hex");
  const payload = JSON.stringify({ runtime_id: "runtime-1", level: "live", timeout_seconds: 0.5, ...requestFields });
  const proof = createHmac("sha256", token).update(`request\n${nonce}\n${payload}`).digest("hex");
  const message = JSON.stringify({ nonce, payload, proof });
  assert.equal(message.includes(endpoint.token), false, "the endpoint secret must stay off the wire");
  return new Promise((resolve, reject) => {
    const socket = connect({ host: "127.0.0.1", port: endpoint.port });
    let output = "";
    socket.setTimeout(1000, () => {
      socket.destroy();
      reject(new Error("test health deadline"));
    });
    socket.on("error", reject);
    socket.on("data", (chunk) => {
      output += chunk;
    });
    socket.on("end", () => {
      try {
        if (!output) return resolve(null);
        const reply = JSON.parse(output);
        assert.deepEqual(Object.keys(reply).sort(), ["payload", "proof"]);
        assert.equal(
          reply.proof,
          createHmac("sha256", token).update(`response\n${nonce}\n${reply.payload}`).digest("hex"),
        );
        resolve(JSON.parse(reply.payload));
      } catch (error) {
        reject(error);
      }
    });
    socket.on("connect", () => socket.write(`${message}\n`));
  });
}

test("health endpoint is private, authenticated, and selects one runtime", async (t) => {
  const probes = [];
  const { endpoint, path } = await setup(t, async (level) => {
    probes.push(level);
    return passed();
  });
  if (process.platform !== "win32") assert.equal((await stat(path)).mode & 0o777, 0o600);
  assert.equal(endpoint.token.length, 64);
  assert.equal(await call(endpoint, { token: "wrong" }), null);
  assert.deepEqual(probes, []);
  assert.equal((await call(endpoint, { runtime_id: "other" })).reason_code, "runtime_unavailable");
  assert.deepEqual(probes, []);
  assert.equal((await call(endpoint)).outcome, "passed");
  assert.deepEqual(probes, ["live"]);
});

test("health timeouts leave probes and the channel running", async (t) => {
  const release = Promise.withResolvers();
  let completed = false;
  const { endpoint } = await setup(t, async () => {
    await release.promise;
    completed = true;
    return passed();
  });
  const result = await call(endpoint, { timeout_seconds: 0.01 });
  assert.equal(result.reason_code, "probe_timeout");
  assert.equal(completed, false);
  release.resolve();
  assert.equal((await call(endpoint)).outcome, "passed");
  assert.equal(completed, true);
});

test("health exceptions and invalid results emit fixed evidence", async (t) => {
  let mode = "throw";
  const { endpoint } = await setup(t, async () => {
    if (mode === "throw") throw new Error("PRIVATE credentials and prompt");
    return { ...passed(), source: "PRIVATE", raw: "PRIVATE" };
  });
  for (const value of ["throw", "invalid"]) {
    mode = value;
    const result = await call(endpoint);
    assert.equal(result.reason_code, "probe_error");
    assert.equal(JSON.stringify(result).includes("PRIVATE"), false);
  }
});
