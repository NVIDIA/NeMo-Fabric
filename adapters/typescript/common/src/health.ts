// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Authenticated loopback transport for adapter health probes.

import { createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import { writeFile, unlink } from "node:fs/promises";
import { createRequire } from "node:module";
import { createServer, type Socket } from "node:net";
import { join } from "node:path";
import type { AdapterCheckResult, CheckLevel, CheckReason } from "nemo-fabric-adapter-contract";

const require = createRequire(import.meta.url);
const Ajv = (require("ajv/dist/2020.js") as { default: typeof import("ajv/dist/2020.js").default }).default;
const validateResult = new Ajv({ strict: false }).compile(
  require("nemo-fabric-adapter-contract/schemas/adapter-check-result") as object,
);
const LIMIT = 16 * 1024;

export function unobservedHealth(reason: CheckReason): AdapterCheckResult {
  return { outcome: "unknown", reason_code: reason, activity: "unknown", source: "fabric.adapter", observed_at: null };
}

export async function serveHealth(
  runtimeId: string,
  directory: string,
  probe: (level: CheckLevel) => Promise<AdapterCheckResult>,
): Promise<{ close(): Promise<void> }> {
  const token = randomBytes(32).toString("hex");
  const sockets = new Set<Socket>();
  const server = createServer((socket) => {
    sockets.add(socket);
    socket.on("close", () => sockets.delete(socket));
    socket.on("error", () => socket.destroy());
    socket.setEncoding("utf8");
    socket.setTimeout(3000, () => socket.destroy());
    let input = "";
    let dispatched = false;
    socket.on("data", (chunk: string) => {
      if (dispatched) return;
      input += chunk;
      if (Buffer.byteLength(input) > LIMIT) {
        socket.destroy();
        return;
      }
      if (!input.includes("\n")) return;
      dispatched = true;
      let request: Record<string, unknown>;
      let nonce: string;
      try {
        const value: unknown = JSON.parse(input);
        if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error();
        const envelope = value as Record<string, unknown>;
        if (
          Object.keys(envelope).sort().join(",") !== "nonce,payload,proof" ||
          typeof envelope.nonce !== "string" ||
          !/^[a-f0-9]{64}$/.test(envelope.nonce) ||
          typeof envelope.payload !== "string" ||
          typeof envelope.proof !== "string" ||
          !/^[a-f0-9]{64}$/.test(envelope.proof)
        )
          throw new Error();
        nonce = envelope.nonce;
        const expected = createHmac("sha256", token).update(`request\n${nonce}\n${envelope.payload}`).digest();
        if (!timingSafeEqual(Buffer.from(envelope.proof, "hex"), expected)) throw new Error();
        const parsed: unknown = JSON.parse(envelope.payload);
        if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) throw new Error();
        request = parsed as Record<string, unknown>;
        if (
          Object.keys(request).sort().join(",") !== "level,runtime_id,timeout_seconds" ||
          typeof request.timeout_seconds !== "number" ||
          !Number.isFinite(request.timeout_seconds) ||
          request.timeout_seconds <= 0
        )
          throw new Error();
      } catch {
        socket.destroy();
        return;
      }
      const budget = request.timeout_seconds as number;
      socket.setTimeout(Math.min(Math.ceil(budget * 1000) + 100, 2_147_483_647), () => socket.destroy());
      const started = performance.now();
      void (async () => {
        let timer: NodeJS.Timeout | undefined;
        let result: AdapterCheckResult;
        try {
          if (request.runtime_id !== runtimeId) result = unobservedHealth("runtime_unavailable");
          else if (!["live", "active", "ready"].includes(request.level as string)) {
            result = { ...unobservedHealth("unsupported_level"), outcome: "unsupported" };
          } else {
            result = await Promise.race([
              probe(request.level as CheckLevel),
              new Promise<AdapterCheckResult>((resolve) => {
                timer = setTimeout(
                  () => resolve(unobservedHealth("probe_timeout")),
                  Math.min(Math.ceil(budget * 1000), 2_147_483_647),
                );
              }),
            ]);
            if (
              !validateResult(result) ||
              (result.outcome === "passed") !== (result.reason_code === null) ||
              (result.outcome === "passed" &&
                (typeof result.observed_at !== "string" || !Number.isFinite(Date.parse(result.observed_at))))
            ) {
              result = unobservedHealth("probe_error");
            }
            if (performance.now() - started >= budget * 1000) result = unobservedHealth("probe_timeout");
          }
        } catch {
          result = unobservedHealth("probe_error");
        } finally {
          if (timer !== undefined) clearTimeout(timer);
        }
        if (!socket.destroyed) {
          const payload = JSON.stringify(result);
          const proof = createHmac("sha256", token).update(`response\n${nonce}\n${payload}`).digest("hex");
          socket.end(`${JSON.stringify({ payload, proof })}\n`);
        }
      })();
    });
  });
  const path = join(directory, "health.endpoint.json");
  try {
    await new Promise<void>((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", resolve);
    });
    const address = server.address();
    if (address === null || typeof address === "string") throw new Error("health channel unavailable");
    await writeFile(
      path,
      JSON.stringify({ protocol_version: "fabric.check/v1alpha1", runtime_id: runtimeId, port: address.port, token }),
      { flag: "wx", mode: 0o600 },
    );
    server.unref();
  } catch (error) {
    server.close();
    throw error;
  }
  return {
    async close() {
      for (const socket of sockets) socket.destroy();
      await new Promise<void>((resolve) => server.close(() => resolve()));
      await unlink(path).catch(() => undefined);
    },
  };
}
