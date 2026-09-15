// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// OpenCode v2.0.3 always emits prompt_cache_key for its OpenAI-compatible
// protocol. Some compatible providers reject that OpenAI-specific extension,
// so a loopback proxy removes it only for explicitly configured endpoints.

import { createServer, type IncomingHttpHeaders, type Server } from "node:http";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

function hopByHopHeaderNames(headers: IncomingHttpHeaders): Set<string> {
  const connection = headers.connection;
  const connectionValues = Array.isArray(connection) ? connection : [connection];
  return new Set([
    ...HOP_BY_HOP_HEADERS,
    ...connectionValues
      .filter((value): value is string => typeof value === "string")
      .flatMap((value) => value.split(","))
      .map((value) => value.trim().toLowerCase())
      .filter((value) => value.length > 0),
  ]);
}

export function forwardedHeaders(headers: IncomingHttpHeaders): Headers {
  const result = new Headers();
  const hopByHopHeaders = hopByHopHeaderNames(headers);
  for (const [name, value] of Object.entries(headers)) {
    const normalizedName = name.toLowerCase();
    if (
      value === undefined ||
      normalizedName === "host" ||
      normalizedName === "content-length" ||
      hopByHopHeaders.has(normalizedName)
    ) {
      continue;
    }
    result.set(name, Array.isArray(value) ? value.join(", ") : value);
  }
  return result;
}

async function bodyWithoutPromptCacheKey(request: AsyncIterable<Buffer | string>): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  const body = Buffer.concat(chunks);
  try {
    const payload: unknown = JSON.parse(body.toString("utf8"));
    if (typeof payload === "object" && payload !== null && !Array.isArray(payload)) {
      delete (payload as Record<string, unknown>).prompt_cache_key;
      return Buffer.from(JSON.stringify(payload));
    }
  } catch {
    // Forward non-JSON payloads unchanged.
  }
  return body;
}

function targetUrl(baseUrl: string, path: string): string {
  const base = new URL(baseUrl);
  const request = new URL(path, "http://127.0.0.1");
  base.pathname = `${base.pathname.replace(/\/$/u, "")}${request.pathname}`;
  for (const [name, value] of request.searchParams) {
    base.searchParams.append(name, value);
  }
  return base.toString();
}

function forwardedResponseHeaders(headers: Headers): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [name, value] of headers.entries()) {
    if (name === "content-encoding" || name === "content-length" || HOP_BY_HOP_HEADERS.has(name)) {
      continue;
    }
    result[name] = value;
  }
  return result;
}

export class ModelEndpointProxy {
  private readonly server: Server;
  readonly url: string;

  private constructor(server: Server, url: string) {
    this.server = server;
    this.url = url;
  }

  static async create(baseUrl: string): Promise<ModelEndpointProxy> {
    const server = createServer(async (request, response) => {
      try {
        const body = await bodyWithoutPromptCacheKey(request);
        const upstream = await fetch(targetUrl(baseUrl, request.url ?? "/"), {
          method: request.method,
          headers: forwardedHeaders(request.headers),
          ...(body.length === 0 ? {} : { body: body.toString("utf8") }),
        });
        response.writeHead(upstream.status, forwardedResponseHeaders(upstream.headers));
        if (upstream.body === null) {
          response.end();
          return;
        }
        await pipeline(
          Readable.fromWeb(upstream.body as unknown as import("node:stream/web").ReadableStream),
          response,
        );
      } catch {
        if (response.headersSent) {
          response.destroy();
          return;
        }
        response.writeHead(502, { "content-type": "text/plain" });
        response.end("OpenCode model-provider proxy could not reach the configured endpoint");
      }
    });
    await new Promise<void>((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", () => {
        server.off("error", reject);
        resolve();
      });
    });
    const address = server.address();
    if (address === null || typeof address === "string") {
      server.close();
      throw new Error("OpenCode model-provider proxy did not bind a local port");
    }
    return new ModelEndpointProxy(server, `http://127.0.0.1:${address.port}`);
  }

  async close(): Promise<void> {
    if (!this.server.listening) {
      return;
    }
    this.server.closeAllConnections?.();
    await new Promise<void>((resolve, reject) => {
      this.server.close((error) => {
        if (error === undefined || (error as NodeJS.ErrnoException).code === "ERR_SERVER_NOT_RUNNING") {
          resolve();
          return;
        }
        reject(error);
      });
    });
  }
}
