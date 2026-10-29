// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Experimental Kiro 2.34.2 service protocol. Keep this seam adapter-local.
import { randomUUID } from "node:crypto";
import { createServer, type IncomingMessage } from "node:http";
import { crc32 } from "node:zlib";

import type { AgentUsage } from "nemo-fabric-adapter-contract";
import { LifecycleError } from "nemo-fabric-adapters-common";

type ObjectValue = Record<string, unknown>;
const LIMIT = 8 * 1024 * 1024;

function object(value: unknown): ObjectValue {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("Invalid object");
  return value as ObjectValue;
}

function text(value: unknown): string {
  if (typeof value !== "string") throw new Error("Invalid text");
  return value;
}

function array(value: unknown): unknown[] {
  if (!Array.isArray(value)) throw new Error("Invalid list");
  return value;
}

/** Preserve native history and tool results; unsupported content fails closed. */
export function translateKiroRequest(body: unknown, model: string): ObjectValue {
  const state = object(object(body).conversationState);
  const current = object(object(state.currentMessage).userInputMessage);
  const messages: ObjectValue[] = [];
  for (const entry of [...array(state.history ?? []), state.currentMessage]) {
    const message = object(entry);
    if (message.userInputMessage !== undefined) {
      const user = object(message.userInputMessage);
      if (array(user.images ?? []).length || array(user.documents ?? []).length) throw new Error("Unsupported content");
      const context = object(user.userInputMessageContext ?? {});
      const results = array(context.toolResults ?? []);
      for (const result of results) {
        const tool = object(result);
        const content = array(tool.content)
          .map((part) => text(object(part).text))
          .join("\n");
        if (tool.status !== "success" && tool.status !== "error") throw new Error("Unsupported tool status");
        messages.push({
          role: "tool",
          tool_call_id: text(tool.toolUseId),
          content: tool.status === "error" ? `Tool error:\n${content}` : content,
        });
      }
      const content = text(user.content);
      if (content.length > 0 || results.length === 0) messages.push({ role: "user", content });
    } else if (message.assistantResponseMessage !== undefined) {
      const assistant = object(message.assistantResponseMessage);
      const tools = array(assistant.toolUses ?? []).map((value) => {
        const tool = object(value);
        return {
          id: text(tool.toolUseId),
          type: "function",
          function: {
            name: text(tool.name),
            arguments: JSON.stringify(object(tool.input)),
          },
        };
      });
      messages.push({
        role: "assistant",
        content: text(assistant.content),
        ...(tools.length === 0 ? {} : { tool_calls: tools }),
      });
    } else throw new Error("Unsupported message");
  }
  const tools = array(object(current.userInputMessageContext ?? {}).tools ?? []).map((value) => {
    const tool = object(object(value).toolSpecification);
    return {
      type: "function",
      function: {
        name: text(tool.name),
        description: text(tool.description),
        parameters: object(object(tool.inputSchema).json),
      },
    };
  });
  return {
    model,
    messages,
    stream: false,
    ...(tools.length === 0 ? {} : { tools }),
  };
}

export function encodeKiroEvent(type: string, payload: unknown): Buffer {
  const headers = Buffer.concat(
    Object.entries({
      ":message-type": "event",
      ":event-type": type,
      ":content-type": "application/json",
    }).map(([name, value]) => {
      const header = Buffer.alloc(1 + Buffer.byteLength(name) + 3 + Buffer.byteLength(value));
      header.writeUInt8(Buffer.byteLength(name), 0);
      header.write(name, 1);
      const offset = 1 + Buffer.byteLength(name);
      header.writeUInt8(7, offset);
      header.writeUInt16BE(Buffer.byteLength(value), offset + 1);
      header.write(value, offset + 3);
      return header;
    }),
  );
  const data = Buffer.from(JSON.stringify(payload));
  const frame = Buffer.alloc(16 + headers.length + data.length);
  frame.writeUInt32BE(frame.length, 0);
  frame.writeUInt32BE(headers.length, 4);
  frame.writeUInt32BE(crc32(frame.subarray(0, 8)), 8);
  headers.copy(frame, 12);
  data.copy(frame, 12 + headers.length);
  frame.writeUInt32BE(crc32(frame.subarray(0, -4)), frame.length - 4);
  return frame;
}

async function readBody(body: AsyncIterable<Uint8Array>): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of body) {
    size += chunk.length;
    if (size > LIMIT) throw new Error("Body too large");
    chunks.push(Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}

function operation(request: IncomingMessage): string {
  if (request.url === "/")
    return (
      String(request.headers["x-amz-target"] ?? "")
        .split(".")
        .at(-1) ?? ""
    );
  return (
    (
      {
        "/List-Available-Models": "ListAvailableModels",
        "/Get-Profile": "GetProfile",
        "/generateAssistantResponse": "GenerateAssistantResponse",
      } as Record<string, string>
    )[request.url ?? ""] ?? ""
  );
}

export interface KiroByomBridge {
  readonly url: string;
  readonly token: string;
  beginInvocation(): void;
  endInvocation(): AgentUsage | undefined;
  stop(): Promise<void>;
}

export async function startKiroByomBridge(options: {
  model: string;
  baseUrl: string;
  apiKey?: string;
}): Promise<KiroByomBridge> {
  const token = randomUUID();
  const controllers = new Set<AbortController>();
  let active = false;
  let stopped = false;
  let complete = true;
  let calls = 0;
  let input = 0;
  let output = 0;
  let cached = 0;
  let cacheKnown = true;
  let generation = 0;
  // Native background/subagent lifetimes are not established by a root ACP
  // terminal. Keep complete totals unknown for the rest of such a session.
  let scopeUnverified = false;
  const count = (value: unknown): value is number => Number.isSafeInteger(value) && Number(value) >= 0;
  const server = createServer(async (request, response) => {
    try {
      if (request.headers.authorization !== `Bearer ${token}`) {
        response.writeHead(401).end();
        return;
      }
      const op = operation(request);
      const json = (body: unknown) =>
        response.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify(body));
      if (
        request.method === "POST" &&
        request.url === "/service/KiroWebBearerService/operation/GetResolvedConfigManifest"
      ) {
        await readBody(request);
        response.writeHead(200, { "content-type": "application/cbor" }).end(Buffer.from([0xa0]));
        return;
      }
      if (request.method !== "POST" && !(request.method === "GET" && op === "ListAvailableModels")) {
        response.writeHead(405).end();
        return;
      }
      if (op === "ListAvailableModels") {
        json({
          models: [
            {
              modelId: options.model,
              modelName: options.model,
              provider: "OpenAI compatible (experimental)",
              tokenLimits: { maxInputTokens: 131072, maxOutputTokens: 16384 },
            },
          ],
          defaultModel: { modelId: options.model },
        });
        return;
      }
      if (op === "GetProfile") {
        json({
          profile: {
            profileArn: "arn:aws:codewhisperer:us-east-1:000000000000:profile/fabric-byom",
          },
        });
        return;
      }
      if (op === "GetEffectiveEnterpriseManagementResourceWithToken") {
        json({});
        return;
      }
      if (op === "InvokeMCP") {
        json({ jsonrpc: "2.0", id: 1, result: { tools: [] } });
        return;
      }
      if (op !== "GenerateAssistantResponse") {
        response.writeHead(404).end();
        return;
      }
      if (!active || stopped) {
        scopeUnverified = true;
        response.writeHead(409).end();
        return;
      }
      const invocation = generation;
      const controller = new AbortController();
      controllers.add(controller);
      response.on("close", () => {
        if (!response.writableFinished) controller.abort();
      });
      try {
        const body = translateKiroRequest(JSON.parse((await readBody(request)).toString()), options.model);
        if (!active || stopped || invocation !== generation) throw new Error("Invocation ended");
        const upstream = await fetch(`${options.baseUrl.replace(/\/$/u, "")}/chat/completions`, {
          method: "POST",
          redirect: "error",
          signal: controller.signal,
          headers: {
            "content-type": "application/json",
            ...(options.apiKey === undefined ? {} : { authorization: `Bearer ${options.apiKey}` }),
          },
          body: JSON.stringify(body),
        });
        if (!upstream.ok || upstream.body === null) {
          await upstream.body?.cancel();
          throw new Error("Provider failed");
        }
        const result = object(JSON.parse((await readBody(upstream.body)).toString()));
        calls++;
        try {
          const usage = object(result.usage);
          if (!count(usage.prompt_tokens) || !count(usage.completion_tokens)) complete = false;
          else {
            input += usage.prompt_tokens;
            output += usage.completion_tokens;
            if (
              usage.total_tokens !== undefined &&
              usage.total_tokens !== usage.prompt_tokens + usage.completion_tokens
            )
              complete = false;
            const details = usage.prompt_tokens_details;
            const cache = details === undefined || details === null ? undefined : object(details).cached_tokens;
            if (!count(cache) || cache > usage.prompt_tokens) cacheKnown = false;
            else cached += cache;
            if (!count(input) || !count(output) || !count(input + output)) complete = false;
          }
        } catch {
          complete = false;
        }
        const choice = object(array(result.choices)[0]);
        const message = object(choice.message);
        const frames: Buffer[] = [];
        if (message.content !== undefined && message.content !== null) {
          frames.push(
            encodeKiroEvent("assistantResponseEvent", {
              content: text(message.content),
              modelId: options.model,
            }),
          );
        }
        const tools = array(message.tool_calls ?? []);
        for (const value of tools) {
          const tool = object(value);
          if (tool.type !== "function") throw new Error("Unsupported tool");
          const fn = object(tool.function);
          const fields = { name: text(fn.name), toolUseId: text(tool.id) };
          const args = text(fn.arguments);
          const input = object(JSON.parse(args));
          if (fn.name === "invoke_sub_agent" || (fn.name === "execute_bash" && input.run_in_background === true)) {
            scopeUnverified = true;
          }
          frames.push(
            encodeKiroEvent("toolUseEvent", fields),
            encodeKiroEvent("toolUseEvent", { ...fields, input: args }),
            encodeKiroEvent("toolUseEvent", { ...fields, stop: true }),
          );
        }
        const stopReason =
          choice.finish_reason === "tool_calls" && tools.length > 0
            ? "TOOL_USE"
            : choice.finish_reason === "stop" && tools.length === 0
              ? "END_TURN"
              : choice.finish_reason === "length" && tools.length === 0
                ? "MAX_TOKENS"
                : undefined;
        if (stopReason === undefined) throw new Error("Unsupported completion");
        frames.push(encodeKiroEvent("metadataEvent", { stopReason }));
        response.writeHead(200, {
          "content-type": "application/vnd.amazon.eventstream",
          "x-amzn-requestid": randomUUID(),
        });
        response.end(Buffer.concat(frames));
      } catch {
        complete = false;
        response.writeHead(502).end();
      } finally {
        controllers.delete(controller);
      }
    } catch {
      complete = false;
      if (!response.headersSent) response.writeHead(400);
      response.end();
    }
  });
  server.requestTimeout = 30_000;
  server.headersTimeout = 10_000;
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("Missing bridge address");
  return {
    url: `http://127.0.0.1:${address.port}`,
    token,
    beginInvocation() {
      if (active || stopped || controllers.size > 0)
        throw new LifecycleError("kiro_byom_state_invalid", "Kiro BYOM bridge is unavailable");
      generation++;
      active = true;
      complete = true;
      calls = 0;
      input = 0;
      output = 0;
      cached = 0;
      cacheKnown = true;
    },
    endInvocation() {
      active = false;
      generation++;
      if (controllers.size > 0) {
        complete = false;
        for (const controller of controllers) controller.abort();
      }
      return complete && !scopeUnverified && calls > 0
        ? {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
            input_tokens_include_cache: true,
            ...(cacheKnown ? { cached_input_tokens: cached } : {}),
          }
        : undefined;
    },
    async stop() {
      stopped = true;
      active = false;
      for (const controller of controllers) controller.abort();
      await new Promise<void>((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
        server.closeAllConnections();
      });
    },
  };
}
