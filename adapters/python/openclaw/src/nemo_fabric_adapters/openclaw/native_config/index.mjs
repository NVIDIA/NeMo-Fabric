// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getGlobalHookRunner } from "openclaw/plugin-sdk/plugin-runtime";
import { normalizeAgentId } from "openclaw/plugin-sdk/routing";

/** Add the managed instruction while retaining OpenClaw's native prompt. */
export default {
  id: "fabric-shared-config",
  name: "NeMo Fabric shared configuration",
  register(api) {
    const { agentId, systemInstruction } = api.pluginConfig;
    const managedAgentId = normalizeAgentId(agentId);
    const instructionContext = `## Conversation Context\n${systemInstruction}`;
    api.on("before_prompt_build", (_event, context) => {
      if (context.agentId === managedAgentId) {
        return { appendSystemContext: instructionContext };
      }
    }, { priority: 1000 });
    api.registerHttpRoute({
      path: "/fabric/shared-config/health",
      auth: "gateway",
      match: "exact",
      async handler(req, res) {
        if (req.method !== "GET") {
          res.writeHead(405, { Allow: "GET" });
          res.end();
          return;
        }
        let ready = false;
        try {
          const runner = getGlobalHookRunner();
          const result = await runner?.runBeforePromptBuild(
            { prompt: "", messages: [] },
            { agentId: managedAgentId },
          );
          ready = result?.appendSystemContext === instructionContext
            && result?.systemPrompt === undefined;
        } catch {
          // Startup fails closed without returning instructions or conversation data.
        }
        res.writeHead(ready ? 200 : 503, {
          "Content-Type": "application/json",
          "Cache-Control": "no-store",
        });
        res.end(JSON.stringify({ ready, agent_id: managedAgentId }));
      },
    });
  },
};
