// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { normalizeAgentId } from "openclaw/plugin-sdk/routing";

/** Apply the managed agent's replacement prompt at the shared Gateway boundary. */
export default {
  id: "fabric-shared-config",
  name: "NeMo Fabric shared configuration",
  register(api) {
    const { agentId, systemInstruction } = api.pluginConfig;
    const managedAgentId = normalizeAgentId(agentId);
    api.on("before_prompt_build", (_event, context) => {
      if (context.agentId === managedAgentId) {
        return { systemPrompt: systemInstruction };
      }
    }, { priority: 1000 });
  },
};
