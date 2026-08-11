// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AdapterConfigSupport,
  AdapterDescriptor,
} from "./generated/adapter-descriptor.js";
import type { AgentInstructionConfig } from "./generated/agent-config.js";
import type { EnvironmentHandle } from "./generated/runtime-context.js";
import { ADAPTER_CONTRACT_VERSION } from "./version.js";

export type * from "./generated/adapter-descriptor.js";
export type * from "./generated/agent-config.js";
export type * from "./generated/runtime-context.js";
export type {
  JsonArray,
  JsonObject,
  JsonPrimitive,
  JsonValue,
} from "./json.js";

/** Version literal accepted by the negotiated adapter descriptor contract. */
export type AdapterContractVersion = typeof ADAPTER_CONTRACT_VERSION;

/** Adapter implementation kind. */
export type AdapterKind = AdapterDescriptor["adapter_kind"];

/** Configuration object delivered to an adapter lifecycle host. */
export type AdapterConfigInput = NonNullable<AdapterConfigSupport["input"]>;

/** How an instruction value is applied to the selected harness. */
export type InstructionMode = NonNullable<AgentInstructionConfig["mode"]>;

/** Where NeMo Fabric control code runs relative to the environment. */
export type ControlLocation = EnvironmentHandle["control_location"];

/** Whether NeMo Fabric owns the underlying environment resource. */
export type EnvironmentOwnership = EnvironmentHandle["ownership"];

export { ADAPTER_CONTRACT_VERSION };
