// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Public entry point for TypeScript process adapters. It exposes the lifecycle
// host, adapter-facing runtime interfaces, and normalized lifecycle errors
// without exposing the host's internal protocol machinery.

export {
  LifecycleError,
  serve,
  type AdapterRuntime,
  type AdapterRuntimeFactory,
  type AdapterStartInput,
  type LifecycleHostOptions,
} from "./lifecycle.js";

export { insecureHttpModelHosts, modelEndpointAllowed } from "./model-endpoint.js";

export {
  AcpRunnerError,
  startAcpSession,
  type AcpConfigOption,
  type AcpMcpServer,
  type AcpPermissionDecision,
  type AcpPermissionResolver,
  type AcpPromptResult,
  type AcpSession,
  type AcpSessionOptions,
  type AcpSessionReadinessNotification,
} from "./acp-runner.js";
export {
  VERSIONS_REPORT_KEY, boundedText, missingFact, packageVersionFact,
  packageVersionReport, withPackageVersion, type ServedModelFact,
} from "./provenance.js";
export { unobservedHealth } from "./health.js";
