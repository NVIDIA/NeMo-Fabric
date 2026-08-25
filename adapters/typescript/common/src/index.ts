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
