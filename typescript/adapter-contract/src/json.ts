// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is generated from the canonical adapter-contract JSON Schemas.
// Do not edit it directly; run `npm run generate` instead.

/** A JSON scalar value. */
export type JsonPrimitive = string | number | boolean | null;

/** A JSON object with recursively JSON-compatible values. */
export interface JsonObject {
  [key: string]: JsonValue;
}

/** A JSON array with recursively JSON-compatible values. */
export type JsonArray = JsonValue[];

/** Any value representable by JSON. */
export type JsonValue = JsonPrimitive | JsonObject | JsonArray;
