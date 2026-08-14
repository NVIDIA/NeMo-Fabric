// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFile, mkdir, readdir, writeFile } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { compile } from "json-schema-to-typescript";

import {
  assertAdapterSchemaInventory,
  assertRunResultConditionals,
} from "./projection-guards.mjs";

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = resolve(packageRoot, "../..");
const schemaDirectory = resolve(repositoryRoot, "schemas/adapter-contract");
const checkOnly = process.argv.slice(2).includes("--check");

const unexpectedArguments = process.argv
  .slice(2)
  .filter((argument) => argument !== "--check");
if (unexpectedArguments.length > 0) {
  throw new Error(`Unexpected arguments: ${unexpectedArguments.join(", ")}`);
}

const currentYear = new Date().getFullYear();
const banner = `// SPDX-FileCopyrightText: Copyright (c) ${currentYear}, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is generated from the canonical adapter-contract JSON Schemas.
// Do not edit it directly; run \`npm run generate\` instead.`;

const schemaSpecs = [
  {
    name: "adapter-descriptor",
    output: "adapter-descriptor.ts",
    project: projectAdapterDescriptor,
  },
  { name: "agent-config", output: "agent-config.ts" },
  {
    name: "agent-run-request",
    output: "agent-run-request.ts",
    project: projectRunRequest,
  },
  {
    name: "agent-run-result",
    output: "agent-run-result.ts",
    generate: generateRunResult,
  },
  { name: "runtime-context", output: "runtime-context.ts" },
];

// These local-host transport schemas are canonical southbound contracts, but
// they are not part of the dependency-free TypeScript model package. Keep the
// exclusion explicit now that all adapter schemas share one flat directory.
const transportSchemaNames = [
  "adapter-invocation.schema.json",
  "openai-stream-invocation.schema.json",
  "openai-stream-record.schema.json",
];

const pendingFiles = new Map();
let contractVersion;

const schemaInventory = (await readdir(schemaDirectory, { withFileTypes: true }))
  .filter(
    (entry) => entry.isFile() && entry.name.endsWith(".schema.json"),
  )
  .map((entry) => entry.name);
assertAdapterSchemaInventory(
  schemaInventory,
  [
    ...schemaSpecs.map((spec) => `${spec.name}.schema.json`),
    ...transportSchemaNames,
  ],
);

for (const spec of schemaSpecs) {
  const sourcePath = resolve(schemaDirectory, `${spec.name}.schema.json`);
  const schemaBytes = await readFile(sourcePath);
  const schema = JSON.parse(schemaBytes.toString("utf8"));

  if (spec.name === "adapter-descriptor") {
    contractVersion = requireString(
      schema.properties?.contract_version?.const,
      "adapter-descriptor contract_version.const",
    );
  }

  const generated = spec.generate
    ? await spec.generate(schema)
    : await generateSchema(spec.project ? spec.project(schema) : schema);

  pendingFiles.set(
    resolve(packageRoot, "src/generated", spec.output),
    generated,
  );
  pendingFiles.set(
    resolve(packageRoot, "schemas", `${spec.name}.schema.json`),
    schemaBytes,
  );
}

pendingFiles.set(resolve(packageRoot, "src/json.ts"), generateJsonTypes());
pendingFiles.set(
  resolve(packageRoot, "src/version.ts"),
  generateVersion(
    requireString(contractVersion, "resolved adapter contract version"),
  ),
);

const mismatches = [];
for (const [path, expected] of pendingFiles) {
  if (checkOnly) {
    let actual;
    try {
      actual = await readFile(path);
    } catch (error) {
      if (error.code !== "ENOENT") {
        throw error;
      }
    }
    const expectedBytes = Buffer.isBuffer(expected)
      ? expected
      : Buffer.from(expected, "utf8");
    if (actual === undefined || !actual.equals(expectedBytes)) {
      mismatches.push(relative(packageRoot, path));
    }
    continue;
  }

  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, expected);
}

if (mismatches.length > 0) {
  throw new Error(
    `Generated adapter-contract files are stale:\n${mismatches
      .map((path) => `  - ${path}`)
      .join("\n")}\nRun \`npm run generate\` and commit the result.`,
  );
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function requireString(value, label) {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`Expected ${label} to be a non-empty string`);
  }
  return value;
}

function requireStringEnum(value, label) {
  if (
    !Array.isArray(value) ||
    value.length === 0 ||
    value.some((entry) => typeof entry !== "string")
  ) {
    throw new Error(`Expected ${label} to be a non-empty string enum`);
  }
  return value;
}

function projectJsonTypes(value) {
  if (Array.isArray(value)) {
    value.forEach(projectJsonTypes);
    return;
  }
  if (value === null || typeof value !== "object") {
    return;
  }

  const hasNamedProperties =
    value.properties !== undefined && Object.keys(value.properties).length > 0;
  const isObject =
    value.type === "object" ||
    (Array.isArray(value.type) && value.type.includes("object"));
  if (isObject && value.additionalProperties === true && !hasNamedProperties) {
    value.tsType =
      Array.isArray(value.type) && value.type.includes("null")
        ? "JsonObject | null"
        : "JsonObject";
  }

  Object.values(value).forEach(projectJsonTypes);
}

function projectAdapterDescriptor(original) {
  const schema = deepClone(original);
  const properties = schema.properties;
  const definitions = schema.$defs;
  if (properties === undefined || definitions === undefined) {
    throw new Error("AdapterDescriptor must define properties and $defs");
  }

  const contractVersion = requireString(
    properties.contract_version?.const,
    "adapter-descriptor contract_version.const",
  );
  properties.contract_version.tsType = JSON.stringify(contractVersion);

  const extensionPoints = requireStringEnum(
    properties.extension_schemas?.propertyNames?.enum,
    "adapter-descriptor extension_schemas.propertyNames.enum",
  );
  const telemetryProviders = requireStringEnum(
    definitions.AdapterTelemetrySupport?.properties?.providers?.propertyNames
      ?.enum,
    "AdapterTelemetrySupport providers.propertyNames.enum",
  );
  schema.__projectionPrefix = `${renderStringUnion(
    "AdapterExtensionPoint",
    extensionPoints,
    "Southbound extension location supported by an adapter descriptor.",
  )}\n\n${renderStringUnion(
    "TelemetryProvider",
    telemetryProviders,
    "Telemetry provider supported by an adapter descriptor.",
  )}`;
  schema.__propertyNameProjections = [
    {
      property: "extension_schemas",
      type: "Partial<Record<AdapterExtensionPoint, JsonObject>>",
    },
    {
      property: "providers",
      type: "Partial<Record<TelemetryProvider, AdapterTelemetryProviderSupport>>",
    },
  ];
  return schema;
}

function projectRunRequest(original) {
  const schema = deepClone(original);
  const input = schema.properties?.input;
  if (
    input === undefined ||
    Object.keys(input).some((key) => key !== "description")
  ) {
    throw new Error(
      "AgentRunRequest.input is no longer unconstrained JSON; update the TypeScript projection",
    );
  }
  input.tsType = "JsonValue";
  return schema;
}

async function generateRunResult(original) {
  const schema = deepClone(original);
  const properties = schema.properties;
  if (properties === undefined) {
    throw new Error("AgentRunResult must define properties");
  }

  const output = properties.output;
  if (
    output === undefined ||
    Object.keys(output).some((key) => key !== "description")
  ) {
    throw new Error(
      "AgentRunResult.output is no longer unconstrained JSON; update the TypeScript projection",
    );
  }
  output.tsType = "JsonValue";

  const statuses = requireStringEnum(
    schema.$defs?.AgentRunStatus?.oneOf?.map((choice) => choice.const),
    "AgentRunStatus variants",
  );
  const expectedStatuses = ["succeeded", "failed", "cancelled"];
  if (JSON.stringify(statuses) !== JSON.stringify(expectedStatuses)) {
    throw new Error(
      `AgentRunStatus changed from ${expectedStatuses.join(", ")}; update the discriminated union projection`,
    );
  }
  assertRunResultConditionals(schema.allOf);

  delete schema.allOf;
  delete schema.$defs.AgentRunStatus;
  delete properties.status;
  delete properties.error;
  schema.required = schema.required.filter(
    (field) => field !== "status" && field !== "error",
  );
  schema.title = "AgentRunResultCommon";
  schema.description =
    "Fields shared by every preview terminal result variant.";
  schema.__projectionPrefix = `/** Preview southbound terminal result. */
export type AgentRunResult =
  | AgentRunSucceeded
  | AgentRunFailed
  | AgentRunCancelled;

/** Successful terminal result. Successful results can carry only a null error. */
export interface AgentRunSucceeded extends AgentRunResultCommon {
  status: "succeeded";
  error?: null;
}

/** Failed terminal result. Failed results must carry a non-null error. */
export interface AgentRunFailed extends AgentRunResultCommon {
  status: "failed";
  error: AgentRunError;
}

/** Cancelled terminal result. Cancellation details are optional. */
export interface AgentRunCancelled extends AgentRunResultCommon {
  status: "cancelled";
  error?: AgentRunError | null;
}`;

  return generateSchema(schema, { unreachableDefinitions: true });
}

function renderStringUnion(name, values, description) {
  return `/** ${description} */\nexport type ${name} =\n${values
    .map((value, index) => {
      const suffix = index === values.length - 1 ? ";" : "";
      return `  | ${JSON.stringify(value)}${suffix}`;
    })
    .join("\n")}`;
}

async function generateSchema(input, options = {}) {
  const schema = deepClone(input);
  const projectionPrefix = schema.__projectionPrefix;
  const propertyNameProjections = schema.__propertyNameProjections ?? [];
  delete schema.__projectionPrefix;
  delete schema.__propertyNameProjections;
  projectJsonTypes(schema);

  let output = await compile(schema, schema.title, {
    additionalProperties: true,
    bannerComment: banner,
    enableConstEnums: false,
    format: true,
    unknownAny: true,
    unreachableDefinitions: options.unreachableDefinitions ?? false,
  });

  for (const projection of propertyNameProjections) {
    output = replaceGeneratedPropertyType(
      output,
      projection.property,
      projection.type,
    );
  }
  output = projectOpenInterfaces(output);

  const referencedJsonTypes = ["JsonObject", "JsonValue"].filter((name) =>
    new RegExp(`\\b${name}\\b`).test(output.slice(banner.length)),
  );
  const additions = [];
  if (referencedJsonTypes.length > 0) {
    additions.push(
      `import type { ${referencedJsonTypes.join(", ")} } from "../json.js";`,
    );
  }
  if (projectionPrefix !== undefined) {
    additions.push(projectionPrefix);
  }
  if (additions.length > 0) {
    if (!output.startsWith(banner)) {
      throw new Error("json-schema-to-typescript changed banner placement");
    }
    output = `${banner}\n\n${additions.join("\n\n")}${output.slice(banner.length)}`;
  }

  const unsafeAny = output.match(/(:|<|\[|\|)\s*any\b/);
  if (unsafeAny !== null) {
    const matchIndex = unsafeAny.index ?? 0;
    throw new Error(
      `${schema.title} generated an unsafe any type near ${JSON.stringify(
        output.slice(Math.max(0, matchIndex - 80), matchIndex + 80),
      )}`,
    );
  }
  return output;
}

function replaceGeneratedPropertyType(output, property, projectedType) {
  const escapedProperty = property.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(
    `(  ${escapedProperty}\\?: )\\{\\n(?: {4}[^\\n]*\\n)+?  \\};`,
    "g",
  );
  const matches = [...output.matchAll(pattern)];
  if (matches.length !== 1) {
    throw new Error(
      `Expected exactly one generated ${property} map, found ${matches.length}`,
    );
  }
  return output.replace(pattern, `$1${projectedType};`);
}

function projectOpenInterfaces(output) {
  const openInterface =
    /export interface ([A-Za-z][A-Za-z0-9]*) \{\n((?:(?!^export (?:interface|type) ).)*?)  \[k: string\]: unknown;\n\}/gms;
  const projected = output.replace(
    openInterface,
    (_match, name, fields) =>
      `export type ${name} = {\n${fields}} & JsonObject;`,
  );
  if (projected.includes("[k: string]: unknown;")) {
    throw new Error(
      "Generated output contains an open object that was not projected to JsonObject",
    );
  }
  return projected;
}

function generateJsonTypes() {
  return `${banner}

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
`;
}

function generateVersion(version) {
  return `${banner}

/** The negotiated adapter descriptor contract version. */
export const ADAPTER_CONTRACT_VERSION = ${JSON.stringify(version)} as const;
`;
}
