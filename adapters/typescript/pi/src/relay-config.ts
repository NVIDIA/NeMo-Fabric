// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Relay configuration owned by the Pi adapter. The Fabric runtime supplies a
// JSON plugin document; this module validates its observability boundary,
// normalizes local artifact paths, and writes the explicit TOML files consumed
// by the Relay gateway.

import type { Dirent } from "node:fs";
import { mkdir, readFile, readdir, realpath, stat, writeFile } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";

import type { AdapterStartInput } from "nemo-fabric-adapters-common";

export type RelayPluginConfig = Record<string, unknown>;

export interface RelayConfigPaths {
  configPath: string;
  pluginConfigPath: string;
}

export interface RelayArtifact {
  kind: "atof" | "atif";
  path: string;
}

export interface RelayAtifMatcher {
  directory: string;
  local: boolean;
  pattern: RegExp;
  recursive: boolean;
  template: string;
}

type ReadDirectory = (directory: string, options: { withFileTypes: true }) => Promise<Dirent[]>;

const OTEL_ENDPOINT_TYPES = new Set(["full", "gen_ai", "openinference"]);
const TOML_INTEGER_MIN = -(1n << 63n);
const TOML_INTEGER_MAX = (1n << 63n) - 1n;
const LEGACY_FLAT_OTEL_FIELDS = new Set([
  "attribute_mappings",
  "capture_content",
  "endpoint",
  "header_env",
  "headers",
  "instrumentation_scope",
  "mark_exclude_names",
  "mark_projection",
  "resource_attributes",
  "semantic_selector",
  "service_name",
  "service_namespace",
  "service_version",
  "timeout_millis",
  "transport",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function components(pluginConfig: RelayPluginConfig): unknown[] {
  const value = pluginConfig.components;
  if (value === undefined || value === null) {
    return [];
  }
  if (!Array.isArray(value)) {
    throw new Error("NeMo Relay plugin components must be a list");
  }
  return value;
}

export function validateRelayObservabilityV3(pluginConfig: RelayPluginConfig): void {
  for (const component of components(pluginConfig)) {
    if (!isRecord(component) || component.enabled === false) {
      continue;
    }
    const kind = component.kind;
    if (kind !== "observability") {
      continue;
    }
    const config = component.config;
    if (!isRecord(config)) {
      throw new Error("NeMo Relay observability component config must be an object");
    }
    if ("version" in config && config.version !== 3) {
      throw new Error(
        `unsupported NeMo Relay observability config version ${JSON.stringify(config.version)}; expected version 3`,
      );
    }
    if ("openinference" in config) {
      throw new Error(
        "NeMo Relay observability config version 3 removed the standalone openinference section; " +
          "use an opentelemetry endpoint with type 'openinference'",
      );
    }
    if (!("opentelemetry" in config)) {
      continue;
    }
    const opentelemetry = config.opentelemetry;
    if (!isRecord(opentelemetry)) {
      throw new Error("NeMo Relay opentelemetry config must be an object");
    }
    const legacyFields = Object.keys(opentelemetry)
      .filter((field) => LEGACY_FLAT_OTEL_FIELDS.has(field))
      .sort();
    if (legacyFields.length > 0) {
      throw new Error(
        "NeMo Relay observability config version 3 requires exporter fields inside " +
          `opentelemetry.endpoints: ${legacyFields.join(", ")}`,
      );
    }
    const enabled = opentelemetry.enabled ?? false;
    if (typeof enabled !== "boolean") {
      throw new Error("NeMo Relay opentelemetry.enabled must be a boolean");
    }
    const endpoints = opentelemetry.endpoints;
    if (endpoints !== undefined && !Array.isArray(endpoints)) {
      throw new Error("NeMo Relay opentelemetry.endpoints must be a list");
    }
    if (enabled && (!Array.isArray(endpoints) || endpoints.length === 0)) {
      throw new Error("enabled NeMo Relay OpenTelemetry requires at least one endpoint");
    }
    for (const [index, endpoint] of (endpoints ?? []).entries()) {
      if (!isRecord(endpoint)) {
        throw new Error(`NeMo Relay OpenTelemetry endpoint must be an object for opentelemetry.endpoints[${index}]`);
      }
      if (typeof endpoint.type !== "string" || !OTEL_ENDPOINT_TYPES.has(endpoint.type)) {
        throw new Error(
          "NeMo Relay OpenTelemetry endpoint type must be one of 'full', 'gen_ai', or " +
            `'openinference' for opentelemetry.endpoints[${index}].type`,
        );
      }
      if (typeof endpoint.endpoint !== "string" || endpoint.endpoint.trim().length === 0) {
        throw new Error(
          `NeMo Relay OpenTelemetry endpoint must be a non-empty string for opentelemetry.endpoints[${index}]`,
        );
      }
    }
  }
}

function validateUniqueRelayComponentKinds(pluginConfig: RelayPluginConfig): void {
  const seenKinds = new Set<string>();
  for (const component of components(pluginConfig)) {
    if (!isRecord(component)) {
      continue;
    }
    const kind = component.kind;
    if (typeof kind !== "string") {
      continue;
    }
    if (seenKinds.has(kind)) {
      throw new Error(`duplicate NeMo Relay plugin component kind '${kind}'`);
    }
    seenKinds.add(kind);
  }
}

function selectedModelName(input: AdapterStartInput): string {
  const models = input.config.models ?? {};
  const entries = Object.values(models);
  return models.default?.model ?? (entries.length === 1 ? entries[0]?.model : undefined) ?? "unknown";
}

export async function normalizeRelayOutputDirs(
  pluginConfig: RelayPluginConfig,
  input: AdapterStartInput,
): Promise<void> {
  validateRelayObservabilityV3(pluginConfig);
  const base = resolve(input.baseDir);
  const runtimeId = input.runtimeContext.runtime_id;

  for (const component of components(pluginConfig)) {
    if (
      !isRecord(component) ||
      component.kind !== "observability" ||
      component.enabled === false ||
      !isRecord(component.config)
    ) {
      continue;
    }
    const config = component.config;
    const atof = config.atof;
    if (isRecord(atof) && atof.enabled === true && Array.isArray(atof.sinks)) {
      for (const sink of atof.sinks) {
        if (!isRecord(sink) || sink.type !== "file") {
          continue;
        }
        const configured = sink.output_directory;
        const root =
          typeof configured === "string" && configured.length > 0
            ? isAbsolute(configured)
              ? configured
              : resolve(base, configured)
            : join(base, "artifacts", "relay");
        const outputDirectory = join(root, runtimeId);
        sink.output_directory = outputDirectory;
        await mkdir(outputDirectory, { recursive: true });
        sink.filename ??= "events.atof.jsonl";
        sink.mode ??= "overwrite";
      }
    }

    const atif = config.atif;
    if (!isRecord(atif) || atif.enabled !== true) {
      continue;
    }
    const configured = atif.output_directory;
    const root =
      typeof configured === "string" && configured.length > 0
        ? isAbsolute(configured)
          ? configured
          : resolve(base, configured)
        : join(base, "artifacts", "relay");
    const outputDirectory = join(root, runtimeId);
    atif.output_directory = outputDirectory;
    await mkdir(outputDirectory, { recursive: true });
    atif.filename_template ??= "trajectory-{session_id}.atif.json";
    atif.agent_name ??= input.agentName;
    atif.model_name ??= selectedModelName(input);
  }
}

export async function loadRelayPluginConfig(input: AdapterStartInput): Promise<RelayPluginConfig> {
  const configPath = process.env.FABRIC_RELAY_CONFIG_PATH;
  if (!configPath) {
    throw new Error("FABRIC_RELAY_CONFIG_PATH is required when Relay is enabled");
  }
  const wrapper: unknown = JSON.parse(await readFile(configPath, "utf8"));
  if (!isRecord(wrapper)) {
    throw new Error("NeMo Fabric Relay runtime configuration must be an object");
  }
  const relay = wrapper.relay;
  if (relay !== undefined && !isRecord(relay)) {
    throw new Error("NeMo Fabric Relay runtime configuration must contain a relay object");
  }
  const raw = isRecord(relay) ? relay.config : undefined;
  if (raw !== undefined && raw !== null && !isRecord(raw)) {
    throw new Error("NeMo Fabric Relay plugin configuration must be an object");
  }
  let pluginConfig: RelayPluginConfig = raw ?? {};
  if (!("components" in pluginConfig)) {
    pluginConfig = {
      version: 1,
      components:
        Object.keys(pluginConfig).length === 0 ? [] : [{ kind: "observability", enabled: true, config: pluginConfig }],
    };
  }
  pluginConfig.version ??= 1;
  pluginConfig.components ??= [];
  await normalizeRelayOutputDirs(pluginConfig, input);
  return pluginConfig;
}

function tomlKey(value: string): string {
  return /^[A-Za-z0-9_-]+$/.test(value) ? value : JSON.stringify(value);
}

function tomlPath(parts: string[]): string {
  return parts.map(tomlKey).join(".");
}

function tomlScalar(value: unknown): string {
  if (typeof value === "string") {
    return JSON.stringify(value);
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    if (Object.is(value, -0)) {
      return "-0.0";
    }
    if (Number.isInteger(value)) {
      const integer = BigInt(value);
      if (integer < TOML_INTEGER_MIN || integer > TOML_INTEGER_MAX) {
        throw new Error("NeMo Relay configuration contains an integer outside TOML's signed 64-bit range");
      }
      return integer.toString();
    }
    return String(value);
  }
  if (Array.isArray(value) && value.every((item) => !isRecord(item))) {
    return `[${value.map(tomlScalar).join(", ")}]`;
  }
  throw new Error("NeMo Relay configuration contains a value unsupported by the local TOML encoder");
}

function isScalar(value: unknown): boolean {
  return (
    typeof value === "string" ||
    typeof value === "boolean" ||
    (typeof value === "number" && Number.isFinite(value)) ||
    (Array.isArray(value) && (value.length === 0 || value.every((item) => !isRecord(item))))
  );
}

function emitTomlTable(lines: string[], value: Record<string, unknown>, path: string[]): void {
  for (const [key, item] of Object.entries(value)) {
    if (isScalar(item)) {
      lines.push(`${tomlKey(key)} = ${tomlScalar(item)}`);
    }
  }
  for (const [key, item] of Object.entries(value)) {
    if (item === null || item === undefined) {
      throw new Error("NeMo Relay configuration contains a value unsupported by the local TOML encoder");
    }
    if (isScalar(item)) {
      continue;
    }
    if (isRecord(item)) {
      lines.push("", `[${tomlPath([...path, key])}]`);
      emitTomlTable(lines, item, [...path, key]);
      continue;
    }
    if (Array.isArray(item) && item.length > 0 && item.every(isRecord)) {
      for (const entry of item) {
        lines.push("", `[[${tomlPath([...path, key])}]]`);
        emitTomlTable(lines, entry, [...path, key]);
      }
      continue;
    }
    throw new Error("NeMo Relay configuration contains a value unsupported by the local TOML encoder");
  }
}

export function encodeToml(value: Record<string, unknown>): string {
  const lines: string[] = [];
  emitTomlTable(lines, value, []);
  return `${lines
    .join("\n")
    .replace(/^\n+/, "")
    .replace(/\n{3,}/g, "\n\n")}\n`;
}

export async function writeRelayConfigs(pluginConfig: RelayPluginConfig): Promise<RelayConfigPaths> {
  const runtimeConfigPath = process.env.FABRIC_RELAY_CONFIG_PATH;
  if (!runtimeConfigPath) {
    throw new Error("FABRIC_RELAY_CONFIG_PATH is required when Relay is enabled");
  }
  const enabledPluginConfig = {
    ...pluginConfig,
    components: components(pluginConfig).filter(
      (component) => !isRecord(component) || component.enabled !== false,
    ),
  };
  validateRelayObservabilityV3(enabledPluginConfig);
  validateUniqueRelayComponentKinds(enabledPluginConfig);
  const configDir = join(dirname(runtimeConfigPath), "relay-config");
  const configPath = join(configDir, "config.toml");
  const pluginConfigPath = join(configDir, "plugins.toml");
  await mkdir(configDir, { recursive: true });
  await Promise.all([
    writeFile(configPath, encodeToml({}), "utf8"),
    writeFile(pluginConfigPath, encodeToml(enabledPluginConfig), "utf8"),
  ]);
  return { configPath, pluginConfigPath };
}

async function artifactDirectory(value: unknown): Promise<string | undefined> {
  if (typeof value !== "string" || value.length === 0) {
    return undefined;
  }
  try {
    const directory = await realpath(value);
    return (await stat(directory)).isDirectory() ? directory : undefined;
  } catch {
    return undefined;
  }
}

async function artifactFile(path: string, directory: string): Promise<string | undefined> {
  try {
    const candidate = await realpath(path);
    const pathFromDirectory = relative(directory, candidate);
    if (
      pathFromDirectory !== ".." &&
      !pathFromDirectory.startsWith(`..${sep}`) &&
      !isAbsolute(pathFromDirectory) &&
      (await stat(candidate)).isFile()
    ) {
      return candidate;
    }
  } catch {
    // Missing or malformed artifact paths are not invocation failures.
  }
  return undefined;
}

async function existingLocalFile(directory: string, filename: unknown): Promise<string | undefined> {
  if (typeof filename !== "string" || filename.length === 0 || basename(filename) !== filename) {
    return undefined;
  }
  return artifactFile(join(directory, filename), directory);
}

async function directoryEntries(directory: string): Promise<string[]> {
  try {
    return await readdir(directory);
  } catch {
    return [];
  }
}

function escapePattern(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function placeholderPattern(placeholder: string): string {
  const expression = placeholder.slice(1, -1);
  const fallbackIndex = expression.indexOf(":-");
  if (fallbackIndex < 0) {
    return "[^/]+";
  }
  const fallback = expression.slice(fallbackIndex + 2).replaceAll("\\", "/");
  return fallback.includes("/") ? `(?:[^/]+|${escapePattern(fallback)})` : "[^/]+";
}

function templatePattern(template: string): { pattern: RegExp; recursive: boolean } {
  const normalized = template.replaceAll("\\", "/");
  const parts = normalized.split(/(\{[^{}]+\})/u);
  const pattern = parts
    .map((part) => (/^\{[^{}]+\}$/u.test(part) ? placeholderPattern(part) : escapePattern(part)))
    .join("");
  return { pattern: new RegExp(`^${pattern}$`, "u"), recursive: normalized.includes("/") };
}

export function matchesRelayAtifPath(matcher: RelayAtifMatcher, path: string): boolean {
  const name = relative(matcher.directory, path).split(sep).join("/");
  return name !== ".." && !name.startsWith("../") && !isAbsolute(name) && matcher.pattern.test(name);
}

export async function prepareRelayAtifMatchers(pluginConfig: RelayPluginConfig): Promise<RelayAtifMatcher[]> {
  const matchers: RelayAtifMatcher[] = [];
  for (const component of components(pluginConfig)) {
    if (
      !isRecord(component) ||
      component.kind !== "observability" ||
      component.enabled === false ||
      !isRecord(component.config)
    ) {
      continue;
    }
    const atif = component.config.atif;
    if (!isRecord(atif) || atif.enabled !== true) {
      continue;
    }
    const template = atif.filename_template;
    if (typeof template !== "string" || template.length === 0) {
      continue;
    }
    const directory = await artifactDirectory(atif.output_directory);
    if (directory === undefined) {
      throw new Error("NeMo Relay ATIF output directory could not be prepared");
    }
    const { pattern, recursive } = templatePattern(template);
    matchers.push({
      directory,
      local: !Array.isArray(atif.storage) || atif.storage.length === 0,
      pattern,
      recursive,
      template,
    });
  }
  return matchers;
}

async function matchingAtifFiles(
  matcher: RelayAtifMatcher,
  strict = false,
  readDirectory: ReadDirectory = readdir,
): Promise<string[]> {
  const files: string[] = [];
  const visit = async (directory: string, root: boolean): Promise<void> => {
    let entries;
    try {
      entries = await readDirectory(directory, { withFileTypes: true });
    } catch (error) {
      if (strict && root) {
        throw error;
      }
      return;
    }
    for (const entry of entries) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) {
        if (matcher.recursive) {
          await visit(path, false);
        }
        continue;
      }
      if (matchesRelayAtifPath(matcher, path)) {
        const artifact = await artifactFile(path, matcher.directory);
        if (artifact !== undefined) {
          files.push(artifact);
        }
      }
    }
  };
  await visit(matcher.directory, true);
  return files;
}

export async function collectAtifArtifacts(
  matchers: RelayAtifMatcher[],
  options: { localOnly?: boolean; readDirectory?: ReadDirectory; strict?: boolean } = {},
): Promise<RelayArtifact[]> {
  const artifacts: RelayArtifact[] = [];
  for (const matcher of matchers) {
    if (options.localOnly === true && !matcher.local) {
      continue;
    }
    for (const path of await matchingAtifFiles(matcher, options.strict, options.readDirectory)) {
      artifacts.push({ kind: "atif", path });
    }
  }
  return artifacts.sort((left, right) => left.path.localeCompare(right.path));
}

export async function collectRelayArtifacts(
  pluginConfig: RelayPluginConfig,
  atifMatchers?: RelayAtifMatcher[],
): Promise<RelayArtifact[]> {
  const artifacts: RelayArtifact[] = [];
  for (const component of components(pluginConfig)) {
    if (
      !isRecord(component) ||
      component.kind !== "observability" ||
      component.enabled === false ||
      !isRecord(component.config)
    ) {
      continue;
    }
    const atof = component.config.atof;
    if (isRecord(atof) && atof.enabled === true && Array.isArray(atof.sinks)) {
      for (const sink of atof.sinks) {
        if (!isRecord(sink) || sink.type !== "file") {
          continue;
        }
        const directory = await artifactDirectory(sink.output_directory);
        if (directory === undefined) {
          continue;
        }
        const filenames =
          sink.filename === undefined
            ? (await directoryEntries(directory)).filter((entry) => entry.endsWith(".jsonl"))
            : [sink.filename];
        for (const filename of filenames) {
          const path = await existingLocalFile(directory, filename);
          if (path !== undefined) {
            artifacts.push({ kind: "atof", path });
          }
        }
      }
    }
  }
  const matchers = atifMatchers ?? (await prepareRelayAtifMatchers(pluginConfig).catch(() => []));
  artifacts.push(...(await collectAtifArtifacts(matchers)));
  return artifacts.sort((left, right) => left.path.localeCompare(right.path));
}
