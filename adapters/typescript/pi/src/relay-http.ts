// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isIP } from "node:net";
import { LifecycleError, modelEndpointAllowed } from "nemo-fabric-adapters-common";
import type { JsonObject } from "nemo-fabric-adapter-contract";
import type { RelayCliContract } from "./relay-gateway.js";

const HOSTNAME =
  /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*\.?$/;

export function validateRelayModelEndpoint(model: { api: string; baseUrl: string }): void {
  // Relay's Pi extension leaves other API families on their original endpoint.
  // Those requests bypass the gateway's HTTP credential stripping and host policy.
  const proxied = ["openai-completions", "openai-responses", "anthropic-messages"];
  if (!proxied.includes(model.api) && !modelEndpointAllowed(model.baseUrl, [])) {
    throw new LifecycleError(
      "pi_relay_unsupported_http_model_api",
      "This Pi model API requires HTTPS or loopback HTTP when Relay is enabled; remote HTTP requires openai-completions, openai-responses, or anthropic-messages",
    );
  }
}

export function relayHttpModelConfig(settings: JsonObject, contract: RelayCliContract): JsonObject {
  const key = "insecure_http_model_hosts";
  if (!Object.hasOwn(settings, key)) {
    return {};
  }
  const invalid = (detail: string) =>
    new LifecycleError("pi_relay_invalid_http_model_hosts", `harness.settings.${key} ${detail}`);
  const hosts = settings[key];
  if (!Array.isArray(hosts)) {
    throw invalid("must be a list of bare hostnames or IP addresses");
  }
  const normalized: string[] = [];
  for (const [index, host] of hosts.entries()) {
    if (typeof host !== "string" || host.length === 0 || host.length > 253 || host.includes("%")) {
      throw invalid(`[${index}] must be a bare hostname or IP address`);
    }
    let value: string;
    if (isIP(host) === 6) {
      value = new URL(`http://[${host}]`).hostname.slice(1, -1);
    } else {
      if (!HOSTNAME.test(host)) {
        throw invalid(`[${index}] must not include a scheme, port, path, or wildcard`);
      }
      value = host.toLowerCase().replace(/\.$/, "");
      // WHATWG parsing detects alternate IPv4 spellings and invalid numeric DNS tails.
      let parsed: URL;
      try {
        parsed = new URL(`http://${value}`);
      } catch {
        throw invalid(`[${index}] must be a bare hostname or canonical IP address`);
      }
      if (parsed.hostname !== value) {
        throw invalid(`[${index}] must use a canonical IP address`);
      }
    }
    if (!normalized.includes(value)) {
      normalized.push(value);
    }
  }
  if (contract.version[0] === 0 && contract.version[1] < 11) {
    throw invalid("requires NeMo Relay CLI >=0.11.0; upgrade Relay or remove the setting");
  }
  return { upstream: { [key]: normalized } };
}
