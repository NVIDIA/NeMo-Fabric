// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isIP } from "node:net";

import { LifecycleError } from "./lifecycle.js";

const HOSTNAME =
  /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*\.?$/u;

function normalizedHostname(hostname: string): string {
  return hostname.toLowerCase().replace(/\.$/u, "");
}

export function insecureHttpModelHosts(
  settings: Record<string, unknown> | undefined,
  errorCode: string,
): string[] {
  const hosts = settings?.insecure_http_model_hosts;
  if (hosts === undefined) return [];
  if (
    !Array.isArray(hosts) ||
    hosts.some(
      (host) =>
        typeof host !== "string" || host.length > 253 || !HOSTNAME.test(host),
    )
  ) {
    throw new LifecycleError(
      errorCode,
      "insecure_http_model_hosts must contain hostnames without schemes, ports, paths, or wildcards",
    );
  }
  return hosts.map(normalizedHostname);
}

/** Allow HTTPS, loopback HTTP, and explicitly trusted HTTP model hosts. */
export function modelEndpointAllowed(
  value: string,
  insecureHosts: readonly string[],
  allowPrivateHttp = false,
): boolean {
  try {
    const url = new URL(value);
    if (url.protocol === "https:") return true;
    if (url.protocol !== "http:") return false;
    const hostname = url.hostname;
    const loopback =
      hostname === "localhost" ||
      hostname === "[::1]" ||
      /^127(?:\.\d{1,3}){3}$/u.test(hostname);
    const [first = 0, second = 0] = hostname.split(".").map(Number);
    const privateIpv4 =
      isIP(hostname) === 4 &&
      (first === 10 ||
        (first === 172 && second >= 16 && second <= 31) ||
        (first === 192 && second === 168));
    return (
      loopback ||
      insecureHosts.includes(normalizedHostname(hostname)) ||
      (allowPrivateHttp && privateIpv4)
    );
  } catch {
    return false;
  }
}
