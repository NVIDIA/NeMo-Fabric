// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import Ajv from "ajv";

import { insecureHttpModelHosts, modelEndpointAllowed } from "../dist/index.js";

const trusted = insecureHttpModelHosts(
  {
    insecure_http_model_hosts: [
      "Host.Docker.Internal",
      "release-nhx-api",
      "gateway.ns.svc.cluster.local.",
    ],
  },
  "invalid_settings",
);

test("model endpoints retain HTTPS and loopback defaults", () => {
  for (const url of [
    "https://provider.example/v1",
    "http://localhost:8000/v1",
    "http://127.0.0.2/v1",
    "http://[::1]:8080/v1",
  ]) {
    assert.equal(modelEndpointAllowed(url, []), true, url);
  }
  for (const url of [
    "http://host.docker.internal:8080/apis/inference-gateway/",
    "http://release-nhx-api:8080/v1",
    "http://10.86.19.10:8000/v1",
    "http://169.254.169.254/latest",
    "http://localhost.:8000/v1",
    "http://8.8.8.8/v1",
    "ftp://localhost/v1",
    "invalid",
  ]) {
    assert.equal(modelEndpointAllowed(url, []), false, url);
  }
});

test("HTTP host trust is exact, case insensitive, and independent of port and path", () => {
  for (const url of [
    "http://HOST.DOCKER.INTERNAL:8080/apis/inference-gateway/",
    "http://host.docker.internal:9000/custom",
    "http://release-nhx-api:8080/v1",
    "http://gateway.ns.svc.cluster.local/v1",
    "http://gateway.ns.svc.cluster.local.:8000/v1",
  ]) {
    assert.equal(modelEndpointAllowed(url, trusted), true, url);
  }
  for (const url of [
    "http://sub.host.docker.internal/v1",
    "http://host.docker.internal.evil/v1",
    "http://host.docker.internal@evil.example/v1",
    "http://other-release-nhx-api/v1",
    "http://10.86.19.10:8000/v1",
    "ftp://host.docker.internal/v1",
  ]) {
    assert.equal(modelEndpointAllowed(url, trusted), false, url);
  }
  assert.equal(
    modelEndpointAllowed("http://10.86.19.10:8000/v1", ["10.86.19.10"]),
    true,
  );
});

test("private IPv4 opt-in retains RFC1918 boundaries and does not trust DNS names", () => {
  for (const host of [
    "10.0.0.0",
    "10.255.255.255",
    "172.16.0.0",
    "172.31.255.255",
    "192.168.0.0",
    "192.168.255.255",
  ]) {
    assert.equal(
      modelEndpointAllowed(`http://${host}/v1`, [], true),
      true,
      host,
    );
  }
  for (const host of [
    "9.255.255.255",
    "11.0.0.0",
    "172.15.255.255",
    "172.32.0.0",
    "192.167.255.255",
    "192.169.0.0",
    "169.254.169.254",
    "host.docker.internal",
    "[fd00::1]",
  ]) {
    assert.equal(
      modelEndpointAllowed(`http://${host}/v1`, [], true),
      false,
      host,
    );
  }
});

const invalidHosts = [
  null,
  true,
  "host.docker.internal",
  [42],
  [null],
  [""],
  ["https://gateway"],
  ["gateway:8000"],
  ["gateway/v1"],
  ["*.internal"],
  [" gateway"],
  ["gateway "],
  ["gateway@evil"],
  ["-gateway"],
  ["gateway..internal"],
  ["a".repeat(64)],
  ["a.".repeat(127) + "a"],
];

test("invalid allowlists fail closed even before endpoint selection", () => {
  assert.deepEqual(insecureHttpModelHosts(undefined, "invalid_settings"), []);
  assert.deepEqual(
    insecureHttpModelHosts(
      { insecure_http_model_hosts: [] },
      "invalid_settings",
    ),
    [],
  );
  for (const hosts of invalidHosts) {
    assert.throws(
      () =>
        insecureHttpModelHosts(
          { insecure_http_model_hosts: hosts },
          "invalid_settings",
        ),
      (error) => error.code === "invalid_settings",
      JSON.stringify(hosts),
    );
  }
});

for (const name of ["opencode", "qwen", "kilo"]) {
  test(`${name} descriptor admits explicit hosts and rejects malformed allowlists`, async () => {
    const descriptor = JSON.parse(
      await readFile(
        new URL(`../../${name}/${name}.fabric-adapter.json`, import.meta.url),
      ),
    );
    const validate = new Ajv().compile(descriptor.settings_schema);
    assert.equal(
      validate({
        insecure_http_model_hosts: [
          "host.docker.internal",
          "release-nhx-api",
          "gateway.ns.svc.cluster.local",
          "10.86.19.10",
        ],
      }),
      true,
    );
    for (const hosts of invalidHosts) {
      assert.equal(
        validate({ insecure_http_model_hosts: hosts }),
        false,
        JSON.stringify(hosts),
      );
    }
  });
}
