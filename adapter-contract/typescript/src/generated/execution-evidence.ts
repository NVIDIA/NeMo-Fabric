// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is generated from the canonical NeMo Fabric JSON Schemas.
// Do not edit it directly; run `npm run generate` instead.

/**
 * Engine name.
 */
export type VersionObservation =
  | {
      /**
       * Original RFC 3339 UTC capture time; forwarding does not change it.
       */
      observed_at: string;
      /**
       * How the named source obtained this observation.
       */
      origin: "observed" | "reported";
      reason?: null;
      /**
       * Observer and method, such as `adapter:importlib.metadata`.
       */
      source: string;
      status: "available";
      value: string;
    }
  | {
      /**
       * Original RFC 3339 UTC capture time; forwarding does not change it.
       */
      observed_at: string;
      /**
       * How the named source obtained this observation.
       */
      origin: "observed" | "reported";
      reason: string;
      /**
       * Observer and method, such as `adapter:importlib.metadata`.
       */
      source: string;
      status: "unavailable" | "unsupported" | "not_collected" | "not_applicable";
      value?: null;
    };
/**
 * Server version, never a CLI or API version.
 */
export type VersionObservation1 =
  | {
      /**
       * Original RFC 3339 UTC capture time; forwarding does not change it.
       */
      observed_at: string;
      /**
       * How the named source obtained this observation.
       */
      origin: "observed" | "reported";
      reason?: null;
      /**
       * Observer and method, such as `adapter:importlib.metadata`.
       */
      source: string;
      status: "available";
      value: string;
    }
  | {
      /**
       * Original RFC 3339 UTC capture time; forwarding does not change it.
       */
      observed_at: string;
      /**
       * How the named source obtained this observation.
       */
      origin: "observed" | "reported";
      reason: string;
      /**
       * Observer and method, such as `adapter:importlib.metadata`.
       */
      source: string;
      status: "unavailable" | "unsupported" | "not_collected" | "not_applicable";
      value?: null;
    };
/**
 * Provider name.
 */
export type VersionObservation2 =
  | {
      /**
       * Original RFC 3339 UTC capture time; forwarding does not change it.
       */
      observed_at: string;
      /**
       * How the named source obtained this observation.
       */
      origin: "observed" | "reported";
      reason?: null;
      /**
       * Observer and method, such as `adapter:importlib.metadata`.
       */
      source: string;
      status: "available";
      value: string;
    }
  | {
      /**
       * Original RFC 3339 UTC capture time; forwarding does not change it.
       */
      observed_at: string;
      /**
       * How the named source obtained this observation.
       */
      origin: "observed" | "reported";
      reason: string;
      /**
       * Observer and method, such as `adapter:importlib.metadata`.
       */
      source: string;
      status: "unavailable" | "unsupported" | "not_collected" | "not_applicable";
      value?: null;
    };
/**
 * Gateway/service version, not a guest package version.
 */
export type VersionObservation3 =
  | {
      /**
       * Original RFC 3339 UTC capture time; forwarding does not change it.
       */
      observed_at: string;
      /**
       * How the named source obtained this observation.
       */
      origin: "observed" | "reported";
      reason?: null;
      /**
       * Observer and method, such as `adapter:importlib.metadata`.
       */
      source: string;
      status: "available";
      value: string;
    }
  | {
      /**
       * Original RFC 3339 UTC capture time; forwarding does not change it.
       */
      observed_at: string;
      /**
       * How the named source obtained this observation.
       */
      origin: "observed" | "reported";
      reason: string;
      /**
       * Observer and method, such as `adapter:importlib.metadata`.
       */
      source: string;
      status: "unavailable" | "unsupported" | "not_collected" | "not_applicable";
      value?: null;
    };
/**
 * An observed installation/resolution context, never inferred from equal versions.
 */
export type SoftwareEnvironment =
  | {
      complete: true;
      /**
       * Opaque result-local join key; has no machine or role semantics.
       */
      context_ref: string;
      /**
       * Selected software only; this is not a dependency scan.
       */
      entries: SoftwareEntry[];
      /**
       * Components using this installation, such as `caller_sdk` or `adapter`.
       *
       * @minItems 1
       */
      participants: [string, ...string[]];
      /**
       * Why coverage is unavailable or incomplete.
       */
      reason?: string | null;
      status: "available";
    }
  | {
      complete: false;
      /**
       * Opaque result-local join key; has no machine or role semantics.
       */
      context_ref: string;
      /**
       * Selected software only; this is not a dependency scan.
       */
      entries: SoftwareEntry[];
      /**
       * Components using this installation, such as `caller_sdk` or `adapter`.
       *
       * @minItems 1
       */
      participants: [string, ...string[]];
      reason: string;
      status: "available";
    }
  | {
      complete: false;
      /**
       * Opaque result-local join key; has no machine or role semantics.
       */
      context_ref: string;
      /**
       * Selected software only; this is not a dependency scan.
       *
       * @maxItems 0
       */
      entries: [];
      /**
       * Components using this installation, such as `caller_sdk` or `adapter`.
       *
       * @minItems 1
       */
      participants: [string, ...string[]];
      reason: string;
      status: "unavailable" | "unsupported" | "not_collected";
    };
/**
 * Full version observation.
 */
export type VersionObservation4 =
  | {
      /**
       * Original RFC 3339 UTC capture time; forwarding does not change it.
       */
      observed_at: string;
      /**
       * How the named source obtained this observation.
       */
      origin: "observed" | "reported";
      reason?: null;
      /**
       * Observer and method, such as `adapter:importlib.metadata`.
       */
      source: string;
      status: "available";
      value: string;
    }
  | {
      /**
       * Original RFC 3339 UTC capture time; forwarding does not change it.
       */
      observed_at: string;
      /**
       * How the named source obtained this observation.
       */
      origin: "observed" | "reported";
      reason: string;
      /**
       * Observer and method, such as `adapter:importlib.metadata`.
       */
      source: string;
      status: "unavailable" | "unsupported" | "not_collected" | "not_applicable";
      value?: null;
    };

/**
 * Selected software evidence included inline in a RunResult.
 */
export interface ExecutionEvidence {
  /**
   * Existing provider/engine reports, omitted when none were supplied.
   */
  infrastructure?: InfrastructureSoftware | null;
  /**
   * One inventory per observed installation/resolution context.
   *
   * @maxItems 16
   */
  software_environments:
    | []
    | [SoftwareEnvironment]
    | [SoftwareEnvironment, SoftwareEnvironment]
    | [SoftwareEnvironment, SoftwareEnvironment, SoftwareEnvironment]
    | [SoftwareEnvironment, SoftwareEnvironment, SoftwareEnvironment, SoftwareEnvironment]
    | [SoftwareEnvironment, SoftwareEnvironment, SoftwareEnvironment, SoftwareEnvironment, SoftwareEnvironment]
    | [
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment
      ]
    | [
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment
      ]
    | [
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment
      ]
    | [
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment
      ]
    | [
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment
      ]
    | [
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment
      ]
    | [
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment
      ]
    | [
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment
      ]
    | [
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment
      ]
    | [
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment
      ]
    | [
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment,
        SoftwareEnvironment
      ];
}
/**
 * Conditional infrastructure software reports.
 */
export interface InfrastructureSoftware {
  /**
   * Included only when an owner supplied an engine report.
   */
  container?: ContainerSoftwareReport | null;
  /**
   * Included only when an owner supplied a sandbox report.
   */
  sandbox?: SandboxSoftwareReport | null;
}
/**
 * Existing environment-owner report of a container engine server.
 */
export interface ContainerSoftwareReport {
  /**
   * Installation contexts served by this engine.
   *
   * @minItems 1
   */
  context_refs: [string, ...string[]];
  engine: VersionObservation;
  engine_version: VersionObservation1;
}
/**
 * Existing sandbox/provider report, without any new service discovery.
 */
export interface SandboxSoftwareReport {
  /**
   * Installation contexts served by this sandbox.
   *
   * @minItems 1
   */
  context_refs: [string, ...string[]];
  provider: VersionObservation2;
  service_version: VersionObservation3;
}
/**
 * One selected component in an installation inventory.
 */
export interface SoftwareEntry {
  /**
   * `python_distribution`, `npm_package`, `executable`, or `component`.
   */
  kind: string;
  /**
   * Distribution, executable, or component name.
   */
  name: string;
  /**
   * Component category, such as `fabric_sdk`, `adapter`, or `harness`.
   */
  role: string;
  version: VersionObservation4;
}
