// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Transient retry evidence that Kilo does not persist in message history. */
export class KiloUsageEvents {
  private readonly latestAssistant = new Map<string, string>();
  private readonly completed = new Set<string>();
  private readonly changed = new Set<() => void>();
  readonly retried = new Set<string>();
  readonly unattributedRetries = new Set<string>();
  connected = false;
  failed = false;

  observe(value: unknown): void {
    if (typeof value !== "object" || value === null || Array.isArray(value)) return;
    const event = value as Record<string, unknown>;
    if (event.type === "server.connected") this.connected = true;
    const properties = event.properties as Record<string, unknown> | undefined;
    if (properties && typeof properties === "object") {
      const info = properties.info as Record<string, unknown> | undefined;
      if (event.type === "message.updated" && info?.role === "assistant" && typeof info.sessionID === "string" && typeof info.id === "string") {
        this.latestAssistant.set(info.sessionID, info.id);
        const time = info.time as Record<string, unknown> | undefined;
        if (time?.completed !== undefined || info.error !== undefined) this.completed.add(info.id);
      }
      const status = properties.status as Record<string, unknown> | undefined;
      if (event.type === "session.status" && status?.type === "retry" && typeof properties.sessionID === "string") {
        const message = this.latestAssistant.get(properties.sessionID);
        if (message === undefined) this.unattributedRetries.add(properties.sessionID);
        else this.retried.add(message);
      }
    }
    for (const notify of this.changed) notify();
  }

  end(): void {
    this.failed = true;
    for (const notify of this.changed) notify();
  }

  /** Event receipt through the completed message proves earlier retries drained. */
  async waitFor(predicate: () => boolean, timeoutMs: number): Promise<boolean> {
    if (predicate()) return true;
    if (this.failed || timeoutMs <= 0) return false;
    return await new Promise<boolean>((resolve) => {
      const finish = (value: boolean): void => {
        clearTimeout(timer);
        this.changed.delete(check);
        resolve(value);
      };
      const check = (): void => {
        if (predicate()) finish(true);
        else if (this.failed) finish(false);
      };
      const timer = setTimeout(() => finish(false), timeoutMs);
      this.changed.add(check);
      check();
    });
  }

  hasCompleted(messageId: string): boolean { return this.completed.has(messageId); }

  async through(messageId: string, timeoutMs: number): Promise<boolean> {
    return await this.waitFor(() => this.completed.has(messageId), timeoutMs);
  }
}
