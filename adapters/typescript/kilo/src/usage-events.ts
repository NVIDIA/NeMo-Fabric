// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Transient retry evidence that Kilo does not persist in message history. */
export class KiloUsageEvents {
  private readonly retries = new Map<string, number>();
  private readonly completed = new Set<string>();
  private readonly changed = new Set<() => void>();
  connected = false;
  failed = false;
  memoryObserved = false;

  observe(value: unknown): void {
    if (typeof value !== "object" || value === null || Array.isArray(value)) return;
    const event = value as Record<string, unknown>;
    if (event.type === "server.connected") this.connected = true;
    const properties = event.properties as Record<string, unknown> | undefined;
    if (properties && typeof properties === "object") {
      if (event.type === "memory.status" || event.type === "memory.updated" || event.type === "memory.error") {
        if (properties.enabled === true) this.memoryObserved = true;
      }
      const info = properties.info as Record<string, unknown> | undefined;
      if (event.type === "message.updated" && info?.role === "assistant" && typeof info.sessionID === "string" && typeof info.id === "string") {
        const time = info.time as Record<string, unknown> | undefined;
        if (time?.completed !== undefined) this.completed.add(info.id);
      }
      const status = properties.status as Record<string, unknown> | undefined;
      if (event.type === "session.status" && status?.type === "retry" && typeof properties.sessionID === "string") {
        this.retries.set(properties.sessionID, this.retryCount(properties.sessionID) + 1);
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

  retryCount(sessionId: string): number { return this.retries.get(sessionId) ?? 0; }

  async through(messageId: string, timeoutMs: number): Promise<boolean> {
    return await this.waitFor(() => this.completed.has(messageId), timeoutMs);
  }
}
