"use client";

import type { SessionRunResult } from "./results";

const STORAGE_KEY = "relay.session-history.v1";

export function loadHistory(): SessionRunResult[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveRun(result: SessionRunResult): SessionRunResult[] {
  const next = [result, ...loadHistory()].slice(0, 50);
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  return next;
}

export function trySaveRun(result: SessionRunResult): {
  history: SessionRunResult[];
  persisted: boolean;
} {
  try {
    return { history: saveRun(result), persisted: true };
  } catch {
    return { history: loadHistory(), persisted: false };
  }
}

export function deleteRun(id: string): SessionRunResult[] {
  const next = loadHistory().filter((item) => item.id !== id);
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    return next;
  } catch {
    return loadHistory();
  }
}

export function clearHistory(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Storage access can be unavailable in strict/private browser contexts.
  }
}
