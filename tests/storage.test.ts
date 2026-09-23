import { afterEach, describe, expect, it, vi } from "vitest";
import { createDemoSession } from "../src/domain/templates";
import { createRunResult } from "../src/session/results";
import { trySaveRun } from "../src/session/storage";

describe("local session storage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });

  it("keeps a completed result usable when browser storage rejects the write", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("Quota exceeded", "QuotaExceededError");
    });
    const result = createRunResult(
      createDemoSession(),
      new Date(0).toISOString(),
      "replay",
      [],
      true,
    );
    const saved = trySaveRun(result);
    expect(saved.persisted).toBe(false);
    expect(saved.history).toEqual([]);
    expect(result.completed).toBe(true);
  });
});
