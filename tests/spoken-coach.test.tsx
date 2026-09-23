import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useSpokenCoach } from "../src/hooks/use-spoken-coach";

describe("spoken coaching preferences", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });

  it("remains usable when browser storage is unavailable", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("Storage blocked", "SecurityError");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("Storage blocked", "SecurityError");
    });

    const { result } = renderHook(() => useSpokenCoach());
    expect(result.current.muted).toBe(false);

    act(() => result.current.setMuted(true));
    expect(result.current.muted).toBe(true);
  });
});
