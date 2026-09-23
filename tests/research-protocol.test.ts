import { describe, expect, it } from "vitest";
import {
  freezeObservableProtocol,
  LATERAL_ARM_RAISE_REFERENCE_V1,
  validateObservableProtocol,
} from "../src/research/protocol";

describe("observable movement protocol", () => {
  it("freezes a deterministic, nonclinical reference-relative pilot", async () => {
    const first = await freezeObservableProtocol(LATERAL_ARM_RAISE_REFERENCE_V1);
    const second = await freezeObservableProtocol(
      structuredClone(LATERAL_ARM_RAISE_REFERENCE_V1),
    );
    expect(first).toEqual(second);
    expect(first.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(first.protocol.target.source).toBe("captured-reference");
    expect(first.protocol.observability.prohibitedClaims).toContain(
      "clinical correctness or safety",
    );
  });

  it("rejects an unbounded or contradictory target", () => {
    const invalid = structuredClone(LATERAL_ARM_RAISE_REFERENCE_V1);
    invalid.target.comparison = "within-range";
    invalid.target.range = { minimum: Number.NaN, maximum: 10 };
    expect(validateObservableProtocol(invalid)).toEqual(
      expect.arrayContaining([
        "A finite ordered range is required for within-range targets.",
        "Captured-reference targets must use match-reference comparison.",
      ]),
    );
  });

  it("requires explicit claim and abstention boundaries", () => {
    const invalid = structuredClone(LATERAL_ARM_RAISE_REFERENCE_V1);
    invalid.observability.prohibitedClaims = [];
    invalid.observability.abstainWhen = [];
    expect(validateObservableProtocol(invalid)).toEqual(
      expect.arrayContaining([
        "At least one prohibited claim must be declared.",
        "At least one abstention condition must be declared.",
      ]),
    );
  });
});
