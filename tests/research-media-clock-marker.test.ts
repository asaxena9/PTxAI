import { describe, expect, it } from "vitest";
import {
  decodeMediaClockMarkerLuminances,
  encodeMediaClockMarkerBits,
} from "../src/research/media-clock-marker";

describe("media-clock visual marker", () => {
  it("round-trips representative frame identities under luminance compression", () => {
    for (const identity of [0, 1, 42, 1024, 65535]) {
      const luminances = encodeMediaClockMarkerBits(identity).map((bit, index) =>
        bit ? 0.82 + (index % 3) * 0.03 : 0.18 - (index % 3) * 0.03,
      );
      expect(decodeMediaClockMarkerLuminances(luminances)).toMatchObject({
        valid: true,
        frameIdentity: identity,
      });
    }
  });

  it("rejects checksum corruption rather than repairing it", () => {
    const bits = encodeMediaClockMarkerBits(1234);
    bits[bits.length - 1] = bits.at(-1) === 1 ? 0 : 1;
    const decoded = decodeMediaClockMarkerLuminances(
      bits.map((bit) => (bit ? 1 : 0)),
    );
    expect(decoded).toMatchObject({ valid: false, frameIdentity: null });
    expect(decoded.reason).toMatch(/checksum/);
  });

  it("rejects invalid inputs and out-of-range identities", () => {
    expect(decodeMediaClockMarkerLuminances([0, 1])).toMatchObject({
      valid: false,
    });
    expect(() => encodeMediaClockMarkerBits(-1)).toThrow(/\[0,65535\]/);
    expect(() => encodeMediaClockMarkerBits(65_536)).toThrow(/\[0,65535\]/);
  });
});
