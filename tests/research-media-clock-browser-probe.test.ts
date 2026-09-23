import { describe, expect, it, vi } from "vitest";
import {
  SYNTHETIC_MEDIA_CLOCK_PROBE_CLAIM_BOUNDARY,
  SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG,
  calibrationObservationsFromRaw,
  selectSupportedMediaRecorderMimeType,
  type SyntheticMediaClockRawObservation,
} from "../src/research/media-clock-browser-probe";
import {
  decodeMediaClockMarkerLuminances,
  encodeMediaClockMarkerBits,
} from "../src/research/media-clock-marker";

function rawObservation(
  identity: number,
  mediaTimeMs: number,
  trace: SyntheticMediaClockRawObservation["trace"] = "source-presented",
): SyntheticMediaClockRawObservation {
  return {
    trace,
    callbackMonotonicTimeMs: 10_000 + mediaTimeMs,
    mediaTimeMs,
    expectedDisplayTimeMs: 10_010 + mediaTimeMs,
    presentedFrames: identity,
    decoded: decodeMediaClockMarkerLuminances(
      encodeMediaClockMarkerBits(identity).map((bit) => (bit ? 0.9 : 0.1)),
    ),
  };
}

describe("synthetic media-clock browser probe", () => {
  it("chooses the first explicitly supported recorder MIME type", () => {
    const support = vi.fn((mimeType: string) =>
      ["video/webm;codecs=vp8", "video/mp4"].includes(mimeType),
    );

    expect(selectSupportedMediaRecorderMimeType(support)).toBe(
      "video/webm;codecs=vp8",
    );
    expect(support.mock.calls.map(([mimeType]) => mimeType)).toEqual([
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
    ]);
    expect(selectSupportedMediaRecorderMimeType(() => false)).toBeNull();
  });

  it("retains decoded observation order and duplicates for fail-closed fitting", () => {
    const observations = [
      rawObservation(0, 0),
      rawObservation(1, 50),
      rawObservation(1, 51),
      rawObservation(2, 100),
      rawObservation(49, 2_400),
    ];

    expect(calibrationObservationsFromRaw(observations)).toEqual([
      expect.objectContaining({ frameIdentity: 1, mediaTimeMs: 50 }),
      expect.objectContaining({ frameIdentity: 1, mediaTimeMs: 51 }),
      expect.objectContaining({ frameIdentity: 2, mediaTimeMs: 100 }),
    ]);
  });

  it("bounds one trial and states that it cannot calibrate camera admission", () => {
    expect(
      SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.frameCount *
        SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.targetFrameIntervalMs,
    ).toBe(2_400);
    expect(SYNTHETIC_MEDIA_CLOCK_PROBE_CONFIG.maximumWallTimeMs).toBe(20_000);
    expect(SYNTHETIC_MEDIA_CLOCK_PROBE_CLAIM_BOUNDARY.join(" ")).toMatch(
      /does not establish transfer to a physical camera stream/,
    );
    expect(SYNTHETIC_MEDIA_CLOCK_PROBE_CLAIM_BOUNDARY.join(" ")).toMatch(
      /cannot set encodedReplayTimelineCalibrated=true/,
    );
  });
});
