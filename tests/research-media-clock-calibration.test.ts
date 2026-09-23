import { describe, expect, it } from "vitest";
import {
  evaluateMediaClockLocalizationTolerance,
  fitMediaClockCalibration,
  type MediaClockFrameObservation,
} from "../src/research/media-clock-calibration";

function observation(
  frameIdentity: number,
  mediaTimeMs: number,
): MediaClockFrameObservation {
  return {
    frameIdentity,
    mediaTimeMs,
    callbackMonotonicTimeMs: 10_000 + mediaTimeMs,
    identityConfidence: 1,
  };
}

describe("media-clock affine calibration", () => {
  it("recovers a known offset/rate with dropped frames and reports an outlier", () => {
    const source = Array.from({ length: 20 }, (_, index) =>
      observation(index, index * 40),
    );
    const encoded = source
      .filter((entry) => ![3, 11].includes(entry.frameIdentity))
      .map((entry) =>
        observation(
          entry.frameIdentity,
          27.5 + 1.00025 * entry.mediaTimeMs +
            (entry.frameIdentity === 17 ? 9 : 0),
        ),
      );

    const fit = fitMediaClockCalibration(source, encoded);

    expect(fit.status).toBe("FIT_REPORTED_NOT_ADMISSION_CALIBRATED");
    expect(fit.matchedFrameCount).toBe(18);
    expect(fit.sourceUnmatchedCount).toBe(2);
    expect(fit.encodedUnmatchedCount).toBe(0);
    expect(fit.affineMap?.interceptMs).toBeCloseTo(27.5, 9);
    expect(fit.affineMap?.slope).toBeCloseTo(1.00025, 9);
    expect(fit.residuals?.medianAbsoluteMs).toBeCloseTo(0, 9);
    expect(fit.residuals?.maximumAbsoluteMs).toBeCloseTo(9, 9);
    expect(fit.claimBoundary.join(" ")).toMatch(/does not establish transfer/);
  });

  it("rejects duplicate, backward, and insufficient identity traces", () => {
    const duplicate = [observation(0, 0), observation(0, 10)];
    const backward = [observation(0, 0), observation(2, 20), observation(1, 30)];

    expect(fitMediaClockCalibration(duplicate, duplicate)).toMatchObject({
      status: "REJECTED",
      sourceDuplicateIdentities: [0],
      encodedDuplicateIdentities: [0],
    });
    expect(
      fitMediaClockCalibration(backward, [
        observation(0, 2),
        observation(1, 12),
        observation(2, 22),
      ]).issues.join(" "),
    ).toMatch(/frameIdentity must increase strictly/);
    expect(
      fitMediaClockCalibration(
        [observation(0, 0), observation(1, 10)],
        [observation(0, 2), observation(1, 12)],
      ).issues.join(" "),
    ).toMatch(/At least three/);
  });

  it("never invents a localization tolerance", () => {
    const source = [observation(0, 0), observation(1, 10), observation(2, 20)];
    const encoded = [observation(0, 4), observation(1, 14), observation(2, 25)];
    const fit = fitMediaClockCalibration(source, encoded);

    expect(evaluateMediaClockLocalizationTolerance(fit, 0.49).status).toBe(
      "REJECTED",
    );
    expect(evaluateMediaClockLocalizationTolerance(fit, 2).status).toBe(
      "WITHIN_DECLARED_LOCALIZATION_TOLERANCE",
    );
    expect(
      evaluateMediaClockLocalizationTolerance(fit, Number.NaN).reason,
    ).toMatch(/justified outside/);
  });
});
