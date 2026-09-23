/**
 * Pure source-to-encoded timeline calibration mechanics. This module does not
 * decide that a browser/camera path is calibrated: it fits and reports a map
 * from retained, independently decoded frame identities. A separate decision
 * must supply the maximum localization error its downstream use can tolerate.
 */

export const MEDIA_CLOCK_CALIBRATION_SCHEMA =
  "relay.media-clock-calibration/1" as const;

export const MEDIA_CLOCK_QUANTILE_METHOD =
  "nearest-rank-ceiling-on-sorted-absolute-residuals" as const;

export interface MediaClockFrameObservation {
  /** Exact integer recovered from the frame's compression-robust visual code. */
  readonly frameIdentity: number;
  /** Media time reported by requestVideoFrameCallback, converted to ms. */
  readonly mediaTimeMs: number;
  /** Callback time on the page's performance monotonic clock. */
  readonly callbackMonotonicTimeMs: number;
  /** Decoder-specific confidence in [0,1], retained but not thresholded here. */
  readonly identityConfidence: number;
}

export interface MediaClockMatchedFrame {
  readonly frameIdentity: number;
  readonly sourceMediaTimeMs: number;
  readonly encodedMediaTimeMs: number;
  readonly predictedEncodedMediaTimeMs: number;
  readonly residualMs: number;
  readonly sourceIdentityConfidence: number;
  readonly encodedIdentityConfidence: number;
}

export interface MediaClockCalibrationFit {
  readonly schemaVersion: typeof MEDIA_CLOCK_CALIBRATION_SCHEMA;
  readonly status: "FIT_REPORTED_NOT_ADMISSION_CALIBRATED" | "REJECTED";
  readonly evidenceClass: "MEASURED_INPUT_REQUIRED";
  readonly issues: readonly string[];
  readonly sourceObservationCount: number;
  readonly encodedObservationCount: number;
  readonly matchedFrameCount: number;
  readonly sourceUnmatchedCount: number;
  readonly encodedUnmatchedCount: number;
  readonly sourceDuplicateIdentities: readonly number[];
  readonly encodedDuplicateIdentities: readonly number[];
  readonly sourceSpanMs: number | null;
  readonly encodedSpanMs: number | null;
  /** `encoded_ms = intercept_ms + slope * source_ms`. */
  readonly affineMap: {
    readonly interceptMs: number;
    readonly slope: number;
    readonly estimator: "theil-sen-median-slope-and-median-intercept";
  } | null;
  readonly residuals: {
    readonly quantileMethod: typeof MEDIA_CLOCK_QUANTILE_METHOD;
    readonly medianAbsoluteMs: number;
    readonly p95AbsoluteMs: number;
    readonly maximumAbsoluteMs: number;
    readonly signedMeanMs: number;
  } | null;
  readonly matchedFrames: readonly MediaClockMatchedFrame[];
  readonly claimBoundary: readonly string[];
}

export interface MediaClockLocalizationDecision {
  readonly status: "WITHIN_DECLARED_LOCALIZATION_TOLERANCE" | "REJECTED";
  readonly toleranceMs: number;
  readonly p95AbsoluteResidualMs: number | null;
  readonly maximumAbsoluteResidualMs: number | null;
  readonly reason: string;
}

const CLAIM_BOUNDARY = [
  "A fitted canvas or synthetic-stream map does not establish transfer to a physical camera stream.",
  "Frame-identity decoding and affine residuals calibrate timing mechanics, not movement or annotation truth.",
  "Admission requires a separately justified localization tolerance and an applicability match to the retained browser, codec, recorder, and capture path.",
] as const;

function finiteNonnegative(value: number): boolean {
  return Number.isFinite(value) && value >= 0;
}

function observationIssues(
  observations: readonly MediaClockFrameObservation[],
  label: "source" | "encoded",
): string[] {
  const issues: string[] = [];
  if (observations.length === 0) {
    issues.push(`${label} observations are empty.`);
    return issues;
  }
  observations.forEach((entry, index) => {
    const path = `${label}[${index}]`;
    if (!Number.isSafeInteger(entry.frameIdentity) || entry.frameIdentity < 0) {
      issues.push(`${path}.frameIdentity must be a non-negative safe integer.`);
    }
    if (!finiteNonnegative(entry.mediaTimeMs)) {
      issues.push(`${path}.mediaTimeMs must be finite and non-negative.`);
    }
    if (!finiteNonnegative(entry.callbackMonotonicTimeMs)) {
      issues.push(
        `${path}.callbackMonotonicTimeMs must be finite and non-negative.`,
      );
    }
    if (
      !Number.isFinite(entry.identityConfidence) ||
      entry.identityConfidence < 0 ||
      entry.identityConfidence > 1
    ) {
      issues.push(`${path}.identityConfidence must be finite and within [0,1].`);
    }
    if (index > 0) {
      const prior = observations[index - 1];
      if (entry.mediaTimeMs < prior.mediaTimeMs) {
        issues.push(`${path}.mediaTimeMs moved backward.`);
      }
      if (entry.callbackMonotonicTimeMs < prior.callbackMonotonicTimeMs) {
        issues.push(`${path}.callbackMonotonicTimeMs moved backward.`);
      }
      if (entry.frameIdentity <= prior.frameIdentity) {
        issues.push(`${path}.frameIdentity must increase strictly.`);
      }
    }
  });
  return issues;
}

function duplicateIdentities(
  observations: readonly MediaClockFrameObservation[],
): number[] {
  const counts = new Map<number, number>();
  for (const entry of observations) {
    counts.set(entry.frameIdentity, (counts.get(entry.frameIdentity) ?? 0) + 1);
  }
  return Array.from(counts)
    .filter(([, count]) => count > 1)
    .map(([identity]) => identity)
    .sort((left, right) => left - right);
}

function median(values: readonly number[]): number {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function nearestRank(
  sortedAscending: readonly number[],
  probability: number,
): number {
  if (sortedAscending.length === 0) return Number.NaN;
  const rank = Math.max(
    1,
    Math.min(sortedAscending.length, Math.ceil(probability * sortedAscending.length)),
  );
  return sortedAscending[rank - 1];
}

function span(observations: readonly MediaClockFrameObservation[]): number | null {
  if (observations.length < 2) return null;
  return observations.at(-1)!.mediaTimeMs - observations[0].mediaTimeMs;
}

/**
 * Fit a robust affine map using only identities unique in both traces. Inputs
 * are never sorted or repaired; malformed order rejects the fit.
 */
export function fitMediaClockCalibration(
  source: readonly MediaClockFrameObservation[],
  encoded: readonly MediaClockFrameObservation[],
): MediaClockCalibrationFit {
  const sourceDuplicates = duplicateIdentities(source);
  const encodedDuplicates = duplicateIdentities(encoded);
  const issues = [
    ...observationIssues(source, "source"),
    ...observationIssues(encoded, "encoded"),
  ];
  if (sourceDuplicates.length > 0) {
    issues.push("Source frame identities are duplicated and therefore ambiguous.");
  }
  if (encodedDuplicates.length > 0) {
    issues.push("Encoded frame identities are duplicated and therefore ambiguous.");
  }

  const encodedMap = new Map(encoded.map((entry) => [entry.frameIdentity, entry]));
  const matchedRaw = source
    .filter((entry) => encodedMap.has(entry.frameIdentity))
    .map((sourceEntry) => ({
      source: sourceEntry,
      encoded: encodedMap.get(sourceEntry.frameIdentity)!,
    }));

  if (matchedRaw.length < 3) {
    issues.push("At least three uniquely matched frame identities are required.");
  }
  for (let index = 1; index < matchedRaw.length; index += 1) {
    if (
      matchedRaw[index].encoded.mediaTimeMs <=
      matchedRaw[index - 1].encoded.mediaTimeMs
    ) {
      issues.push("Matched encoded frame times must increase strictly by identity.");
      break;
    }
  }

  const sourceSpanMs = span(source);
  const encodedSpanMs = span(encoded);
  const matchedSourceSpan =
    matchedRaw.length >= 2
      ? matchedRaw.at(-1)!.source.mediaTimeMs - matchedRaw[0].source.mediaTimeMs
      : 0;
  if (matchedRaw.length >= 3 && matchedSourceSpan <= 0) {
    issues.push("Matched source frame identities have no positive time span.");
  }

  if (issues.length > 0) {
    return {
      schemaVersion: MEDIA_CLOCK_CALIBRATION_SCHEMA,
      status: "REJECTED",
      evidenceClass: "MEASURED_INPUT_REQUIRED",
      issues: Array.from(new Set(issues)),
      sourceObservationCount: source.length,
      encodedObservationCount: encoded.length,
      matchedFrameCount: matchedRaw.length,
      sourceUnmatchedCount: source.length - matchedRaw.length,
      encodedUnmatchedCount: encoded.length - matchedRaw.length,
      sourceDuplicateIdentities: sourceDuplicates,
      encodedDuplicateIdentities: encodedDuplicates,
      sourceSpanMs,
      encodedSpanMs,
      affineMap: null,
      residuals: null,
      matchedFrames: [],
      claimBoundary: CLAIM_BOUNDARY,
    };
  }

  const slopes: number[] = [];
  for (let left = 0; left < matchedRaw.length; left += 1) {
    for (let right = left + 1; right < matchedRaw.length; right += 1) {
      const sourceDelta =
        matchedRaw[right].source.mediaTimeMs -
        matchedRaw[left].source.mediaTimeMs;
      if (sourceDelta <= 0) continue;
      slopes.push(
        (matchedRaw[right].encoded.mediaTimeMs -
          matchedRaw[left].encoded.mediaTimeMs) /
          sourceDelta,
      );
    }
  }
  const slope = median(slopes);
  const interceptMs = median(
    matchedRaw.map(
      (pair) => pair.encoded.mediaTimeMs - slope * pair.source.mediaTimeMs,
    ),
  );
  if (!Number.isFinite(slope) || slope <= 0 || !Number.isFinite(interceptMs)) {
    return {
      schemaVersion: MEDIA_CLOCK_CALIBRATION_SCHEMA,
      status: "REJECTED",
      evidenceClass: "MEASURED_INPUT_REQUIRED",
      issues: ["The robust affine estimator produced an invalid map."],
      sourceObservationCount: source.length,
      encodedObservationCount: encoded.length,
      matchedFrameCount: matchedRaw.length,
      sourceUnmatchedCount: source.length - matchedRaw.length,
      encodedUnmatchedCount: encoded.length - matchedRaw.length,
      sourceDuplicateIdentities: sourceDuplicates,
      encodedDuplicateIdentities: encodedDuplicates,
      sourceSpanMs,
      encodedSpanMs,
      affineMap: null,
      residuals: null,
      matchedFrames: [],
      claimBoundary: CLAIM_BOUNDARY,
    };
  }

  const matchedFrames = matchedRaw.map(({ source: sourceEntry, encoded: encodedEntry }) => {
    const predictedEncodedMediaTimeMs =
      interceptMs + slope * sourceEntry.mediaTimeMs;
    return {
      frameIdentity: sourceEntry.frameIdentity,
      sourceMediaTimeMs: sourceEntry.mediaTimeMs,
      encodedMediaTimeMs: encodedEntry.mediaTimeMs,
      predictedEncodedMediaTimeMs,
      residualMs: encodedEntry.mediaTimeMs - predictedEncodedMediaTimeMs,
      sourceIdentityConfidence: sourceEntry.identityConfidence,
      encodedIdentityConfidence: encodedEntry.identityConfidence,
    };
  });
  const absoluteResiduals = matchedFrames
    .map((entry) => Math.abs(entry.residualMs))
    .sort((left, right) => left - right);
  const signedMeanMs =
    matchedFrames.reduce((sum, entry) => sum + entry.residualMs, 0) /
    matchedFrames.length;

  return {
    schemaVersion: MEDIA_CLOCK_CALIBRATION_SCHEMA,
    status: "FIT_REPORTED_NOT_ADMISSION_CALIBRATED",
    evidenceClass: "MEASURED_INPUT_REQUIRED",
    issues: [],
    sourceObservationCount: source.length,
    encodedObservationCount: encoded.length,
    matchedFrameCount: matchedFrames.length,
    sourceUnmatchedCount: source.length - matchedFrames.length,
    encodedUnmatchedCount: encoded.length - matchedFrames.length,
    sourceDuplicateIdentities: sourceDuplicates,
    encodedDuplicateIdentities: encodedDuplicates,
    sourceSpanMs,
    encodedSpanMs,
    affineMap: {
      interceptMs,
      slope,
      estimator: "theil-sen-median-slope-and-median-intercept",
    },
    residuals: {
      quantileMethod: MEDIA_CLOCK_QUANTILE_METHOD,
      medianAbsoluteMs: median(absoluteResiduals),
      p95AbsoluteMs: nearestRank(absoluteResiduals, 0.95),
      maximumAbsoluteMs: absoluteResiduals.at(-1)!,
      signedMeanMs,
    },
    matchedFrames,
    claimBoundary: CLAIM_BOUNDARY,
  };
}

/**
 * Apply an externally justified localization tolerance. This function does not
 * derive or choose that tolerance and therefore cannot silently promote a fit.
 */
export function evaluateMediaClockLocalizationTolerance(
  fit: MediaClockCalibrationFit,
  toleranceMs: number,
): MediaClockLocalizationDecision {
  if (!Number.isFinite(toleranceMs) || toleranceMs < 0) {
    return {
      status: "REJECTED",
      toleranceMs,
      p95AbsoluteResidualMs: fit.residuals?.p95AbsoluteMs ?? null,
      maximumAbsoluteResidualMs: fit.residuals?.maximumAbsoluteMs ?? null,
      reason: "Localization tolerance must be finite, non-negative, and justified outside the calibration fit.",
    };
  }
  if (fit.status !== "FIT_REPORTED_NOT_ADMISSION_CALIBRATED" || !fit.residuals) {
    return {
      status: "REJECTED",
      toleranceMs,
      p95AbsoluteResidualMs: null,
      maximumAbsoluteResidualMs: null,
      reason: "No valid calibration fit is available.",
    };
  }
  const within = fit.residuals.maximumAbsoluteMs <= toleranceMs;
  return {
    status: within
      ? "WITHIN_DECLARED_LOCALIZATION_TOLERANCE"
      : "REJECTED",
    toleranceMs,
    p95AbsoluteResidualMs: fit.residuals.p95AbsoluteMs,
    maximumAbsoluteResidualMs: fit.residuals.maximumAbsoluteMs,
    reason: within
      ? "Every retained matched-frame residual is within the externally declared tolerance."
      : "At least one retained matched-frame residual exceeds the externally declared tolerance.",
  };
}
