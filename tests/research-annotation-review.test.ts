import { describe, expect, it } from "vitest";
import {
  compareResearchAnnotations,
  deriveResearchAnnotationReview,
  type BlindedIndependentReviewDeclaration,
  type ResearchAnnotationAdjudication,
  type ResearchAnnotationReviewEntry,
} from "../src/research/annotation-review";
import type { ResearchAnnotation } from "../src/research/capture-bundle";

const EPISODE_ID = "episode-001";
const ANCESTOR_ID = "P001/S001/E001";
const PROTOCOL_SHA256 = "a".repeat(64);
const EXECUTION_SHA256 = "b".repeat(64);

function annotation(
  annotationId: string,
  raterCode: string,
  overrides: Partial<ResearchAnnotation> = {},
): ResearchAnnotation {
  return {
    annotationId,
    raterCode,
    raterRole: "movement-professional",
    reviewStatus: "draft-unreviewed",
    movementIdentity: "squat",
    assessment: "needs-correction",
    phaseStartMs: 300,
    phaseEndMs: 900,
    realizedErrorType: "insufficient observable depth",
    evidenceJoints: ["left-hip", "left-knee", "left-ankle"],
    evidenceStartMs: 420,
    evidenceEndMs: 780,
    metric: "left knee flexion",
    observedValue: 76,
    targetValue: 90,
    delta: 14,
    unit: "degrees",
    desiredTargetSource: "supplied-protocol",
    cue: "Bend the knee farther toward the supplied 90 degree target.",
    confidence: 0.8,
    protocolSha256: PROTOCOL_SHA256,
    protocolExecutionSha256: EXECUTION_SHA256,
    referenceEvidenceSequenceNumber: 10,
    attemptEvidenceSequenceNumber: 20,
    createdAt: "2026-08-12T00:00:01.000Z",
    ...overrides,
  };
}

function declaration(
  value: ResearchAnnotation,
  overrides: Partial<BlindedIndependentReviewDeclaration> = {},
): BlindedIndependentReviewDeclaration {
  return {
    annotationId: value.annotationId,
    raterCode: value.raterCode,
    episodeId: EPISODE_ID,
    episodeAncestorId: ANCESTOR_ID,
    blinded: true,
    independentlyProduced: true,
    declaredAt: "2026-08-12T00:02:00.000Z",
    ...overrides,
  };
}

function entry(
  value: ResearchAnnotation,
  overrides: Partial<ResearchAnnotationReviewEntry> = {},
): ResearchAnnotationReviewEntry {
  return {
    episodeId: EPISODE_ID,
    episodeAncestorId: ANCESTOR_ID,
    annotation: value,
    independentReviewDeclaration: declaration(value),
    ...overrides,
  };
}

function independentPair(): readonly [
  ResearchAnnotationReviewEntry,
  ResearchAnnotationReviewEntry,
] {
  return [entry(annotation("A1", "R1")), entry(annotation("A2", "R2"))];
}

describe("append-only research annotation review", () => {
  it("keeps an empty review draft and a lone declaration awaiting a second rater", () => {
    expect(deriveResearchAnnotationReview([]).state).toBe("draft");
    expect(
      deriveResearchAnnotationReview([entry(annotation("A1", "R1"))]).state,
    ).toBe("needs-second-review");
  });

  it("completes matching blinded independent annotations by agreement", () => {
    const entries = independentPair();
    const before = JSON.stringify(entries);

    const result = deriveResearchAnnotationReview(entries);

    expect(result).toMatchObject({
      state: "adjudicated",
      resolution: "independent-agreement",
      primaryAnnotationIds: ["A1", "A2"],
      raterCodes: ["R1", "R2"],
      disagreements: [],
      numericTolerance: 0,
      clinicalTruthStatus: "not-established-by-review",
    });
    expect(JSON.stringify(entries)).toBe(before);
  });

  it("lists every claim-bearing disagreement and requests adjudication", () => {
    const first = entry(annotation("A1", "R1"));
    const second = entry(
      annotation("A2", "R2", {
        movementIdentity: "lunge",
        assessment: "uncertain",
        realizedErrorType: "different error",
        evidenceJoints: ["right-hip", "right-knee", "right-ankle"],
        evidenceStartMs: 450,
        evidenceEndMs: 800,
        metric: "right knee flexion",
        targetValue: 95,
        desiredTargetSource: "captured-reference",
        unit: "radians",
        cue: "Different cue.",
      }),
    );

    const result = deriveResearchAnnotationReview([first, second]);

    expect(result.state).toBe("needs-adjudication");
    expect(result.disagreements.map(({ field }) => field)).toEqual(
      expect.arrayContaining([
        "movementIdentity",
        "assessment",
        "realizedErrorType",
        "evidenceJoints",
        "evidenceStartMs",
        "evidenceEndMs",
        "metric",
        "targetValue",
        "desiredTargetSource",
        "unit",
        "cue",
      ]),
    );
  });

  it("rejects duplicate identity, non-blinded review, and self-review claims", () => {
    const selfReviewed = entry(
      annotation("A1", "R1", { reviewStatus: "independently-reviewed" }),
    );
    expect(() => deriveResearchAnnotationReview([selfReviewed])).toThrow(
      /cannot self-mark as independently reviewed/,
    );

    const first = entry(annotation("A1", "R1"));
    const duplicate = entry(annotation("A1", "R1"), {
      independentReviewDeclaration: declaration(annotation("A1", "R1"), {
        blinded: false,
      }),
    });
    const deriveDuplicateReview = () =>
      deriveResearchAnnotationReview([first, duplicate]);
    expect(deriveDuplicateReview).toThrow(/Duplicate annotation ID/);
    expect(deriveDuplicateReview).toThrow(/Duplicate rater code/);
    expect(deriveDuplicateReview).toThrow(/blinded/);
    expect(() =>
      compareResearchAnnotations(first.annotation, first.annotation),
    ).toThrow(/Duplicate annotation ID/);

    expect(() =>
      deriveResearchAnnotationReview([
        first,
        {
          ...entry(annotation("A2", "R2")),
          independentReviewDeclaration: undefined,
        },
      ]),
    ).toThrow(/requires a blinded independent review declaration/);
  });

  it("requires a third distinct rater and exact references for adjudication", () => {
    const entries = [
      entry(annotation("A1", "R1")),
      entry(annotation("A2", "R2", { targetValue: 92, delta: 16 })),
    ] as const;
    const adjudication: ResearchAnnotationAdjudication = {
      episodeId: EPISODE_ID,
      episodeAncestorId: ANCESTOR_ID,
      annotation: annotation("A3", "R3", { targetValue: 91, delta: 15 }),
      reviewedAnnotationIds: ["A1", "A2"],
      declaredAt: "2026-08-12T00:04:00.000Z",
    };

    const result = deriveResearchAnnotationReview(entries, { adjudication });
    expect(result).toMatchObject({
      state: "adjudicated",
      resolution: "third-rater-adjudication",
      adjudicationAnnotationId: "A3",
      raterCodes: ["R1", "R2", "R3"],
    });
    expect(result.disagreements.map(({ field }) => field)).toContain(
      "targetValue",
    );

    expect(() =>
      deriveResearchAnnotationReview(entries, {
        adjudication: {
          ...adjudication,
          annotation: annotation("A3", "R2"),
          reviewedAnnotationIds: ["A1", "not-A2"],
        },
      }),
    ).toThrow(
      /third distinct rater[\s\S]*reference both primary annotation IDs/,
    );
  });

  it("uses exact numeric comparison by default and only an explicit tolerance", () => {
    const first = annotation("A1", "R1");
    const second = annotation("A2", "R2", {
      targetValue: 90.05,
      delta: 14.05,
    });

    expect(
      compareResearchAnnotations(first, second).map(({ field }) => field),
    ).toEqual(expect.arrayContaining(["targetValue", "delta"]));
    expect(compareResearchAnnotations(first, second, 0.1)).toEqual([]);
    expect(() => compareResearchAnnotations(first, second, Number.NaN)).toThrow(
      /finite, non-negative/,
    );
  });

  it("fails closed on episode or ancestry mutation and nonfinite values", () => {
    const [first, second] = independentPair();
    const mutated = {
      ...second,
      episodeId: "episode-mutated",
      episodeAncestorId: "P999/S999/E999",
      annotation: { ...second.annotation, observedValue: Number.POSITIVE_INFINITY },
    };

    const deriveMutatedReview = () =>
      deriveResearchAnnotationReview([first, mutated]);
    expect(deriveMutatedReview).toThrow(/episode ID does not match/);
    expect(deriveMutatedReview).toThrow(/ancestor ID does not match/);
    expect(deriveMutatedReview).toThrow(/observedValue must be finite/);
    expect(() =>
      deriveResearchAnnotationReview([
        entry(annotation("A1", "R1", { confidence: Number.NaN })),
      ]),
    ).toThrow(/confidence must be finite/);
  });

  it("rejects matching evidence-free claim annotations instead of deriving agreement", () => {
    const evidenceFree = (annotationId: string, raterCode: string) =>
      entry(
        annotation(annotationId, raterCode, {
          movementIdentity: "",
          evidenceJoints: [],
          evidenceStartMs: undefined,
          evidenceEndMs: undefined,
          metric: undefined,
          observedValue: undefined,
          targetValue: undefined,
          delta: undefined,
          unit: undefined,
          cue: undefined,
          realizedErrorType: undefined,
          protocolSha256: undefined,
          protocolExecutionSha256: undefined,
          referenceEvidenceSequenceNumber: undefined,
          attemptEvidenceSequenceNumber: undefined,
        }),
      );

    expect(() =>
      deriveResearchAnnotationReview([
        evidenceFree("A1", "R1"),
        evidenceFree("A2", "R2"),
      ]),
    ).toThrow(/movement identity is required[\s\S]*evidence interval is required/);
  });

  it("requires abstention semantics and forbids observable fields on unobservable labels", () => {
    expect(() =>
      deriveResearchAnnotationReview([
        entry(
          annotation("A1", "R1", {
            assessment: "unobservable",
            desiredTargetSource: "captured-reference",
            abstainReason: undefined,
          }),
        ),
      ]),
    ).toThrow(/abstention reason[\s\S]*target source unknown[\s\S]*omit metric/);
  });

  it("does not promote agreement or adjudication into invented clinical truth", () => {
    const result = deriveResearchAnnotationReview(independentPair());
    const untyped = result as unknown as Record<string, unknown>;

    expect(result.clinicalTruthStatus).toBe("not-established-by-review");
    expect(untyped.consensusAnnotation).toBeUndefined();
    expect(untyped.assessment).toBeUndefined();
    expect(untyped.targetValue).toBeUndefined();
  });
});
