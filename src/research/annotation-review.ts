import type { ResearchAnnotation } from "./capture-bundle";

export const EXACT_ANNOTATION_NUMERIC_TOLERANCE = 0;

export type ResearchAnnotationReviewState =
  | "draft"
  | "needs-second-review"
  | "needs-adjudication"
  | "adjudicated";

export type ResearchAnnotationDisagreementField =
  | "movementIdentity"
  | "assessment"
  | "phaseStartMs"
  | "phaseEndMs"
  | "realizedErrorType"
  | "evidenceJoints"
  | "evidenceStartMs"
  | "evidenceEndMs"
  | "evidenceClock"
  | "metric"
  | "observedValue"
  | "targetValue"
  | "delta"
  | "unit"
  | "desiredTargetSource"
  | "cue"
  | "confidence"
  | "abstainReason"
  | "protocolSha256"
  | "protocolExecutionSha256"
  | "referenceEvidenceSequenceNumber"
  | "attemptEvidenceSequenceNumber";

export type ResearchAnnotationSnapshot = Readonly<
  Omit<ResearchAnnotation, "evidenceJoints">
> & {
  readonly evidenceJoints: readonly ResearchAnnotation["evidenceJoints"][number][];
};

/**
 * A declaration belongs to one immutable annotation submission. Repeating the
 * annotation, rater, episode, and ancestor identifiers makes accidental lineage
 * changes detectable when append-only records are later combined.
 */
export interface BlindedIndependentReviewDeclaration {
  readonly annotationId: string;
  readonly raterCode: string;
  readonly episodeId: string;
  readonly episodeAncestorId: string;
  readonly blinded: boolean;
  readonly independentlyProduced: boolean;
  readonly declaredAt: string;
}

export interface ResearchAnnotationReviewEntry {
  readonly episodeId: string;
  readonly episodeAncestorId: string;
  readonly annotation: ResearchAnnotationSnapshot;
  readonly independentReviewDeclaration?: BlindedIndependentReviewDeclaration;
}

/**
 * Adjudication is another annotation, not a mutation or synthetic merge of the
 * first two. The adjudicator is intentionally not declared blinded: this record
 * states that they considered both annotations named in reviewedAnnotationIds.
 */
export interface ResearchAnnotationAdjudication {
  readonly episodeId: string;
  readonly episodeAncestorId: string;
  readonly annotation: ResearchAnnotationSnapshot;
  readonly reviewedAnnotationIds: readonly [string, string];
  readonly declaredAt: string;
}

export interface ResearchAnnotationDisagreement {
  field: ResearchAnnotationDisagreementField;
  firstAnnotationId: string;
  secondAnnotationId: string;
  firstValue: unknown;
  secondValue: unknown;
}

export interface ResearchAnnotationReviewOptions {
  /** Absolute tolerance for every compared numeric field. Omitted means exact. */
  numericTolerance?: number;
  adjudication?: ResearchAnnotationAdjudication;
}

export interface ResearchAnnotationReviewResult {
  state: ResearchAnnotationReviewState;
  resolution:
    | "none"
    | "independent-agreement"
    | "third-rater-adjudication";
  episodeId?: string;
  episodeAncestorId?: string;
  primaryAnnotationIds: readonly string[];
  raterCodes: readonly string[];
  adjudicationAnnotationId?: string;
  disagreements: readonly ResearchAnnotationDisagreement[];
  numericTolerance: number;
  /** Rater agreement or adjudication is evidence provenance, not clinical truth. */
  clinicalTruthStatus: "not-established-by-review";
}

export class ResearchAnnotationReviewValidationError extends Error {
  readonly issues: readonly string[];

  constructor(issues: readonly string[]) {
    super(issues.join(" "));
    this.name = "ResearchAnnotationReviewValidationError";
    this.issues = [...issues];
  }
}

const NUMERIC_FIELDS = [
  "phaseStartMs",
  "phaseEndMs",
  "evidenceStartMs",
  "evidenceEndMs",
  "observedValue",
  "targetValue",
  "delta",
] as const satisfies readonly (keyof ResearchAnnotation)[];

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const OBSERVABLE_ASSESSMENTS = new Set(["acceptable", "needs-correction"]);
const CLAIM_BEARING_ASSESSMENTS = new Set([
  "acceptable",
  "needs-correction",
  "uncertain",
]);

const SCALAR_COMPARISON_FIELDS = [
  "movementIdentity",
  "assessment",
  "phaseStartMs",
  "phaseEndMs",
  "realizedErrorType",
  "evidenceStartMs",
  "evidenceEndMs",
  "evidenceClock",
  "metric",
  "observedValue",
  "targetValue",
  "delta",
  "unit",
  "desiredTargetSource",
  "cue",
  "confidence",
  "abstainReason",
  "protocolSha256",
  "protocolExecutionSha256",
  "referenceEvidenceSequenceNumber",
  "attemptEvidenceSequenceNumber",
] as const satisfies readonly ResearchAnnotationDisagreementField[];

function canonicalIdentifier(value: string): string {
  return value.trim().toLocaleLowerCase("en-US");
}

function pushRequiredIdentifierError(
  issues: string[],
  value: string,
  label: string,
): void {
  if (!value.trim()) issues.push(`${label} is required.`);
}

function hasText(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function pushPairedRangeIssues(
  issues: string[],
  start: number | undefined,
  end: number | undefined,
  label: string,
  required: boolean,
): void {
  if (start === undefined && end === undefined) {
    if (required) issues.push(`${label} is required.`);
    return;
  }
  if (start === undefined || end === undefined) {
    issues.push(`${label} start and end must be supplied together.`);
    return;
  }
  if (!Number.isFinite(start) || !Number.isFinite(end)) return;
  if (start < 0 || end < 0) issues.push(`${label} must be non-negative.`);
  if (end < start) issues.push(`${label} end cannot precede its start.`);
}

function pushRequiredFiniteField(
  issues: string[],
  annotation: ResearchAnnotationSnapshot,
  field: "observedValue" | "targetValue" | "delta",
  context: string,
): void {
  if (!Number.isFinite(annotation[field])) {
    issues.push(`${context} ${field} is required and must be finite.`);
  }
}

function claimSourceBindingIssues(
  annotation: ResearchAnnotationSnapshot,
  context: string,
): string[] {
  if (!OBSERVABLE_ASSESSMENTS.has(annotation.assessment)) return [];
  const issues: string[] = [];
  if (!hasText(annotation.protocolSha256) || !SHA256_PATTERN.test(annotation.protocolSha256)) {
    issues.push(
      `${context} observable assessment must bind a lowercase SHA-256 protocol digest.`,
    );
  }
  if (
    !hasText(annotation.protocolExecutionSha256) ||
    !SHA256_PATTERN.test(annotation.protocolExecutionSha256)
  ) {
    issues.push(
      `${context} observable assessment must bind a lowercase SHA-256 protocol-execution digest.`,
    );
  }
  for (const field of [
    "referenceEvidenceSequenceNumber",
    "attemptEvidenceSequenceNumber",
  ] as const) {
    const value = annotation[field];
    if (!Number.isSafeInteger(value) || (value as number) < 0) {
      issues.push(
        `${context} observable assessment requires a non-negative safe-integer ${field}.`,
      );
    }
  }
  return issues;
}

function annotationValidationIssues(
  annotation: ResearchAnnotationSnapshot,
  context: string,
): string[] {
  const issues: string[] = [];
  pushRequiredIdentifierError(issues, annotation.annotationId, `${context} annotation ID`);
  pushRequiredIdentifierError(issues, annotation.raterCode, `${context} rater code`);
  if (!hasText(annotation.movementIdentity)) {
    issues.push(`${context} movement identity is required.`);
  }

  if (
    annotation.assessment !== "acceptable" &&
    annotation.assessment !== "needs-correction" &&
    annotation.assessment !== "uncertain" &&
    annotation.assessment !== "unobservable"
  ) {
    issues.push(`${context} assessment is not recognized.`);
  }
  if (
    annotation.desiredTargetSource !== "captured-reference" &&
    annotation.desiredTargetSource !== "supplied-protocol" &&
    annotation.desiredTargetSource !== "analytic-oracle" &&
    annotation.desiredTargetSource !== "unknown"
  ) {
    issues.push(`${context} desired target source is not recognized.`);
  }

  if (annotation.reviewStatus !== "draft-unreviewed") {
    issues.push(
      `${context} annotation ${annotation.annotationId || "(missing ID)"} cannot self-mark as independently reviewed; review state is derived from distinct append-only records.`,
    );
  }

  if (
    !Number.isFinite(annotation.confidence) ||
    annotation.confidence < 0 ||
    annotation.confidence > 1
  ) {
    issues.push(
      `${context} annotation ${annotation.annotationId || "(missing ID)"} confidence must be finite and between 0 and 1.`,
    );
  }

  for (const field of NUMERIC_FIELDS) {
    const value = annotation[field];
    if (value !== undefined && !Number.isFinite(value)) {
      issues.push(
        `${context} annotation ${annotation.annotationId || "(missing ID)"} ${field} must be finite when present.`,
      );
    }
  }

  if (!Number.isFinite(Date.parse(annotation.createdAt))) {
    issues.push(`${context} annotation createdAt must be a valid timestamp.`);
  }

  pushPairedRangeIssues(
    issues,
    annotation.phaseStartMs,
    annotation.phaseEndMs,
    `${context} phase interval`,
    false,
  );
  pushPairedRangeIssues(
    issues,
    annotation.evidenceStartMs,
    annotation.evidenceEndMs,
    `${context} evidence interval`,
    CLAIM_BEARING_ASSESSMENTS.has(annotation.assessment),
  );
  if (
    Number.isFinite(annotation.phaseStartMs) &&
    Number.isFinite(annotation.phaseEndMs) &&
    Number.isFinite(annotation.evidenceStartMs) &&
    Number.isFinite(annotation.evidenceEndMs) &&
    ((annotation.evidenceStartMs as number) < (annotation.phaseStartMs as number) ||
      (annotation.evidenceEndMs as number) > (annotation.phaseEndMs as number))
  ) {
    issues.push(`${context} evidence interval must lie within its supplied phase interval.`);
  }

  if (!Array.isArray(annotation.evidenceJoints)) {
    issues.push(`${context} evidence joints must be an array.`);
  } else {
    if (
      CLAIM_BEARING_ASSESSMENTS.has(annotation.assessment) &&
      annotation.evidenceJoints.length === 0
    ) {
      issues.push(`${context} claim-bearing assessment requires at least one evidence joint.`);
    }
    if (new Set(annotation.evidenceJoints).size !== annotation.evidenceJoints.length) {
      issues.push(`${context} evidence joints cannot contain duplicates.`);
    }
  }

  for (const [field, value] of [
    ["protocolSha256", annotation.protocolSha256],
    ["protocolExecutionSha256", annotation.protocolExecutionSha256],
  ] as const) {
    if (value !== undefined && !SHA256_PATTERN.test(value)) {
      issues.push(`${context} ${field} must be a lowercase SHA-256 digest when present.`);
    }
  }
  for (const field of [
    "referenceEvidenceSequenceNumber",
    "attemptEvidenceSequenceNumber",
  ] as const) {
    const value = annotation[field];
    if (value !== undefined && (!Number.isSafeInteger(value) || value < 0)) {
      issues.push(`${context} ${field} must be a non-negative safe integer when present.`);
    }
  }

  if (OBSERVABLE_ASSESSMENTS.has(annotation.assessment)) {
    if (!hasText(annotation.metric)) {
      issues.push(`${context} observable assessment requires a metric.`);
    }
    pushRequiredFiniteField(issues, annotation, "observedValue", context);
    pushRequiredFiniteField(issues, annotation, "targetValue", context);
    pushRequiredFiniteField(issues, annotation, "delta", context);
    if (!hasText(annotation.unit)) {
      issues.push(`${context} observable assessment requires a unit.`);
    }
    if (annotation.desiredTargetSource === "unknown") {
      issues.push(`${context} observable assessment requires an explicit target source.`);
    }
    if (hasText(annotation.abstainReason)) {
      issues.push(`${context} observable assessment cannot also declare an abstention reason.`);
    }
    if (
      Number.isFinite(annotation.observedValue) &&
      Number.isFinite(annotation.targetValue) &&
      Number.isFinite(annotation.delta)
    ) {
      const expected =
        (annotation.targetValue as number) - (annotation.observedValue as number);
      const scale = Math.max(
        1,
        Math.abs(annotation.observedValue as number),
        Math.abs(annotation.targetValue as number),
        Math.abs(annotation.delta as number),
      );
      if (
        Math.abs((annotation.delta as number) - expected) >
        Number.EPSILON * 16 * scale
      ) {
        issues.push(`${context} delta must equal target value minus observed value.`);
      }
    }
  }

  if (annotation.assessment === "needs-correction") {
    if (!hasText(annotation.realizedErrorType)) {
      issues.push(`${context} correction assessment requires a realized error type.`);
    }
    if (!hasText(annotation.cue)) {
      issues.push(`${context} correction assessment requires a grounded cue.`);
    }
  }

  if (annotation.assessment === "uncertain") {
    if (!hasText(annotation.realizedErrorType)) {
      issues.push(`${context} uncertain assessment requires an observable description.`);
    }
    if (annotation.desiredTargetSource === "unknown") {
      issues.push(`${context} uncertain assessment requires an explicit comparator source.`);
    }
  }

  if (annotation.assessment === "unobservable") {
    if (!hasText(annotation.abstainReason)) {
      issues.push(`${context} unobservable assessment requires an abstention reason.`);
    }
    if (annotation.desiredTargetSource !== "unknown") {
      issues.push(`${context} unobservable assessment must declare its target source unknown.`);
    }
    if (
      annotation.metric !== undefined ||
      annotation.observedValue !== undefined ||
      annotation.targetValue !== undefined ||
      annotation.delta !== undefined ||
      annotation.unit !== undefined ||
      annotation.cue !== undefined
    ) {
      issues.push(
        `${context} unobservable assessment must omit metric, values, unit, and cue.`,
      );
    }
  }

  return issues;
}

function declarationValidationIssues(
  entry: ResearchAnnotationReviewEntry,
  context: string,
  required: boolean,
): string[] {
  const declaration = entry.independentReviewDeclaration;
  if (!declaration) {
    return required
      ? [`${context} requires a blinded independent review declaration.`]
      : [];
  }

  const issues: string[] = [];
  if (!declaration.blinded || !declaration.independentlyProduced) {
    issues.push(
      `${context} declaration must affirm both blinded and independently produced review.`,
    );
  }
  if (declaration.annotationId !== entry.annotation.annotationId) {
    issues.push(`${context} declaration annotation ID does not match its annotation.`);
  }
  if (declaration.raterCode !== entry.annotation.raterCode) {
    issues.push(`${context} declaration rater code does not match its annotation.`);
  }
  if (declaration.episodeId !== entry.episodeId) {
    issues.push(`${context} declaration episode ID does not match its review entry.`);
  }
  if (declaration.episodeAncestorId !== entry.episodeAncestorId) {
    issues.push(
      `${context} declaration episode ancestor ID does not match its review entry.`,
    );
  }
  if (!Number.isFinite(Date.parse(declaration.declaredAt))) {
    issues.push(`${context} declaration timestamp must be valid.`);
  }
  return issues;
}

function valuesAgree(
  first: unknown,
  second: unknown,
  numericTolerance: number,
): boolean {
  if (typeof first === "number" || typeof second === "number") {
    return (
      typeof first === "number" &&
      typeof second === "number" &&
      Math.abs(first - second) <= numericTolerance
    );
  }
  return Object.is(first, second);
}

function jointListsAgree(
  first: ResearchAnnotationSnapshot["evidenceJoints"],
  second: ResearchAnnotationSnapshot["evidenceJoints"],
): boolean {
  return (
    first.length === second.length &&
    first.every((joint, index) => joint === second[index])
  );
}

function copiedValue(value: unknown): unknown {
  return Array.isArray(value) ? [...value] : value;
}

function assertValidNumericTolerance(numericTolerance: number): void {
  if (!Number.isFinite(numericTolerance) || numericTolerance < 0) {
    throw new ResearchAnnotationReviewValidationError([
      "Numeric tolerance must be a finite, non-negative number.",
    ]);
  }
}

/**
 * Compares all claim-bearing annotation fields. Rater identity, role, timestamps,
 * notes, and the self-asserted review status are provenance rather than label
 * agreement fields. Evidence-joint order is significant under exact comparison.
 */
export function compareResearchAnnotations(
  first: ResearchAnnotationSnapshot,
  second: ResearchAnnotationSnapshot,
  numericTolerance = EXACT_ANNOTATION_NUMERIC_TOLERANCE,
): ResearchAnnotationDisagreement[] {
  assertValidNumericTolerance(numericTolerance);
  const issues = [
    ...annotationValidationIssues(first, "First"),
    ...annotationValidationIssues(second, "Second"),
  ];
  if (
    canonicalIdentifier(first.annotationId) &&
    canonicalIdentifier(first.annotationId) ===
      canonicalIdentifier(second.annotationId)
  ) {
    issues.push(`Duplicate annotation ID: ${second.annotationId}.`);
  }
  if (
    canonicalIdentifier(first.raterCode) &&
    canonicalIdentifier(first.raterCode) === canonicalIdentifier(second.raterCode)
  ) {
    issues.push(`Duplicate rater code: ${second.raterCode}.`);
  }
  if (issues.length > 0) throw new ResearchAnnotationReviewValidationError(issues);

  const disagreements: ResearchAnnotationDisagreement[] = [];
  const addDisagreement = (
    field: ResearchAnnotationDisagreementField,
    firstValue: unknown,
    secondValue: unknown,
  ) => {
    disagreements.push({
      field,
      firstAnnotationId: first.annotationId,
      secondAnnotationId: second.annotationId,
      firstValue: copiedValue(firstValue),
      secondValue: copiedValue(secondValue),
    });
  };

  for (const field of SCALAR_COMPARISON_FIELDS) {
    const firstValue = first[field];
    const secondValue = second[field];
    if (!valuesAgree(firstValue, secondValue, numericTolerance)) {
      addDisagreement(field, firstValue, secondValue);
    }
  }

  if (!jointListsAgree(first.evidenceJoints, second.evidenceJoints)) {
    addDisagreement("evidenceJoints", first.evidenceJoints, second.evidenceJoints);
  }

  return disagreements;
}

function reviewValidationIssues(
  entries: readonly ResearchAnnotationReviewEntry[],
  adjudication: ResearchAnnotationAdjudication | undefined,
): string[] {
  const issues: string[] = [];
  if (entries.length > 2) {
    issues.push(
      "Independent review accepts at most two primary annotations; a third annotation must be supplied as adjudication.",
    );
  }

  const seenAnnotationIds = new Set<string>();
  const seenRaterCodes = new Set<string>();
  for (const [index, entry] of entries.entries()) {
    const context = `Review entry ${index + 1}`;
    pushRequiredIdentifierError(issues, entry.episodeId, `${context} episode ID`);
    pushRequiredIdentifierError(
      issues,
      entry.episodeAncestorId,
      `${context} episode ancestor ID`,
    );
    issues.push(...annotationValidationIssues(entry.annotation, context));
    issues.push(...claimSourceBindingIssues(entry.annotation, context));
    issues.push(...declarationValidationIssues(entry, context, entries.length === 2));

    const annotationId = canonicalIdentifier(entry.annotation.annotationId);
    if (annotationId && seenAnnotationIds.has(annotationId)) {
      issues.push(`Duplicate annotation ID: ${entry.annotation.annotationId}.`);
    }
    seenAnnotationIds.add(annotationId);

    const raterCode = canonicalIdentifier(entry.annotation.raterCode);
    if (raterCode && seenRaterCodes.has(raterCode)) {
      issues.push(`Duplicate rater code: ${entry.annotation.raterCode}.`);
    }
    seenRaterCodes.add(raterCode);

    if (index > 0) {
      const first = entries[0];
      if (entry.episodeId !== first.episodeId) {
        issues.push(`${context} episode ID does not match the first review entry.`);
      }
      if (entry.episodeAncestorId !== first.episodeAncestorId) {
        issues.push(
          `${context} episode ancestor ID does not match the first review entry.`,
        );
      }
    }
  }

  if (!adjudication) return issues;
  if (entries.length !== 2) {
    issues.push("Adjudication requires exactly two primary annotations.");
  }

  pushRequiredIdentifierError(
    issues,
    adjudication.episodeId,
    "Adjudication episode ID",
  );
  pushRequiredIdentifierError(
    issues,
    adjudication.episodeAncestorId,
    "Adjudication episode ancestor ID",
  );
  if (!Number.isFinite(Date.parse(adjudication.declaredAt))) {
    issues.push("Adjudication timestamp must be valid.");
  }
  issues.push(
    ...annotationValidationIssues(adjudication.annotation, "Adjudication"),
  );
  issues.push(
    ...claimSourceBindingIssues(adjudication.annotation, "Adjudication"),
  );

  const adjudicationAnnotationId = canonicalIdentifier(
    adjudication.annotation.annotationId,
  );
  if (seenAnnotationIds.has(adjudicationAnnotationId)) {
    issues.push(
      `Duplicate annotation ID: ${adjudication.annotation.annotationId}.`,
    );
  }
  const adjudicatorCode = canonicalIdentifier(adjudication.annotation.raterCode);
  if (seenRaterCodes.has(adjudicatorCode)) {
    issues.push(
      `Adjudication requires a third distinct rater; duplicate rater code: ${adjudication.annotation.raterCode}.`,
    );
  }

  if (entries.length > 0) {
    if (adjudication.episodeId !== entries[0].episodeId) {
      issues.push("Adjudication episode ID does not match the review entries.");
    }
    if (adjudication.episodeAncestorId !== entries[0].episodeAncestorId) {
      issues.push(
        "Adjudication episode ancestor ID does not match the review entries.",
      );
    }
  }

  const reviewedIds = adjudication.reviewedAnnotationIds.map(canonicalIdentifier);
  if (reviewedIds.length !== 2 || reviewedIds[0] === reviewedIds[1]) {
    issues.push("Adjudication must reference two distinct primary annotation IDs.");
  }
  const primaryIds = entries.map((entry) =>
    canonicalIdentifier(entry.annotation.annotationId),
  );
  if (
    primaryIds.length === 2 &&
    !(
      primaryIds.every((id) => reviewedIds.includes(id)) &&
      reviewedIds.every((id) => primaryIds.includes(id))
    )
  ) {
    issues.push("Adjudication must reference both primary annotation IDs exactly.");
  }

  return issues;
}

export function validateResearchAnnotationReview(
  entries: readonly ResearchAnnotationReviewEntry[],
  options: ResearchAnnotationReviewOptions = {},
): string[] {
  const numericTolerance =
    options.numericTolerance ?? EXACT_ANNOTATION_NUMERIC_TOLERANCE;
  const issues: string[] = [];
  if (!Number.isFinite(numericTolerance) || numericTolerance < 0) {
    issues.push("Numeric tolerance must be a finite, non-negative number.");
  }
  issues.push(...reviewValidationIssues(entries, options.adjudication));
  return Array.from(new Set(issues));
}

/**
 * Derives review state without mutating or merging any annotation. Two matching
 * independent annotations are complete by agreement; a disagreement remains
 * visible even after a third rater appends an adjudication annotation.
 */
export function deriveResearchAnnotationReview(
  entries: readonly ResearchAnnotationReviewEntry[],
  options: ResearchAnnotationReviewOptions = {},
): ResearchAnnotationReviewResult {
  const numericTolerance =
    options.numericTolerance ?? EXACT_ANNOTATION_NUMERIC_TOLERANCE;
  const issues = validateResearchAnnotationReview(entries, options);
  if (issues.length > 0) throw new ResearchAnnotationReviewValidationError(issues);

  const first = entries[0];
  const disagreements =
    entries.length === 2
      ? compareResearchAnnotations(
          entries[0].annotation,
          entries[1].annotation,
          numericTolerance,
        )
      : [];

  let state: ResearchAnnotationReviewState = "draft";
  let resolution: ResearchAnnotationReviewResult["resolution"] = "none";
  if (entries.length === 1) {
    state = "needs-second-review";
  } else if (entries.length === 2 && disagreements.length === 0) {
    state = "adjudicated";
    resolution = "independent-agreement";
  } else if (entries.length === 2) {
    state = "needs-adjudication";
  }

  if (options.adjudication) {
    state = "adjudicated";
    resolution = "third-rater-adjudication";
  }

  return {
    state,
    resolution,
    episodeId: first?.episodeId,
    episodeAncestorId: first?.episodeAncestorId,
    primaryAnnotationIds: entries.map((entry) => entry.annotation.annotationId),
    raterCodes: [
      ...entries.map((entry) => entry.annotation.raterCode),
      ...(options.adjudication
        ? [options.adjudication.annotation.raterCode]
        : []),
    ],
    adjudicationAnnotationId: options.adjudication?.annotation.annotationId,
    disagreements,
    numericTolerance,
    clinicalTruthStatus: "not-established-by-review",
  };
}
