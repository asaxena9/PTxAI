import type { JointName } from "../motion/types";
import { canonicalJson, sha256Text } from "./capture-bundle";

export type ObservableTargetSource =
  | "captured-reference"
  | "supplied-protocol"
  | "analytic-oracle";

export interface ObservableMovementProtocol {
  schemaVersion: "relay.observable-movement-protocol/1";
  protocolId: string;
  protocolVersion: string;
  title: string;
  domain: "daily-movement" | "sport-fitness" | "rehabilitation-research" | "other";
  movementFamily: string;
  instruction: string;
  cameraView: "front" | "left" | "right" | "oblique" | "other";
  target: {
    source: ObservableTargetSource;
    metricId: string;
    equation: string;
    unit: "degrees" | "percent" | "seconds" | "count";
    coordinateSource: "image-2d" | "world-3d";
    aggregation: "maximum-over-interval" | "median-over-interval";
    directionConvention: "target-minus-observed";
    referenceBinding:
      | "reference-interval-from-shared-continuous-session"
      | "supplied-fixed-target"
      | "analytic-oracle-target";
    evidenceJoints: JointName[];
    comparison: "match-reference" | "within-range" | "increase" | "decrease";
    range?: { minimum: number; maximum: number };
  };
  observability: {
    requiredSignals: string[];
    permittedClaims: string[];
    prohibitedClaims: string[];
    abstainWhen: string[];
  };
  provenance: {
    evidenceClass: "EXPLORATORY" | "SOURCED" | "MEASURED_CALIBRATED";
    author: string;
    createdAt: string;
    sourceReference?: string;
  };
}

export interface FrozenObservableProtocol {
  protocol: ObservableMovementProtocol;
  canonicalJson: string;
  sha256: string;
}

function finite(value: number): boolean {
  return Number.isFinite(value);
}

export function validateObservableProtocol(
  protocol: ObservableMovementProtocol,
): string[] {
  const errors: string[] = [];
  const requiredText: Array<[string, string]> = [
    ["protocolId", protocol.protocolId],
    ["protocolVersion", protocol.protocolVersion],
    ["title", protocol.title],
    ["movementFamily", protocol.movementFamily],
    ["instruction", protocol.instruction],
    ["target.metricId", protocol.target.metricId],
    ["target.equation", protocol.target.equation],
    ["provenance.author", protocol.provenance.author],
    ["provenance.createdAt", protocol.provenance.createdAt],
  ];
  for (const [name, value] of requiredText) {
    if (!value.trim()) errors.push(`${name} is required.`);
  }
  if (protocol.target.evidenceJoints.length < 3) {
    errors.push("At least three evidence joints are required.");
  }
  if (
    protocol.target.source === "captured-reference" &&
    protocol.target.referenceBinding !==
      "reference-interval-from-shared-continuous-session"
  ) {
    errors.push(
      "Captured-reference targets must bind to the reference interval from the shared continuous session.",
    );
  }
  if (new Set(protocol.target.evidenceJoints).size !== protocol.target.evidenceJoints.length) {
    errors.push("Evidence joints must be unique.");
  }
  if (protocol.target.comparison === "within-range") {
    const range = protocol.target.range;
    if (
      !range ||
      !finite(range.minimum) ||
      !finite(range.maximum) ||
      range.minimum > range.maximum
    ) {
      errors.push("A finite ordered range is required for within-range targets.");
    }
  } else if (protocol.target.range) {
    errors.push("A target range is only allowed for within-range comparisons.");
  }
  if (
    protocol.target.source === "captured-reference" &&
    protocol.target.comparison !== "match-reference"
  ) {
    errors.push("Captured-reference targets must use match-reference comparison.");
  }
  if (protocol.observability.requiredSignals.length === 0) {
    errors.push("At least one required signal must be declared.");
  }
  if (protocol.observability.permittedClaims.length === 0) {
    errors.push("At least one permitted claim must be declared.");
  }
  if (protocol.observability.prohibitedClaims.length === 0) {
    errors.push("At least one prohibited claim must be declared.");
  }
  if (protocol.observability.abstainWhen.length === 0) {
    errors.push("At least one abstention condition must be declared.");
  }
  return errors;
}

export async function freezeObservableProtocol(
  protocol: ObservableMovementProtocol,
): Promise<FrozenObservableProtocol> {
  const errors = validateObservableProtocol(protocol);
  if (errors.length > 0) throw new Error(errors.join(" "));
  const serialized = canonicalJson(protocol);
  return {
    protocol: structuredClone(protocol),
    canonicalJson: serialized,
    sha256: await sha256Text(serialized),
  };
}

/**
 * EXPLORATORY first capture protocol. It tests observable body geometry and
 * reference-relative correction, not clinical correctness. The same-person
 * reference must itself be independently accepted before the target is usable.
 */
export const LATERAL_ARM_RAISE_REFERENCE_V1: ObservableMovementProtocol = {
  schemaVersion: "relay.observable-movement-protocol/1",
  protocolId: "relay.lateral-arm-raise.reference-relative",
  protocolVersion: "1.0.0",
  title: "Reference-relative left lateral arm raise",
  domain: "daily-movement",
  movementFamily: "left-lateral-arm-raise",
  instruction:
    "From a front camera view, perform one comfortable left-arm lateral raise. Record an accepted same-person reference, return to neutral, then record an attempt without changing camera placement.",
  cameraView: "front",
  target: {
    source: "captured-reference",
    metricId: "left-shoulder-lateral-raise-included-angle",
    equation:
      "included_angle(left_hip, left_shoulder, left_elbow)",
    unit: "degrees",
    coordinateSource: "image-2d",
    aggregation: "maximum-over-interval",
    directionConvention: "target-minus-observed",
    referenceBinding: "reference-interval-from-shared-continuous-session",
    evidenceJoints: [
      "left-hip",
      "left-shoulder",
      "left-elbow",
    ],
    comparison: "match-reference",
  },
  observability: {
    requiredSignals: [
      "timestamped image landmarks",
      "landmark visibility and presence",
      "front camera view",
      "accepted same-person reference",
    ],
    permittedClaims: [
      "observed movement identity under this protocol",
      "reference-relative left-shoulder-angle difference",
      "evidence joint and time localization when timing replay passes",
    ],
    prohibitedClaims: [
      "clinical correctness or safety",
      "pain, load, force, contact, intent, diagnosis, or treatment need",
      "anatomical ground-truth angle from monocular inferred landmarks",
    ],
    abstainWhen: [
      "required shoulder, elbow, or hip landmarks are missing or low-confidence",
      "the body is not approximately front-facing",
      "the accepted reference or camera geometry changed",
      "the requested judgment depends on pain, load, force, contact, or intent",
    ],
  },
  provenance: {
    evidenceClass: "EXPLORATORY",
    author: "PTxAI body-native research",
    createdAt: "2026-08-12T00:00:00.000Z",
  },
};
