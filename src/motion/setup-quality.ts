import type { ExercisePrescription, MetricId } from "../domain/types";
import type {
  MotionMeasurements,
  TrackingStatus,
} from "./types";

export interface SetupFeatureReadiness {
  status: TrackingStatus;
  reasons: string[];
  requiredMetrics: MetricId[];
}

function primaryMetricFor(
  exerciseId: ExercisePrescription["exerciseId"],
): MetricId {
  if (exerciseId === "straight-leg-raise") return "leg-lift";
  if (exerciseId === "seated-knee-extension") return "extension-progress";
  return "knee-flexion";
}

/**
 * Camera setup must establish every body feature used by a confirmed
 * non-temporal rule. During movement, those features may abstain
 * independently without stopping unrelated assessments.
 */
export function evaluateSetupFeatureReadiness(
  prescription: ExercisePrescription,
  measurements: MotionMeasurements,
): SetupFeatureReadiness {
  const requiredMetrics = Array.from(
    new Set([
      primaryMetricFor(prescription.exerciseId),
      ...prescription.rules
        .filter((rule) => rule.scope.kind !== "temporal")
        .map((rule) => rule.metric),
    ]),
  );
  const requiredFeatures = requiredMetrics.map((metric) => ({
    metric,
    feature: measurements.features.find(
      (candidate) => candidate.metric === metric,
    ),
  }));
  const lost = requiredFeatures.filter(
    ({ feature }) => !feature || feature.quality === "lost" || feature.value === undefined,
  );
  const limited = requiredFeatures.filter(
    ({ feature }) => feature?.quality === "limited",
  );

  if (lost.length > 0) {
    return {
      status: "lost",
      reasons: lost.map(
        ({ metric, feature }) =>
          `${feature?.description ?? metric} is not reliably visible.`,
      ),
      requiredMetrics,
    };
  }
  if (limited.length > 0) {
    return {
      status: "limited",
      reasons: limited.map(
        ({ metric, feature }) =>
          `${feature?.description ?? metric} needs a clearer view.`,
      ),
      requiredMetrics,
    };
  }
  return {
    status: "good",
    reasons: ["All confirmed body-feature checks are visible."],
    requiredMetrics,
  };
}
