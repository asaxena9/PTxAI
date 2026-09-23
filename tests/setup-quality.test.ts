import { describe, expect, it } from "vitest";
import { createDemoSession } from "../src/domain/templates";
import { measurePose } from "../src/motion/geometry";
import { createReplayFrames } from "../src/motion/replay-fixtures";
import { evaluateSetupFeatureReadiness } from "../src/motion/setup-quality";

describe("camera setup feature readiness", () => {
  it("requires every confirmed full-body relationship, not only the primary leg chain", () => {
    const exercise = createDemoSession().exercises[1];
    const frame = createReplayFrames({
      exerciseId: exercise.exerciseId,
      side: exercise.side,
      reps: 1,
      holdSeconds: exercise.dose.holdSeconds,
    })[0];
    delete frame.image["right-shoulder"];
    delete frame.world["right-shoulder"];

    const measurements = measurePose(
      frame,
      exercise.exerciseId,
      exercise.side,
    );
    expect(measurements.confidence.status).toBe("good");

    const readiness = evaluateSetupFeatureReadiness(exercise, measurements);
    expect(readiness.status).toBe("lost");
    expect(readiness.requiredMetrics).toContain("trunk-compensation");
    expect(readiness.reasons.join(" ")).toMatch(/shoulder|hip/i);
  });
});
