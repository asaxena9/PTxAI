import assert from "node:assert/strict";
import { readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { describe, it } from "node:test";
import { createReplayFrames } from "../../src/motion/replay-fixtures.ts";
import {
  advanceTemporalSampleSchedule,
  TemporalMovementModel,
} from "../../src/motion/temporal-model.ts";

const artifactPath = resolve(
  process.cwd(),
  "public/models/temporal/relay-temporal-v1.json",
);
const artifact = JSON.parse(readFileSync(artifactPath, "utf8"));

function model() {
  return new TemporalMovementModel(structuredClone(artifact));
}

describe("TemporalMovementModel", () => {
  for (const exerciseId of [
    "heel-slide",
    "straight-leg-raise",
    "seated-knee-extension",
  ]) {
    it(`recognizes all phases in the shared ${exerciseId} replay`, () => {
      const temporal = model();
      const seen = new Set();
      let finalPrediction;
      for (const frame of createReplayFrames({
        exerciseId,
        side: "right",
        scenario: "correct",
        holdSeconds: 1,
        intervalMs: 100,
      })) {
        finalPrediction = temporal.observe(frame, "right", exerciseId);
        if (finalPrediction.shouldApplyPhase) seen.add(finalPrediction.phase);
      }

      assert.deepEqual(
        seen,
        new Set(["ready", "moving", "holding", "returning"]),
      );
      assert.equal(finalPrediction.exercise, exerciseId);
      assert.equal(finalPrediction.phase, "ready");
      assert.equal(finalPrediction.state, "confident");
    });
  }

  it("abstains during warmup and current tracking loss", () => {
    const temporal = model();
    const predictions = createReplayFrames({
      exerciseId: "heel-slide",
      side: "right",
      scenario: "tracking-loss",
      holdSeconds: 1,
      intervalMs: 100,
    }).map((frame) => temporal.observe(frame, "right", "heel-slide"));

    assert.equal(predictions[0].state, "unavailable");
    assert.equal(predictions[0].shouldApplyPhase, false);
    assert.ok(
      predictions.some(
        (prediction) =>
          prediction.state === "unavailable" &&
          prediction.reasons[0].includes("not currently visible"),
      ),
    );
    assert.ok(
      predictions.some((prediction) => prediction.state === "confident"),
    );
  });

  it("ships a compact, explicitly synthetic artifact", () => {
    assert.ok(statSync(artifactPath).size < 10_000);
    assert.equal(artifact.training.dataOrigin, "synthetic");
    assert.equal(artifact.training.containsRealPeople, false);
    assert.match(
      artifact.limitations.join(" "),
      /not camera recordings of people/,
    );
    assert.equal(artifact.input.featureNames.length, 22);
    assert.equal(artifact.labels.joint.length, 15);
  });

  it("rejects a feature-contract mismatch before inference", () => {
    const incompatible = structuredClone(artifact);
    incompatible.input.featureNames[0] = "wrong-feature";
    assert.throws(
      () => new TemporalMovementModel(incompatible),
      /features do not match/,
    );
  });

  it("decimates a 15 Hz provider near the trained 10 Hz cadence", () => {
    let nextSampleAtMs = null;
    const accepted = [];
    for (let timestampMs = 0; timestampMs < 1_000; timestampMs += 66) {
      const decision = advanceTemporalSampleSchedule(
        timestampMs,
        nextSampleAtMs,
        artifact.input.sampleRateHz,
      );
      nextSampleAtMs = decision.nextSampleAtMs;
      if (decision.shouldSample) accepted.push(timestampMs);
    }
    assert.equal(accepted.length, 10);
    assert.deepEqual(accepted.slice(0, 5), [0, 132, 264, 330, 462]);
  });
});
