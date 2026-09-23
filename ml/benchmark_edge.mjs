#!/usr/bin/env node
/**
 * Benchmarks artifact decode plus pose-feature extraction and temporal inference.
 *
 * This is a Node/V8 CPU microbenchmark, not a phone-browser pose benchmark.
 */
import { readFileSync, statSync, writeFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { platform, release, arch } from "node:os";
import { createReplayFrames } from "../src/motion/replay-fixtures.ts";
import { TemporalMovementModel } from "../src/motion/temporal-model.ts";

const iterationsIndex = process.argv.indexOf("--iterations");
const iterations =
  iterationsIndex >= 0 ? Number(process.argv[iterationsIndex + 1]) : 5_000;
const outputIndex = process.argv.indexOf("--output");
const output =
  outputIndex >= 0
    ? process.argv[outputIndex + 1]
    : "ml/results/edge-benchmark.json";
const artifactPath = "public/models/temporal/relay-temporal-v1.json";

const readStarted = performance.now();
const artifactText = readFileSync(artifactPath, "utf8");
const artifact = JSON.parse(artifactText);
const readAndParseMs = performance.now() - readStarted;

const constructStarted = performance.now();
const model = new TemporalMovementModel(artifact);
const constructMs = performance.now() - constructStarted;

const fixtures = [
  ...createReplayFrames({
    exerciseId: "heel-slide",
    side: "right",
    scenario: "correct",
    holdSeconds: 1,
    intervalMs: 100,
  }),
  ...createReplayFrames({
    exerciseId: "straight-leg-raise",
    side: "right",
    scenario: "correct",
    holdSeconds: 1,
    intervalMs: 100,
  }),
  ...createReplayFrames({
    exerciseId: "seated-knee-extension",
    side: "right",
    scenario: "correct",
    holdSeconds: 1,
    intervalMs: 100,
  }),
];
for (let index = 0; index < 250; index += 1) {
  model.observe(fixtures[index % fixtures.length], "right");
}

const heapBefore = process.memoryUsage().heapUsed;
const samples = new Float64Array(iterations);
const benchmarkStarted = performance.now();
for (let index = 0; index < iterations; index += 1) {
  const started = performance.now();
  model.observe(fixtures[index % fixtures.length], "right");
  samples[index] = performance.now() - started;
}
const durationMs = performance.now() - benchmarkStarted;
const heapAfter = process.memoryUsage().heapUsed;
const sorted = Array.from(samples).sort((left, right) => left - right);
const percentile = (value) =>
  sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * value))];
const result = {
  benchmarkVersion: 1,
  scope: "TypeScript feature extraction plus int8-weight temporal inference",
  excludes: [
    "camera capture",
    "pose-landmarker inference",
    "React rendering",
    "speech synthesis",
  ],
  warning:
    "Node/V8 desktop CPU microbenchmark; verify separately on target phone browsers.",
  iterations,
  artifact: {
    path: artifactPath,
    bytes: statSync(artifactPath).size,
  },
  startup: {
    fileReadAndJsonParseMs: readAndParseMs,
    tensorDecodeAndValidationMs: constructMs,
    combinedMs: readAndParseMs + constructMs,
  },
  inference: {
    meanMs: durationMs / iterations,
    medianMs: percentile(0.5),
    p95Ms: percentile(0.95),
    p99Ms: percentile(0.99),
    throughputPerSecond: (iterations / durationMs) * 1_000,
  },
  memory: {
    heapDeltaBytesDuringLoop: heapAfter - heapBefore,
    caveat: "GC was not forced; this is a coarse allocation signal, not peak RSS.",
  },
  environment: {
    node: process.version,
    v8: process.versions.v8,
    os: `${platform()} ${release()}`,
    architecture: arch(),
  },
};
writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify(result, null, 2));
