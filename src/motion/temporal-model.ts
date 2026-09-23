import type { BodySide, ExerciseId } from "../domain/types";
import type { PoseFrame, PoseLandmark } from "./types";

export type TemporalPhase =
  | "ready"
  | "moving"
  | "holding"
  | "returning"
  | "paused";

export type TemporalConfidenceState =
  | "confident"
  | "caution"
  | "unavailable";

interface QuantizedTensorArtifact {
  shape: number[];
  scale: number;
  encoding: "base64-int8";
  data: string;
}

export interface TemporalModelArtifact {
  schemaVersion: 1;
  modelId: string;
  modelKind: "causal-temporal-convolution";
  input: {
    windowSize: number;
    sampleRateHz: number;
    featureNames: string[];
  };
  labels: {
    exercises: ExerciseId[];
    phases: TemporalPhase[];
    joint: string[];
  };
  normalization: {
    mean: number[];
    scale: number[];
  };
  architecture: {
    hiddenChannels: number;
    kernelSize: number;
    pooling: "temporal-mean-plus-last";
    activation: "relu";
  };
  tensors: {
    "conv1.weight": QuantizedTensorArtifact;
    "conv1.bias": number[];
    "conv2.weight": QuantizedTensorArtifact;
    "conv2.bias": number[];
    "head.weight": QuantizedTensorArtifact;
    "head.bias": number[];
  };
  calibration: {
    temperature: number;
    selectionSource: string;
  };
  abstention: {
    minimumConfidence: number;
    minimumMargin: number;
    minimumCurrentVisibility: number;
    minimumWarmupFrames: number;
    selectionSource: string;
  };
  training: {
    seed: number;
    dataOrigin: "synthetic" | "recorded";
    containsRealPeople: boolean;
    generatorVersion?: string;
  };
  limitations: string[];
}

export interface TemporalPrediction {
  exercise: ExerciseId;
  exerciseConfidence: number;
  phase: TemporalPhase;
  confidence: number;
  margin: number;
  state: TemporalConfidenceState;
  shouldApplyPhase: boolean;
  expectedExercise?: ExerciseId;
  modelId: string;
  framesObserved: number;
  reasons: string[];
  probabilities: {
    exercise: Record<ExerciseId, number>;
    phase: Record<TemporalPhase, number>;
  };
}

export interface TemporalSamplingDecision {
  shouldSample: boolean;
  nextSampleAtMs: number;
}

interface DecodedTensor {
  shape: number[];
  values: Float32Array;
}

const DERIVATIVE_FEATURE_INDICES = [0, 1, 4, 5] as const;
const FEATURE_NAMES = [
  "knee_flexion_image",
  "knee_flexion_world",
  "image_hip_to_knee_x",
  "image_hip_to_knee_y",
  "image_knee_to_ankle_x",
  "image_knee_to_ankle_y",
  "world_hip_to_knee_x",
  "world_hip_to_knee_y",
  "world_hip_to_knee_z",
  "world_knee_to_ankle_x",
  "world_knee_to_ankle_y",
  "world_knee_to_ankle_z",
  "image_shoulder_to_hip_x",
  "image_shoulder_to_hip_y",
  "minimum_visibility",
  "mean_visibility",
  "angle_agreement",
  "framing_margin",
  "delta_knee_flexion_image",
  "delta_knee_flexion_world",
  "delta_image_knee_to_ankle_x",
  "delta_image_knee_to_ankle_y",
] as const;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function point(
  frame: PoseFrame,
  space: "image" | "world",
  side: BodySide,
  joint: "shoulder" | "hip" | "knee" | "ankle",
): PoseLandmark {
  return (
    frame[space][`${side}-${joint}`] ?? {
      x: 0,
      y: 0,
      z: 0,
      visibility: 0,
      presence: 0,
    }
  );
}

function distance(a: number[], b: number[]): number {
  return Math.hypot(...a.map((value, index) => value - b[index]));
}

function angle(a: number[], vertex: number[], c: number[]): number {
  const first = a.map((value, index) => value - vertex[index]);
  const second = c.map((value, index) => value - vertex[index]);
  const denominator = Math.hypot(...first) * Math.hypot(...second);
  if (denominator < 1e-8) return 0;
  const dot = first.reduce(
    (sum, value, index) => sum + value * second[index],
    0,
  );
  return (
    Math.acos(clamp(dot / denominator, -1, 1)) *
    (180 / Math.PI)
  );
}

export function extractTemporalFeatures(
  frame: PoseFrame,
  side: BodySide,
): Float32Array {
  const joints = ["shoulder", "hip", "knee", "ankle"] as const;
  const image = Object.fromEntries(
    joints.map((joint) => [joint, point(frame, "image", side, joint)]),
  ) as Record<(typeof joints)[number], PoseLandmark>;
  const world = Object.fromEntries(
    joints.map((joint) => [joint, point(frame, "world", side, joint)]),
  ) as Record<(typeof joints)[number], PoseLandmark>;
  const direction = side === "left" ? -1 : 1;
  const imageCoordinates = Object.fromEntries(
    joints.map((joint) => [
      joint,
      [
        image[joint].x * frame.frameWidth,
        image[joint].y * frame.frameHeight,
        image[joint].z * frame.frameWidth,
      ],
    ]),
  ) as Record<(typeof joints)[number], number[]>;
  const worldCoordinates = Object.fromEntries(
    joints.map((joint) => [
      joint,
      [world[joint].x, world[joint].y, world[joint].z],
    ]),
  ) as Record<(typeof joints)[number], number[]>;

  const imageScale = Math.max(
    1e-6,
    distance(imageCoordinates.hip, imageCoordinates.knee) +
      distance(imageCoordinates.knee, imageCoordinates.ankle),
  );
  const worldScale = Math.max(
    1e-6,
    distance(worldCoordinates.hip, worldCoordinates.knee) +
      distance(worldCoordinates.knee, worldCoordinates.ankle),
  );
  const imageVector = (
    start: (typeof joints)[number],
    end: (typeof joints)[number],
  ): [number, number] => [
    (direction *
      (imageCoordinates[end][0] - imageCoordinates[start][0])) /
      imageScale,
    (imageCoordinates[end][1] - imageCoordinates[start][1]) / imageScale,
  ];
  const worldVector = (
    start: (typeof joints)[number],
    end: (typeof joints)[number],
  ): [number, number, number] => [
    (direction *
      (worldCoordinates[end][0] - worldCoordinates[start][0])) /
      worldScale,
    (worldCoordinates[end][1] - worldCoordinates[start][1]) / worldScale,
    (direction *
      (worldCoordinates[end][2] - worldCoordinates[start][2])) /
      worldScale,
  ];

  const imageFlexion =
    (180 -
      angle(
        imageCoordinates.hip,
        imageCoordinates.knee,
        imageCoordinates.ankle,
      )) /
    180;
  const worldFlexion =
    (180 -
      angle(
        worldCoordinates.hip,
        worldCoordinates.knee,
        worldCoordinates.ankle,
      )) /
    180;
  const imageHipKnee = imageVector("hip", "knee");
  const imageKneeAnkle = imageVector("knee", "ankle");
  const worldHipKnee = worldVector("hip", "knee");
  const worldKneeAnkle = worldVector("knee", "ankle");
  const shoulderHip = imageVector("shoulder", "hip");
  const visibilities = joints.map((joint) =>
    Math.min(image[joint].visibility, image[joint].presence),
  );
  const margins = joints.flatMap((joint) => [
    image[joint].x,
    1 - image[joint].x,
    image[joint].y,
    1 - image[joint].y,
  ]);

  return Float32Array.from([
    imageFlexion,
    worldFlexion,
    ...imageHipKnee,
    ...imageKneeAnkle,
    ...worldHipKnee,
    ...worldKneeAnkle,
    ...shoulderHip,
    Math.min(...visibilities),
    visibilities.reduce((sum, value) => sum + value, 0) /
      visibilities.length,
    Math.abs(imageFlexion - worldFlexion),
    clamp(Math.min(...margins) / 0.12, 0, 1),
  ]);
}

function decodeTensor(tensor: QuantizedTensorArtifact): DecodedTensor {
  if (tensor.encoding !== "base64-int8") {
    throw new Error(`Unsupported temporal tensor encoding: ${tensor.encoding}`);
  }
  const binary = globalThis.atob(tensor.data);
  const expected = tensor.shape.reduce((product, size) => product * size, 1);
  if (binary.length !== expected) {
    throw new Error(
      `Temporal tensor size mismatch: expected ${expected}, got ${binary.length}`,
    );
  }
  const values = new Float32Array(expected);
  for (let index = 0; index < binary.length; index += 1) {
    const unsigned = binary.charCodeAt(index);
    const signed = unsigned > 127 ? unsigned - 256 : unsigned;
    values[index] = signed * tensor.scale;
  }
  return { shape: tensor.shape, values };
}

function assertArtifact(artifact: TemporalModelArtifact): void {
  if (artifact.schemaVersion !== 1) {
    throw new Error(
      `Unsupported temporal model schema: ${artifact.schemaVersion}`,
    );
  }
  if (
    artifact.input.featureNames.length !== FEATURE_NAMES.length ||
    artifact.input.featureNames.some(
      (feature, index) => feature !== FEATURE_NAMES[index],
    )
  ) {
    throw new Error(
      "Temporal artifact features do not match this runtime version.",
    );
  }
  if (
    artifact.normalization.mean.length !== FEATURE_NAMES.length ||
    artifact.normalization.scale.length !== FEATURE_NAMES.length
  ) {
    throw new Error("Temporal normalization shape is invalid.");
  }
  if (
    artifact.labels.joint.length !==
    artifact.labels.exercises.length * artifact.labels.phases.length
  ) {
    throw new Error("Temporal joint-label shape is invalid.");
  }
}

function causalConvolution(
  input: number[][],
  tensor: DecodedTensor,
  bias: number[],
): number[][] {
  const [outputChannels, inputChannels, kernelSize] = tensor.shape;
  const result = Array.from({ length: input.length }, () =>
    Array<number>(outputChannels).fill(0),
  );
  for (let time = 0; time < input.length; time += 1) {
    for (let output = 0; output < outputChannels; output += 1) {
      let sum = bias[output];
      for (let kernel = 0; kernel < kernelSize; kernel += 1) {
        const sourceTime = time + kernel - kernelSize + 1;
        if (sourceTime < 0) continue;
        for (let feature = 0; feature < inputChannels; feature += 1) {
          const weightIndex =
            (output * inputChannels + feature) * kernelSize + kernel;
          sum += input[sourceTime][feature] * tensor.values[weightIndex];
        }
      }
      result[time][output] = Math.max(0, sum);
    }
  }
  return result;
}

function linear(
  input: number[],
  tensor: DecodedTensor,
  bias: number[],
): number[] {
  const [outputs, inputs] = tensor.shape;
  return Array.from({ length: outputs }, (_, output) => {
    let sum = bias[output];
    for (let index = 0; index < inputs; index += 1) {
      sum += input[index] * tensor.values[output * inputs + index];
    }
    return sum;
  });
}

function softmax(logits: number[], temperature: number): number[] {
  const adjusted = logits.map((value) => value / temperature);
  const maximum = Math.max(...adjusted);
  const exponentials = adjusted.map((value) => Math.exp(value - maximum));
  const denominator = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / denominator);
}

function topTwo(probabilities: number[]): {
  index: number;
  confidence: number;
  margin: number;
} {
  const order = probabilities
    .map((value, index) => ({ value, index }))
    .sort((left, right) => right.value - left.value);
  return {
    index: order[0].index,
    confidence: order[0].value,
    margin: order[0].value - (order[1]?.value ?? 0),
  };
}

export function advanceTemporalSampleSchedule(
  timestampMs: number,
  nextSampleAtMs: number | null,
  sampleRateHz: number,
): TemporalSamplingDecision {
  if (!Number.isFinite(timestampMs) || timestampMs < 0) {
    throw new Error("Temporal sample timestamp must be finite and non-negative.");
  }
  if (!Number.isFinite(sampleRateHz) || sampleRateHz <= 0) {
    throw new Error("Temporal sample rate must be positive.");
  }
  const intervalMs = 1_000 / sampleRateHz;
  if (nextSampleAtMs !== null && timestampMs < nextSampleAtMs) {
    return { shouldSample: false, nextSampleAtMs };
  }
  let scheduled = (nextSampleAtMs ?? timestampMs) + intervalMs;
  while (scheduled <= timestampMs) scheduled += intervalMs;
  return { shouldSample: true, nextSampleAtMs: scheduled };
}

export class TemporalMovementModel {
  readonly artifact: TemporalModelArtifact;
  private readonly conv1: DecodedTensor;
  private readonly conv2: DecodedTensor;
  private readonly head: DecodedTensor;
  private readonly frames: Float32Array[] = [];

  constructor(artifact: TemporalModelArtifact) {
    assertArtifact(artifact);
    this.artifact = artifact;
    this.conv1 = decodeTensor(artifact.tensors["conv1.weight"]);
    this.conv2 = decodeTensor(artifact.tensors["conv2.weight"]);
    this.head = decodeTensor(artifact.tensors["head.weight"]);
  }

  reset(): void {
    this.frames.length = 0;
  }

  observe(
    frame: PoseFrame,
    side: BodySide,
    expectedExercise?: ExerciseId,
  ): TemporalPrediction {
    const features = extractTemporalFeatures(frame, side);
    this.frames.push(features);
    if (this.frames.length > this.artifact.input.windowSize) {
      this.frames.shift();
    }
    return this.predict(expectedExercise);
  }

  predict(expectedExercise?: ExerciseId): TemporalPrediction {
    if (this.frames.length === 0) {
      throw new Error("Observe at least one pose frame before prediction.");
    }
    const baseWindow = this.paddedWindow();
    const input = baseWindow.map((features, time) => {
      const derivatives = DERIVATIVE_FEATURE_INDICES.map((index) =>
        time === 0 ? 0 : features[index] - baseWindow[time - 1][index],
      );
      return [...features, ...derivatives].map(
        (value, index) =>
          (value - this.artifact.normalization.mean[index]) /
          this.artifact.normalization.scale[index],
      );
    });
    const first = causalConvolution(
      input,
      this.conv1,
      this.artifact.tensors["conv1.bias"],
    );
    const second = causalConvolution(
      first,
      this.conv2,
      this.artifact.tensors["conv2.bias"],
    );
    const hidden = second[0].length;
    const pooled = Array<number>(hidden * 2).fill(0);
    for (const timestep of second) {
      for (let channel = 0; channel < hidden; channel += 1) {
        pooled[channel] += timestep[channel] / second.length;
      }
    }
    for (let channel = 0; channel < hidden; channel += 1) {
      pooled[hidden + channel] = second.at(-1)?.[channel] ?? 0;
    }
    const probabilities = softmax(
      linear(pooled, this.head, this.artifact.tensors["head.bias"]),
      this.artifact.calibration.temperature,
    );
    return this.toPrediction(probabilities, expectedExercise);
  }

  private paddedWindow(): Float32Array[] {
    const first = this.frames[0];
    const missing = this.artifact.input.windowSize - this.frames.length;
    return [
      ...Array.from({ length: missing }, () => first),
      ...this.frames,
    ];
  }

  private toPrediction(
    jointProbabilities: number[],
    expectedExercise?: ExerciseId,
  ): TemporalPrediction {
    const exerciseProbabilities = this.artifact.labels.exercises.map(
      (_, exerciseIndex) =>
        this.artifact.labels.phases.reduce(
          (sum, _phase, phaseIndex) =>
            sum +
            jointProbabilities[
              exerciseIndex * this.artifact.labels.phases.length + phaseIndex
            ],
          0,
        ),
    );
    const exerciseTop = topTwo(exerciseProbabilities);
    const predictedExercise =
      this.artifact.labels.exercises[exerciseTop.index];
    const expectedIndex =
      expectedExercise === undefined
        ? undefined
        : this.artifact.labels.exercises.indexOf(expectedExercise);
    const phaseProbabilities = this.artifact.labels.phases.map(
      (_phase, phaseIndex) => {
        if (expectedIndex !== undefined && expectedIndex >= 0) {
          return jointProbabilities[
            expectedIndex * this.artifact.labels.phases.length + phaseIndex
          ];
        }
        return this.artifact.labels.exercises.reduce(
          (sum, _exercise, exerciseIndex) =>
            sum +
            jointProbabilities[
              exerciseIndex * this.artifact.labels.phases.length + phaseIndex
            ],
          0,
        );
      },
    );
    const phaseDenominator = phaseProbabilities.reduce(
      (sum, value) => sum + value,
      0,
    );
    const normalizedPhase = phaseProbabilities.map(
      (value) => value / Math.max(phaseDenominator, 1e-8),
    );
    const phaseTop = topTwo(normalizedPhase);
    const currentVisibility = this.frames.at(-1)?.[14] ?? 0;
    const reasons: string[] = [];
    let state: TemporalConfidenceState = "confident";

    if (
      currentVisibility <
      this.artifact.abstention.minimumCurrentVisibility
    ) {
      state = "unavailable";
      reasons.push(
        "Required landmarks are not currently visible enough for temporal assessment.",
      );
    } else if (
      this.frames.length < this.artifact.abstention.minimumWarmupFrames
    ) {
      state = "unavailable";
      reasons.push("Collecting a short movement history.");
    } else if (
      phaseTop.confidence < this.artifact.abstention.minimumConfidence ||
      phaseTop.margin < this.artifact.abstention.minimumMargin
    ) {
      state = "caution";
      reasons.push("The temporal phase evidence is ambiguous.");
    }
    if (
      expectedExercise !== undefined &&
      predictedExercise !== expectedExercise &&
      exerciseTop.confidence >= this.artifact.abstention.minimumConfidence
    ) {
      state = state === "unavailable" ? state : "caution";
      reasons.push(
        `Observed movement does not clearly match the expected ${expectedExercise}.`,
      );
    }
    if (reasons.length === 0) {
      reasons.push("Temporal movement evidence is consistent.");
    }

    return {
      exercise: predictedExercise,
      exerciseConfidence: exerciseTop.confidence,
      phase: this.artifact.labels.phases[phaseTop.index],
      confidence: phaseTop.confidence,
      margin: phaseTop.margin,
      state,
      shouldApplyPhase: state === "confident",
      expectedExercise,
      modelId: this.artifact.modelId,
      framesObserved: this.frames.length,
      reasons,
      probabilities: {
        exercise: Object.fromEntries(
          this.artifact.labels.exercises.map((exercise, index) => [
            exercise,
            exerciseProbabilities[index],
          ]),
        ) as Record<ExerciseId, number>,
        phase: Object.fromEntries(
          this.artifact.labels.phases.map((phase, index) => [
            phase,
            normalizedPhase[index],
          ]),
        ) as Record<TemporalPhase, number>,
      },
    };
  }
}

export async function loadTemporalMovementModel(
  url = "/models/temporal/relay-temporal-v1.json",
): Promise<TemporalMovementModel> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Unable to load temporal model (${response.status} ${response.statusText}).`,
    );
  }
  return new TemporalMovementModel(
    (await response.json()) as TemporalModelArtifact,
  );
}
