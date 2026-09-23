import { z } from "zod";

const movementRuleSchema = z
  .object({
    id: z.string().min(1),
    scope: z.object({
      kind: z.enum(["joint", "relationship", "whole-body", "temporal"]),
      joints: z.array(
        z.enum([
          "nose",
          "shoulder",
          "elbow",
          "wrist",
          "hip",
          "knee",
          "ankle",
          "heel",
          "foot-index",
        ]),
      ),
      relationship: z.string().min(1),
      description: z.string().min(1),
    }),
    metric: z.string().min(1),
    kind: z.enum([
      "target",
      "minimum",
      "maximum",
      "range",
      "visible-condition",
    ]),
    unit: z.enum(["degrees", "seconds", "percent", "observation"]),
    value: z.number().finite().optional(),
    min: z.number().finite().optional(),
    max: z.number().finite().optional(),
    hard: z.boolean(),
    label: z.string().min(1),
    sourceText: z.string().optional(),
  })
  .superRefine((rule, context) => {
    if (rule.kind === "range" && (rule.min === undefined || rule.max === undefined)) {
      context.addIssue({
        code: "custom",
        message: "Range rules require both min and max.",
      });
    }
    if (
      rule.kind === "range" &&
      rule.min !== undefined &&
      rule.max !== undefined &&
      rule.min > rule.max
    ) {
      context.addIssue({
        code: "custom",
        message: "A range minimum cannot be greater than its maximum.",
      });
    }
    if (rule.kind !== "range" && rule.value === undefined) {
      context.addIssue({
        code: "custom",
        message: "Non-range rules require a value.",
      });
    }
  });

const exerciseSchema = z.object({
  id: z.string().min(1),
  exerciseId: z.enum([
    "heel-slide",
    "straight-leg-raise",
    "seated-knee-extension",
  ]),
  name: z.string().min(1),
  side: z.enum(["left", "right"]),
  dose: z.object({
    sets: z.number().int().min(1).max(20),
    reps: z.number().int().min(1).max(100),
    holdSeconds: z.number().min(0).max(120),
    restSeconds: z.number().min(0).max(900).optional(),
    tempo: z.object({
      cue: z.enum(["comfortable", "slow", "controlled"]),
      minRepDurationSeconds: z.number().positive().max(120).optional(),
      liftSeconds: z.number().positive().max(120).optional(),
      lowerSeconds: z.number().positive().max(120).optional(),
    }),
  }),
  rules: z.array(movementRuleSchema),
  notes: z.array(z.string()),
});

export const confirmedSessionSchema = z.object({
  schemaVersion: z.literal(1),
  id: z.string().min(1),
  title: z.string().min(1),
  createdAt: z.string(),
  updatedAt: z.string(),
  source: z.enum(["typed", "spoken", "upload", "manual", "template"]),
  sourceText: z.string(),
  exercises: z.array(exerciseSchema).min(1),
  clarifications: z.array(z.unknown()),
  confirmed: z.boolean(),
  estimatesAcknowledged: z.boolean(),
});
