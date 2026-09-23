import { describe, expect, it } from "vitest";
import { confirmedSessionSchema } from "../src/domain/schema";
import { createDemoSession } from "../src/domain/templates";

describe("confirmed movement schema", () => {
  it("rejects an inverted body-feature range", () => {
    const session = createDemoSession();
    session.exercises[0].rules[0] = {
      ...session.exercises[0].rules[0],
      kind: "range",
      value: undefined,
      min: 100,
      max: 60,
    };
    const result = confirmedSessionSchema.safeParse(session);
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues[0]?.message).toContain(
        "minimum cannot be greater",
      );
    }
  });

  it("accepts a provider-defined body-feature metric", () => {
    const session = createDemoSession();
    session.exercises[0].rules.push({
      id: "ankle-rule",
      scope: {
        kind: "joint",
        joints: ["ankle", "foot-index"],
        relationship: "range",
        description: "Provider-defined ankle relationship",
      },
      metric: "ankle-dorsiflexion",
      kind: "target",
      unit: "degrees",
      value: 15,
      hard: false,
      label: "Aim for 15° ankle dorsiflexion",
    });
    expect(confirmedSessionSchema.safeParse(session).success).toBe(true);
  });
});
