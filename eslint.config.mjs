import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "public/mediapipe/**",
    "public/models/**",
    "public/pdf.worker.min.mjs",
    "experiments/pose-benchmark/.venv/**",
    "experiments/**/__pycache__/**",
    "ml/**/__pycache__/**",
    "ml/.generated/**",
    "next-env.d.ts",
  ]),
]);

export default eslintConfig;
