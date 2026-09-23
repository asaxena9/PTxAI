import type { Metadata } from "next";
import ResearchCapture from "../../src/components/ResearchCapture";

export const metadata: Metadata = {
  title: "Research capture",
  description:
    "Locally capture paired body-motion evidence with explicit lineage, consent, structured observations, and content hashes.",
  robots: { index: false, follow: false },
};

export default function ResearchCapturePage() {
  return <ResearchCapture />;
}
