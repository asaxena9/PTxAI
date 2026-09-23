import type { Metadata } from "next";
import ResearchReview from "../../src/components/ResearchReview";

export const metadata: Metadata = {
  title: "Independent research review",
  description:
    "Locally verify and independently review a quarantined continuous body-motion episode before revealing prior judgments.",
  robots: { index: false, follow: false },
};

export default function ResearchReviewPage() {
  return <ResearchReview />;
}
