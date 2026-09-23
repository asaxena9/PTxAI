import type { Metadata } from "next";
import MediaClockBrowserProbe from "../../src/components/MediaClockBrowserProbe";

export const metadata: Metadata = {
  title: "Synthetic media-clock probe",
  description:
    "A local, privacy-free browser mechanics probe for measuring source-to-encoded video timing without camera or participant data.",
  robots: { index: false, follow: false },
};

export default function ResearchClockCalibrationPage() {
  return <MediaClockBrowserProbe />;
}
