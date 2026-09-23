import type { Metadata } from "next";
import RelayApp from "../src/components/RelayApp";

export const metadata: Metadata = {
  title: "Relay · Body signals, grounded evidence",
  description:
    "A camera-guided movement experience and private body-signal research pilot with explicit evidence boundaries.",
};

export default function Home() {
  return <RelayApp />;
}
