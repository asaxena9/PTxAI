import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL(
    "https://relay-body-movement.adityasaxena.chatgpt.site",
  ),
  title: {
    default: "Relay · Body signals, grounded evidence",
    template: "%s · Relay",
  },
  description:
    "A private research pilot for retaining continuous body signals, derived kinematics, and independently reviewable evidence without making clinical claims.",
  icons: {
    icon: "/icon.png",
    apple: "/apple-touch-icon.png",
  },
  manifest: "/manifest.webmanifest",
  applicationName: "Relay",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "Relay",
  },
  formatDetection: { telephone: false },
  openGraph: {
    type: "website",
    title: "Relay · Body signals, grounded evidence",
    description:
      "Continuous body signals, deterministic kinematics, and independently reviewable evidence for a body-native research program.",
    images: [
      {
        url: "/og-body-native.png",
        width: 1200,
        height: 630,
        alt: "Relay: a person moving at home with a phone camera and full-body pose landmarks.",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Relay · Body signals, grounded evidence",
    description:
      "A private, claim-limited pilot for continuous body signals and grounded evidence.",
    images: ["/og-body-native.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="theme-color" content="#173c34" />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
