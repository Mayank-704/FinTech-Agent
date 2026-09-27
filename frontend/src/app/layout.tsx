import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aegis – Financial Risk Orchestrator",
  description:
    "AI-powered personal finance agent with Ghost Ledger simulation, GRPO optimization, and Human-in-the-Loop execution gates.",
  keywords: ["fintech", "AI finance", "risk management", "budget planning"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>{children}</body>
    </html>
  );
}
