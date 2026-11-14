import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "WATTWISE — AI energy intelligence for buildings",
  description:
    "WATTWISE learns normal electricity patterns and identifies unexpectedly high appliance energy consumption for investigation.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

