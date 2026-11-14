import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#10140f",
        paper: "#f3f5ed",
        moss: "#8cff5b",
        limewash: "#d9ffc7",
        skywash: "#dff2ef",
      },
      boxShadow: {
        soft: "0 24px 80px rgba(20, 35, 22, 0.12)",
      },
    },
  },
  plugins: [],
};

export default config;

