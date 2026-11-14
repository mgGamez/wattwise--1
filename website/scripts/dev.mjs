import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const websiteRoot = path.resolve(scriptDirectory, "..");
const argumentsFromLauncher = process.argv.slice(2);

const readArgument = (name, fallback) => {
  const index = argumentsFromLauncher.indexOf(name);
  return index >= 0 && argumentsFromLauncher[index + 1]
    ? argumentsFromLauncher[index + 1]
    : fallback;
};

const host = readArgument("--host", "127.0.0.1");
const port = readArgument("--port", "3000");
const nextBinary = path.join(websiteRoot, "node_modules", "next", "dist", "bin", "next");

const child = spawn(process.execPath, [nextBinary, "dev", "-H", host, "-p", port], {
  cwd: websiteRoot,
  stdio: "inherit",
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}

child.on("exit", (code) => process.exit(code ?? 0));
