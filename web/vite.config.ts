import { URL as NodeURL, fileURLToPath } from "node:url"; // Node's URL: the test environment has its own
import { defineConfig } from "vitest/config";
import { cspPlugin } from "./build/csp.ts";

// The api listens on 127.0.0.1 only. Not "localhost": Node may resolve it to the IPv6
// address ::1, where nothing listens (web.md section 8.3).
const API = "http://127.0.0.1:8000";
const WEB_ROOT = fileURLToPath(new NodeURL(".", import.meta.url));
const SHARED = fileURLToPath(new NodeURL("../shared", import.meta.url)); // quality.json and i18n/, shared with the api

export default defineConfig({
  plugins: [cspPlugin()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: { "/analyze": API, "/health": API },
    fs: { allow: [WEB_ROOT, SHARED] },
  },
  preview: { host: "127.0.0.1" },
  build: { outDir: "dist", emptyOutDir: true },
  test: {
    environment: "happy-dom",
    include: ["tests/**/*.test.ts"],
  },
});
