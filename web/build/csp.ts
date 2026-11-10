// Content-Security-Policy of the built page. Only the build gets it:
// the dev server injects inline scripts for hot reload, which this policy would block.
//
// MediaPipe (the capture screens): tasks-vision adds a <script> of its wasm loader from the CDN
// and fetches the wasm from there, and fetches the model from the host of the model
// file, so the CDN goes in script-src and connect-src and the model host in
// connect-src; 'wasm-unsafe-eval' lets the page compile WebAssembly. The hosts come
// from the URLs in src/constants.ts, so they cannot drift apart.
import type { Plugin } from "vite";
import { FACE_LANDMARKER_MODEL_URL, MEDIAPIPE_WASM_URL } from "../src/constants.ts";

export const MEDIAPIPE_CDN = new URL(MEDIAPIPE_WASM_URL).origin;
export const MODEL_HOST = new URL(FACE_LANDMARKER_MODEL_URL).origin;

export const CONTENT_SECURITY_POLICY = [
  "default-src 'self'",
  `script-src 'self' 'wasm-unsafe-eval' ${MEDIAPIPE_CDN}`,
  "img-src 'self' blob: data:",
  "style-src 'self' https://fonts.googleapis.com",
  "font-src https://fonts.gstatic.com",
  `connect-src 'self' ${MEDIAPIPE_CDN} ${MODEL_HOST}`,
].join("; ");

const CHARSET = /<meta charset="UTF-8"\s*\/?>/i;

/** index.html with the policy right after <meta charset>, before anything it governs. */
export function withPolicy(html: string): string {
  if (!CHARSET.test(html)) throw new Error('index.html needs <meta charset="UTF-8"> first in <head>');
  const tag = `<meta http-equiv="Content-Security-Policy" content="${CONTENT_SECURITY_POLICY}" />`;
  return html.replace(CHARSET, (charset) => `${charset}\n    ${tag}`);
}

export function cspPlugin(): Plugin {
  return {
    name: "facemeasure-csp",
    apply: "build",
    transformIndexHtml: { order: "pre", handler: withPolicy },
  };
}
