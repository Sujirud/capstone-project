// Start the page: the frame, GET /health and the home screen (web.md section 8.2).
// ?demo=<fixture> shows a stored result while developing; the build leaves it out,
// since import.meta.env.DEV is false there and the branch is removed.
import "./styles/tokens.css";
import "./styles/components.css";
import { App } from "./app.ts";
import { DEMO_PARAM } from "./constants.ts";

const root = document.getElementById("app");
if (!root) throw new Error("index.html has no #app");
const app = new App(root, location.search);
const demo = new URLSearchParams(location.search).get(DEMO_PARAM);

if (import.meta.env.DEV && demo !== null) {
  import("./demo.ts").then(({ startDemo }) => startDemo(app, demo));
} else {
  void app.start();
}
