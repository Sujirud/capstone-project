// The screens of web.md section 2. Each one builds its own element; the App puts it in
// <main> and moves the focus to its heading.
import type { App } from "../app.ts";
import { captureScreen } from "./capture.ts";
import { homeScreen } from "./home.ts";
import { resultsScreen } from "./results.ts";
import { reviewScreen } from "./review.ts";

export type ScreenName = "home" | "frontal" | "profile" | "review" | "results";

export interface ScreenOptions {
  /** Review: send at once, as after a change of language. */
  send?: boolean;
}

export interface Screen {
  element: HTMLElement;
  /** The heading that gets the focus when the screen is shown (tabindex -1). */
  heading: HTMLElement;
  /** Called when another screen replaces this one: stop requests, timers and cameras. */
  dispose?: () => void;
}

export type ScreenFactory = (app: App, options: ScreenOptions) => Screen;

export const SCREENS: Record<ScreenName, ScreenFactory> = {
  home: homeScreen,
  frontal: (app) => captureScreen(app, "frontal"),
  profile: (app) => captureScreen(app, "profile"),
  review: reviewScreen,
  results: resultsScreen,
};
