// Pieces that several screens share: the back button and the step line.
import type { App } from "../app.ts";
import { button, el } from "../dom.ts";

/** "Step n of 3": frontal, profile, then review and send. */
export const STEP = { frontal: 1, profile: 2, review: 3 } as const;
export const STEPS = 3;

export function stepLine(app: App, step: number): HTMLElement {
  return el("p", { class: "page__step", text: app.t("ui.step", { n: step, total: STEPS }) });
}

/** The back button at the top of a screen; going back keeps every image and value. */
export function backButton(app: App, onClick: () => void): HTMLButtonElement {
  return button(app.t("ui.nav.back"), onClick, { class: "btn btn--link page__back", attrs: { "data-action": "back" } });
}
