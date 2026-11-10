// Review and send (web.md sections 2 and 4.2): both thumbnails, the send button, and
// while the request runs a message in an aria-live area with a cancel button. To change
// an image the user goes back. Errors keep the images, so "try again" sends the same
// ones.
import type { App } from "../app.ts";
import type { AnalyzeOutcome } from "../api.ts";
import { button, el } from "../dom.ts";
import { uploadFlag } from "../results/cards.ts";
import { archImage } from "../results/overlay.ts";
import type { ImageName } from "../types.ts";
import type { Screen, ScreenOptions } from "./index.ts";
import { STEP, backButton, stepLine } from "./parts.ts";

type Failure = Exclude<AnalyzeOutcome["kind"], "ok">;

const FAILURE_TEXT: Record<Failure, string> = {
  bad_image: "ui.error.bad_image",
  too_large: "ui.error.too_large",
  invalid: "ui.error.invalid",
  server: "ui.error.server",
  timeout: "ui.error.timeout",
  cancelled: "ui.review.cancelled",
};
const RETRY_AFTER: ReadonlySet<Failure> = new Set(["server", "timeout"]);

const IMAGE_TEXT: Record<ImageName, string> = { frontal: "ui.image.frontal", profile: "ui.image.profile" };

export function reviewScreen(app: App, options: ScreenOptions): Screen {
  const t = app.t;
  const s = app.session;
  const heading = el("h1", { text: t("ui.review.title"), attrs: { tabindex: -1 } });
  let controller: AbortController | null = null;

  const thumb = (name: ImageName) => {
    const entry = s.images[name];
    const picture = entry
      ? archImage(entry.url, "")
      : el("div", { class: "thumb__empty" }, [el("p", { text: t(name === "profile" ? "ui.review.no_profile" : "ui.review.no_frontal") })]);
    // the pictures sit on one line at the bottom, so both captions start at the same height
    return el("figure", { class: "thumb", attrs: { "data-image": name } }, [
      el("div", { class: "thumb__picture" }, [picture]),
      el("figcaption", { class: "thumb__caption" }, [
        el("p", { class: "thumb__name" }, [
          document.createTextNode(t(IMAGE_TEXT[name])),
          entry?.source === "upload" ? uploadFlag(app.i18n) : null,
        ]),
      ]),
    ]);
  };

  const status = el("div", { class: "status", attrs: { role: "status", "aria-live": "polite" } });
  const send = button(t("ui.review.send"), () => void run(), { class: "btn btn--primary", attrs: { "data-action": "send" } });
  const cancel = button(t("ui.review.cancel"), () => controller?.abort(), { class: "btn btn--secondary", attrs: { "data-action": "cancel", hidden: true } });
  const progress = el("div", { class: "progress", attrs: { hidden: true, "aria-hidden": "true" } });
  const canSend = app.requestImages() !== null;
  send.disabled = !canSend;

  const say = (text: string, error = false, retry = false) => {
    status.classList.toggle("status--error", error);
    const children: Node[] = [el("p", { text })];
    if (retry) children.push(button(t("ui.review.retry"), () => void run(), { class: "btn btn--secondary", attrs: { "data-action": "retry" } }));
    status.replaceChildren(...children);
  };

  const busy = (on: boolean) => {
    send.disabled = on || !canSend;
    cancel.hidden = !on;
    progress.hidden = !on;
    element.setAttribute("aria-busy", String(on));
  };

  async function run(): Promise<void> {
    if (controller) return;
    controller = new AbortController();
    busy(true);
    say(t("ui.review.analyzing"));
    const outcome = await app.analyze(controller.signal, () => say(t("ui.review.shrinking")));
    controller = null;
    if (!element.isConnected) return; // another screen took over
    busy(false);
    if (outcome.kind === "ok") {
      app.show("results");
      return;
    }
    say(t(FAILURE_TEXT[outcome.kind]), outcome.kind !== "cancelled", RETRY_AFTER.has(outcome.kind));
  }

  const element = el("section", { class: "page page--medium review", attrs: { "data-screen": "review" } }, [
    // going back while a request runs cancels it (dispose)
    el("header", { class: "page__head" }, [
      backButton(app, () => app.show("profile")),
      stepLine(app, STEP.review),
      heading,
      el("p", { class: "lead", text: t("ui.review.intro") }),
    ]),
    el("div", { class: "thumbs" }, [thumb("frontal"), thumb("profile")]),
    el("div", { class: "send" }, [el("div", { class: "actions" }, [send, cancel]), progress, status]),
  ]);

  if (!canSend) say(t("ui.review.no_frontal"), true);
  else if (options.send) queueMicrotask(() => void run());

  return {
    element,
    heading,
    dispose: () => controller?.abort(),
  };
}
