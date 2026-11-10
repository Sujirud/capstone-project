// The frontal and the profile screens (web.md sections 2, 3 and 3.4), in three parts:
//
// - the top: the heading and the panel of what makes an image usable;
// - the middle, one of: the camera block when the camera is on (frontal_camera.ts with
//   the live pose guide, profile_camera.ts with the countdown); the image there already
//   is with "use the same image", which goes on without changing it (web.md section
//   4.3); or nothing;
// - the bottom: open or close the camera (one button whose label changes), choose an
//   image from the computer, and on the profile screen skip it, which leaves the chin
//   and the nasal dorsum not assessable.
//
// A photo from the camera takes the place of the image, as a chosen file does. The
// camera opens only when the button is pressed; it is turned off by the same button, by
// choosing a file (or a photo could be taken while the file dialog is open), by hiding
// the tab, and by leaving the screen (dispose).
import type { App } from "../app.ts";
import { FILE_ACCEPT } from "../constants.ts";
import { button, el, uid } from "../dom.ts";
import { CAMERA_PROBLEM_TEXT } from "../camera.ts";
import { archImage } from "../results/overlay.ts";
import type { ImageName } from "../types.ts";
import type { Screen } from "./index.ts";
import { cameraSupported, type CameraPart } from "./camera_view.ts";
import { frontalCamera } from "./frontal_camera.ts";
import { STEP, backButton, stepLine } from "./parts.ts";
import { profileCamera } from "./profile_camera.ts";

interface CaptureTexts {
  title: string;
  intro: string;
  image: string;
  tips: string[];
}

const TEXTS: Record<ImageName, CaptureTexts> = {
  frontal: {
    title: "ui.frontal.title",
    intro: "ui.frontal.intro",
    image: "ui.image.frontal",
    tips: ["ui.frontal.tip_glasses", "ui.frontal.tip_hair", "ui.frontal.tip_light", "ui.frontal.tip_face"],
  },
  profile: {
    title: "ui.profile.title",
    intro: "ui.profile.intro",
    image: "ui.image.profile",
    tips: [
      "ui.profile.tip_turn",
      "ui.profile.tip_eyes",
      "ui.profile.tip_mouth",
      "ui.profile.tip_forehead",
      "ui.profile.tip_glasses",
      "ui.profile.tip_wall",
    ],
  },
};

type Mode = "empty" | "image" | "camera";

/** Space kept between the edge of the window and what is scrolled into view. */
const SCROLL_ROOM_PX = 16;

export function captureScreen(app: App, name: ImageName): Screen {
  const t = app.t;
  const texts = TEXTS[name];
  const heading = el("h1", { text: t(texts.title), attrs: { tabindex: -1 } });
  const status = el("p", { class: "status", attrs: { role: "status", "aria-live": "polite" } });

  const next = () => app.show(name === "frontal" ? "profile" : "review");
  const back = () => app.show(name === "frontal" ? "home" : "frontal");

  // ---- the middle: the camera, the image there is, or nothing
  const main = el("div", { class: "capture__main", attrs: { "data-capture-main": "" } });
  const handlers = { onDone: next, onClosed: () => showIdle() };
  const camera: CameraPart | null = cameraSupported(app) ? (name === "frontal" ? frontalCamera(app, handlers) : profileCamera(app, handlers)) : null;
  let mode: Mode = "empty";

  /** The image of this screen, centred, with the button that keeps it. */
  function currentImage(): HTMLElement | null {
    const image = app.session.images[name];
    if (!image) return null;
    const picture = el("div", { class: "capture__picture" }, [archImage(image.url, t(texts.image))]);
    picture.style.setProperty("--image-ratio", String(image.width / image.height));
    return el("div", { class: "capture__current", attrs: { "data-current-image": "" } }, [
      picture,
      button(t("ui.capture.keep"), next, { class: "btn btn--secondary", attrs: { "data-action": "keep" } }),
    ]);
  }

  // ---- the bottom: the actions
  const input = el("input", { class: "visually-hidden", attrs: { type: "file", accept: FILE_ACCEPT, tabindex: -1, "aria-hidden": "true" } });
  const choose = button(
    t("ui.capture.choose_file"),
    () => {
      if (mode === "camera") showIdle(); // no photo while the file dialog is open
      input.click();
    },
    { class: "btn btn--secondary", attrs: { "data-action": "choose-file" } },
  );
  input.addEventListener("change", async () => {
    const file = input.files?.[0];
    input.value = "";
    if (!file) return;
    choose.disabled = true;
    say(t("ui.capture.reading"));
    const result = await app.useUpload(name, file);
    choose.disabled = false;
    if (!result.ok) {
      say(result.message, true);
      return;
    }
    say("");
    next();
  });

  const toggle = camera ? button(t("ui.camera.start"), () => (mode === "camera" ? showIdle() : showCamera()), { class: "btn btn--primary" }) : null;
  const skip =
    name === "profile"
      ? button(
          t("ui.profile.skip"),
          () => {
            app.session.clearImage("profile");
            app.session.profileSkipped = true;
            app.show("review");
          },
          { class: "btn btn--plain", attrs: { "data-action": "skip-profile" } },
        )
      : null;
  const unsupported = camera ? null : el("p", { class: "notice notice--camera", text: t(CAMERA_PROBLEM_TEXT.unsupported), attrs: { "data-camera-notice": "unsupported" } });
  const skipNote = name === "profile" ? el("p", { class: "hint", text: t("ui.profile.skip_note") }) : null;

  function say(text: string, error = false): void {
    status.textContent = text;
    status.classList.toggle("status--error", error);
  }

  /** The same button opens and closes the camera, so the focus stays where it is. */
  function setToggle(on: boolean): void {
    if (!toggle) return;
    toggle.querySelector("span")!.textContent = t(on ? "ui.camera.stop" : "ui.camera.start");
    toggle.dataset.action = on ? "camera-close" : "camera-open";
    toggle.classList.toggle("btn--primary", !on); // while it is on, the camera's own button is the main one
    toggle.classList.toggle("btn--secondary", on);
  }

  function setMode(to: Mode, child: HTMLElement | null): void {
    mode = to;
    main.dataset.mode = to;
    main.replaceChildren(...(child ? [child] : []));
    setToggle(to === "camera");
  }

  function showIdle(): void {
    camera?.close();
    const image = currentImage();
    setMode(image ? "image" : "empty", image);
  }

  function showCamera(): void {
    if (!camera) return;
    say("");
    setMode("camera", camera.element);
    camera.open();
    bringIntoView();
  }

  /**
   * Scroll so that the camera block is on screen, and the buttons under it too when they
   * fit, but never past the top of the camera. A jump instead of a glide for reduced motion.
   */
  function bringIntoView(): void {
    const room = SCROLL_ROOM_PX;
    const top = main.getBoundingClientRect().top - room;
    const below = buttons.getBoundingClientRect().bottom + room - globalThis.innerHeight;
    const by = Math.min(top, Math.max(0, below));
    if (Math.abs(by) < 1) return;
    const still = globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? true;
    globalThis.scrollBy?.({ top: by, behavior: still ? "auto" : "smooth" });
  }

  const buttons = el("div", { class: "actions" }, [toggle, choose, skip, input]);
  showIdle();

  const tipsTitle = uid("tips");
  const element = el("section", { class: `page capture capture--${name}`, attrs: { "data-screen": name } }, [
    el("header", { class: "page__head" }, [backButton(app, back), stepLine(app, STEP[name]), heading, el("p", { class: "lead", text: t(texts.intro) })]),
    el("section", { class: "panel capture__tips", attrs: { "aria-labelledby": tipsTitle } }, [
      el("h2", { class: "panel__title", text: t("ui.capture.tips_title"), attrs: { id: tipsTitle } }),
      el("ul", { class: "checklist checklist--columns" }, texts.tips.map((key) => el("li", { text: t(key) }))),
    ]),
    main,
    el("div", { class: "capture__actions" }, [buttons, unsupported, skipNote, status]),
  ]);
  return { element, heading, dispose: () => camera?.dispose() };
}
