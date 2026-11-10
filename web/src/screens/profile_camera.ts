// The camera of the profile screen (web.md section 3.3). Face Landmarker does not suit a
// face turned past about 80 degrees, so there is no live check here: the outline of a
// head from the side in the style of the frontal guide (guide_layer.ts), a button that
// counts down 3 seconds with a beep each second and
// takes one frame, then the photo to keep or take again. The api checks that it is a
// profile (api.md section 4). Either side will do, since the api flips the image itself.
// The kept photo takes the place of the image there was.
//
// capture.ts shows this block and calls open() when the user presses the camera button,
// and close() when it is pressed again, which also drops a photo not kept yet. Hiding
// the tab turns the camera off and tells the screen (onClosed).
import type { App } from "../app.ts";
import { COUNTDOWN_SECONDS } from "../constants.ts";
import { button, el } from "../dom.ts";
import type { Size } from "../images.ts";
import { startCountdown } from "../sound.ts";
import { LiveCamera, type CameraPart, type CameraPartHandlers } from "./camera_view.ts";

export function profileCamera(app: App, handlers: CameraPartHandlers): CameraPart {
  const t = app.t;
  const beeper = app.deps.beeper;
  let disposed = false;
  let countdown: { cancel: () => void } | null = null;
  let photo: { blob: Blob; size: Size; url: string } | null = null;

  const count = el("p", { class: "camera__count", attrs: { "aria-hidden": "true", hidden: true } });
  const spoken = el("p", { class: "visually-hidden", attrs: { "aria-live": "assertive" } });
  const start = button(t("ui.profile.countdown", { seconds: COUNTDOWN_SECONDS }), () => (countdown ? stopCountdown() : runCountdown()), {
    class: "btn btn--primary",
    attrs: { "data-action": "countdown" },
  });
  start.disabled = true;

  const live = new LiveCamera(app, "profile", {
    onOpen: () => {
      start.disabled = false;
    },
    onClose: () => {
      stopCountdown();
      start.disabled = true;
    },
    onHidden: () => {
      if (!photo) handlers.onClosed();
    },
  });
  live.stage.append(count);

  const setLabel = (key: "ui.profile.countdown" | "ui.profile.countdown_cancel") => {
    start.querySelector("span")!.textContent = key === "ui.profile.countdown" ? t(key, { seconds: COUNTDOWN_SECONDS }) : t(key);
  };

  function stopCountdown(): void {
    countdown?.cancel();
    countdown = null;
    count.hidden = true;
    spoken.textContent = "";
    setLabel("ui.profile.countdown");
  }

  function runCountdown(): void {
    if (!live.active) return;
    beeper.prepare(); // inside the click, so the browser lets the page play sound
    setLabel("ui.profile.countdown_cancel");
    count.hidden = false;
    countdown = startCountdown(COUNTDOWN_SECONDS, {
      onTick: (left) => {
        count.textContent = app.i18n.numberUpTo(left, 0);
        spoken.textContent = t("ui.profile.countdown_left", { n: left });
        beeper.tick();
      },
      onDone: () => {
        countdown = null;
        count.hidden = true;
        spoken.textContent = "";
        setLabel("ui.profile.countdown");
        beeper.shot();
        void take();
      },
    });
  }

  async function take(): Promise<void> {
    if (disposed || !live.active) return;
    start.disabled = true;
    try {
      const shot = await live.camera.capture(app.deps.makeCanvas);
      if (disposed) return;
      live.flash();
      live.stop(); // the light goes off while the photo is checked
      photo = { ...shot, url: app.deps.urlApi.createObjectURL(shot.blob) };
      showPhoto();
    } catch {
      if (!disposed) start.disabled = !live.active;
    }
  }

  function dropPhoto(): void {
    if (photo) app.deps.urlApi.revokeObjectURL(photo.url);
    photo = null;
  }

  const liveBox = el("div", { class: "camera-live" }, [
    live.element,
    el("p", { class: "camera__note", text: t("ui.profile.guide_note") }),
    el("div", { class: "actions actions--center" }, [start]),
    spoken,
  ]);
  const element = el("div", { class: "camera-part", attrs: { "data-camera-part": "profile" } }, [liveBox]);

  function goLive(): void {
    dropPhoto();
    element.replaceChildren(liveBox);
    void live.start();
  }

  function showPhoto(): void {
    if (!photo) return;
    const keep = button(
      t("ui.profile.use_photo"),
      async () => {
        if (!photo) return;
        keep.disabled = true;
        const { blob, size } = photo;
        dropPhoto();
        const kept = await app.useCapture("profile", blob, size);
        if (!disposed && kept.ok) handlers.onDone();
      },
      { class: "btn btn--primary", attrs: { "data-action": "use-photo" } },
    );
    element.replaceChildren(
      el("div", { class: "camera-shot", attrs: { "data-camera-shot": "" } }, [
        el("img", { class: "camera-shot__image", attrs: { src: photo.url, alt: t("ui.profile.photo_alt") } }),
        el("p", { text: t("ui.profile.check_photo") }),
        el("div", { class: "actions actions--center" }, [keep, button(t("ui.profile.retake"), goLive, { class: "btn btn--secondary", attrs: { "data-action": "retake" } })]),
      ]),
    );
  }

  function close(): void {
    stopCountdown();
    live.stop();
    dropPhoto();
    element.replaceChildren(liveBox);
  }

  return {
    element,
    open: goLive,
    close,
    dispose: () => {
      disposed = true;
      stopCountdown();
      live.dispose();
      dropPhoto();
    },
  };
}
