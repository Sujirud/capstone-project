// The camera of the frontal screen (web.md sections 3.2, 6.3 and 7): the video with the
// outline of a head that fits a face at about 1 m (guide_layer.ts), the band under the
// video with one message at a time in an aria-live area, a dot for each group of checks,
// the button that takes the photo by hand at any time, and in lab mode the live values.
// Every check passing for about one second takes one frame with a short flash; the photo
// takes the place of the image there was, and the screen goes on to the profile. The
// review screen shows the photo before it is sent.
//
// capture.ts shows this block and calls open() when the user presses the camera button,
// and close() when it is pressed again. Hiding the tab turns the camera off and tells
// the screen (onClosed), which shows the button again.
import type { App } from "../app.ts";
import { AUTO_CAPTURE_HOLD_MS, FLASH_MS } from "../constants.ts";
import { button, el } from "../dom.ts";
import { CHECK_GROUPS, Guide, GuidanceLoop, issueText, type CheckGroup, type CheckState, type LandmarkerLike, type LiveState } from "../guidance.ts";
import type { Size } from "../images.ts";
import type { FrontalValues } from "../metrics.ts";
import { LiveCamera, type CameraPart, type CameraPartHandlers } from "./camera_view.ts";

const CHECK_TEXT: Record<CheckGroup, string> = {
  face: "ui.guide.check.face",
  distance: "ui.guide.check.distance",
  pose: "ui.guide.check.pose",
  expression: "ui.guide.check.expression",
};
const STATE_TEXT: Record<CheckState, string> = { pass: "ui.guide.state.pass", fail: "ui.guide.state.fail", unknown: "ui.guide.state.unknown" };

/** Lab mode: the values under the picture (web.md section 7). */
function labReadout(app: App) {
  const i18n = app.i18n;
  const cells = new Map<string, HTMLElement>();
  const row = (name: string) => {
    const value = el("dd", { text: i18n.t("ui.lab.no_value") });
    cells.set(name, value);
    return [el("dt", { class: "code", text: name }), value];
  };
  const names = ["yaw_deg", "pitch_deg", "roll_deg", "iod_px", "face_width_px", "smile", "jaw_open", "blink_left", "blink_right", "stream", "fps"];
  const element = el("section", { class: "lab lab-live", attrs: { "data-lab": "live" } }, [
    el("h2", { class: "panel__title", text: i18n.t("ui.lab.live") }),
    el("dl", { class: "lab-live__values" }, names.flatMap(row)),
  ]);
  const set = (name: string, text: string) => {
    const cell = cells.get(name)!;
    if (cell.textContent !== text) cell.textContent = text;
  };
  return {
    element,
    stream(size: Size | null) {
      set("stream", size ? `${size.width}×${size.height}` : i18n.t("ui.lab.no_value"));
    },
    update(state: LiveState) {
      const v = state.values;
      const fmt = (name: keyof FrontalValues, digits: number) => (v ? i18n.number(v[name], digits) : i18n.t("ui.lab.no_value"));
      for (const name of ["yaw_deg", "pitch_deg", "roll_deg", "iod_px", "face_width_px"] as const) set(name, fmt(name, 1));
      for (const name of ["smile", "jaw_open", "blink_left", "blink_right"] as const) set(name, fmt(name, 2));
      set("fps", i18n.number(state.fps, 0));
    },
  };
}

export function frontalCamera(app: App, handlers: CameraPartHandlers): CameraPart {
  const t = app.t;
  const s = app.session;
  let disposed = false;
  let landmarker: LandmarkerLike | null = null;
  let loading = false;
  let loop: GuidanceLoop | null = null;
  let size: Size | null = null;
  let latest: FrontalValues | null = null;
  let taking = false;
  let doneTimer: ReturnType<typeof setTimeout> | null = null;

  const message = el("p", { class: "camera__message", text: t("ui.camera.preparing"), attrs: { "aria-live": "polite", role: "status", "data-guide-message": "" } });
  const hold = el("div", { class: "camera__hold", attrs: { "aria-hidden": "true" } });
  const band = el("div", { class: "camera__band" }, [message, hold]);
  const dots = new Map<CheckGroup, { item: HTMLElement; state: HTMLElement }>();
  const checks = el(
    "ul",
    { class: "guide-checks", attrs: { "aria-label": t("ui.guide.checks") } },
    CHECK_GROUPS.map((group) => {
      const state = el("span", { class: "visually-hidden", text: t(STATE_TEXT.unknown) });
      const item = el("li", { class: "guide-check", attrs: { "data-check": group, "data-state": "unknown" } }, [
        el("span", { class: "guide-check__dot", attrs: { "aria-hidden": "true" } }),
        el("span", { text: t(CHECK_TEXT[group]) }),
        state,
      ]);
      dots.set(group, { item, state });
      return item;
    }),
  );
  const shoot = button(t("ui.guide.capture"), () => void take(latest), { class: "btn btn--primary", attrs: { "data-action": "capture" } });
  shoot.disabled = true;
  const lab = s.lab ? labReadout(app) : null;

  const say = (text: string) => {
    if (message.textContent !== text) message.textContent = text;
  };

  const showState = (state: LiveState) => {
    latest = state.values;
    say(state.issue ? issueText(state.issue, t) : t("ui.guide.hold"));
    for (const group of CHECK_GROUPS) {
      const dot = dots.get(group)!;
      if (dot.item.dataset.state !== state.checks[group]) {
        dot.item.dataset.state = state.checks[group];
        dot.state.textContent = t(STATE_TEXT[state.checks[group]]);
      }
    }
    hold.style.setProperty("--hold", String(Math.min(1, state.holdMs / AUTO_CAPTURE_HOLD_MS)));
    lab?.update(state);
    if (state.capture) void take(state.values);
  };

  const live = new LiveCamera(app, "frontal", {
    onOpen: (opened) => {
      size = opened;
      shoot.disabled = false;
      lab?.stream(opened);
      startLoop();
    },
    onClose: () => {
      loop?.stop();
      shoot.disabled = true;
      lab?.stream(null);
    },
    onHidden: () => {
      if (!taking) handlers.onClosed();
    },
  });

  function startLoop(): void {
    if (disposed || !landmarker || !live.active || !size) return;
    loop ??= new GuidanceLoop(landmarker, live.camera, new Guide(), app.deps.frames, {
      onState: showState,
      onError: () => say(t("ui.guide.unavailable")),
    });
    loop.start();
  }

  async function take(values: FrontalValues | null): Promise<void> {
    if (taking || disposed || !live.active) return;
    taking = true;
    loop?.stop();
    shoot.disabled = true;
    try {
      const shot = await live.camera.capture(app.deps.makeCanvas);
      if (disposed) return;
      if (!live.active) throw new Error("closed while taking"); // the camera button was pressed meanwhile
      live.flash();
      const kept = await app.useCapture("frontal", shot.blob, shot.size, values);
      if (disposed) return;
      if (!kept.ok) throw new Error(kept.message);
      live.stop();
      doneTimer = setTimeout(() => {
        if (!disposed) handlers.onDone(); // the image is replaced: go on
      }, FLASH_MS);
    } catch {
      taking = false;
      if (disposed) return;
      shoot.disabled = !live.active;
      if (live.active) loop?.start();
    }
  }

  /** The user pressed the button: open the camera, and load MediaPipe the first time. */
  function open(): void {
    say(t("ui.camera.preparing"));
    for (const { item, state } of dots.values()) {
      item.dataset.state = "unknown";
      state.textContent = t(STATE_TEXT.unknown);
    }
    hold.style.setProperty("--hold", "0");
    void live.start();
    if (landmarker || loading) return;
    loading = true;
    app.deps.loadLandmarker().then(
      (ready) => {
        loading = false;
        if (disposed) return;
        landmarker = ready;
        startLoop();
      },
      () => {
        loading = false;
        if (!disposed) say(t("ui.guide.unavailable"));
      },
    );
  }

  function close(): void {
    loop?.stop();
    live.stop();
  }

  live.element.append(band); // the band sits right under the picture, inside the camera block
  const note = el("p", { class: "hint camera__note", text: t("ui.guide.auto_note", { seconds: app.i18n.numberUpTo(AUTO_CAPTURE_HOLD_MS / 1000, 1) }) });
  const element = el("div", { class: "camera-part", attrs: { "data-camera-part": "frontal" } }, [
    live.element,
    checks,
    el("div", { class: "actions actions--center" }, [shoot]),
    note,
    lab?.element,
  ]);

  return {
    element,
    open,
    close,
    dispose: () => {
      disposed = true;
      loop?.stop();
      live.dispose();
      if (doneTimer) clearTimeout(doneTimer);
    },
  };
}
