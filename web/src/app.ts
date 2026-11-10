// The page: the frame around the screens, the switch between screens, the language,
// the /health check, and the images of this round (web.md sections 2, 4 and 8.4).
// Everything outside the browser comes in through AppDeps, so the tests run it with a
// fake fetch, fake image tools, a fake camera and a fake landmarker.
import { analyzeWithRetry, checkHealth, type AnalyzeImages, type AnalyzeOutcome, type FetchFn, type HealthOutcome } from "./api.ts";
import { browserCanvas, type CanvasLike, type MediaDevicesLike } from "./camera.ts";
import { ANALYZE_TIMEOUT_MS, IMAGE_BUDGET_BYTES, LAB_PARAM, LANG_PARAM } from "./constants.ts";
import { button, el, uid } from "./dom.ts";
import { BUNDLED_LANGUAGES, I18n, applyDocumentLang, langFromSearch, type Params, type Texts } from "./i18n.ts";
import { browserClock, sharedLandmarker, type FrameClock, type LandmarkerLike } from "./guidance.ts";
import { browserImageTools, chooseToShrink, loadUpload, shrinkImage, type ImageTools, type Size } from "./images.ts";
import type { FrontalValues } from "./metrics.ts";
import { WebAudioBeeper, type Beeper } from "./sound.ts";
import { Session, type UrlApi } from "./state.ts";
import { SCREENS, type ScreenName, type ScreenOptions, type Screen } from "./screens/index.ts";
import type { ImageName, Meta } from "./types.ts";

export interface AppDeps {
  fetch: FetchFn;
  imageTools: ImageTools;
  urlApi: UrlApi;
  copy: (text: string) => Promise<void>;
  languages: Record<string, Texts>;
  /** navigator.mediaDevices; null where the page cannot use a camera (not on localhost or HTTPS). */
  media: MediaDevicesLike | null;
  loadLandmarker: () => Promise<LandmarkerLike>;
  frames: FrameClock;
  makeCanvas: () => CanvasLike;
  beeper: Beeper;
}

function defaultDeps(): AppDeps {
  return {
    fetch: (input, init) => globalThis.fetch(input, init),
    imageTools: browserImageTools,
    urlApi: URL,
    copy: (text) => navigator.clipboard.writeText(text),
    languages: BUNDLED_LANGUAGES,
    media: globalThis.navigator?.mediaDevices ?? null,
    loadLandmarker: () => sharedLandmarker(),
    frames: browserClock,
    makeCanvas: browserCanvas,
    beeper: new WebAudioBeeper(),
  };
}

export type UploadResult = { ok: true } | { ok: false; message: string };

export class App {
  readonly root: HTMLElement;
  readonly deps: AppDeps;
  readonly i18n: I18n;
  readonly session: Session;
  /** The demo mode shows a stored result; there is nothing to send again. */
  demo = false;
  health: HealthOutcome | null = null;
  current: { name: ScreenName; options: ScreenOptions; screen: Screen } | null = null;
  private main: HTMLElement | null = null;
  private healthBox: HTMLElement | null = null;
  private firstScreen = true;

  constructor(root: HTMLElement, search: string, deps: Partial<AppDeps> = {}) {
    this.root = root;
    this.deps = { ...defaultDeps(), ...deps };
    this.i18n = new I18n(this.deps.languages, langFromSearch(search, LANG_PARAM));
    const lab = new URLSearchParams(search).get(LAB_PARAM) === "1";
    this.session = new Session({ lang: this.i18n.lang, lab }, this.deps.urlApi);
  }

  t = (key: string, params?: Params): string => this.i18n.t(key, params);

  /** Draw the frame, check /health and show the home screen. */
  async start(): Promise<void> {
    this.mount();
    this.show("home");
    await this.checkHealth();
  }

  /** The frame: the Greek key band, the language picker, the /health notice and the screen area. */
  mount(): void {
    applyDocumentLang(this.i18n);
    document.title = this.t("ui.app.name");
    this.healthBox = el("div", { class: "health", attrs: { "aria-live": "polite" } });
    this.main = el("main", { class: "screen", attrs: { id: "main" } });
    this.root.replaceChildren(
      el("div", { class: "greek-key", attrs: { "aria-hidden": "true" } }),
      this.languagePicker() ?? "",
      this.healthBox,
      this.main,
    );
    this.renderHealth();
  }

  /** The language picker under the Greek key band, only when there is more than one language file. */
  private languagePicker(): HTMLElement | null {
    const codes = this.i18n.codes;
    if (codes.length < 2) return null;
    const id = uid("lang");
    const select = el("select", { class: "select", attrs: { id } });
    for (const code of codes) {
      const option = el("option", { text: this.i18n.nameOf(code), attrs: { value: code, lang: code } });
      option.selected = code === this.i18n.lang;
      select.append(option);
    }
    select.addEventListener("change", () => this.setLang(select.value));
    return el("div", { class: "lang-picker" }, [el("label", { text: this.t("ui.lang.label"), attrs: { for: id } }), el("div", { class: "select-wrapper" }, [select])]);
  }

  show(name: ScreenName, options: ScreenOptions = {}): Screen {
    this.current?.screen.dispose?.();
    const screen = SCREENS[name](this, options);
    this.current = { name, options, screen };
    this.main?.replaceChildren(screen.element);
    if (this.firstScreen) this.firstScreen = false;
    else screen.heading.focus();
    window.scrollTo?.(0, 0);
    return screen;
  }

  /**
   * Switch the language. After a result, the same images are analysed again so the texts
   * of the result come in the new language too (web.md section 8.4).
   */
  setLang(code: string): void {
    this.session.lang = this.i18n.setLang(code);
    const name = this.current?.name ?? "home";
    const again = name === "results" && !this.demo && this.requestImages() !== null;
    this.mount();
    if (again) this.show("review", { send: true });
    else this.show(name, this.current?.options ?? {});
  }

  // ---------------------------------------------------------------- /health

  async checkHealth(): Promise<HealthOutcome> {
    this.health = await checkHealth({ fetch: this.deps.fetch });
    this.renderHealth();
    return this.health;
  }

  private renderHealth(): void {
    if (!this.healthBox) return;
    if (!this.health || this.health.ok) {
      this.healthBox.replaceChildren();
      return;
    }
    this.healthBox.replaceChildren(
      el("div", { class: "notice notice--alert", attrs: { "data-health": "down" } }, [
        el("p", { text: this.t("ui.health.down") }),
        button(this.t("ui.health.retry"), () => void this.checkHealth(), { class: "btn btn--secondary" }),
      ]),
    );
  }

  // ---------------------------------------------------------------- the round

  meta(): Meta {
    const s = this.session;
    if (s.age === null || s.sex === null) throw new Error("age and sex are not set");
    const meta: Meta = {
      age: s.age,
      sex: s.sex,
      frontal_source: s.images.frontal?.source ?? "camera",
      lang: this.i18n.lang,
    };
    if (s.lab) meta.lab = true;
    return meta;
  }

  /** The images to send, or null without a frontal image. */
  requestImages(): AnalyzeImages | null {
    const { frontal, profile } = this.session.images;
    if (!frontal?.blob) return null;
    return { frontal: frontal.blob, profile: profile?.blob ?? null };
  }

  /** Check a chosen file, keep it, and re-encode images until the request fits. */
  async useUpload(name: ImageName, file: Blob): Promise<UploadResult> {
    const loaded = await loadUpload(file, this.deps.imageTools);
    if (!loaded.ok) {
      return { ok: false, message: this.t(loaded.reason === "not_jpeg_png" ? "ui.upload.not_jpeg_png" : "ui.upload.unreadable") };
    }
    this.session.setImage(name, loaded.blob, { width: loaded.width, height: loaded.height, source: "upload", shrunk: false });
    if (name === "frontal") this.session.frontalValues = null; // the guide did not see this one
    return this.fitRequest();
  }

  /**
   * Keep a photo from the camera, at the size of the stream (camera.ts). values is what
   * the pose guide measured when the frontal photo was taken, kept for lab mode.
   */
  async useCapture(name: ImageName, blob: Blob, size: Size, values: FrontalValues | null = null): Promise<UploadResult> {
    this.session.setImage(name, blob, { width: size.width, height: size.height, source: "camera", shrunk: false });
    if (name === "frontal") this.session.frontalValues = values;
    return this.fitRequest();
  }

  /** Re-encode images, one at a time and the largest first, until the request fits. */
  private async fitRequest(): Promise<UploadResult> {
    try {
      for (let names = this.toShrink(IMAGE_BUDGET_BYTES); names.length; names = this.toShrink(IMAGE_BUDGET_BYTES)) {
        await this.shrink(names.slice(0, 1));
      }
    } catch {
      return { ok: false, message: this.t("ui.upload.unreadable") };
    }
    return { ok: true };
  }

  /** Images to re-encode so the request fits budget; prefers none. */
  private toShrink(budget: number): ImageName[] {
    const sized = (["frontal", "profile"] as const).flatMap((name) => {
      const entry = this.session.images[name];
      return entry?.blob ? [{ name, bytes: entry.blob.size, shrunk: entry.shrunk }] : [];
    });
    return chooseToShrink(sized, budget);
  }

  private async shrink(names: ImageName[]): Promise<void> {
    for (const name of names) {
      const entry = this.session.images[name];
      if (!entry?.blob) continue;
      const out = await shrinkImage(entry.blob, this.deps.imageTools);
      this.session.setImage(name, out.blob, { width: out.width, height: out.height, source: entry.source, shrunk: true });
    }
  }

  /** After a 413: re-encode every image not re-encoded yet; null when there is none. */
  async shrinkForRetry(): Promise<AnalyzeImages | null> {
    const names = (["frontal", "profile"] as const).filter((n) => {
      const entry = this.session.images[n];
      return entry?.blob && !entry.shrunk;
    });
    if (names.length === 0) return null;
    await this.shrink(names);
    return this.requestImages();
  }

  /** POST /analyze with the images of this round (web.md section 4.2). */
  async analyze(signal: AbortSignal, onShrink?: () => void): Promise<AnalyzeOutcome> {
    const images = this.requestImages();
    if (!images) return { kind: "invalid" };
    const outcome = await analyzeWithRetry(images, this.meta(), () => this.shrinkForRetry(), {
      fetch: this.deps.fetch,
      signal,
      timeoutMs: ANALYZE_TIMEOUT_MS,
      onShrink,
    });
    if (outcome.kind === "ok") this.session.result = outcome.result;
    return outcome;
  }

  /** Start over: every image and value of the round is dropped. */
  restart(): void {
    this.session.reset();
    this.demo = false;
    this.show("home");
  }
}
