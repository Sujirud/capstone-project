// The screens together: /health, home, choosing images, sending and the results page
// (web.md sections 2, 3.4, 4 and 8.4), with a fake fetch and fake image tools.
import { afterEach, describe, expect, it, vi } from "vitest";
import { App, type AppDeps } from "../src/app.ts";
import { BUNDLED_LANGUAGES } from "../src/i18n.ts";
import { IMAGE_BUDGET_BYTES } from "../src/constants.ts";
import { JPEG_HEAD, PNG_HEAD, TH, fakeFile, fakeImageTools, fakeMedia, fakeUrlApi, fixture, hangingFetch, jsonResponse, scriptedFetch, settle, th } from "./helpers.ts";

const HEALTH = { status: "ok", versions: {}, stub_heads: [], device: "cpu", languages: ["th"] };
const RESULT = fixture("female_clear").result;

type Answer = Response | Error;

/** A fetch that answers /health with health and /analyze with the answers in order. */
function apiFetch(health: Answer, ...analyze: Answer[]) {
  const analyzeFetch = analyze.length ? scriptedFetch(...analyze) : hangingFetch();
  const healthFetch = scriptedFetch(health);
  return vi.fn((input: string, init?: RequestInit) => (input === "/health" ? healthFetch(input, init) : analyzeFetch(input, init)));
}

function makeApp(deps: Partial<AppDeps> = {}, search = "") {
  document.body.innerHTML = '<div id="app"></div>';
  const root = document.getElementById("app")!;
  const app = new App(root, search, {
    fetch: apiFetch(jsonResponse(200, HEALTH)),
    imageTools: fakeImageTools(),
    urlApi: fakeUrlApi(),
    copy: async () => {},
    ...deps,
  });
  return { app, root };
}

function $(root: ParentNode, selector: string): HTMLElement {
  const found = root.querySelector<HTMLElement>(selector);
  if (!found) throw new Error(`nothing matches ${selector}`);
  return found;
}

function fillHome(root: HTMLElement, age = "30", sex = "female") {
  ($(root, 'input[name="age"]') as HTMLInputElement).value = age;
  if (sex) ($(root, `input[name="sex"][value="${sex}"]`) as HTMLInputElement).checked = true;
  $(root, "form").dispatchEvent(new Event("submit", { cancelable: true }));
}

async function chooseFile(root: HTMLElement, file: File) {
  const input = $(root, 'input[type="file"]') as HTMLInputElement;
  Object.defineProperty(input, "files", { value: [file], configurable: true });
  input.dispatchEvent(new Event("change"));
  await settle();
}

/** Home, a frontal upload, skip the profile: the review screen. */
async function toReview(root: HTMLElement, frontal = fakeFile(JPEG_HEAD)) {
  fillHome(root);
  await chooseFile(root, frontal);
  $(root, '[data-action="skip-profile"]').click();
}

function screenName(root: HTMLElement): string | undefined {
  return root.querySelector<HTMLElement>("[data-screen]")?.dataset.screen;
}

afterEach(() => {
  vi.useRealTimers();
});

describe("/health", () => {
  it("says the api is not running, with a button that tries again", async () => {
    const fetch = vi.fn().mockRejectedValueOnce(new TypeError("Failed to fetch")).mockResolvedValue(jsonResponse(200, HEALTH));
    const { app, root } = makeApp({ fetch });
    await app.start();
    const notice = $(root, '[data-health="down"]');
    expect(notice.textContent).toContain(th("ui.health.down"));
    expect(notice.closest("[aria-live]")).not.toBeNull();
    $(notice, "button").click();
    await settle();
    expect(root.querySelector('[data-health="down"]')).toBeNull();
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("shows nothing when the api answers", async () => {
    const { app, root } = makeApp();
    await app.start();
    expect(root.querySelector('[data-health="down"]')).toBeNull();
  });
});

describe("home", () => {
  it("explains the system without a consent step and asks for age and sex", async () => {
    const { app, root } = makeApp();
    await app.start();
    expect(screenName(root)).toBe("home");
    const text = root.textContent!;
    for (const key of ["ui.home.point_regions", "ui.home.point_not_diagnosis", "ui.home.point_local"]) expect(text).toContain(th(key));
    expect(root.querySelectorAll('input[type="checkbox"]').length).toBe(0);
    expect(document.documentElement.lang).toBe("th");
    expect(document.title).toBe(th("ui.app.name"));
    expect(root.querySelector("header.masthead")).toBeNull(); // no header above the Greek key band
    expect(root.firstElementChild!.classList.contains("greek-key")).toBe(true);
  });

  it("gives every input a label", async () => {
    const { app, root } = makeApp();
    await app.start();
    for (const input of root.querySelectorAll<HTMLInputElement>("input:not([type=file])")) {
      expect(root.querySelector(`label[for="${input.id}"]`), input.name).not.toBeNull();
    }
  });

  it("marks both fields as required", async () => {
    const { app, root } = makeApp();
    await app.start();
    expect(root.querySelectorAll(".field .required-mark").length).toBe(2);
    expect(($(root, 'input[name="age"]') as HTMLInputElement).required).toBe(true);
    expect([...root.querySelectorAll<HTMLInputElement>('input[name="sex"]')].every((i) => i.required)).toBe(true);
    expect($(root, ".form__note").textContent).toContain(th("ui.form.required_note"));
  });

  it.each([
    ["", "female", "ui.home.age_invalid", null],
    ["2.5", "female", "ui.home.age_invalid", null],
    ["121", "female", "ui.home.age_invalid", null],
    ["30", "", null, "ui.home.sex_missing"],
    ["", "", "ui.home.age_invalid", "ui.home.sex_missing"],
  ])("refuses age %j and sex %j with an error under each field", async (age, sex, ageKey, sexKey) => {
    const { app, root } = makeApp();
    await app.start();
    fillHome(root, age, sex);
    expect(screenName(root)).toBe("home");
    const ageInput = $(root, 'input[name="age"]');
    const ageError = document.getElementById(ageInput.getAttribute("aria-describedby")!.split(" ")[1])!;
    const sexError = document.getElementById($(root, "fieldset").getAttribute("aria-describedby")!)!;
    expect(ageError.closest(".field")).toBe(ageInput.closest(".field"));
    expect(ageError.textContent).toBe(ageKey ? th(ageKey) : "");
    expect(ageInput.getAttribute("aria-invalid")).toBe(String(ageKey !== null));
    expect(sexError.textContent).toBe(sexKey ? th(sexKey) : "");
    expect(document.activeElement).toBe(ageKey ? ageInput : $(root, 'input[name="sex"]'));
  });

  it("clears an error once its field is filled in", async () => {
    const { app, root } = makeApp();
    await app.start();
    fillHome(root, "", "");
    const age = $(root, 'input[name="age"]') as HTMLInputElement;
    age.value = "41";
    age.dispatchEvent(new Event("input"));
    expect(age.getAttribute("aria-invalid")).toBe("false");
    const male = $(root, 'input[name="sex"][value="male"]') as HTMLInputElement;
    male.checked = true;
    male.dispatchEvent(new Event("change"));
    expect([...root.querySelectorAll(".field__error")].map((e) => e.textContent)).toEqual(["", ""]);
  });

  it("starts the round and moves the focus to the next heading", async () => {
    const { app, root } = makeApp();
    await app.start();
    fillHome(root, "0");
    expect(screenName(root)).toBe("frontal");
    expect(document.activeElement).toBe($(root, "h1"));
    expect(app.session.age).toBe(0);
  });
});

describe("choosing images", () => {
  it("has the camera button and the file button", async () => {
    const { app, root } = makeApp({ media: fakeMedia() });
    await app.start();
    fillHome(root);
    expect($(root, '.capture__actions [data-action="camera-open"]').textContent).toBe(th("ui.camera.start"));
    expect($(root, '.capture__actions [data-action="choose-file"]').textContent).toBe(th("ui.capture.choose_file"));
  });

  it("says why there is no camera button where the page cannot use a camera", async () => {
    // media is null, as on a page not opened from localhost: a message, and the file button still works
    const { app, root } = makeApp({ media: null });
    await app.start();
    fillHome(root);
    expect(root.querySelector('[data-action="camera-open"]')).toBeNull();
    expect($(root, '[data-camera-notice="unsupported"]').textContent).toBe(th("ui.camera.unsupported"));
    expect(($(root, 'input[type="file"]') as HTMLInputElement).accept).toBe("image/jpeg,image/png");
  });

  it("asks to convert a file that is not JPEG or PNG", async () => {
    const { app, root } = makeApp();
    await app.start();
    fillHome(root);
    await chooseFile(root, fakeFile([0, 0, 0, 0x18, 0x66, 0x74, 0x79, 0x70], 64, "image/heic"));
    expect(screenName(root)).toBe("frontal");
    expect($(root, ".status").textContent).toBe(th("ui.upload.not_jpeg_png"));
    expect(app.session.images.frontal).toBeNull();
  });

  it("goes on to the profile, which can be skipped", async () => {
    const { app, root } = makeApp();
    await app.start();
    fillHome(root);
    await chooseFile(root, fakeFile(PNG_HEAD, 4096, "image/png"));
    expect(screenName(root)).toBe("profile");
    expect(app.session.images.frontal).toMatchObject({ width: 1280, height: 1600, source: "upload", shrunk: false });
    expect(root.textContent).toContain(th("ui.profile.skip_note"));
    $(root, '[data-action="skip-profile"]').click();
    expect(screenName(root)).toBe("review");
    expect(app.session.profileSkipped).toBe(true);
    expect(root.textContent).toContain(th("ui.review.no_profile"));
  });

  it("re-encodes an image when both together are over the request size", async () => {
    const tools = fakeImageTools();
    const { app, root } = makeApp({ imageTools: tools });
    await app.start();
    fillHome(root);
    await chooseFile(root, fakeFile(JPEG_HEAD, 1024));
    expect(tools.encodeJpeg).not.toHaveBeenCalled();
    await chooseFile(root, fakeFile(JPEG_HEAD, IMAGE_BUDGET_BYTES)); // the profile, too large with the frontal
    expect(tools.encodeJpeg).toHaveBeenCalledTimes(1);
    expect(app.session.images.profile).toMatchObject({ shrunk: true, source: "upload" });
    expect(app.session.images.frontal!.shrunk).toBe(false);
    expect(screenName(root)).toBe("review");
  });
});

describe("sending", () => {
  it("sends meta with the upload source, then shows the results", async () => {
    const fetch = apiFetch(jsonResponse(200, HEALTH), jsonResponse(200, RESULT));
    const { app, root } = makeApp({ fetch });
    await app.start();
    await toReview(root);
    $(root, '[data-action="send"]').click();
    await settle();
    const form = fetch.mock.calls.find((c) => c[0] === "/analyze")![1]!.body as FormData;
    expect(JSON.parse(form.get("meta") as string)).toEqual({ age: 30, sex: "female", frontal_source: "upload", lang: "th" });
    expect(form.has("profile")).toBe(false);
    expect(screenName(root)).toBe("results");
    expect(document.activeElement).toBe($(root, "h1"));
    expect(root.querySelectorAll(".card").length).toBe(5);
  });

  it("sends lab true in lab mode only", async () => {
    const fetch = apiFetch(jsonResponse(200, HEALTH), jsonResponse(200, { ...RESULT, lab: true }));
    const { app, root } = makeApp({ fetch }, "?lab=1");
    await app.start();
    await toReview(root);
    $(root, '[data-action="send"]').click();
    await settle();
    const form = fetch.mock.calls.find((c) => c[0] === "/analyze")![1]!.body as FormData;
    expect(JSON.parse(form.get("meta") as string).lab).toBe(true);
  });

  it("says it is analysing in an aria-live area and can be cancelled", async () => {
    const { app, root } = makeApp({ fetch: apiFetch(jsonResponse(200, HEALTH)) });
    await app.start();
    await toReview(root);
    $(root, '[data-action="send"]').click();
    await settle(1);
    const status = $(root, ".status");
    expect(status.getAttribute("aria-live")).toBe("polite");
    expect(status.textContent).toBe(th("ui.review.analyzing"));
    const cancel = $(root, '[data-action="cancel"]') as HTMLButtonElement;
    expect(cancel.hidden).toBe(false);
    expect(($(root, '[data-action="send"]') as HTMLButtonElement).disabled).toBe(true);
    cancel.click();
    await settle();
    expect(status.textContent).toBe(th("ui.review.cancelled"));
    expect(cancel.hidden).toBe(true);
    expect(screenName(root)).toBe("review");
  });

  it("tries again with the same images after a server error, without a new photo", async () => {
    const fetch = apiFetch(jsonResponse(200, HEALTH), jsonResponse(500, {}), jsonResponse(200, RESULT));
    const { app, root } = makeApp({ fetch });
    await app.start();
    await toReview(root);
    $(root, '[data-action="send"]').click();
    await settle();
    expect($(root, ".status").textContent).toContain(th("ui.error.server"));
    $(root, '[data-action="retry"]').click();
    await settle();
    expect(screenName(root)).toBe("results");
    const sent = fetch.mock.calls.filter((c) => c[0] === "/analyze").map((c) => ((c[1]!.body as FormData).get("frontal") as File).size);
    expect(sent).toEqual([1024, 1024]);
  });

  it.each([
    [400, "ui.error.bad_image"],
    [415, "ui.error.bad_image"],
    [422, "ui.error.invalid"],
  ])("explains HTTP %i", async (status, key) => {
    const { app, root } = makeApp({ fetch: apiFetch(jsonResponse(200, HEALTH), jsonResponse(status, {})) });
    await app.start();
    await toReview(root);
    $(root, '[data-action="send"]').click();
    await settle();
    expect($(root, ".status").textContent).toBe(th(key));
    expect(root.querySelector('[data-action="retry"]')).toBeNull();
  });

  it("re-encodes and sends once more after 413, then shows the re-encoded image", async () => {
    const tools = fakeImageTools({ width: 6000, height: 8000 });
    const fetch = apiFetch(jsonResponse(200, HEALTH), jsonResponse(413, {}), jsonResponse(200, RESULT));
    const { app, root } = makeApp({ fetch, imageTools: tools });
    await app.start();
    await toReview(root);
    $(root, '[data-action="send"]').click();
    await settle(10);
    expect(tools.encodeJpeg).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls.filter((c) => c[0] === "/analyze").length).toBe(2);
    expect(app.session.images.frontal).toMatchObject({ width: 3024, height: 4032, shrunk: true });
    expect(screenName(root)).toBe("results");
    expect($(root, 'svg[data-image="frontal"]').getAttribute("viewBox")).toBe("0 0 3024 4032");
  });

  it("times out after 60 seconds with a retry button", async () => {
    vi.useFakeTimers();
    const { app, root } = makeApp({ fetch: apiFetch(jsonResponse(200, HEALTH)) });
    await app.start();
    fillHome(root);
    const input = $(root, 'input[type="file"]') as HTMLInputElement;
    Object.defineProperty(input, "files", { value: [fakeFile(JPEG_HEAD)] });
    input.dispatchEvent(new Event("change"));
    await vi.advanceTimersByTimeAsync(10);
    $(root, '[data-action="skip-profile"]').click();
    $(root, '[data-action="send"]').click();
    await vi.advanceTimersByTimeAsync(60_000);
    expect($(root, ".status").textContent).toContain(th("ui.error.timeout"));
    expect(root.querySelector('[data-action="retry"]')).not.toBeNull();
  });
});

describe("results", () => {
  async function toResults(fetch = apiFetch(jsonResponse(200, HEALTH), jsonResponse(200, fixture("no_face_no_profile").result))) {
    const made = makeApp({ fetch });
    await made.app.start();
    await toReview(made.root);
    $(made.root, '[data-action="send"]').click();
    await settle();
    return { ...made, fetch };
  }

  /** From the results: back to the review, back to the profile screen, choose a file. */
  async function chooseProfileFromReview(root: HTMLElement, file: File) {
    $(root, '[data-action="back"]').click();
    $(root, '[data-action="back"]').click();
    await chooseFile(root, file);
  }

  it("has no button to take an image again; back leads to the review with the images kept", async () => {
    const { app, root, fetch } = await toResults();
    expect(root.querySelectorAll("[data-retake]").length).toBe(0);
    $(root, '[data-action="back"]').click();
    expect(screenName(root)).toBe("review");
    expect(document.activeElement).toBe($(root, "h1"));
    expect(app.session.images.frontal).not.toBeNull();
    expect(root.querySelectorAll("[data-retake]").length).toBe(0);
    expect(fetch.mock.calls.filter((c) => c[0] === "/analyze").length).toBe(1); // not sent again by itself
  });

  it("changes only the frontal image by going back, keeping the profile", async () => {
    const { app, root } = await toResults();
    await chooseProfileFromReview(root, fakeFile(JPEG_HEAD, 3000)); // add the profile that was skipped
    expect(screenName(root)).toBe("review");
    $(root, '[data-action="back"]').click(); // review -> profile
    expect(screenName(root)).toBe("profile");
    $(root, '[data-action="back"]').click(); // profile -> frontal
    expect(screenName(root)).toBe("frontal");
    await chooseFile(root, fakeFile(JPEG_HEAD, 2000));
    expect(screenName(root)).toBe("profile");
    $(root, '[data-action="keep"]').click();
    expect(screenName(root)).toBe("review");
    expect(app.session.images.frontal!.blob!.size).toBe(2000);
    expect(app.session.images.profile!.blob!.size).toBe(3000);
  });

  it("offers to keep an image only when there is one", async () => {
    const { app, root } = makeApp();
    await app.start();
    fillHome(root);
    expect(root.querySelector('[data-action="keep"]')).toBeNull(); // no frontal image yet
    await chooseFile(root, fakeFile(JPEG_HEAD));
    expect(screenName(root)).toBe("profile");
    expect(root.querySelector('[data-action="keep"]')).toBeNull(); // no profile image yet
    $(root, '[data-action="back"]').click();
    expect(screenName(root)).toBe("frontal");
    expect($(root, '[data-action="keep"]').textContent).toBe(th("ui.capture.keep"));
    $(root, '[data-action="keep"]').click();
    expect(screenName(root)).toBe("profile");
  });

  it("starts over with nothing kept", async () => {
    const urls = fakeUrlApi();
    const fetch = apiFetch(jsonResponse(200, HEALTH), jsonResponse(200, RESULT));
    const { app, root } = makeApp({ fetch, urlApi: urls });
    await app.start();
    await toReview(root);
    $(root, '[data-action="send"]').click();
    await settle();
    $(root, '[data-action="restart"]').click();
    expect(screenName(root)).toBe("home");
    expect(app.session.result).toBeNull();
    expect(app.session.images.frontal).toBeNull();
    expect(urls.revokeObjectURL).toHaveBeenCalledWith("blob:test/1");
    expect(($(root, 'input[name="age"]') as HTMLInputElement).value).toBe("");
  });
});

describe("languages", () => {
  const EN = { "language.name": "English", "ui.home.title": "Compare face proportions", disclaimer: "Not a diagnosis" };

  it("shows the picker only when there is more than one language", async () => {
    const one = makeApp({ languages: { th: TH } });
    await one.app.start();
    expect(one.root.querySelector(".lang-picker")).toBeNull();
    const two = makeApp({ languages: { th: TH, en: EN } });
    await two.app.start();
    const picker = $(two.root, ".lang-picker");
    expect(picker.previousElementSibling!.classList.contains("greek-key")).toBe(true); // under the Greek key band
    const select = $(picker, "select") as HTMLSelectElement;
    expect([...select.options].map((o) => o.value)).toEqual(["th", "en"]);
    expect(two.root.querySelector(`label[for="${select.id}"]`)).not.toBeNull();
  });

  it("shows the screens before the results with no Thai in English, apart from the picker", async () => {
    const THAI = /[\u0E00-\u0E7F]/; // code points, so this file has no Thai letters
    const thaiOutsidePicker = (root: HTMLElement) => {
      const copy = root.cloneNode(true) as HTMLElement;
      copy.querySelector(".lang-picker")?.remove();
      const attrs = [...copy.querySelectorAll("*")].flatMap((e) => ["aria-label", "alt", "title"].map((a) => e.getAttribute(a) ?? ""));
      return [copy.textContent ?? "", ...attrs].filter((t) => THAI.test(t));
    };
    const { app, root } = makeApp({ languages: BUNDLED_LANGUAGES }, "?lang=en");
    await app.start();
    const seen: string[] = [];
    const look = () => {
      seen.push(screenName(root) ?? "");
      expect(thaiOutsidePicker(root), screenName(root)).toEqual([]);
    };
    look();
    fillHome(root);
    look();
    await chooseFile(root, fakeFile(JPEG_HEAD));
    look();
    $(root, '[data-action="skip-profile"]').click();
    look();
    expect(seen).toEqual(["home", "frontal", "profile", "review"]);
  });

  it("takes ?lang= and keeps Thai for the keys the language lacks", async () => {
    const { app, root } = makeApp({ languages: { th: TH, en: EN } }, "?lang=en");
    await app.start();
    expect($(root, "h1").textContent).toBe("Compare face proportions");
    expect($(root, "button[type=submit]").textContent).toBe(th("ui.home.start"));
    expect(document.documentElement.lang).toBe("en");
  });

  it("analyses the same images again when the language changes after a result", async () => {
    const fetch = apiFetch(jsonResponse(200, HEALTH), jsonResponse(200, RESULT));
    const { app, root } = makeApp({ fetch, languages: { th: TH, en: EN } });
    await app.start();
    await toReview(root);
    $(root, '[data-action="send"]').click();
    await settle();
    expect(screenName(root)).toBe("results");
    const select = $(root, ".lang-picker select") as HTMLSelectElement;
    select.value = "en";
    select.dispatchEvent(new Event("change"));
    await settle();
    const calls = fetch.mock.calls.filter((c) => c[0] === "/analyze");
    expect(calls.length).toBe(2);
    expect(JSON.parse((calls[1][1]!.body as FormData).get("meta") as string).lang).toBe("en");
    expect(screenName(root)).toBe("results");
    expect(app.session.lang).toBe("en");
  });
});
