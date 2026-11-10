// Demo mode, while developing only: ?demo=<name> shows the results page of
// tests/fixtures/<name>.json on plain pictures of the image sizes in the file, and
// ?demo alone lists the files. main.ts loads this module only when import.meta.env.DEV
// is true, so neither the code nor the fixtures are in the build.
import type { App } from "./app.ts";
import { el } from "./dom.ts";
import type { AnalyzeResult, ImageName, ImageSource, Meta } from "./types.ts";

export interface Fixture {
  name: string;
  meta: Required<Meta>;
  images: Record<ImageName, { width: number; height: number; source: ImageSource } | null>;
  result: AnalyzeResult;
}

const FILES = import.meta.glob<Fixture>("../tests/fixtures/*.json", { eager: true, import: "default" });

export const FIXTURES: Record<string, Fixture> = Object.fromEntries(
  Object.values(FILES)
    .filter((f) => f && typeof f === "object" && "result" in f)
    .map((f) => [f.name, f]),
);

const PICTURE_FILL: Record<ImageName, string> = { frontal: "#D8C3B0", profile: "#CDB8A6" };

/** A plain picture of the given size, standing in for the image that was sent. */
export function plainPicture(name: ImageName, width: number, height: number): string {
  const markup = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><rect width="100%" height="100%" fill="${PICTURE_FILL[name]}"/></svg>`;
  return `data:image/svg+xml,${encodeURIComponent(markup)}`;
}

/** Put a fixture in the session as if it had just come back from the api. */
export function loadFixture(app: App, fixture: Fixture): void {
  const s = app.session;
  s.reset();
  app.demo = true;
  s.age = fixture.meta.age;
  s.sex = fixture.meta.sex;
  s.lab = fixture.result.lab;
  for (const name of ["frontal", "profile"] as const) {
    const image = fixture.images[name];
    if (image) s.setShownImage(name, plainPicture(name, image.width, image.height), { ...image, shrunk: false });
  }
  s.result = fixture.result;
}

function listScreen(app: App): void {
  const heading = el("h1", { text: app.t("ui.demo.title"), attrs: { tabindex: -1 } });
  const items = Object.keys(FIXTURES)
    .sort()
    .map((name) => el("li", {}, [el("a", { text: name, attrs: { href: `?demo=${encodeURIComponent(name)}` } })]));
  const main = app.root.querySelector("main");
  main?.replaceChildren(el("section", { class: "demo" }, [heading, el("p", { text: app.t("ui.demo.intro") }), el("ul", {}, items)]));
}

export function startDemo(app: App, name: string): void {
  app.mount();
  const fixture = FIXTURES[name];
  if (!fixture) {
    listScreen(app);
    return;
  }
  loadFixture(app, fixture);
  app.show("results");
}
