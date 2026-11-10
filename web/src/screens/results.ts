// The results screen (web.md sections 2 and 5): the result of this round on the images
// that were sent, "back" to the review to change an image, and "start over".
import type { App } from "../app.ts";
import { button, el } from "../dom.ts";
import { resultBody } from "../results/view.ts";
import type { Screen } from "./index.ts";
import { backButton } from "./parts.ts";

export function resultsScreen(app: App): Screen {
  const t = app.t;
  const s = app.session;
  const heading = el("h1", { text: t("ui.results.title"), attrs: { tabindex: -1 } });
  const element = el("section", { class: "page results", attrs: { "data-screen": "results" } }, [
    el("header", { class: "page__head" }, [backButton(app, () => app.show("review")), heading]),
  ]);
  const result = s.result;
  if (!result) {
    element.append(el("p", { text: t("ui.results.none") }));
  } else {
    const images = {
      frontal: s.images.frontal ? { ...s.images.frontal } : null,
      profile: s.images.profile ? { ...s.images.profile } : null,
    };
    element.append(
      ...resultBody(result, {
        i18n: app.i18n,
        images,
        copy: app.deps.copy,
        lab: { frontalValues: s.images.frontal?.source === "camera" ? s.frontalValues : null },
      }),
    );
  }
  element.append(
    el("div", { class: "actions actions--end" }, [
      button(t("ui.results.restart"), () => app.restart(), { class: "btn btn--primary", attrs: { "data-action": "restart" } }),
    ]),
  );
  return { element, heading };
}
