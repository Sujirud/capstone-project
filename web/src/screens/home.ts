// Home (web.md sections 1.1 and 2): what the system does, where the images are
// processed, that the result is not a diagnosis, and the age and sex, both required.
// There is no consent step. Each field shows its own error under it.
import type { App } from "../app.ts";
import { el, uid } from "../dom.ts";
import { SEXES, type Sex } from "../types.ts";
import type { Screen } from "./index.ts";

export const MAX_AGE = 120; // api.md section 2.1

/** A whole number of years 0-120, or null. */
export function parseAge(text: string): number | null {
  const trimmed = text.trim();
  if (!/^\d{1,3}$/.test(trimmed)) return null;
  const age = Number(trimmed);
  return age <= MAX_AGE ? age : null;
}

const SEX_TEXT: Record<Sex, string> = { female: "sex.female", male: "sex.male" };

/** The red mark of a required field; the inputs carry `required` for screen readers. */
export function requiredMark(): HTMLElement {
  return el("span", { class: "required-mark", text: "*", attrs: { "aria-hidden": "true" } });
}

/** A field's error line, empty until the field is checked. */
function errorLine(id: string): HTMLElement {
  return el("p", { class: "field__error", attrs: { id } });
}

function setError(input: HTMLElement[], line: HTMLElement, text: string | null): void {
  line.textContent = text ?? "";
  for (const i of input) i.setAttribute("aria-invalid", String(text !== null));
}

export function homeScreen(app: App): Screen {
  const t = app.t;
  const s = app.session;
  const heading = el("h1", { text: t("ui.home.title"), attrs: { tabindex: -1 } });

  const ageId = uid("age");
  const ageHint = uid("age-hint");
  const ageError = uid("age-error");
  const age = el("input", {
    class: "input input--age",
    attrs: {
      id: ageId,
      name: "age",
      type: "number",
      inputmode: "numeric",
      min: 0,
      max: MAX_AGE,
      step: 1,
      required: true,
      autocomplete: "off",
      "aria-describedby": `${ageHint} ${ageError}`,
    },
  });
  if (s.age !== null) age.value = String(s.age);
  const ageLine = errorLine(ageError);

  const sexError = uid("sex-error");
  const sexLine = errorLine(sexError);
  const radios = SEXES.map((sex) => {
    const id = uid(`sex-${sex}`);
    const input = el("input", { attrs: { type: "radio", name: "sex", value: sex, id, required: true, "aria-describedby": sexError } });
    input.checked = s.sex === sex;
    return { sex, input, field: el("div", { class: "segmented__option" }, [input, el("label", { text: t(SEX_TEXT[sex]), attrs: { for: id } })]) };
  });
  const radioInputs = radios.map((r) => r.input);

  const start = el("button", { class: "btn btn--primary", text: t("ui.home.start"), attrs: { type: "submit" } });
  const form = el("form", { class: "panel form", attrs: { novalidate: true } }, [
    el("p", { class: "form__note" }, [requiredMark(), document.createTextNode(` ${t("ui.form.required_note")}`)]),
    el("div", { class: "field" }, [
      el("label", { class: "field__label", attrs: { for: ageId } }, [document.createTextNode(t("ui.home.age")), requiredMark()]),
      age,
      el("p", { class: "field__hint", text: t("ui.home.age_hint"), attrs: { id: ageHint } }),
      ageLine,
    ]),
    el("fieldset", { class: "field", attrs: { "aria-describedby": sexError } }, [
      el("legend", { class: "field__label" }, [document.createTextNode(t("ui.home.sex")), requiredMark()]),
      el("div", { class: "segmented" }, radios.map((r) => r.field)),
      sexLine,
    ]),
    el("div", { class: "form__actions" }, [start]),
  ]);

  // an error goes away as soon as its field is filled in correctly
  age.addEventListener("input", () => {
    if (ageLine.textContent && parseAge(age.value) !== null) setError([age], ageLine, null);
  });
  for (const r of radioInputs) r.addEventListener("change", () => setError(radioInputs, sexLine, null));

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const years = parseAge(age.value);
    const sex = radios.find((r) => r.input.checked)?.sex ?? null;
    setError([age], ageLine, years === null ? t("ui.home.age_invalid") : null);
    setError(radioInputs, sexLine, sex === null ? t("ui.home.sex_missing") : null);
    if (years === null || sex === null) {
      (years === null ? age : radioInputs[0]).focus();
      return;
    }
    s.age = years;
    s.sex = sex;
    app.show("frontal");
  });

  const element = el("section", { class: "page page--narrow home", attrs: { "data-screen": "home" } }, [
    el("header", { class: "page__head" }, [heading, el("p", { class: "lead", text: t("ui.home.lead") })]),
    el("ul", { class: "points" }, [
      el("li", { text: t("ui.home.point_regions") }),
      el("li", { text: t("ui.home.point_not_diagnosis") }),
      el("li", { text: t("ui.home.point_local") }),
    ]),
    form,
  ]);
  return { element, heading };
}
