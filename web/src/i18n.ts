// Texts from the language files shared/i18n/<code>.json, the same files the api uses
// (web.md section 8.4). Every file is bundled at build time, so adding a language is
// adding a file and building again. Thai is the main language: a key that another
// language lacks falls back to Thai, and a key missing from Thai as well shows the key
// itself with a warning in the console. Placeholders are {name}, as in i18n.py.

export type Texts = Record<string, string>;
export type Params = Record<string, string | number>;

export const DEFAULT_LANG = "th";
const PLACEHOLDER = /\{([A-Za-z_][A-Za-z0-9_]*)\}/g;
const LANG_CODE = /^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$/; // the pattern of meta.lang (api/schemas.py)

const FILES = import.meta.glob<Texts>("../../shared/i18n/*.json", { eager: true, import: "default" });

/** Language code -> texts, from the paths of the bundled files. */
export function languagesFromFiles(files: Record<string, Texts>): Record<string, Texts> {
  const out: Record<string, Texts> = {};
  for (const [path, texts] of Object.entries(files)) {
    const code = path.split("/").pop()!.replace(/\.json$/, "");
    if (LANG_CODE.test(code)) out[code] = texts;
  }
  return out;
}

export function placeholders(text: string): string[] {
  return [...text.matchAll(PLACEHOLDER)].map((m) => m[1]).sort();
}

export class I18n {
  readonly languages: Record<string, Texts>;
  private current: string = DEFAULT_LANG;
  private formats = new Map<string, Intl.NumberFormat>();
  private warned = new Set<string>();

  constructor(languages: Record<string, Texts>, lang?: string | null) {
    if (!(DEFAULT_LANG in languages)) throw new Error(`no ${DEFAULT_LANG}.json in shared/i18n`);
    this.languages = languages;
    this.current = this.resolve(lang);
  }

  /** Codes of the languages, Thai first. */
  get codes(): string[] {
    return [DEFAULT_LANG, ...Object.keys(this.languages).filter((c) => c !== DEFAULT_LANG).sort()];
  }

  get lang(): string {
    return this.current;
  }

  /** lang if it has a file, else Thai. */
  resolve(lang: string | null | undefined): string {
    return lang && lang in this.languages ? lang : DEFAULT_LANG;
  }

  /** Switch the language; returns the one in use. */
  setLang(lang: string | null | undefined): string {
    this.current = this.resolve(lang);
    return this.current;
  }

  /** The name of a language in that language (language.name). */
  nameOf(code: string): string {
    return this.languages[code]?.["language.name"] ?? this.languages[DEFAULT_LANG]["language.name"] ?? code;
  }

  has(key: string): boolean {
    return key in this.languages[DEFAULT_LANG];
  }

  /** The text of key in the current language, with each {name} replaced. */
  t(key: string, params: Params = {}): string {
    const template = this.languages[this.current]?.[key] ?? this.languages[DEFAULT_LANG][key];
    if (template === undefined) {
      this.warn(`missing text ${key}`);
      return key;
    }
    return template.replace(PLACEHOLDER, (whole, name: string) => {
      if (!(name in params)) {
        this.warn(`${key} needs the value {${name}}`);
        return whole;
      }
      return String(params[name]);
    });
  }

  /** A number with exactly digits decimals, in the current language. */
  number(value: number, digits: number): string {
    return this.format(digits, digits).format(value);
  }

  /** A number with at most digits decimals and no trailing zeros, such as 83 or 1.15. */
  numberUpTo(value: number, digits: number): string {
    return this.format(0, digits).format(value);
  }

  private format(min: number, max: number): Intl.NumberFormat {
    const id = `${this.current}:${min}:${max}`;
    let format = this.formats.get(id);
    if (!format) {
      format = new Intl.NumberFormat(this.current, { minimumFractionDigits: min, maximumFractionDigits: max });
      this.formats.set(id, format);
    }
    return format;
  }

  /** Items joined as a list in the current language, such as "a, b and c". */
  list(items: string[]): string {
    return new Intl.ListFormat(this.current, { style: "long", type: "conjunction" }).format(items);
  }

  private warn(message: string): void {
    if (this.warned.has(message)) return;
    this.warned.add(message);
    console.warn(`i18n: ${message}`);
  }
}

/** The language of ?lang=, or null. */
export function langFromSearch(search: string, param = "lang"): string | null {
  return new URLSearchParams(search).get(param);
}

/** Set <html lang> to the language in use. */
export function applyDocumentLang(i18n: I18n, doc: Document = document): void {
  doc.documentElement.lang = i18n.lang;
}

export const BUNDLED_LANGUAGES = languagesFromFiles(FILES);
