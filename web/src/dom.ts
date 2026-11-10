// Small helpers that build the page with createElement and textContent only; data from
// the api never goes through innerHTML.

type Attrs = Record<string, string | number | boolean | null | undefined>;

export interface ElOptions {
  class?: string;
  text?: string;
  attrs?: Attrs;
  lang?: string;
}

function setAttrs(node: Element, attrs: Attrs | undefined): void {
  if (!attrs) return;
  for (const [name, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    node.setAttribute(name, value === true ? "" : String(value));
  }
}

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  options: ElOptions = {},
  children: (Node | null | undefined | false)[] = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (options.class) node.className = options.class;
  if (options.text !== undefined) node.textContent = options.text;
  if (options.lang) node.lang = options.lang;
  setAttrs(node, options.attrs);
  for (const child of children) if (child) node.append(child);
  return node;
}

const SVG_NS = "http://www.w3.org/2000/svg";

export function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  children: (Node | null | undefined | false)[] = [],
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  setAttrs(node, attrs);
  for (const child of children) if (child) node.append(child);
  return node;
}

export function button(text: string, onClick: (event: MouseEvent) => void, options: ElOptions = {}): HTMLButtonElement {
  const node = el("button", { ...options, attrs: { type: "button", ...options.attrs } }, [el("span", { text })]);
  node.addEventListener("click", onClick);
  return node;
}

let counter = 0;

/** An id unique within the page, for aria-controls, aria-labelledby and labels. */
export function uid(prefix: string): string {
  counter += 1;
  return `${prefix}-${counter}`;
}
