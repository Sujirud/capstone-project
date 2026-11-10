// A small accessibility audit run inside the page (web.md sections 6 and 9):
// the contrast of every visible piece of text against the colour behind it (WCAG AA), the
// smallest text (14 px, web.md section 6.2), the height of buttons (42 px, web.md
// section 6.3) and a name for every control. It does not replace Lighthouse or a screen
// reader; it catches the same kinds of problem on every screen of the end-to-end round.
import type { Page } from "@playwright/test";

export interface Finding {
  rule: "contrast" | "font-size" | "button-height" | "name";
  where: string;
  detail: string;
}

export const LIMITS = {
  contrastText: 4.5,
  contrastLarge: 3, // 24 px, or 18.66 px bold (WCAG 1.4.3)
  minFontPx: 14,
  minButtonPx: 42,
};

export async function audit(page: Page): Promise<Finding[]> {
  return page.evaluate((limits) => {
    type Rgba = [number, number, number, number];
    const findings: { rule: string; where: string; detail: string }[] = [];
    const parse = (c: string): Rgba => {
      const n = (c.match(/[\d.]+/g) ?? []).map(Number);
      return [n[0] ?? 0, n[1] ?? 0, n[2] ?? 0, n[3] ?? 1];
    };
    const over = (top: Rgba, under: Rgba): Rgba => {
      const a = top[3];
      return [top[0] * a + under[0] * (1 - a), top[1] * a + under[1] * (1 - a), top[2] * a + under[2] * (1 - a), 1];
    };
    const luminance = ([r, g, b]: Rgba) => {
      const ch = (v: number) => {
        const s = v / 255;
        return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
      };
      return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b);
    };
    const ratio = (a: Rgba, b: Rgba) => {
      const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
      return (hi + 0.05) / (lo + 0.05);
    };
    const where = (el: Element) => {
      const tag = el.tagName.toLowerCase();
      const cls = el.getAttribute("class")?.split(/\s+/).slice(0, 2).join(".") ?? "";
      const text = (el.textContent ?? "").trim().slice(0, 30);
      return `${tag}${cls ? "." + cls : ""} "${text}"`;
    };
    const media = Array.from(document.querySelectorAll("img, video, canvas")).map((m) => m.getBoundingClientRect());
    const overlaps = (a: DOMRect, b: DOMRect) => a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
    /** The colour behind el, or null when an image, a video or a gradient is behind it. */
    const backdrop = (el: Element): Rgba | null => {
      const rect = el.getBoundingClientRect();
      if (media.some((m) => m.width > 1 && m.height > 1 && overlaps(rect, m))) return null;
      const layers: Rgba[] = [];
      for (let e: Element | null = el; e; e = e.parentElement) {
        const s = getComputedStyle(e);
        if (s.backgroundImage !== "none") return null;
        const bg = parse(s.backgroundColor);
        if (bg[3] > 0) layers.push(bg);
        if (bg[3] >= 1) break;
      }
      let out: Rgba = [255, 255, 255, 1];
      for (const layer of layers.reverse()) out = over(layer, out);
      return out;
    };
    const hidden = (el: Element, rect: DOMRect) => {
      const s = getComputedStyle(el);
      return s.visibility === "hidden" || s.display === "none" || rect.width <= 1 || rect.height <= 1 || el.closest("[hidden], [aria-hidden='true']") !== null;
    };

    for (const el of Array.from(document.body.querySelectorAll("*"))) {
      const rect = el.getBoundingClientRect();
      if (hidden(el, rect) || el.closest("svg")) continue;
      const s = getComputedStyle(el);
      const ownText = Array.from(el.childNodes).some((n) => n.nodeType === Node.TEXT_NODE && (n.textContent ?? "").trim() !== "");
      if (ownText) {
        const size = parseFloat(s.fontSize);
        if (size < limits.minFontPx) findings.push({ rule: "font-size", where: where(el), detail: `${size}px` });
        const behind = backdrop(el);
        if (behind) {
          const fg = over(parse(s.color), behind);
          const large = size >= 24 || (size >= 18.66 && Number(s.fontWeight) >= 700);
          const r = ratio(fg, behind);
          const need = large ? limits.contrastLarge : limits.contrastText;
          if (r < need) findings.push({ rule: "contrast", where: where(el), detail: `${r.toFixed(2)} < ${need} (${s.color} on rgb(${behind.slice(0, 3).join(", ")}))` });
        }
      }
      if (el instanceof HTMLButtonElement && el.classList.contains("btn") && rect.height < limits.minButtonPx - 0.5) {
        findings.push({ rule: "button-height", where: where(el), detail: `${rect.height.toFixed(1)}px` });
      }
      const control = el.matches("button, input:not([type=hidden]), select, textarea, a[href], video");
      if (control) {
        const input = el as HTMLInputElement;
        const named =
          (el.getAttribute("aria-label") ?? "").trim() !== "" ||
          el.getAttribute("aria-labelledby") !== null ||
          (input.labels?.length ?? 0) > 0 ||
          (el.textContent ?? "").trim() !== "" ||
          (el.getAttribute("title") ?? "").trim() !== "";
        if (!named) findings.push({ rule: "name", where: where(el), detail: "no accessible name" });
      }
    }
    for (const img of Array.from(document.querySelectorAll("img"))) {
      if (!img.hasAttribute("alt")) findings.push({ rule: "name", where: where(img), detail: "img without alt" });
    }
    return findings;
  }, LIMITS) as Promise<Finding[]>;
}
