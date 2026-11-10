// The build settings: the CSP only in the build, the proxy to 127.0.0.1,
// the shared folder, the fonts, the theme colours of web.md section 6.1, and no demo
// code in web/dist.
import { existsSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { CONTENT_SECURITY_POLICY, cspPlugin, withPolicy } from "../build/csp.ts";
import config from "../vite.config.ts";
import { FIXTURES, ROOT, WEB, filesUnder, readText } from "./helpers.ts";

describe("Content-Security-Policy", () => {
  it("is the policy with the hosts of MediaPipe", () => {
    expect(CONTENT_SECURITY_POLICY).toBe(
      "default-src 'self'; script-src 'self' 'wasm-unsafe-eval' https://cdn.jsdelivr.net; img-src 'self' blob: data:; style-src 'self' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; connect-src 'self' https://cdn.jsdelivr.net https://storage.googleapis.com",
    );
  });

  it("goes into index.html at build time only, before the fonts it governs", () => {
    const plugin = cspPlugin();
    expect(plugin.apply).toBe("build");
    const html = withPolicy(readText(join(WEB, "index.html")));
    const meta = `<meta http-equiv="Content-Security-Policy" content="${CONTENT_SECURITY_POLICY}" />`;
    expect(html).toContain(meta);
    expect(html.indexOf(meta)).toBeGreaterThan(html.indexOf("<meta charset"));
    expect(html.indexOf(meta)).toBeLessThan(html.indexOf("fonts.googleapis.com"));
  });
});

describe("vite.config.ts", () => {
  it("proxies /analyze and /health to 127.0.0.1:8000, not localhost", () => {
    expect(config.server?.proxy).toEqual({ "/analyze": "http://127.0.0.1:8000", "/health": "http://127.0.0.1:8000" });
  });

  it("lets the dev server read shared/", () => {
    const allow = (config.server?.fs?.allow ?? []).map((p) => p.replace(/[\\/]+$/, ""));
    expect(allow).toContain(join(ROOT, "shared"));
  });

  it("builds into web/dist", () => {
    expect(config.build?.outDir).toBe("dist");
  });
});

describe("index.html", () => {
  const html = readText(join(WEB, "index.html"));

  it("loads the three fonts of web.md section 6.2 from Google Fonts with display=swap", () => {
    expect(html).toContain('<link rel="preconnect" href="https://fonts.googleapis.com" />');
    expect(html).toContain('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />');
    for (const family of ["Trirong:wght@500", "Cormorant+Garamond:ital", "Noto+Sans+Thai:wght@400;500"]) expect(html).toContain(family);
    expect(html).toContain("display=swap");
  });

  it("has no CSP of its own, since the dev server needs inline scripts", () => {
    expect(html).not.toContain("Content-Security-Policy");
  });
});

describe("theme", () => {
  const tokens = readText(join(WEB, "src", "styles", "tokens.css"));
  const components = readText(join(WEB, "src", "styles", "components.css"));

  /** `--name` -> `#VALUE` of the colour table of web.md section 6.1. */
  function docColours(): Record<string, string> {
    const doc = readText(join(ROOT, "docs", "web.md"));
    const section = doc.split("\n### 6.1", 2)[1].split("\n### ", 1)[0];
    return Object.fromEntries([...section.matchAll(/^\| `(--[a-z-]+)` \| `(#[0-9A-Fa-f]{6})` \|/gm)].map((m) => [m[1], m[2].toUpperCase()]));
  }

  it("has every colour of web.md section 6.1, and no other colour but white", () => {
    const doc = docColours();
    expect(Object.keys(doc).length).toBeGreaterThanOrEqual(11);
    for (const [name, value] of Object.entries(doc)) {
      const m = tokens.match(new RegExp(`${name}:\\s*(#[0-9A-Fa-f]{6})`));
      expect(m?.[1].toUpperCase(), name).toBe(value);
    }
    const inTokens = [...tokens.matchAll(/(--[a-z-]+):\s*#[0-9A-Fa-f]{6}/g)].map((m) => m[1]);
    expect(inTokens.filter((name) => !(name in doc))).toEqual(["--white"]);
    expect(components).not.toMatch(/#[0-9A-Fa-f]{3,6}\b/); // colours come from tokens.css only
  });

  it("draws every border and rule 1px wide", () => {
    const widths = [...components.matchAll(/border(?:-(?:top|right|bottom|left))?(?:-width)?:\s*([^;]+);/g)].map((m) => m[1]);
    for (const value of widths) {
      for (const size of value.match(/\d+(?:\.\d+)?px/g) ?? []) expect(size, value).toBe("1px");
    }
    expect(tokens).toMatch(/--border:\s*1px/);
  });

  it("uses red only for form errors and the mark of required fields", () => {
    const rules = [...components.matchAll(/([^{}]+)\{([^{}]*)\}/g)].filter((m) => m[2].includes("var(--error)"));
    expect(rules.length).toBeGreaterThan(0);
    for (const [, selector] of rules) expect(selector.trim(), selector).toMatch(/^(\.required-mark|\.field__error|\.input\[aria-invalid="true"\]|\.segmented:has\(\[aria-invalid="true"\]\))$/);
  });

  it("uses --cocoa-soft only on rules without a --sky or --blush background", () => {
    const rules = [...components.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({ selector: m[1].trim(), body: m[2] }));
    const soft = rules.filter((r) => r.body.includes("var(--cocoa-soft)"));
    expect(soft.length).toBeGreaterThan(0);
    const onColour = new Set(rules.filter((r) => /background[^;]*var\(--(sky|blush)\)/.test(r.body)).map((r) => r.selector));
    for (const r of soft) {
      expect(/background[^;]*var\(--(sky|blush)\)/.test(r.body), r.selector).toBe(false);
      for (const selector of onColour) expect(r.selector.includes(selector), `${r.selector} inside ${selector}`).toBe(false);
    }
  });

  it("makes buttons at least 42 px high with an --ink-blue focus ring off their edge", () => {
    expect(Number(tokens.match(/--button-height:\s*(\d+)px/)![1])).toBeGreaterThanOrEqual(42); // web.md section 6.3
    const offset = Number(tokens.match(/--focus-offset:\s*(\d+)px/)![1]);
    expect(offset).toBeGreaterThanOrEqual(2);
    expect(components).toMatch(/:focus-visible\s*\{[^}]*outline:[^;]*var\(--ink-blue\)[^}]*outline-offset:\s*var\(--focus-offset\)/);
    expect(components).toMatch(/\.btn\s*\{[^}]*min-height:\s*var\(--button-height\)/);
  });

  it("keeps text at 14 px or more", () => {
    for (const m of [...tokens.matchAll(/--size-[a-z0-9]+:\s*(\d+)px/g), ...components.matchAll(/font-size:\s*(\d+)px/g)]) {
      expect(Number(m[1])).toBeGreaterThanOrEqual(14);
    }
  });
});

describe("web/dist", () => {
  const dist = join(WEB, "dist");

  it.skipIf(!existsSync(dist))("has neither the demo code nor the example results", () => {
    const text = filesUnder(dist, [".js", ".html", ".css"]).map(readText).join("\n");
    expect(text).not.toContain("tests/fixtures");
    for (const f of Object.values(FIXTURES)) expect(text).not.toContain(f.result.request_id);
    expect(text).toContain("Content-Security-Policy");
  });
});
