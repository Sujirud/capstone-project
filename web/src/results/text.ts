// Texts of the results page from the codes and numbers of the api (web.md sections 5.2-5.4).
import { DIGITS } from "../constants.ts";
import type { I18n } from "../i18n.ts";
import { CHIN_GRADES, GRADES, type Decision, type Grade, type Measurement, type Region, type RegionResult, type Unit } from "../types.ts";

export type Badge = "clear" | "borderline" | "values_only" | "not_assessable";

/** The status badge of a region (the badge table of web.md section 5.2). */
export function badgeOf(region: RegionResult): Badge {
  if (region.status === "not_assessable") return "not_assessable";
  if (region.decision === null) return "values_only";
  return region.decision.level;
}

export const BADGE_TEXT: Record<Badge, string> = {
  clear: "level.clear",
  borderline: "level.borderline",
  values_only: "level.values_only",
  not_assessable: "level.not_assessable",
};

/** The grades of a region in order: "0" to "3" for the chin, "0", "1" and "2+" otherwise. */
export function gradesOf(region: Region): readonly Grade[] {
  return region === "chin" ? CHIN_GRADES : GRADES;
}

export function gradeText(region: Region, grade: Grade, i18n: I18n): string {
  return i18n.t(region === "chin" ? `grade.chin.${grade}` : `grade.default.${grade}`);
}

/** One grade with its meaning: ui.grade.line with the grade and grade.<...>.<grade>. */
export function gradeLine(region: Region, grade: Grade, i18n: I18n): string {
  return i18n.t("ui.grade.line", { grade, text: gradeText(region, grade, i18n) });
}

export interface LevelText {
  summary: string;
  /** For a borderline result, the meaning of every grade in the range, lowest first. */
  details: string[];
}

/**
 * What the decision means. Clear: the grade and its meaning. Borderline: "between grade
 * 0 and 1" and the meaning of each grade; a range over two cutpoints, such as the
 * under-eye 0 to 2+, says "from 0 to 2+" and lists every grade in it (web.md section 5.2).
 */
export function levelText(region: Region, decision: Decision | null, i18n: I18n): LevelText | null {
  if (decision === null) return null;
  if (decision.level === "clear" && decision.grade !== null) {
    return { summary: gradeLine(region, decision.grade, i18n), details: [] };
  }
  const range = decision.grade_range;
  if (range === null) return null;
  const order = gradesOf(region);
  const [low, high] = range;
  const grades = order.slice(order.indexOf(low), order.indexOf(high) + 1);
  const key = grades.length > 2 ? "grade.between_wide" : "grade.between";
  return { summary: i18n.t(key, { low, high }), details: grades.map((g) => gradeLine(region, g, i18n)) };
}

const UNIT_TEXT: Record<Unit, { key: string; digits: number }> = {
  "%": { key: "ui.unit.percent", digits: DIGITS.percent },
  deg: { key: "ui.unit.deg", digits: DIGITS.deg },
  ratio: { key: "ui.unit.ratio", digits: DIGITS.ratio },
};

/** A value with its unit: percent and degrees with one decimal, ratios with two. */
export function formatValue(value: number, unit: Unit, i18n: I18n): string {
  const u = UNIT_TEXT[unit];
  return i18n.t(u.key, { value: i18n.number(value, u.digits) });
}

/** A cutoff with its unit, without trailing zeros: 83% rather than 83.0%. */
export function formatCutoff(value: number, unit: Unit, i18n: I18n): string {
  const u = UNIT_TEXT[unit];
  return i18n.t(u.key, { value: i18n.numberUpTo(value, u.digits) });
}

/** The criterion and its source under the value, such as "<reference> [3]" (web.md section 5.2). */
export function criterionLine(m: Measurement, i18n: I18n): string {
  return i18n.t("ui.measure.criterion", { reference: m.reference, source: m.source });
}

/** A few words for the summary: the grade, the range, or the value when there is no grade. */
export function shortLevel(region: RegionResult, i18n: I18n): string {
  const d = region.decision;
  if (region.status !== "assessed") return "";
  if (d === null) {
    const m = region.measurements[0];
    return m ? formatValue(m.value, m.unit, i18n) : "";
  }
  if (d.level === "clear" && d.grade !== null) return i18n.t("ui.grade.short", { grade: d.grade });
  if (d.grade_range) return i18n.t("ui.grade.range_short", { low: d.grade_range[0], high: d.grade_range[1] });
  return "";
}

export function formatProbability(p: number, i18n: I18n): string {
  return i18n.number(p, DIGITS.probability);
}

/** k of P(grade >= k): the key "2+" of at_least is grade 2 and above. */
export function atLeastLabel(key: string): string {
  return key.replace("+", "");
}
