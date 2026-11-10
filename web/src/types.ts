// Types of the requests and answers of the api (api.md sections 2.1-2.3), checked against
// /openapi.json of the api (api/schemas.py). The code lists mirror facemeasure/codes.py;
// tests/texts.test.ts compares them with tests/fixtures/codes.json.

export const REGIONS = ["jaw", "chin", "undereye_left", "undereye_right", "nose_base", "nose_dorsum"] as const;
export type Region = (typeof REGIONS)[number];
export const FRONTAL_REGIONS = ["jaw", "undereye_left", "undereye_right", "nose_base"] as const satisfies readonly Region[];
export const PROFILE_REGIONS = ["chin", "nose_dorsum"] as const satisfies readonly Region[];

export const IMAGES = ["frontal", "profile"] as const;
export type ImageName = (typeof IMAGES)[number];

export const SEXES = ["female", "male"] as const;
export type Sex = (typeof SEXES)[number];
export const FRONTAL_SOURCES = ["camera", "upload"] as const;
export type ImageSource = (typeof FRONTAL_SOURCES)[number];

export const MEASURES = ["facial_taperness", "legan_angle", "alar_intercanthal_ratio", "nasofacial_angle"] as const;
export type Measure = (typeof MEASURES)[number];
export type Unit = "%" | "deg" | "ratio";
export type Method = "model" | "measured";

export const FRONTAL_CODES = [
  "no_face",
  "multiple_faces",
  "face_too_small",
  "frontal_pose",
  "smiling",
  "eyes_closed",
  "hair_occlusion",
] as const;
export type FrontalCode = (typeof FRONTAL_CODES)[number];
export const PROFILE_CODES = ["profile_invalid", "missing_input"] as const;
export type ProfileCode = (typeof PROFILE_CODES)[number];
export type QualityCode = FrontalCode | ProfileCode;
export const STATUS_CODES = [...FRONTAL_CODES, ...PROFILE_CODES, "model_not_ready"] as const;
export type StatusCode = (typeof STATUS_CODES)[number];

export const PROCEDURES = [
  "botox_masseter",
  "jaw_reduction",
  "chin_filler",
  "chin_implant",
  "genioplasty",
  "undereye_filler",
  "alar_reduction",
  "nose_filler",
  "rhinoplasty_implant",
] as const;
export type ProcedureCode = (typeof PROCEDURES)[number];

export const NOTES = [
  "jaw_cause_unknown",
  "chin_check_occlusion",
  "chin_check_mandible",
  "undereye_scope",
  "nose_base_indications",
  "nose_filler_vision_risk",
] as const;
export type NoteCode = (typeof NOTES)[number];

export const WITHHELD = [
  "male_jaw_no_criterion",
  "distance_unknown",
  "age_under_18",
  "age_21_or_under",
  "no_procedure_at_grade",
] as const;
export type WithheldCode = (typeof WITHHELD)[number];

export const GRADES = ["0", "1", "2+"] as const; // jaw, under-eye and nose
export const CHIN_GRADES = ["0", "1", "2", "3"] as const;
export type Grade = (typeof GRADES)[number] | (typeof CHIN_GRADES)[number];
export type UndereyeGrade = (typeof GRADES)[number];

export type Status = "assessed" | "not_assessable";
export type Level = "clear" | "borderline";

// ---------------------------------------------------------------- request (api.md section 2.1)

export interface Meta {
  age: number; // whole years, 0-120
  sex: Sex;
  frontal_source?: ImageSource; // default "camera"
  lab?: boolean; // default false
  lang?: string; // default "th"
}

// ---------------------------------------------------------------- result (api.md section 2.2)

export interface Versions {
  api: string;
  model_bundle: string;
}

export interface RetakeItem {
  image: ImageName;
  code: QualityCode;
  regions: Region[];
  text: string;
}

export interface FrontalQuality {
  iod_px: number | null;
  yaw_deg: number | null;
  pitch_deg: number | null;
  roll_deg: number | null;
  smile: number | null;
  jaw_open: number | null;
  blink_left: number | null;
  blink_right: number | null;
  hair_jaw: number | null;
  hair_undereye_left: number | null;
  hair_undereye_right: number | null;
  failed: FrontalCode[];
}

export interface ProfileQuality {
  face_found: boolean | null;
  face_yaw_deg: number | null;
  silhouette_found: boolean | null;
  flipped: boolean | null;
  failed: ProfileCode[];
}

export interface Quality {
  frontal: FrontalQuality;
  profile: ProfileQuality;
}

export interface Cutoff {
  value: number; // in the unit of the measurement
  band: number; // half width of the borderline band around value
}

export interface Measurement {
  name: Measure;
  value: number;
  unit: Unit;
  method: Method;
  landmark_value: number | null; // shown in lab mode only
  cutoffs: Cutoff[];
  reference: string;
  source: string;
}

export type Point = [number, number];

export interface Overlay {
  image: ImageName;
  points: Record<string, Point>; // pixels of the image as sent; names end in _l and _r for image sides
  lines: [string, string][];
}

export interface Probabilities {
  grades: Partial<Record<UndereyeGrade, number>>;
  at_least: Partial<Record<"1" | "2+", number>>; // shown in lab mode only
}

export interface Decision {
  grade: Grade | null;
  grade_range: [Grade, Grade] | null;
  level: Level;
}

export interface Procedure {
  code: ProcedureCode;
  name: string;
  when: string;
  checks: string | null;
}

export interface Note {
  code: NoteCode;
  text: string;
  with_procedures: boolean;
}

export interface RegionResult {
  region: Region;
  status: Status;
  status_reasons: StatusCode[];
  warnings: StatusCode[];
  measurements: Measurement[];
  overlay: Overlay | null;
  probabilities: Probabilities | null;
  decision: Decision | null;
  procedures: Procedure[];
  procedures_withheld: WithheldCode[];
  notes: Note[];
}

export interface AnalyzeResult {
  request_id: string;
  versions: Versions;
  lab: boolean;
  retake: RetakeItem[];
  quality: Quality;
  regions: RegionResult[];
  disclaimer: string;
}

// ---------------------------------------------------------------- health

export interface HealthVersions {
  api: string;
  model_bundle: string;
  facemeasure: string;
  mediapipe: string;
  torch: string;
}

export interface Health {
  status: "ok";
  versions: HealthVersions;
  stub_heads: ("jaw" | "nose_base" | "undereye")[];
  device: string;
  languages: string[];
}
