// Images chosen from the computer (web.md section 3.4).
//
// - The type comes from the first bytes, not from the name or the type the browser
//   reports, the same way the api checks it: only JPEG and PNG.
// - createImageBitmap with imageOrientation "from-image" gives the size after the EXIF
//   rotation, which is the frame of the coordinates in the api's result.
// - When the images together are over the request size, the largest ones are
//   re-encoded as JPEG 0.92 with the long side at most 4032 px, from the rotated
//   bitmap. The page then shows the re-encoded image, since its coordinates are the
//   ones the api returns.
import { IMAGE_BUDGET_BYTES, SHRINK_JPEG_QUALITY, SHRINK_MAX_SIDE_PX, SNIFF_BYTES } from "./constants.ts";
import type { ImageName } from "./types.ts";

export type ImageType = "image/jpeg" | "image/png";

const JPEG_MAGIC = [0xff, 0xd8, 0xff];
const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

function startsWith(bytes: Uint8Array, magic: number[]): boolean {
  return bytes.length >= magic.length && magic.every((b, i) => bytes[i] === b);
}

/** JPEG or PNG from the first bytes of a file, or null for anything else. */
export function sniff(head: Uint8Array): ImageType | null {
  if (startsWith(head, JPEG_MAGIC)) return "image/jpeg";
  if (startsWith(head, PNG_MAGIC)) return "image/png";
  return null;
}

export async function readHead(blob: Blob, count = SNIFF_BYTES): Promise<Uint8Array> {
  return new Uint8Array(await blob.slice(0, count).arrayBuffer());
}

export interface Size {
  width: number;
  height: number;
}

/** Decoding and re-encoding, done by the browser; tests pass their own. */
export interface ImageTools {
  /** Size after the EXIF rotation. */
  size(blob: Blob): Promise<Size>;
  /** The rotated image drawn at size and encoded as JPEG SHRINK_JPEG_QUALITY. */
  encodeJpeg(blob: Blob, size: Size): Promise<Blob>;
}

/** size scaled down so that its long side is at most maxSide; never scaled up. */
export function fitWithin(size: Size, maxSide = SHRINK_MAX_SIDE_PX): Size {
  const long = Math.max(size.width, size.height);
  if (long <= maxSide) return { width: size.width, height: size.height };
  const scale = maxSide / long;
  return { width: Math.max(1, Math.round(size.width * scale)), height: Math.max(1, Math.round(size.height * scale)) };
}

export type LoadOutcome =
  | { ok: true; blob: Blob; type: ImageType; width: number; height: number }
  | { ok: false; reason: "not_jpeg_png" | "unreadable" };

/** Check and decode a chosen file. A file whose bytes say JPEG or PNG is sent as it is. */
export async function loadUpload(file: Blob, tools: ImageTools): Promise<LoadOutcome> {
  let type: ImageType | null;
  try {
    type = sniff(await readHead(file));
  } catch {
    return { ok: false, reason: "unreadable" };
  }
  if (type === null) return { ok: false, reason: "not_jpeg_png" };
  try {
    const { width, height } = await tools.size(file);
    const blob = file.type === type ? file : new Blob([file], { type });
    return { ok: true, blob, type, width, height };
  } catch {
    return { ok: false, reason: "unreadable" };
  }
}

export interface SizedImage {
  name: ImageName;
  bytes: number;
  shrunk: boolean;
}

/**
 * Which images to re-encode so that all of them fit in budget bytes: the largest
 * image not yet re-encoded first, until the total fits, counting a re-encoded image
 * as small. An empty list when they fit. The caller re-encodes the first one and asks
 * again with its real size.
 */
export function chooseToShrink(images: SizedImage[], budget = IMAGE_BUDGET_BYTES): ImageName[] {
  let total = images.reduce((sum, im) => sum + im.bytes, 0);
  const out: ImageName[] = [];
  const candidates = images.filter((im) => !im.shrunk).sort((a, b) => b.bytes - a.bytes);
  for (const im of candidates) {
    if (total <= budget) break;
    out.push(im.name);
    total -= im.bytes; // its new size is not known before encoding; at least it is no longer the problem
  }
  return out;
}

export interface Shrunk {
  blob: Blob;
  width: number;
  height: number;
}

/** Re-encode an image as JPEG with its long side at most SHRINK_MAX_SIDE_PX. */
export async function shrinkImage(blob: Blob, tools: ImageTools): Promise<Shrunk> {
  const size = fitWithin(await tools.size(blob));
  return { blob: await tools.encodeJpeg(blob, size), ...size };
}

async function decode(blob: Blob): Promise<ImageBitmap> {
  return createImageBitmap(blob, { imageOrientation: "from-image" });
}

export const browserImageTools: ImageTools = {
  async size(blob) {
    const bitmap = await decode(blob);
    try {
      return { width: bitmap.width, height: bitmap.height };
    } finally {
      bitmap.close();
    }
  },
  async encodeJpeg(blob, size) {
    const bitmap = await decode(blob);
    try {
      const canvas = document.createElement("canvas");
      canvas.width = size.width;
      canvas.height = size.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("no 2d canvas");
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(bitmap, 0, 0, size.width, size.height);
      const out = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", SHRINK_JPEG_QUALITY));
      canvas.width = canvas.height = 0; // let the pixels go at once
      if (!out) throw new Error("the canvas gave no JPEG");
      return out;
    } finally {
      bitmap.close();
    }
  },
};
