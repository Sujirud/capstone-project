// What the page knows about this round, in memory only (web.md section 1.2): no Web
// Storage (local or session), IndexedDB, Cache API or cookies. Closing the tab or
// pressing "start over" (reset) drops every image.
//
// The language, lab mode and the chosen camera are settings of the page, from the URL,
// the language picker or the camera picker, so reset() keeps them; everything about the
// round is cleared.
import type { FrontalValues } from "./metrics.ts";
import type { AnalyzeResult, ImageName, ImageSource, Sex } from "./types.ts";

export interface ImageEntry {
  blob: Blob | null; // what is sent; null only for the plain pictures of the demo mode
  url: string; // object URL of blob, shown on the page
  width: number; // pixels after the EXIF rotation, the frame of the api's coordinates
  height: number;
  source: ImageSource;
  shrunk: boolean; // re-encoded to fit the request size
}

export interface UrlApi {
  createObjectURL(blob: Blob): string;
  revokeObjectURL(url: string): void;
}

export class Session {
  lang: string;
  lab: boolean;
  age: number | null = null;
  sex: Sex | null = null;
  images: Record<ImageName, ImageEntry | null> = { frontal: null, profile: null };
  profileSkipped = false;
  result: AnalyzeResult | null = null;
  cameraId: string | null = null; // the camera chosen in the picker; a setting, kept by reset()
  /** What the pose guide measured on the frontal photo when it was taken, for lab mode (web.md section 7). */
  frontalValues: FrontalValues | null = null;
  private urls = new Set<string>();
  private urlApi: UrlApi;

  constructor(settings: { lang: string; lab: boolean }, urlApi: UrlApi = URL) {
    this.lang = settings.lang;
    this.lab = settings.lab;
    this.urlApi = urlApi;
  }

  /** Keep an image of this round; the image it replaces and its URL are dropped. */
  setImage(name: ImageName, blob: Blob, info: Omit<ImageEntry, "blob" | "url">): ImageEntry {
    const url = this.urlApi.createObjectURL(blob);
    this.urls.add(url);
    return this.putImage(name, { ...info, blob, url });
  }

  /** An image shown from a URL this session does not own, such as a demo picture. */
  setShownImage(name: ImageName, url: string, info: Omit<ImageEntry, "blob" | "url">): ImageEntry {
    return this.putImage(name, { ...info, blob: null, url });
  }

  clearImage(name: ImageName): void {
    this.revoke(this.images[name]?.url);
    this.images[name] = null;
  }

  /** Every object URL this session made that is still alive. */
  get liveUrls(): string[] {
    return [...this.urls];
  }

  /** Forget the round: every object URL is revoked and every value cleared. */
  reset(): void {
    for (const url of this.urls) this.urlApi.revokeObjectURL(url);
    this.urls.clear();
    this.age = null;
    this.sex = null;
    this.images = { frontal: null, profile: null };
    this.profileSkipped = false;
    this.result = null;
    this.frontalValues = null;
  }

  private putImage(name: ImageName, entry: ImageEntry): ImageEntry {
    const old = this.images[name];
    if (old && old.url !== entry.url) this.revoke(old.url);
    this.images[name] = entry;
    if (name === "profile") this.profileSkipped = false;
    return entry;
  }

  private revoke(url: string | undefined): void {
    if (url && this.urls.delete(url)) this.urlApi.revokeObjectURL(url);
  }
}
