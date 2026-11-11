// The full-screen view of one image with its lines (web.md section 5.1): a modal
// <dialog> with the whole image, not cut by the arch, as large as the screen allows,
// and the same legend as the figure, sharing which lines are hidden.
//
// The line width is set once when it opens, for LINE_PX at the size the image is shown
// then. If the window changes while it is open, the lines scale with the image, as they
// do on the page.
import { el, uid } from "../dom.ts";
import type { RegionResult } from "../types.ts";
import { IMAGE_TEXT, type LineState, type ShownImage, legendList, lineWidth, overlayLayer } from "./overlay.ts";

export interface ViewerOptions {
  image: ShownImage;
  /** The regions whose lines are on this image. */
  drawn: RegionResult[];
  state: LineState;
  /** The element the dialog goes in while it is open, as its first child. */
  host: HTMLElement;
  /** Takes the focus back when the dialog closes. */
  opener: HTMLElement;
}

/** The width an element is shown at, in CSS pixels, or fallback before there is a layout. */
export function shownWidth(element: Element, fallback: number): number {
  const width = element.getBoundingClientRect().width;
  return width > 0 ? width : fallback;
}

export function openViewer({ image, drawn, state, host, opener }: ViewerOptions): HTMLDialogElement {
  const { i18n, name } = state;
  const titleId = uid("viewer-title");
  const frame = el("div", { class: "viewer__frame" }, [el("img", { class: "viewer__image", attrs: { src: image.url, alt: "", decoding: "async" } })]);
  frame.style.setProperty("--ratio", String(image.width / image.height));
  const close = el("button", { class: "btn btn--plain", text: i18n.t("ui.overlay.close"), attrs: { type: "button", "data-action": "viewer-close" } });
  const legend = state.groups.length ? legendList(state) : null;
  const dialog = el("dialog", { class: "viewer", attrs: { "aria-labelledby": titleId, "data-viewer": name } }, [
    el("div", { class: "viewer__head" }, [el("h2", { class: "viewer__title", text: i18n.t(IMAGE_TEXT[name]), attrs: { id: titleId } }), close]),
    el("div", { class: "viewer__stage" }, [frame]),
    legend,
  ]);
  host.prepend(dialog); // a <figcaption> stays the last child of its <figure>
  dialog.showModal();

  // the frame has its size once the dialog is open
  const layer = overlayLayer(name, image, drawn, lineWidth(image.width, shownWidth(frame, image.width)));
  frame.append(layer);
  state.attach(layer, legend);

  close.addEventListener("click", () => dialog.close());
  dialog.addEventListener(
    "close",
    () => {
      state.detach(layer, legend);
      state.point(null);
      dialog.remove();
      opener.focus();
    },
    { once: true },
  );
  return dialog;
}
