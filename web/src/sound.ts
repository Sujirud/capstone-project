// The beeps of the profile countdown (web.md section 3.3): the person who has turned to
// the side cannot see the screen, so each second and the photo have a sound. They are
// made with Web Audio; there are no sound files.
import { BEEP_GAIN, BEEP_SHOT, BEEP_TICK } from "./constants.ts";

export interface Beeper {
  /** Called from the click that starts the countdown, so the browser lets the page play sound. */
  prepare(): void;
  tick(): void;
  shot(): void;
}

type AudioContextClass = typeof AudioContext;

export class WebAudioBeeper implements Beeper {
  private ctx: AudioContext | null = null;

  prepare(): void {
    if (this.ctx) {
      void this.ctx.resume().catch(() => {});
      return;
    }
    const Ctor = (globalThis as { AudioContext?: AudioContextClass; webkitAudioContext?: AudioContextClass }).AudioContext ??
      (globalThis as { webkitAudioContext?: AudioContextClass }).webkitAudioContext;
    if (!Ctor) return;
    try {
      this.ctx = new Ctor();
    } catch {
      this.ctx = null;
    }
  }

  tick(): void {
    this.beep(BEEP_TICK);
  }

  shot(): void {
    this.beep(BEEP_SHOT);
  }

  private beep(sound: { hz: number; ms: number }): void {
    const ctx = this.ctx;
    if (!ctx) return;
    try {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = sound.hz;
      const start = ctx.currentTime;
      const end = start + sound.ms / 1000;
      gain.gain.setValueAtTime(0, start);
      gain.gain.linearRampToValueAtTime(BEEP_GAIN, start + 0.01);
      gain.gain.setValueAtTime(BEEP_GAIN, Math.max(start + 0.01, end - 0.03));
      gain.gain.linearRampToValueAtTime(0, end);
      osc.connect(gain).connect(ctx.destination);
      osc.start(start);
      osc.stop(end + 0.01);
    } catch {
      // no sound is better than no photo
    }
  }
}

/** A countdown of whole seconds: onTick(seconds left) now and each second, then onDone(). */
export function startCountdown(seconds: number, handlers: { onTick: (left: number) => void; onDone: () => void }): { cancel: () => void } {
  let left = seconds;
  let id: ReturnType<typeof setTimeout> | null = null;
  const step = () => {
    left -= 1;
    if (left <= 0) {
      id = null;
      handlers.onDone();
      return;
    }
    handlers.onTick(left);
    id = setTimeout(step, 1000);
  };
  handlers.onTick(left);
  id = setTimeout(step, 1000);
  return {
    cancel: () => {
      if (id !== null) clearTimeout(id);
      id = null;
    },
  };
}
