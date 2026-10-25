// One shared pointer listener; glass elements subscribe to aim their specular rim.
type Fn = (x: number, y: number) => void;
const subs = new Set<Fn>();
const last = { x: 0.5, y: 0.2, set: false };
let started = false;

export function subscribePointer(fn: Fn): () => void {
  if (!started) {
    started = true;
    let raf = 0;
    window.addEventListener(
      "pointermove",
      (e) => {
        last.x = e.clientX;
        last.y = e.clientY;
        last.set = true;
        if (!raf)
          raf = requestAnimationFrame(() => {
            raf = 0;
            subs.forEach((f) => f(last.x, last.y));
          });
      },
      { passive: true },
    );
  }
  subs.add(fn);
  fn(last.set ? last.x : window.innerWidth * 0.5, last.set ? last.y : -200);
  return () => {
    subs.delete(fn);
  };
}
