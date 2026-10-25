import { useEffect } from "react";
import { animate, motion, useMotionValue, useTransform } from "motion/react";
import { spring } from "./motion";

/** Number that counts up to its value with the gentle spring. */
export function CountUp({ value }: { value: number }) {
  const mv = useMotionValue(0);
  const text = useTransform(mv, (v) => Math.round(v).toString());
  useEffect(() => {
    const c = animate(mv, value, spring.gentle);
    return () => c.stop();
  }, [value, mv]);
  return <motion.span>{text}</motion.span>;
}
