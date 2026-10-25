import { motion, type HTMLMotionProps } from "motion/react";
import { spring } from "./motion";

/** Press: scale 0.97 + fill highlight (CSS :active), releasing with a bouncy spring. */
export function Press({ className = "", ...props }: HTMLMotionProps<"button">) {
  return <motion.button whileTap={{ scale: 0.97 }} transition={spring.bouncy} className={`pressable ${className}`} {...props} />;
}
