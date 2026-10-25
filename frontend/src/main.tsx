import { StrictMode, lazy, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { MotionConfig } from "motion/react";
import { LensDefs } from "./ui/LensDefs";
import { ErrorBoundary } from "./ui/ErrorBoundary";
import { applyAppearance, getAppearance } from "./lib/theme";
import "./styles.css";

// Hub at "/", caregiver at "/care" — FastAPI serves the same index.html for both.
const Hub = lazy(() => import("./hub/Hub"));
const Care = lazy(() => import("./care/Care"));

const isCare = location.pathname.replace(/\/+$/, "").endsWith("/care");
// The hub is always dark (kitchen display); the caregiver app follows the saved appearance.
if (isCare) applyAppearance(getAppearance());

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {/* reducedMotion="user": prefers-reduced-motion → transforms become simple fades */}
    <MotionConfig reducedMotion="user">
      <LensDefs />
      <ErrorBoundary area={isCare ? "caregiver app" : "hub"}>
        <Suspense fallback={null}>{isCare ? <Care /> : <Hub />}</Suspense>
      </ErrorBoundary>
    </MotionConfig>
  </StrictMode>,
);
