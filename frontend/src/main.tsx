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
const How = lazy(() => import("./how/HowItWorks"));

const path = location.pathname.replace(/\/+$/, "");
const isCare = path.endsWith("/care");
const isHow = path.endsWith("/how");
// The hub is always dark (kitchen display); the caregiver app and the How page follow the saved appearance.
if (isCare || isHow) applyAppearance(getAppearance());

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {/* reducedMotion="user": prefers-reduced-motion → transforms become simple fades */}
    <MotionConfig reducedMotion="user">
      <LensDefs />
      <ErrorBoundary area={isCare ? "caregiver app" : isHow ? "how it works" : "hub"}>
        <Suspense fallback={null}>{isCare ? <Care /> : isHow ? <How /> : <Hub />}</Suspense>
      </ErrorBoundary>
    </MotionConfig>
  </StrictMode>,
);
