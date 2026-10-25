import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev: proxy the API, streams, clips and WebSocket to the FastAPI server.
const api = "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": api,
      "/stream": api,
      "/clips": api,
      "/ws": { target: api, ws: true },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
