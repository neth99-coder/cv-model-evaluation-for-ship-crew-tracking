import { defineConfig, loadEnv } from "vite";
import react, { reactCompilerPreset } from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backendHttpTarget = env.VITE_BACKEND_URL || "http://127.0.0.1:8000";
  const backendWsTarget = env.VITE_BACKEND_WS_URL || backendHttpTarget.replace(/^http/i, "ws");

  return {
    plugins: [
      react({
        babel: {
          presets: [reactCompilerPreset()],
        },
      }),
    ],
    server: {
      proxy: {
        "/api": backendHttpTarget,
        "/face-recognition": backendHttpTarget,
        "/ws": { target: backendWsTarget, ws: true },
      },
    },
  };
});
