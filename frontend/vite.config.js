import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy API calls to the FastAPI backend during dev so we avoid CORS issues.
    // The same API_BASE constant in src/api.js is used; adjust it if you deploy.
    proxy: {
      "/query": "http://127.0.0.1:8000",
      "/ingest": "http://127.0.0.1:8000",
    },
  },
});
