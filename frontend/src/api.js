// ── API configuration ──────────────────────────────────────────────────────
// In development, Vite proxies /query and /ingest to http://127.0.0.1:8000
// automatically (see vite.config.js), so we use a relative base here.
//
// When deploying, either:
//   (a) Set API_BASE to the production URL, e.g. "https://api.example.com"
//   (b) Or keep "" and configure your reverse-proxy to forward /query and /ingest.
export const API_BASE = "http://127.0.0.1:8000";   // ← change this for production

// Default timeout in milliseconds (45 s to cover slow graph queries).
const DEFAULT_TIMEOUT_MS = 45_000;

function withTimeout(promise, ms = DEFAULT_TIMEOUT_MS) {
  const timeout = new Promise((_, reject) =>
    setTimeout(() => reject(new Error(`Request timed out after ${ms / 1000}s`)), ms)
  );
  return Promise.race([promise, timeout]);
}

/**
 * POST /query
 * @param {string} question
 * @returns {Promise<object>} Full response JSON from the backend.
 */
export async function queryAPI(question) {
  const response = await withTimeout(
    fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    })
  );
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Server error ${response.status}: ${text}`);
  }
  return response.json();
}

/**
 * POST /ingest
 * @param {File} file
 * @returns {Promise<object>} Ingest result JSON from the backend.
 */
export async function ingestAPI(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await withTimeout(
    fetch(`${API_BASE}/ingest`, {
      method: "POST",
      body: formData,
    }),
    120_000 // ingest can take longer — 2-min timeout
  );
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Server error ${response.status}: ${text}`);
  }
  return response.json();
}
