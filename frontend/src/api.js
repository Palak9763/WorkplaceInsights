// ── API configuration ──────────────────────────────────────────────────────
// In development, Vite proxies /query and /ingest to http://127.0.0.1:8000
// automatically (see vite.config.js), so we use a relative base here.
export const API_BASE = "http://127.0.0.1:8000";   // ← change this for production

// Default timeout in milliseconds (120 s covers slow graph queries + LLM reasoning / fallbacks).
const DEFAULT_TIMEOUT_MS = 120_000;

function withTimeout(promise, ms = DEFAULT_TIMEOUT_MS) {
  const timeout = new Promise((_, reject) =>
    setTimeout(() => reject(new Error(`Request timed out after ${ms / 1000}s`)), ms)
  );
  return Promise.race([promise, timeout]);
}

// ── Auth Token Helpers ──────────────────────────────────────────────────────
const TOKEN_KEY = "graphrag_auth_token";
const USER_KEY = "graphrag_auth_user";

export function getAuthToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setAuthSession(token, user) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  if (user) localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function clearAuthSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export function getSavedUser() {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function authHeaders(extraHeaders = {}) {
  const token = getAuthToken();
  const headers = { "Content-Type": "application/json", ...extraHeaders };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
}

// ── Authentication APIs ─────────────────────────────────────────────────────

/**
 * POST /api/auth/register
 */
export async function registerAPI(email, username, password) {
  const response = await withTimeout(
    fetch(`${API_BASE}/api/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, username, password }),
    })
  );
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Registration failed.");
  }
  if (data.token) {
    setAuthSession(data.token, data.user);
  }
  return data;
}

/**
 * POST /api/auth/login
 */
export async function loginAPI(emailOrUsername, password) {
  const response = await withTimeout(
    fetch(`${API_BASE}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email_or_username: emailOrUsername, password }),
    })
  );
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Invalid email/username or password.");
  }
  if (data.token) {
    setAuthSession(data.token, data.user);
  }
  return data;
}

/**
 * POST /api/auth/oauth
 */
export async function oauthLoginAPI({ provider, email, name, provider_id, avatar_url }) {
  const response = await withTimeout(
    fetch(`${API_BASE}/api/auth/oauth`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider,
        email: email || "",
        name: name || "",
        provider_id: String(provider_id),
        avatar_url: avatar_url || "",
      }),
    })
  );
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `${provider} OAuth login failed.`);
  }
  if (data.token) {
    setAuthSession(data.token, data.user);
  }
  return data;
}

/**
 * GET /api/auth/me
 */
export async function fetchCurrentUserAPI() {
  const token = getAuthToken();
  if (!token) return null;
  const response = await withTimeout(
    fetch(`${API_BASE}/api/auth/me`, {
      headers: authHeaders(),
    }),
    10_000
  );
  if (!response.ok) {
    clearAuthSession();
    return null;
  }
  return response.json();
}

// ── Persistent Chat & Conversations APIs ────────────────────────────────────

/**
 * GET /api/conversations
 */
export async function fetchConversationsAPI() {
  const token = getAuthToken();
  if (!token) return [];
  try {
    const response = await withTimeout(
      fetch(`${API_BASE}/api/conversations`, {
        headers: authHeaders(),
      }),
      10_000
    );
    if (!response.ok) return [];
    return response.json();
  } catch {
    return [];
  }
}

/**
 * POST /api/conversations
 */
export async function createConversationAPI(title = "New Conversation") {
  const token = getAuthToken();
  if (!token) return { id: Date.now().toString(), title };
  const response = await withTimeout(
    fetch(`${API_BASE}/api/conversations`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ title }),
    })
  );
  if (!response.ok) {
    return { id: Date.now().toString(), title };
  }
  return response.json();
}

/**
 * GET /api/conversations/{id}/messages
 */
export async function fetchMessagesAPI(conversationId) {
  const token = getAuthToken();
  if (!token || !conversationId) return [];
  try {
    const response = await withTimeout(
      fetch(`${API_BASE}/api/conversations/${conversationId}/messages`, {
        headers: authHeaders(),
      }),
      10_000
    );
    if (!response.ok) return [];
    return response.json();
  } catch {
    return [];
  }
}

/**
 * DELETE /api/conversations/{id}
 */
export async function deleteConversationAPI(conversationId) {
  const token = getAuthToken();
  if (!token || !conversationId) return;
  try {
    await withTimeout(
      fetch(`${API_BASE}/api/conversations/${conversationId}`, {
        method: "DELETE",
        headers: authHeaders(),
      }),
      10_000
    );
  } catch (err) {
    console.warn("Delete conversation error:", err);
  }
}

// ── Query & Ingest APIs ─────────────────────────────────────────────────────

/**
 * POST /query
 * @param {string} question
 * @param {string|null} conversationId
 * @returns {Promise<object>} Full response JSON from the backend.
 */
export async function queryAPI(question, conversationId = null) {
  const body = { question };
  if (conversationId) {
    body.conversation_id = conversationId;
  }

  const response = await withTimeout(
    fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
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

  const headers = {};
  const token = getAuthToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const response = await withTimeout(
    fetch(`${API_BASE}/ingest`, {
      method: "POST",
      headers,
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
