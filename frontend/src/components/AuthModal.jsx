import { useState } from "react";
import { loginAPI, registerAPI, oauthLoginAPI } from "../api";

export default function AuthModal({ isOpen, onClose, onAuthSuccess }) {
  const [isRegister, setIsRegister] = useState(false);
  const [emailOrUsername, setEmailOrUsername] = useState("");
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [oauthLoading, setOauthLoading] = useState(null); // "google" | "github"
  const [error, setError] = useState("");

  if (!isOpen) return null;

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      let res;
      if (isRegister) {
        if (!email.includes("@")) {
          throw new Error("Please enter a valid email address.");
        }
        if (password.length < 6) {
          throw new Error("Password must be at least 6 characters.");
        }
        res = await registerAPI(email, username || email.split("@")[0], password);
      } else {
        if (!emailOrUsername.trim() || !password) {
          throw new Error("Please enter your credentials.");
        }
        res = await loginAPI(emailOrUsername, password);
      }

      if (res && res.user) {
        onAuthSuccess(res.user);
        onClose();
      }
    } catch (err) {
      setError(err.message || "Authentication failed. Please check credentials.");
    } finally {
      setLoading(false);
    }
  }

  // ── Google OAuth Flow ───────────────────────────────────────────────────
  async function handleGoogleOAuth() {
    setError("");
    setOauthLoading("google");

    try {
      // Simulate Google OAuth response or prompt for test account
      // In production, Google One Tap or OAuth2 popup exchanges code for user profile
      const defaultEmail = "user." + Math.random().toString(36).substring(2, 7) + "@gmail.com";
      const userEmail = window.prompt("Enter your Google Account email for testing Google Sign-In:", defaultEmail);
      if (!userEmail) {
        setOauthLoading(null);
        return;
      }

      const googlePayload = {
        provider: "google",
        email: userEmail.trim(),
        name: userEmail.split("@")[0].replace(".", " ").replace(/\b\w/g, l => l.toUpperCase()),
        provider_id: "goog_" + Math.random().toString(36).substring(2, 10),
        avatar_url: `https://api.dicebear.com/7.x/bottts/svg?seed=${userEmail}`,
      };

      const res = await oauthLoginAPI(googlePayload);
      if (res && res.user) {
        onAuthSuccess(res.user);
        onClose();
      }
    } catch (err) {
      setError(err.message || "Google authentication failed.");
    } finally {
      setOauthLoading(null);
    }
  }

  // ── GitHub OAuth Flow ───────────────────────────────────────────────────
  async function handleGithubOAuth() {
    setError("");
    setOauthLoading("github");

    try {
      const defaultUsername = "dev_" + Math.random().toString(36).substring(2, 7);
      const ghUsername = window.prompt("Enter your GitHub username for testing GitHub Sign-In:", defaultUsername);
      if (!ghUsername) {
        setOauthLoading(null);
        return;
      }

      const githubPayload = {
        provider: "github",
        email: `${ghUsername.trim()}@users.noreply.github.com`,
        name: ghUsername.trim(),
        provider_id: "gh_" + Math.random().toString(36).substring(2, 10),
        avatar_url: `https://github.com/${ghUsername.trim()}.png`,
      };

      const res = await oauthLoginAPI(githubPayload);
      if (res && res.user) {
        onAuthSuccess(res.user);
        onClose();
      }
    } catch (err) {
      setError(err.message || "GitHub authentication failed.");
    } finally {
      setOauthLoading(null);
    }
  }

  return (
    <div className="auth-overlay" onClick={onClose}>
      <div className="auth-card" onClick={e => e.stopPropagation()}>
        {/* Close button */}
        <button className="auth-close-btn" onClick={onClose} title="Close">
          ✕
        </button>

        {/* Header */}
        <div className="auth-header">
          <div className="auth-icon-badge">
            <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
          </div>
          <h2 className="auth-title">
            {isRegister ? "Create your account" : "Welcome back"}
          </h2>
          <p className="auth-subtitle">
            {isRegister
              ? "Join WorkplaceInsights to sync chat history and access knowledge graphs."
              : "Sign in to access your persistent conversations and insights."}
          </p>
        </div>

        {/* OAuth Buttons */}
        <div className="auth-oauth-group">
          {/* Google Button */}
          <button
            type="button"
            className="btn-oauth btn-google"
            onClick={handleGoogleOAuth}
            disabled={loading || oauthLoading !== null}
          >
            {oauthLoading === "google" ? (
              <span className="auth-spinner-sm" />
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24">
                <path
                  fill="#4285F4"
                  d="M23.745 12.27c0-.7-.06-1.4-.19-2.07H12v4.51h6.6c-.29 1.52-1.14 2.82-2.4 3.68v3.05h3.88c2.27-2.09 3.66-5.17 3.66-9.17z"
                />
                <path
                  fill="#34A853"
                  d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.88-3.05c-1.08.72-2.45 1.16-4.05 1.16-3.12 0-5.77-2.1-6.72-4.93H1.25v3.15C3.26 21.36 7.35 24 12 24z"
                />
                <path
                  fill="#FBBC05"
                  d="M5.28 14.27c-.25-.72-.38-1.49-.38-2.27s.13-1.55.38-2.27V6.58H1.25C.45 8.18 0 9.99 0 12s.45 3.82 1.25 5.42l4.03-3.15z"
                />
                <path
                  fill="#EA4335"
                  d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.35 0 3.26 2.64 1.25 6.58l4.03 3.15c.95-2.83 3.6-4.93 6.72-4.93z"
                />
              </svg>
            )}
            Continue with Google
          </button>

          {/* GitHub Button */}
          <button
            type="button"
            className="btn-oauth btn-github"
            onClick={handleGithubOAuth}
            disabled={loading || oauthLoading !== null}
          >
            {oauthLoading === "github" ? (
              <span className="auth-spinner-sm" />
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
              </svg>
            )}
            Continue with GitHub
          </button>
        </div>

        {/* Divider */}
        <div className="auth-divider">
          <span>or continue with email</span>
        </div>

        {/* Error Banner */}
        {error && <div className="auth-error-banner">{error}</div>}

        {/* Form */}
        <form onSubmit={handleSubmit} className="auth-form">
          {isRegister ? (
            <>
              <div className="auth-input-group">
                <label>Email Address</label>
                <input
                  type="email"
                  placeholder="name@company.com"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  required
                />
              </div>
              <div className="auth-input-group">
                <label>Username (Optional)</label>
                <input
                  type="text"
                  placeholder="sarthak"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                />
              </div>
            </>
          ) : (
            <div className="auth-input-group">
              <label>Email or Username</label>
              <input
                type="text"
                placeholder="name@company.com or username"
                value={emailOrUsername}
                onChange={e => setEmailOrUsername(e.target.value)}
                required
              />
            </div>
          )}

          <div className="auth-input-group">
            <div className="auth-label-row">
              <label>Password</label>
              <button
                type="button"
                className="auth-toggle-pwd"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>
            <input
              type={showPassword ? "text" : "password"}
              placeholder="••••••••"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
            />
          </div>

          <button
            type="submit"
            className="auth-submit-btn"
            disabled={loading || oauthLoading !== null}
          >
            {loading ? (
              <span className="auth-spinner" />
            ) : isRegister ? (
              "Create Account"
            ) : (
              "Sign In"
            )}
          </button>
        </form>

        {/* Footer Toggle */}
        <div className="auth-footer">
          {isRegister ? (
            <p>
              Already have an account?{" "}
              <button
                type="button"
                className="auth-switch-btn"
                onClick={() => {
                  setIsRegister(false);
                  setError("");
                }}
              >
                Sign in
              </button>
            </p>
          ) : (
            <p>
              Don't have an account?{" "}
              <button
                type="button"
                className="auth-switch-btn"
                onClick={() => {
                  setIsRegister(true);
                  setError("");
                }}
              >
                Create one
              </button>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
