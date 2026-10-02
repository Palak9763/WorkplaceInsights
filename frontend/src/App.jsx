import { useState, useCallback, useEffect } from "react";
import ChatView from "./components/ChatView";
import UploadView from "./components/UploadView";
import AuthModal from "./components/AuthModal";
import {
  getSavedUser,
  fetchCurrentUserAPI,
  clearAuthSession,
  fetchConversationsAPI,
  fetchMessagesAPI,
  createConversationAPI,
  deleteConversationAPI,
} from "./api";

// ── Icons ──────────────────────────────────────────────────────────────────
const PlusIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
  </svg>
);
const ChatIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
  </svg>
);
const UploadIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="17 8 12 3 7 8"/>
    <line x1="12" y1="3" x2="12" y2="15"/>
  </svg>
);
const TrashIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
  </svg>
);
const UserIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
  </svg>
);
const LogoutIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>
  </svg>
);

export default function App() {
  // "chat" | "upload"
  const [activeTab, setActiveTab] = useState("chat");

  // Auth state
  const [currentUser, setCurrentUser] = useState(() => getSavedUser());
  const [isAuthOpen, setIsAuthOpen] = useState(false);

  // sessions: [{ id, label, history[] }]
  const [sessions, setSessions] = useState([]);
  // null = new unsaved chat; string = id of existing session
  const [activeSessionId, setActiveSessionId] = useState(null);

  // Fetch initial profile & conversations
  useEffect(() => {
    async function initAuth() {
      const user = await fetchCurrentUserAPI();
      if (user) {
        setCurrentUser(user);
      }
    }
    initAuth();
  }, []);

  // Fetch conversations from MongoDB when logged in
  useEffect(() => {
    async function loadConversations() {
      if (currentUser) {
        const convs = await fetchConversationsAPI();
        if (convs && convs.length > 0) {
          const loadedSessions = convs.map(c => ({
            id: c.id,
            label: c.title,
            history: [],
          }));
          setSessions(loadedSessions);
        }
      }
    }
    loadConversations();
  }, [currentUser]);

  // Derived: history for the currently active session
  const activeHistory = sessions.find(s => s.id === activeSessionId)?.history ?? [];

  // When switching to a session, load its messages from MongoDB if empty
  async function handleSelectSession(id) {
    setActiveSessionId(id);
    setActiveTab("chat");

    const session = sessions.find(s => s.id === id);
    if (session && session.history.length === 0 && currentUser) {
      try {
        const msgs = await fetchMessagesAPI(id);
        if (msgs && msgs.length > 0) {
          // Reconstruct Q&A turns from alternating user/assistant messages
          const turns = [];
          for (let i = 0; i < msgs.length; i++) {
            if (msgs[i].role === "user") {
              const question = msgs[i].content;
              const nextMsg = msgs[i + 1]?.role === "assistant" ? msgs[i + 1] : null;
              turns.push({
                question,
                answer: nextMsg ? nextMsg.content : "",
                classification: nextMsg?.metadata?.classification || "hybrid",
                cypher_used: nextMsg?.metadata?.cypher_used || "",
                graph_results: nextMsg?.metadata?.graph_results || [],
                graph_result_count: nextMsg?.metadata?.graph_result_count,
                vector_results: nextMsg?.metadata?.vector_results || [],
                grounding_warning: nextMsg?.metadata?.grounding_warning || [],
              });
              if (nextMsg) i++; // skip assistant message
            }
          }
          setSessions(prev => prev.map(s => s.id === id ? { ...s, history: turns } : s));
        }
      } catch (err) {
        console.warn("Error fetching conversation messages:", err);
      }
    }
  }

  // Called by ChatView when the user submits a message + gets a response
  const handleMessageAdded = useCallback((question, responseData) => {
    const newTurn = {
      question,
      answer: responseData.answer ?? "(no answer)",
      classification: responseData.classification,
      cypher_used: responseData.cypher_used,
      graph_results: responseData.graph_results ?? [],
      graph_result_count: responseData.graph_result_count,
      vector_results: responseData.vector_results ?? [],
      grounding_warning: responseData.grounding_warning ?? [],
    };

    setSessions(prev => {
      const existing = prev.find(s => s.id === activeSessionId);
      if (existing) {
        // Append to existing session
        return prev.map(s =>
          s.id === activeSessionId
            ? { ...s, history: [...s.history, newTurn] }
            : s
        );
      } else {
        // Create new session on first message
        const id = responseData.conversation_id || Date.now().toString();
        const label = question.length > 42 ? question.slice(0, 42) + "…" : question;
        setActiveSessionId(id);
        return [{ id, label, history: [newTurn] }, ...prev];
      }
    });
  }, [activeSessionId]);

  async function handleNewChat() {
    if (currentUser) {
      try {
        const newConv = await createConversationAPI("New Conversation");
        if (newConv && newConv.id) {
          setSessions(prev => [{ id: newConv.id, label: newConv.title, history: [] }, ...prev]);
          setActiveSessionId(newConv.id);
          setActiveTab("chat");
          return;
        }
      } catch (err) {
        console.warn("Create conversation error:", err);
      }
    }
    setActiveSessionId(null);
    setActiveTab("chat");
  }

  async function handleDeleteSession(e, id) {
    e.stopPropagation();
    if (currentUser) {
      await deleteConversationAPI(id);
    }
    setSessions(prev => prev.filter(s => s.id !== id));
    if (activeSessionId === id) {
      setActiveSessionId(null);
    }
  }

  function handleLogout() {
    clearAuthSession();
    setCurrentUser(null);
    setSessions([]);
    setActiveSessionId(null);
  }

  return (
    <div className="app-shell">
      {/* ── Sidebar ─────────────────────────────────── */}
      <aside className="sidebar">
        {/* Logo */}
        <div className="sidebar-logo">
          <svg viewBox="0 0 24 24" width="22" height="22" xmlns="http://www.w3.org/2000/svg">
            <circle cx="5" cy="12" r="2.5" fill="currentColor"/>
            <circle cx="19" cy="6" r="2.5" fill="currentColor"/>
            <circle cx="19" cy="18" r="2.5" fill="currentColor"/>
            <line x1="7.2" y1="11" x2="17" y2="7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            <line x1="7.2" y1="13" x2="17" y2="17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
          GraphRAG
        </div>

        {/* New Chat */}
        <button
          id="btn-new-chat"
          className={`sidebar-btn ${activeTab === "chat" && activeSessionId === null ? "active" : ""}`}
          onClick={handleNewChat}
        >
          <PlusIcon /> New chat
        </button>

        <div className="sidebar-divider" />

        {/* Upload */}
        <button
          id="btn-upload"
          className={`sidebar-btn ${activeTab === "upload" ? "active" : ""}`}
          onClick={() => setActiveTab("upload")}
        >
          <UploadIcon /> Upload files
        </button>

        {/* Chat */}
        <button
          id="btn-chat"
          className={`sidebar-btn ${activeTab === "chat" ? "active" : ""}`}
          onClick={() => { setActiveTab("chat"); }}
        >
          <ChatIcon /> Chat / Query
        </button>

        {/* Recent history */}
        <div className="sidebar-history-container">
          {sessions.length > 0 && (
            <>
              <div className="sidebar-section-label">Recent Chats</div>
              <div className="sidebar-history">
                {sessions.map(s => (
                  <div
                    key={s.id}
                    className={`history-entry-wrapper ${activeSessionId === s.id && activeTab === "chat" ? "active" : ""}`}
                    onClick={() => handleSelectSession(s.id)}
                  >
                    <button
                      className="history-entry"
                      title={s.label}
                    >
                      {s.label}
                    </button>
                    <button
                      className="history-delete-btn"
                      onClick={(e) => handleDeleteSession(e, s.id)}
                      title="Delete chat"
                    >
                      <TrashIcon />
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        {/* ── User Profile / Login Footer ──────────────── */}
        <div className="sidebar-footer">
          {currentUser ? (
            <div className="user-profile-badge">
              <div className="user-avatar-circle">
                {currentUser.avatar_url ? (
                  <img src={currentUser.avatar_url} alt={currentUser.username} className="user-avatar-img" />
                ) : (
                  currentUser.username?.charAt(0).toUpperCase() || "U"
                )}
              </div>
              <div className="user-info-text">
                <span className="user-name-label">{currentUser.username || "User"}</span>
                <span className="user-email-label">{currentUser.email || "Logged in"}</span>
              </div>
              <button
                className="user-logout-btn"
                onClick={handleLogout}
                title="Sign out"
              >
                <LogoutIcon />
              </button>
            </div>
          ) : (
            <button
              className="sidebar-login-btn"
              onClick={() => setIsAuthOpen(true)}
            >
              <UserIcon /> Sign In / Register
            </button>
          )}
        </div>
      </aside>

      {/* ── Main ──────────────────────────────────── */}
      <div className="main-area">
        {activeTab === "chat" && (
          <ChatView
            history={activeHistory}
            activeSessionId={activeSessionId}
            onMessageAdded={handleMessageAdded}
          />
        )}
        {activeTab === "upload" && <UploadView />}
      </div>

      {/* ── Auth Modal ─────────────────────────────── */}
      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
        onAuthSuccess={(user) => {
          setCurrentUser(user);
        }}
      />
    </div>
  );
}
