import { useState, useCallback } from "react";
import ChatView from "./components/ChatView";
import UploadView from "./components/UploadView";

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

export default function App() {
  // "chat" | "upload"
  const [activeTab, setActiveTab] = useState("chat");

  // sessions: [{ id, label, history[] }]
  const [sessions, setSessions] = useState([]);
  // null = new unsaved chat; string = id of existing session
  const [activeSessionId, setActiveSessionId] = useState(null);

  // Derived: history for the currently active session
  const activeHistory = sessions.find(s => s.id === activeSessionId)?.history ?? [];

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
        const id = Date.now().toString();
        const label = question.length > 42 ? question.slice(0, 42) + "…" : question;
        setActiveSessionId(id);
        return [{ id, label, history: [newTurn] }, ...prev];
      }
    });
  }, [activeSessionId]);

  function handleNewChat() {
    setActiveSessionId(null);
    setActiveTab("chat");
  }

  function handleSelectSession(id) {
    setActiveSessionId(id);
    setActiveTab("chat");
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
        {sessions.length > 0 && (
          <>
            <div className="sidebar-section-label">Recent</div>
            <div className="sidebar-history">
              {sessions.map(s => (
                <button
                  key={s.id}
                  className={`history-entry ${activeSessionId === s.id && activeTab === "chat" ? "active" : ""}`}
                  onClick={() => handleSelectSession(s.id)}
                  title={s.label}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </>
        )}
      </aside>

      {/* ── Main ──────────────────────────────────── */}
      <div className="main-area">
        {activeTab === "chat" && (
          <ChatView
            history={activeHistory}
            onMessageAdded={handleMessageAdded}
          />
        )}
        {activeTab === "upload" && <UploadView />}
      </div>
    </div>
  );
}
