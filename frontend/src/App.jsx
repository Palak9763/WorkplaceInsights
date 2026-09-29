import { useState } from "react";
import ChatView from "./components/ChatView";
import UploadView from "./components/UploadView";

const TABS = [
  { id: "chat", label: "💬 Chat / Query" },
  { id: "upload", label: "📤 Upload" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("chat");

  return (
    <div className="app-shell">
      <nav className="app-nav">
        <div className="app-title">
          <span className="dot" />
          GraphRAG
        </div>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`nav-tab ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {activeTab === "chat" && <ChatView />}
      {activeTab === "upload" && <UploadView />}
    </div>
  );
}
