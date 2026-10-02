import { useState, useRef, useEffect } from "react";
import { queryAPI } from "../api";

// ── Icons ──────────────────────────────────────────────────────────────────
const SendIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"/>
    <polygon points="22 2 15 22 11 13 2 9 22 2"/>
  </svg>
);

// ── Sub-components ─────────────────────────────────────────────────────────
function Badge({ cls }) {
  const map = { graph: "badge-graph", vector: "badge-vector", hybrid: "badge-hybrid" };
  return <span className={`badge ${map[cls] ?? "badge-default"}`}>{cls ?? "—"}</span>;
}

function GraphRow({ row }) {
  const entries = Object.entries(row).filter(([k]) => !k.startsWith("_") && k !== "rid");
  if (!entries.length) return null;
  return (
    <table className="kv-table">
      <tbody>
        {entries.map(([k, v]) => (
          <tr key={k}>
            <td>{k}</td>
            <td>{typeof v === "object" ? JSON.stringify(v) : String(v ?? "—")}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SourcesSection({ graphResults, vectorResults, cypherUsed }) {
  const [open, setOpen] = useState(false);
  const total = (graphResults?.length ?? 0) + (vectorResults?.length ?? 0);
  if (!total && !cypherUsed) return null;

  return (
    <div>
      <button className="sources-toggle" onClick={() => setOpen(o => !o)}>
        <span className={`sources-chevron ${open ? "open" : ""}`}>▶</span>
        Sources &amp; evidence {total > 0 && `(${total})`}
      </button>
      {open && (
        <div className="sources-body">
          {cypherUsed && (
            <div>
              <p className="source-label">Cypher used</p>
              <pre className="code-block">{cypherUsed}</pre>
            </div>
          )}
          {vectorResults?.length > 0 && (
            <div>
              <p className="source-label">Document chunks ({vectorResults.length})</p>
              {vectorResults.map((chunk, i) => (
                <div className="source-chunk" key={i}>
                  <div>{chunk.text ?? "(no text)"}</div>
                  <div className="source-chunk-meta">
                    {chunk.source && <span>📄 {chunk.source}</span>}
                    {chunk.date && <span>📅 {chunk.date}</span>}
                    {chunk.score != null && <span>score {Number(chunk.score).toFixed(3)}</span>}
                  </div>
                </div>
              ))}
            </div>
          )}
          {graphResults?.length > 0 && (
            <div>
              <p className="source-label">Graph facts ({graphResults.length})</p>
              {graphResults.map((row, i) => (
                <div className="source-chunk" key={i}><GraphRow row={row} /></div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function UserMessage({ text }) {
  return (
    <div className="msg-user">
      <div className="msg-user-bubble">{text}</div>
    </div>
  );
}

function AssistantMessage({ turn }) {
  return (
    <div className="msg-assistant">
      <div className="assistant-avatar">✦</div>
      <div className="msg-assistant-body">
        <div className="msg-assistant-text">{turn.answer}</div>
        <div className="msg-assistant-meta">
          <Badge cls={turn.classification} />
          {turn.graph_result_count != null && (
            <span className="meta-chip">
              {turn.graph_result_count} graph fact{turn.graph_result_count !== 1 ? "s" : ""}
            </span>
          )}
          {turn.vector_results?.length > 0 && (
            <span className="meta-chip">
              {turn.vector_results.length} doc chunk{turn.vector_results.length !== 1 ? "s" : ""}
            </span>
          )}
        </div>
        <SourcesSection
          graphResults={turn.graph_results}
          vectorResults={turn.vector_results}
          cypherUsed={turn.cypher_used}
        />
        {turn.grounding_warning?.length > 0 && (
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 6 }}>
            ⚠ Possible ungrounded tokens: {turn.grounding_warning.join(", ")}
          </div>
        )}
      </div>
    </div>
  );
}

function ThinkingIndicator({ question }) {
  return (
    <>
      <UserMessage text={question} />
      <div className="msg-assistant">
        <div className="assistant-avatar">✦</div>
        <div className="msg-assistant-body" style={{ paddingTop: 8 }}>
          <div className="thinking-dots">
            <span /><span /><span />
          </div>
        </div>
      </div>
    </>
  );
}

const SUGGESTIONS = [
  { title: "Who works on", sub: "which technology?" },
  { title: "What projects", sub: "are currently active?" },
  { title: "Show connections", sub: "between teams" },
  { title: "Summarize", sub: "recent activities" },
];

// ── Main Component ─────────────────────────────────────────────────────────
// history & onMessageAdded come from App.jsx — ChatView never owns history
export default function ChatView({ history = [], activeSessionId = null, onMessageAdded }) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState("");
  const [error, setError] = useState(null);
  const logEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, loading]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 180) + "px";
  }, [input]);

  async function submit(q) {
    const question = q.trim();
    if (!question || loading) return;

    setInput("");
    setError(null);
    setLoading(true);
    setPendingQuestion(question);

    try {
      const data = await queryAPI(question, activeSessionId);
      onMessageAdded?.(question, data);
    } catch (err) {
      setError(err.message ?? "Unknown error");
    } finally {
      setLoading(false);
      setPendingQuestion("");
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(input);
    }
  }

  const isEmpty = history.length === 0 && !loading;

  return (
    <>
      {/* Top bar */}
      <div className="topbar">
        <button className="model-pill active">
          GraphRAG
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </button>
      </div>

      {/* Scrollable chat area */}
      <div className="chat-area">
        <div className="chat-inner">

          {/* Empty state */}
          {isEmpty && (
            <div className="chat-empty-state">
              <h2>What's on your mind today?</h2>
              <p>Ask anything about your workplace data — retrieved from your Neo4j graph and Qdrant vector store.</p>
              <div className="empty-suggestions">
                {SUGGESTIONS.map((s, i) => (
                  <button
                    key={i}
                    className="suggestion-card"
                    onClick={() => submit(`${s.title} ${s.sub}`)}
                  >
                    <strong>{s.title}</strong>
                    {s.sub}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Message history from App */}
          {history.map((turn, i) => (
            <div key={i} className="msg-turn">
              <UserMessage text={turn.question} />
              <AssistantMessage turn={turn} />
            </div>
          ))}

          {/* In-flight request */}
          {loading && (
            <div className="msg-turn">
              <ThinkingIndicator question={pendingQuestion} />
            </div>
          )}

          {/* Error */}
          {error && <div className="inline-error">⚠ {error}</div>}

          <div ref={logEndRef} />
        </div>
      </div>

      {/* Input bar */}
      <div className="input-area">
        <div className="input-inner">
          <div className="input-box">
            <textarea
              ref={textareaRef}
              id="chat-input"
              className="chat-textarea"
              placeholder="Ask a question…"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={loading}
            />
            <button
              id="chat-send-btn"
              className="send-btn"
              onClick={() => submit(input)}
              disabled={!input.trim() || loading}
              title="Send"
            >
              {loading ? <span className="spinner" /> : <SendIcon />}
            </button>
          </div>
          <p className="input-hint">Enter to send · Shift+Enter for new line</p>
        </div>
      </div>
    </>
  );
}
