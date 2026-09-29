import { useState, useRef, useEffect } from "react";
import { queryAPI } from "../api";

// ── Helpers ──────────────────────────────────────────────────────────────

function classificationBadge(cls) {
  const map = {
    graph: "badge-graph",
    vector: "badge-vector",
    hybrid: "badge-hybrid",
  };
  return (
    <span className={`badge ${map[cls] ?? "badge-default"}`}>
      {cls ?? "—"}
    </span>
  );
}

function GraphRow({ row }) {
  const entries = Object.entries(row).filter(
    ([k]) => !k.startsWith("_") && k !== "rid"
  );
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

  const hasContent =
    (graphResults && graphResults.length > 0) ||
    (vectorResults && vectorResults.length > 0) ||
    cypherUsed;

  if (!hasContent) return null;

  return (
    <div>
      <button
        className={`collapsible-toggle ${open ? "open" : ""}`}
        onClick={() => setOpen((o) => !o)}
      >
        <em className="chevron">▶</em>
        Sources &amp; evidence
        {graphResults?.length || vectorResults?.length
          ? ` (${(graphResults?.length ?? 0) + (vectorResults?.length ?? 0)})`
          : ""}
      </button>

      {open && (
        <div className="collapsible-body">
          {/* Cypher */}
          {cypherUsed && (
            <div>
              <p className="code-label">Cypher used</p>
              <pre className="code-block">{cypherUsed}</pre>
            </div>
          )}

          {/* Vector chunks */}
          {vectorResults?.length > 0 && (
            <div>
              <p className="code-label">
                Document chunks ({vectorResults.length})
              </p>
              {vectorResults.map((chunk, i) => (
                <div className="source-chunk" key={i}>
                  <div>{chunk.text ?? "(no text)"}</div>
                  <div className="source-meta">
                    {chunk.source && <span>📄 {chunk.source}</span>}
                    {chunk.date && <span>📅 {chunk.date}</span>}
                    {chunk.score != null && (
                      <span>score {Number(chunk.score).toFixed(3)}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Graph rows */}
          {graphResults?.length > 0 && (
            <div>
              <p className="code-label">
                Graph facts ({graphResults.length})
              </p>
              {graphResults.map((row, i) => (
                <div className="source-chunk" key={i}>
                  <GraphRow row={row} />
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function MessageBubble({ turn }) {
  return (
    <div className="chat-turn">
      <div className="chat-question">{turn.question}</div>

      <div className="chat-answer-wrap">
        <div className="chat-answer-bubble">
          <div className="answer-text">{turn.answer}</div>
          <div className="answer-meta">
            {classificationBadge(turn.classification)}
            {turn.graph_result_count != null && (
              <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                {turn.graph_result_count} graph fact
                {turn.graph_result_count !== 1 ? "s" : ""}
              </span>
            )}
            {turn.vector_results?.length != null && (
              <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                {turn.vector_results.length} doc chunk
                {turn.vector_results.length !== 1 ? "s" : ""}
              </span>
            )}
          </div>

          <SourcesSection
            graphResults={turn.graph_results}
            vectorResults={turn.vector_results}
            cypherUsed={turn.cypher_used}
          />
        </div>

        {turn.grounding_warning?.length > 0 && (
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4, paddingLeft: 4 }}>
            ⚠ Possible ungrounded tokens: {turn.grounding_warning.join(", ")}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────

export default function ChatView() {
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const logEndRef = useRef(null);

  // Auto-scroll to bottom when history changes
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, loading]);

  async function handleSubmit(e) {
    e?.preventDefault();
    const q = input.trim();
    if (!q || loading) return;

    setInput("");
    setError(null);
    setLoading(true);

    try {
      const data = await queryAPI(q);
      setHistory((prev) => [
        ...prev,
        {
          question: q,
          answer: data.answer ?? "(no answer)",
          classification: data.classification,
          cypher_used: data.cypher_used,
          graph_results: data.graph_results ?? [],
          graph_result_count: data.graph_result_count,
          vector_results: data.vector_results ?? [],
          grounding_warning: data.grounding_warning ?? [],
        },
      ]);
    } catch (err) {
      setError(err.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    // Submit on Enter (not Shift+Enter)
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  return (
    <div className="chat-view">
      {/* Log */}
      <div className="chat-log">
        {history.length === 0 && !loading && (
          <div className="chat-empty">
            <div className="icon">💬</div>
            <strong>Ask anything about your data</strong>
            <span>Results are retrieved from your Neo4j graph and Qdrant vector store.</span>
          </div>
        )}

        {history.map((turn, i) => (
          <MessageBubble key={i} turn={turn} />
        ))}

        {loading && (
          <div className="chat-turn">
            <div
              className="chat-question"
              style={{ alignSelf: "flex-end", opacity: 0.6 }}
            >
              {input || "…"}
            </div>
            <div className="loading-bubble">
              <span className="spinner" />
              Thinking…
            </div>
          </div>
        )}

        {error && <div className="inline-error">⚠ {error}</div>}

        <div ref={logEndRef} />
      </div>

      {/* Input */}
      <form className="chat-input-row" onSubmit={handleSubmit}>
        <textarea
          className="chat-textarea"
          placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={loading}
        />
        <button
          type="submit"
          className="btn-primary"
          disabled={!input.trim() || loading}
        >
          {loading ? (
            <>
              <span className="spinner" />
              Sending
            </>
          ) : (
            "Send →"
          )}
        </button>
      </form>
    </div>
  );
}
