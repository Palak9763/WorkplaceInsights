import { useState, useRef } from "react";
import { ingestAPI } from "../api";

const ACCEPTED = ".pdf,.docx,.png,.jpg,.jpeg,.xlsx,.csv,.md";

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function IngestResultCard({ filename, result }) {
  if (result.error) {
    return (
      <div className="inline-error">
        <strong>Failed:</strong> {result.error}
      </div>
    );
  }

  return (
    <div className="ingest-result-card">
      <h4>✓ "{filename}" ingested successfully</h4>
      <div className="stat-grid">
        <div className="stat-box">
          <div className="stat-value">{result.chunks_processed ?? 0}</div>
          <div className="stat-label">Chunks processed</div>
        </div>
        <div className={`stat-box ${result.chunks_failed ? "has-error" : ""}`}>
          <div className="stat-value">{result.chunks_failed ?? 0}</div>
          <div className="stat-label">Chunks failed</div>
        </div>
        <div className="stat-box">
          <div className="stat-value">{result.entities_extracted ?? 0}</div>
          <div className="stat-label">Entities extracted</div>
        </div>
        <div className="stat-box">
          <div className="stat-value">{result.relationships_extracted ?? 0}</div>
          <div className="stat-label">Relationships</div>
        </div>
      </div>
    </div>
  );
}

export default function UploadView() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [lastResult, setLastResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [sessionHistory, setSessionHistory] = useState([]);
  const inputRef = useRef(null);

  function pickFile(f) {
    if (!f) return;
    setFile(f);
    setLastResult(null);
    setError(null);
  }

  function handleFileInput(e) {
    pickFile(e.target.files?.[0] ?? null);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    pickFile(e.dataTransfer.files?.[0] ?? null);
  }

  async function handleUpload() {
    if (!file || loading) return;
    setLoading(true);
    setError(null);
    setLastResult(null);

    try {
      const data = await ingestAPI(file);
      setLastResult(data);
      setSessionHistory((prev) => [
        { filename: file.name, result: data, ts: Date.now() },
        ...prev,
      ]);
      setFile(null);
      // Reset the hidden input so the same file can be re-selected
      if (inputRef.current) inputRef.current.value = "";
    } catch (err) {
      const msg = err.message ?? "Unknown error";
      setError(msg);
      setSessionHistory((prev) => [
        { filename: file.name, result: { error: msg }, ts: Date.now() },
        ...prev,
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="upload-view">
      {/* Drop zone */}
      <div
        className={`upload-zone ${dragOver ? "drag-over" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        aria-label="File drop zone"
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          onChange={handleFileInput}
          style={{ display: "none" }}
        />
        <div className="upload-icon">📂</div>
        {file ? (
          <strong style={{ fontSize: 14 }}>{file.name}</strong>
        ) : (
          <>
            <strong style={{ fontSize: 14 }}>Click to browse or drag a file here</strong>
            <div className="upload-hint">
              Supported: PDF, DOCX, PNG, JPG, XLSX, CSV, MD
            </div>
          </>
        )}
      </div>

      {/* Selected file pill + actions */}
      {file && (
        <>
          <div className="file-pill">
            <span>📄</span>
            <span className="file-name">{file.name}</span>
            <span className="file-size">{formatBytes(file.size)}</span>
            <button
              className="btn-clear"
              onClick={() => { setFile(null); if (inputRef.current) inputRef.current.value = ""; }}
              title="Remove"
            >
              ×
            </button>
          </div>
          <div className="upload-actions">
            <button
              className="btn-primary"
              onClick={handleUpload}
              disabled={loading}
            >
              {loading ? (
                <><span className="spinner" /> Uploading…</>
              ) : (
                "Upload & Ingest →"
              )}
            </button>
          </div>
        </>
      )}

      {/* Result */}
      {lastResult && (
        <IngestResultCard filename={sessionHistory[0]?.filename} result={lastResult} />
      )}

      {/* Error */}
      {error && <div className="inline-error">⚠ {error}</div>}

      {/* Session history */}
      {sessionHistory.length > 0 && (
        <div className="upload-history card">
          <h3>This session ({sessionHistory.length})</h3>
          {sessionHistory.map((item, i) => (
            <div
              key={i}
              className={`history-item ${item.result.error ? "error" : ""}`}
            >
              <span className="hi-name">
                {item.result.error ? "✗" : "✓"} {item.filename}
              </span>
              <span className="hi-stats">
                {item.result.error
                  ? item.result.error.slice(0, 60)
                  : `${item.result.chunks_processed ?? 0} chunks · ${item.result.entities_extracted ?? 0} entities`}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
