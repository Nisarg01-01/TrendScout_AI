import { useEffect, useMemo, useState } from "react";

type StatusResponse = {
  artifacts: Record<string, boolean>;
  env: Record<string, unknown>;
};

type QueryResponse = {
  answer: string;
  context?: any;
};

async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    apiGet<StatusResponse>("/api/status")
      .then(setStatus)
      .catch((e) => setError(String(e)));
  }, []);

  const artifactsOk = useMemo(() => {
    if (!status) return false;
    return Object.values(status.artifacts).some(Boolean);
  }, [status]);

  async function runQuery() {
    setLoading(true);
    setError("");
    setAnswer("");
    try {
      const result = await apiPost<QueryResponse>("/api/query", { query, return_context: true });
      setAnswer(result.answer || "");
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Arial", padding: 24, maxWidth: 960, margin: "0 auto" }}>
      <h1 style={{ margin: 0 }}>TrendScout AI</h1>
      <p style={{ color: "#444" }}>
        Market intelligence demo UI (React) calling the FastAPI backend.
      </p>

      <section style={{ border: "1px solid #eee", borderRadius: 12, padding: 16, marginBottom: 16 }}>
        <h2 style={{ marginTop: 0, fontSize: 16 }}>System Status</h2>
        {status ? (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div>
              <div style={{ fontWeight: 600, marginBottom: 8 }}>Artifacts</div>
              <ul style={{ margin: 0, paddingLeft: 18 }}>
                {Object.entries(status.artifacts).map(([k, v]) => (
                  <li key={k}>
                    {k}: <span style={{ color: v ? "green" : "#999" }}>{v ? "present" : "missing"}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <div style={{ fontWeight: 600, marginBottom: 8 }}>Environment</div>
              <pre style={{ margin: 0, background: "#fafafa", padding: 12, borderRadius: 8, overflowX: "auto" }}>
                {JSON.stringify(status.env, null, 2)}
              </pre>
            </div>
          </div>
        ) : (
          <div>Loading status...</div>
        )}
        {!artifactsOk && (
          <div style={{ marginTop: 12, color: "#b45309" }}>
            No data artifacts found. Run the pipeline first (or point the app at an existing DATA folder).
          </div>
        )}
      </section>

      <section style={{ border: "1px solid #eee", borderRadius: 12, padding: 16 }}>
        <h2 style={{ marginTop: 0, fontSize: 16 }}>Ask a Question</h2>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={4}
          placeholder="Ask about market trends, competitors, funding, hiring, SWOT..."
          style={{ width: "100%", padding: 12, borderRadius: 8, border: "1px solid #ddd", resize: "vertical" }}
        />
        <div style={{ display: "flex", gap: 12, marginTop: 12 }}>
          <button
            onClick={runQuery}
            disabled={loading || query.trim().length === 0}
            style={{ padding: "10px 14px", borderRadius: 10, border: "1px solid #ddd", background: "#111", color: "#fff" }}
          >
            {loading ? "Running..." : "Run"}
          </button>
          {error && <div style={{ color: "crimson", alignSelf: "center" }}>{error}</div>}
        </div>

        {answer && (
          <div style={{ marginTop: 16 }}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>Answer</div>
            <div style={{ whiteSpace: "pre-wrap", background: "#fafafa", padding: 12, borderRadius: 8, border: "1px solid #eee" }}>
              {answer}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

