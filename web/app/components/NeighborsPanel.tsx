import type { RetrievedNeighbor } from "@/lib/api";

interface Props {
  neighbors: RetrievedNeighbor[];
  usedRag: boolean;
}

function pct(n: number) {
  return (n * 100).toFixed(1) + "%";
}

export default function NeighborsPanel({ neighbors, usedRag }: Props) {
  return (
    <div className="card" style={{ height: "fit-content" }}>
      <div className="flex-between mb-12">
        <h2 style={{ marginBottom: 0 }}>Retrieved similar past resolutions (RAG)</h2>
        {!usedRag && (
          <span className="badge badge-muted" style={{ fontSize: 10 }}>
            RAG off
          </span>
        )}
      </div>

      {neighbors.length === 0 ? (
        <p className="empty-state">
          No similar past resolutions found.
          <br />
          Expected for novel conflicts — the retrieval corpus grows with each
          resolution.
        </p>
      ) : (
        <div>
          {neighbors.map((n, i) => (
            <div key={i} className="neighbor-item">
              <div className="neighbor-meta">
                <span className="neighbor-path" title={n.file_path}>
                  {n.file_path}
                </span>
                <span className="badge badge-muted">{n.language}</span>
                <span className="badge badge-blue">{n.resolution_kind}</span>
                <span
                  className="text-small text-muted"
                  style={{ flexShrink: 0, fontFamily: "var(--mono)" }}
                >
                  sim {pct(n.similarity)}
                </span>
              </div>
              {n.resolved_content_preview && (
                <div className="neighbor-preview">
                  {n.resolved_content_preview}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
