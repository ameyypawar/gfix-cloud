import type { ResolveResponse } from "@/lib/api";

interface Props {
  result: ResolveResponse;
}

function viaBadgeClass(via: string) {
  if (via === "git_automerge") return "badge-green";
  if (via === "mergiraf") return "badge-blue";
  if (via === "ai" || via === "manual") return "badge-warn";
  return "badge-muted";
}

export default function ResolvePanel({ result }: Props) {
  return (
    <div className="card" style={{ height: "fit-content" }}>
      {/* Header row */}
      <div className="flex-between mb-16">
        <h2 style={{ marginBottom: 0 }}>Resolution</h2>
        <span className={`badge ${viaBadgeClass(result.via)}`}>{result.via}</span>
      </div>

      {/* AI unavailable notice */}
      {result.ai_unavailable && (
        <div className="alert alert-warning mb-16">
          <strong>AI suggestion unavailable</strong> — no{" "}
          <code>GEMINI_API_KEY</code> configured.
          <br />
          Set <code>GEMINI_API_KEY</code> to enable the generation path. Retrieval
          still ran — see neighbors.
          {result.ai_unavailable_reason && (
            <div className="mt-8 text-small" style={{ opacity: 0.8 }}>
              {result.ai_unavailable_reason}
            </div>
          )}
        </div>
      )}

      {/* Resolved content */}
      {result.resolved && result.resolved_content ? (
        <>
          <label>Resolved content</label>
          <pre className="code-block mt-4">{result.resolved_content}</pre>
        </>
      ) : (
        !result.ai_unavailable && (
          <div className="alert alert-info mb-12">
            Conflict could not be resolved automatically.
          </div>
        )
      )}

      {/* AI rationale */}
      {result.ai_rationale && (
        <div className="mt-16">
          <label>AI rationale</label>
          <p className="mt-4">{result.ai_rationale}</p>
        </div>
      )}

      {/* AI confidence */}
      {result.ai_confidence !== undefined && result.ai_confidence !== null && (
        <div className="mt-12">
          <label>Confidence</label>
          <p className="mt-4 text-mono">
            {(result.ai_confidence * 100).toFixed(0)}%
          </p>
        </div>
      )}

      {/* Audit ref */}
      {result.audit_ref ? (
        <div className="mt-16">
          <label>gfix audit ref — tamper-evident trace</label>
          <div className="audit-ref mt-4">
            <code>{result.audit_ref}</code>
          </div>
        </div>
      ) : (
        result.resolved && (
          <div className="mt-16">
            <label>gfix audit ref</label>
            <p className="text-dim text-small mt-4">not available</p>
          </div>
        )
      )}

      {/* Merge metadata */}
      <div className="mt-16">
        <label>Merge ID</label>
        <p className="audit-ref mt-4 text-small text-dim">{result.merge_id}</p>
      </div>
    </div>
  );
}
