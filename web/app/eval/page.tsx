"use client";

import { useState, useEffect } from "react";
import { getEvalSummary } from "@/lib/api";

interface Bucket {
  rows: number;
  families?: string[];
  retrieval: {
    mean_context_precision: number;
    threshold: number;
  };
  generation: string | Record<string, unknown>;
}

interface EvalSummary {
  status?: string;
  generated_at?: string;
  provider?: string;
  model?: string;
  golden_set_size?: number;
  rag_top_k?: number;
  buckets?: {
    recurring?: Bucket;
    one_off?: Bucket;
    overall?: Bucket;
  };
}

function pct(n: number) {
  return (n * 100).toFixed(1) + "%";
}

function generationCell(gen: string | Record<string, unknown>) {
  if (gen === "pending-key") {
    return <span className="text-dim">pending key</span>;
  }
  if (typeof gen === "string") return gen;
  // structured gen metrics
  const obj = gen as Record<string, unknown>;
  if (typeof obj.exact_match_rate === "number") {
    return (
      <span>
        exact-match {pct(obj.exact_match_rate as number)}
        {typeof obj.edit_distance_mean === "number" && (
          <>, edit-dist {(obj.edit_distance_mean as number).toFixed(1)}</>
        )}
      </span>
    );
  }
  return JSON.stringify(gen);
}

const BUCKET_LABELS: Record<string, string> = {
  recurring: "recurring",
  one_off: "one-off",
  overall: "overall",
};

export default function EvalPage() {
  const [data, setData] = useState<EvalSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getEvalSummary()
      .then((d) => setData(d as EvalSummary))
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <div className="page-content">
      <div className="mb-24">
        <h1>Eval</h1>
        <p className="mt-4">
          Per-regime retrieval and generation metrics over the golden set.
        </p>
      </div>

      {error && (
        <div className="alert alert-error">
          Failed to load eval summary: {error}
        </div>
      )}

      {!data && !error && (
        <div className="flex-row mt-24">
          <span className="spinner" />
          <span className="text-muted text-small">Loading eval summary…</span>
        </div>
      )}

      {data?.status === "no eval run yet" && (
        <div className="alert alert-info">
          No eval run yet. Run <code>python eval/run_eval.py</code> to produce
          results.
        </div>
      )}

      {data && !data.status && (
        <>
          {/* Meta */}
          <div className="card mb-16">
            <div className="flex-row flex-wrap" style={{ gap: 20 }}>
              {data.model && (
                <div>
                  <label>Model</label>
                  <p className="text-mono">{data.model}</p>
                </div>
              )}
              {data.provider && (
                <div>
                  <label>Provider</label>
                  <p>{data.provider}</p>
                </div>
              )}
              {data.golden_set_size !== undefined && (
                <div>
                  <label>Golden set</label>
                  <p>{data.golden_set_size} conflicts</p>
                </div>
              )}
              {data.rag_top_k !== undefined && (
                <div>
                  <label>RAG top-k</label>
                  <p>{data.rag_top_k}</p>
                </div>
              )}
              {data.generated_at && (
                <div>
                  <label>Run at</label>
                  <p className="text-small text-muted">
                    {new Date(data.generated_at).toLocaleString()}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Per-regime table */}
          {data.buckets && (
            <div className="card mb-20">
              <h2>Per-regime results</h2>
              <table>
                <thead>
                  <tr>
                    <th>Regime</th>
                    <th>Conflicts</th>
                    <th>
                      Context Precision
                      {data.buckets.overall?.retrieval?.threshold !== undefined && (
                        <span className="text-dim text-small" style={{ fontWeight: 400, marginLeft: 4 }}>
                          @{data.buckets.overall.retrieval.threshold}
                        </span>
                      )}
                    </th>
                    <th>Generation</th>
                  </tr>
                </thead>
                <tbody>
                  {(
                    ["recurring", "one_off", "overall"] as const
                  ).map((key) => {
                    const b = data.buckets![key];
                    if (!b) return null;
                    return (
                      <tr key={key}>
                        <td className="mono value">{BUCKET_LABELS[key]}</td>
                        <td className="value">{b.rows}</td>
                        <td className="value">
                          {pct(b.retrieval.mean_context_precision)}
                        </td>
                        <td>{generationCell(b.generation)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Insight */}
          {data.buckets?.recurring && data.buckets?.one_off && (
            <div className="alert alert-info">
              RAG retrieval is ~
              {(
                data.buckets.recurring.retrieval.mean_context_precision /
                Math.max(
                  data.buckets.one_off.retrieval.mean_context_precision,
                  0.001
                )
              ).toFixed(0)}
              x more effective on recurring conflict families than one-offs — the
              expected behavior of retrieval-over-history; it grows as the corpus
              grows.
            </div>
          )}
        </>
      )}
    </div>
  );
}
