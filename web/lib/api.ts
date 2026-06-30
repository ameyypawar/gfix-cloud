const API =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

// ── Types mirroring the FastAPI models ────────────────────────────────────────

export interface ConflictSide {
  content: string;
  oid: string;
  source?: string | null;
}

export interface ConflictDetail {
  conflict_id: string;
  file: string;
  kind: string;
  ours: ConflictSide;
  theirs: ConflictSide;
  base: ConflictSide;
}

export interface RetrievedNeighbor {
  file_path: string;
  language: string;
  resolution_kind: string;
  similarity: number;
  resolved_content_preview: string;
}

export interface ResolveResponse {
  merge_id: string;
  file_path: string;
  resolved_content: string;
  via: string;
  audit_ref?: string | null;
  conflict: ConflictDetail;
  used_rag: boolean;
  neighbors: RetrievedNeighbor[];
  ai_rationale?: string | null;
  ai_confidence?: number | null;
  /** False when AI path needed but no key configured. */
  resolved: boolean;
  ai_unavailable: boolean;
  ai_unavailable_reason?: string | null;
}

export interface ResolveRequest {
  base: string;
  ours: string;
  theirs: string;
  file_path: string;
  rag: boolean;
}

// ── API calls ─────────────────────────────────────────────────────────────────

export async function resolveConflict(
  req: ResolveRequest
): Promise<ResolveResponse> {
  const res = await fetch(`${API}/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json();
}

export async function getEvalSummary(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API}/eval/summary`);
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${res.statusText}`);
  }
  return res.json();
}
