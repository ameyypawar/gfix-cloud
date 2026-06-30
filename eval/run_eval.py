#!/usr/bin/env python3
"""
Leave-one-out RAG eval harness for gfix-cloud.

Keyless path (no ANTHROPIC_API_KEY set):
  1. Ingests golden_set.jsonl into pgvector (idempotent — skips existing rows).
  2. Leave-one-out retrieval: for each row, retrieve top-k neighbors using
     exclude_id=<self> so a conflict never retrieves its own resolution.
  3. Reports mean Context Precision (fraction of retrieved neighbors whose
     resolution is ≥0.7 similar to the gold resolution).
  4. Writes eval/summary.json (retrieval section populated).

With ANTHROPIC_API_KEY:
  5. For each row, produce_suggestion(rag=True) and produce_suggestion(rag=False).
  6. Score both against expected_resolution (exact_match + edit_distance_ratio).
  7. Prints delta table: no-RAG vs RAG (exact-match %, mean edit-distance).
  8. Populates generation section of summary.json.

With --with-gfix-baseline:
  9. Adds a gfix BYOK column (needs GITFIX_BYOK=1 + key set).

Run from project root with the api venv activated:
  cd /path/to/gfix-cloud
  source api/.venv/bin/activate
  docker compose up -d db
  python eval/run_eval.py           # keyless
  python eval/run_eval.py           # with ANTHROPIC_API_KEY for generation delta
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: eval/ lives one level above api/; we import app modules from api/
# ---------------------------------------------------------------------------
_EVAL_DIR = Path(__file__).parent.resolve()
_API_DIR  = _EVAL_DIR / ".." / "api"
sys.path.insert(0, str(_API_DIR))
sys.path.insert(0, str(_EVAL_DIR))

from app.config import settings
from app.db import close_pool, open_pool, run_migrations
from app.embeddings import build_conflict_text, embed_conflict
from app.models import ConflictDetail, ConflictSide, TargetSide
from app.persistence import persist_resolution
from app.retrieval import hybrid_search

from metrics import context_precision, edit_distance_ratio, exact_match  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GOLDEN_SET_PATH = _EVAL_DIR / "golden_set.jsonl"
SUMMARY_PATH    = _EVAL_DIR / "summary.json"

_RAG_TOP_K = settings.rag_top_k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_golden_set() -> list[dict]:
    rows: list[dict] = []
    with open(GOLDEN_SET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _make_conflict_detail(row: dict) -> ConflictDetail:
    return ConflictDetail(
        conflict_id="",
        file=row["file_path"],
        kind="text",
        ours=ConflictSide(content=row["ours"],   oid="", source="ours"),
        theirs=ConflictSide(content=row["theirs"], oid="", source="theirs"),
        base=ConflictSide(content=row["base"],    oid="", source=None),
        target=TargetSide(content=row["ours"],    oid="", exists=True),
    )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

async def ingest_golden_set(pool, rows: list[dict]) -> dict[str, str]:
    """Insert all golden-set rows as past_resolutions (idempotent).

    Returns id_map: row["id"] → resolution_id (UUID str).
    Each row uses rerere_hash="golden:<row_id>" so re-runs skip existing entries.
    """
    print(f"\nIngesting {len(rows)} golden-set rows into pgvector …", flush=True)
    id_map: dict[str, str] = {}

    for row in rows:
        language     = row.get("language", "text")
        conflict_kind = "text"
        rerere_key   = f"golden:{row['id']}"

        # Idempotency check
        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT resolution_id FROM past_resolutions WHERE rerere_hash = $1",
                rerere_key,
            )
        if existing:
            id_map[row["id"]] = str(existing["resolution_id"])
            continue

        embedding     = embed_conflict(
            file_path=row["file_path"],
            language=language,
            conflict_kind=conflict_kind,
            base=row["base"],
            ours=row["ours"],
            theirs=row["theirs"],
        )
        conflict_text = build_conflict_text(
            file_path=row["file_path"],
            language=language,
            conflict_kind=conflict_kind,
            base=row["base"],
            ours=row["ours"],
            theirs=row["theirs"],
        )

        record = {
            "merge_id":        row.get("merge_sha", ""),
            "file_path":       row["file_path"],
            "language":        language,
            "conflict_kind":   conflict_kind,
            "resolution_kind": "ground_truth",
            "base_code":       row["base"],
            "ours_code":       row["ours"],
            "theirs_code":     row["theirs"],
            "resolved_content": row["expected_resolution"],
            "ai_model":        None,
            "ai_confidence":   None,
            "ai_rationale":    None,
            "used_rag":        False,
            "base_oid":        "",
            "ours_oid":        "",
            "theirs_oid":      "",
            "rerere_hash":     rerere_key,
            "embedding":       embedding,
            "conflict_text":   conflict_text,
        }

        resolution_id      = await persist_resolution(pool, record)
        id_map[row["id"]]  = resolution_id
        print(f"  ingested {row['id']}", flush=True)

    already = len(rows) - sum(1 for v in id_map.values() if v)
    print(
        f"Ingestion complete: {len(id_map)} entries "
        f"({len(rows) - already} newly inserted, {already} already present).",
        flush=True,
    )
    return id_map


# ---------------------------------------------------------------------------
# Retrieval eval (keyless)
# ---------------------------------------------------------------------------

async def run_retrieval_eval(
    pool,
    rows: list[dict],
    id_map: dict[str, str],
) -> tuple[float, list[float]]:
    """Leave-one-out retrieval: exclude self, compute Context Precision per row."""
    print("\n--- Retrieval Eval (keyless) ---", flush=True)
    precisions: list[float] = []

    for row in rows:
        resolution_id = id_map.get(row["id"])
        language      = row.get("language", "text")
        gold          = row["expected_resolution"]

        embedding = embed_conflict(
            file_path=row["file_path"],
            language=language,
            conflict_kind="text",
            base=row["base"],
            ours=row["ours"],
            theirs=row["theirs"],
        )
        conflict_text = build_conflict_text(
            file_path=row["file_path"],
            language=language,
            conflict_kind="text",
            base=row["base"],
            ours=row["ours"],
            theirs=row["theirs"],
        )

        try:
            neighbors = await hybrid_search(
                pool=pool,
                query_text=conflict_text,
                query_embedding=embedding,
                language=language,
                k=_RAG_TOP_K,
                exclude_id=resolution_id,
            )
        except Exception as exc:
            print(f"  WARN {row['id']}: retrieval failed: {exc}", flush=True)
            precisions.append(0.0)
            continue

        retrieved_resolutions = [n.resolved_content for n in neighbors]
        cp = context_precision(retrieved_resolutions, gold)
        precisions.append(cp)
        print(
            f"  {row['id']}: {len(neighbors)} neighbors  "
            f"context_precision={cp:.3f}",
            flush=True,
        )

    mean_cp = sum(precisions) / len(precisions) if precisions else 0.0
    print(
        f"\nMean Context Precision (k={_RAG_TOP_K}, threshold=0.7): {mean_cp:.4f}",
        flush=True,
    )
    return mean_cp, precisions


# ---------------------------------------------------------------------------
# Generation eval (needs ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

async def run_generation_eval(pool, rows: list[dict]) -> dict | None:
    """Per-row: produce_suggestion(rag=True) vs (rag=False), score vs gold."""
    from app.rag import produce_suggestion

    print("\n--- Generation Eval (with key) ---", flush=True)
    results: list[dict] = []

    for row in rows:
        gold     = row["expected_resolution"]
        conflict = _make_conflict_detail(row)

        # RAG path
        try:
            sug_rag, _ = await produce_suggestion(
                pool=pool, conflict=conflict, use_rag=True
            )
            em_rag = exact_match(sug_rag.text, gold)
            ed_rag = edit_distance_ratio(sug_rag.text, gold)
        except Exception as exc:
            print(f"  {row['id']} RAG error: {exc}", flush=True)
            em_rag, ed_rag = False, 0.0

        # Baseline (no RAG)
        try:
            sug_norag, _ = await produce_suggestion(
                pool=None, conflict=conflict, use_rag=False
            )
            em_norag = exact_match(sug_norag.text, gold)
            ed_norag = edit_distance_ratio(sug_norag.text, gold)
        except Exception as exc:
            print(f"  {row['id']} no-RAG error: {exc}", flush=True)
            em_norag, ed_norag = False, 0.0

        results.append(
            {
                "id": row["id"],
                "em_rag": em_rag,
                "ed_rag": ed_rag,
                "em_norag": em_norag,
                "ed_norag": ed_norag,
            }
        )
        print(
            f"  {row['id']}: "
            f"EM rag={int(em_rag)}/norag={int(em_norag)}  "
            f"ED rag={ed_rag:.3f}/norag={ed_norag:.3f}",
            flush=True,
        )

    n = len(results)
    if n == 0:
        return None

    mean_em_rag   = sum(r["em_rag"]   for r in results) / n
    mean_ed_rag   = sum(r["ed_rag"]   for r in results) / n
    mean_em_norag = sum(r["em_norag"] for r in results) / n
    mean_ed_norag = sum(r["ed_norag"] for r in results) / n

    w = 25
    print(f"\n{'Metric':<{w}} {'no-RAG':>10} {'RAG':>10} {'Δ (RAG−no-RAG)':>16}")
    print("-" * (w + 38))
    print(
        f"{'Exact Match (%)':<{w}} "
        f"{mean_em_norag*100:>10.1f} {mean_em_rag*100:>10.1f} "
        f"{(mean_em_rag - mean_em_norag)*100:>+16.1f}"
    )
    print(
        f"{'Edit Distance Ratio':<{w}} "
        f"{mean_ed_norag:>10.4f} {mean_ed_rag:>10.4f} "
        f"{(mean_ed_rag - mean_ed_norag):>+16.4f}"
    )

    return {
        "n":             n,
        "mean_em_rag":   mean_em_rag,
        "mean_ed_rag":   mean_ed_rag,
        "mean_em_norag": mean_em_norag,
        "mean_ed_norag": mean_ed_norag,
        "per_row":       results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    has_key = bool(api_key)

    print("gfix-cloud eval harness", flush=True)
    print(
        f"ANTHROPIC_API_KEY: "
        f"{'set' if has_key else 'NOT SET — keyless retrieval eval only'}",
        flush=True,
    )

    rows = load_golden_set()
    print(f"Loaded {len(rows)} golden-set rows from {GOLDEN_SET_PATH}", flush=True)

    await run_migrations(settings.database_url)
    pool = await open_pool(settings.database_url)

    summary: dict = {
        "golden_set_size": len(rows),
        "rag_top_k":       _RAG_TOP_K,
    }

    try:
        id_map = await ingest_golden_set(pool, rows)

        # --- Retrieval eval (keyless) ---
        mean_cp, per_row_cp = await run_retrieval_eval(pool, rows, id_map)
        summary["retrieval"] = {
            "mean_context_precision": mean_cp,
            "threshold":              0.7,
            "per_row":                per_row_cp,
        }

        # --- Generation eval (needs key) ---
        if has_key:
            gen = await run_generation_eval(pool, rows)
            summary["generation"] = gen
        else:
            print(
                "\n[set ANTHROPIC_API_KEY to populate the generation delta table]",
                flush=True,
            )
            summary["generation"] = None

    finally:
        await close_pool(pool)

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {SUMMARY_PATH}", flush=True)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gfix-cloud eval harness")
    parser.add_argument(
        "--with-gfix-baseline",
        action="store_true",
        help="Add gfix BYOK column (needs GITFIX_BYOK=1 + ANTHROPIC_API_KEY)",
    )
    parsed = parser.parse_args()
    sys.exit(asyncio.run(main(parsed)))
