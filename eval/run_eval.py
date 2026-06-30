#!/usr/bin/env python3
"""
Leave-one-out RAG eval harness for gfix-cloud.

Keyless path (no GEMINI_API_KEY set):
  1. Ingests golden_set.jsonl into pgvector (idempotent — skips existing rows).
  2. Leave-one-out retrieval: for each row, retrieve top-k neighbors using
     exclude_id=<self> so a conflict never retrieves its own resolution.
  3. Reports Context Precision per regime bucket (recurring / one-off / overall).
     recurring = basename with ≥3 members; one-off = <3 members.
  4. Writes eval/summary.json (retrieval section populated per-bucket).

With GEMINI_API_KEY (or ANTHROPIC_API_KEY + GENERATION_PROVIDER=anthropic):
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
  python eval/run_eval.py           # keyless — prints per-regime Context Precision
  python eval/run_eval.py           # with GEMINI_API_KEY for generation delta
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
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

# Minimum family size to be classified as "recurring"
_RECURRING_THRESHOLD = 3


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


def _bucket_rows(rows: list[dict]) -> tuple[Counter, dict[str, str]]:
    """Classify each row as 'recurring' or 'one_off'.

    Family key = normalized basename of file_path.
    Families with ≥ _RECURRING_THRESHOLD members → recurring.

    Returns (family_counts, row_id_to_regime).
    """
    family_counts: Counter = Counter(Path(r["file_path"]).name for r in rows)
    row_to_regime: dict[str, str] = {}
    for r in rows:
        name = Path(r["file_path"]).name
        row_to_regime[r["id"]] = (
            "recurring" if family_counts[name] >= _RECURRING_THRESHOLD else "one_off"
        )
    return family_counts, row_to_regime


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
# Retrieval eval (keyless) — per-regime
# ---------------------------------------------------------------------------

async def run_retrieval_eval(
    pool,
    rows: list[dict],
    id_map: dict[str, str],
    row_to_regime: dict[str, str],
) -> dict:
    """Leave-one-out retrieval: exclude self, compute Context Precision per row.

    Returns a dict with per-regime and overall stats.
    """
    print("\n--- Retrieval Eval (keyless, per-regime) ---", flush=True)
    precisions: dict[str, list[float]] = {"recurring": [], "one_off": [], "overall": []}

    for row in rows:
        resolution_id = id_map.get(row["id"])
        language      = row.get("language", "text")
        gold          = row["expected_resolution"]
        regime        = row_to_regime[row["id"]]

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
            precisions[regime].append(0.0)
            precisions["overall"].append(0.0)
            continue

        retrieved_resolutions = [n.resolved_content for n in neighbors]
        cp = context_precision(retrieved_resolutions, gold)
        precisions[regime].append(cp)
        precisions["overall"].append(cp)
        print(
            f"  [{regime:9s}] {row['id']}: {len(neighbors)} neighbors  "
            f"context_precision={cp:.3f}",
            flush=True,
        )

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    stats: dict = {}
    for bucket in ("recurring", "one_off", "overall"):
        ps = precisions[bucket]
        stats[bucket] = {
            "count":                  len(ps),
            "mean_context_precision": _mean(ps),
            "threshold":              0.7,
        }

    # Print clean per-bucket table
    w = 12
    print(f"\n{'Bucket':<{w}} | {'Count':>6} | {'Mean Context Precision':>22}")
    print("-" * (w + 35))
    for bucket in ("recurring", "one_off", "overall"):
        s = stats[bucket]
        print(
            f"{bucket:<{w}} | {s['count']:>6} | {s['mean_context_precision']:>22.4f}"
        )

    return stats


# ---------------------------------------------------------------------------
# Generation eval (needs generation key)
# ---------------------------------------------------------------------------

async def run_generation_eval(
    pool,
    rows: list[dict],
    row_to_regime: dict[str, str],
) -> dict | None:
    """Per-row: produce_suggestion(rag=True) vs (rag=False), score vs gold.
    Returns bucketed results.
    """
    from app.rag import produce_suggestion

    print("\n--- Generation Eval (with key) ---", flush=True)
    results: list[dict] = []

    for row in rows:
        gold     = row["expected_resolution"]
        conflict = _make_conflict_detail(row)
        regime   = row_to_regime[row["id"]]

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
                "id":      row["id"],
                "regime":  regime,
                "em_rag":  em_rag,
                "ed_rag":  ed_rag,
                "em_norag": em_norag,
                "ed_norag": ed_norag,
            }
        )
        print(
            f"  [{regime:9s}] {row['id']}: "
            f"EM rag={int(em_rag)}/norag={int(em_norag)}  "
            f"ED rag={ed_rag:.3f}/norag={ed_norag:.3f}",
            flush=True,
        )

    if not results:
        return None

    def _bucket_stats(subset: list[dict]) -> dict:
        n = len(subset)
        if n == 0:
            return None
        return {
            "n":             n,
            "mean_em_rag":   sum(r["em_rag"]   for r in subset) / n,
            "mean_ed_rag":   sum(r["ed_rag"]   for r in subset) / n,
            "mean_em_norag": sum(r["em_norag"] for r in subset) / n,
            "mean_ed_norag": sum(r["ed_norag"] for r in subset) / n,
        }

    by_regime: dict[str, list[dict]] = {"recurring": [], "one_off": []}
    for r in results:
        by_regime[r["regime"]].append(r)

    gen_buckets = {
        "recurring": _bucket_stats(by_regime["recurring"]),
        "one_off":   _bucket_stats(by_regime["one_off"]),
        "overall":   _bucket_stats(results),
        "per_row":   results,
    }

    # Print overall table
    overall = gen_buckets["overall"]
    w = 25
    print(f"\n{'Metric':<{w}} {'no-RAG':>10} {'RAG':>10} {'Δ (RAG−no-RAG)':>16}")
    print("-" * (w + 38))
    print(
        f"{'Exact Match (%)':<{w}} "
        f"{overall['mean_em_norag']*100:>10.1f} {overall['mean_em_rag']*100:>10.1f} "
        f"{(overall['mean_em_rag'] - overall['mean_em_norag'])*100:>+16.1f}"
    )
    print(
        f"{'Edit Distance Ratio':<{w}} "
        f"{overall['mean_ed_norag']:>10.4f} {overall['mean_ed_rag']:>10.4f} "
        f"{(overall['mean_ed_rag'] - overall['mean_ed_norag']):>+16.4f}"
    )

    return gen_buckets


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> int:
    # Detect which key is active
    gemini_key   = os.environ.get("GEMINI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    provider     = settings.generation_provider
    has_key      = bool(gemini_key) if provider == "gemini" else bool(anthropic_key)

    print("gfix-cloud eval harness", flush=True)
    print(f"provider: {provider}  model: {settings.generation_model}", flush=True)
    if not has_key:
        active_key_name = "GEMINI_API_KEY" if provider == "gemini" else "ANTHROPIC_API_KEY"
        print(
            f"{active_key_name}: NOT SET — keyless retrieval eval only",
            flush=True,
        )

    rows = load_golden_set()
    print(f"Loaded {len(rows)} golden-set rows from {GOLDEN_SET_PATH}", flush=True)

    family_counts, row_to_regime = _bucket_rows(rows)
    recurring_families = sorted(
        f for f, c in family_counts.items() if c >= _RECURRING_THRESHOLD
    )
    print(
        f"Regime split: {sum(1 for v in row_to_regime.values() if v=='recurring')} recurring"
        f" / {sum(1 for v in row_to_regime.values() if v=='one_off')} one-off"
        f"  (recurring families: {recurring_families})",
        flush=True,
    )

    await run_migrations(settings.database_url)
    pool = await open_pool(settings.database_url)

    try:
        id_map = await ingest_golden_set(pool, rows)

        # --- Retrieval eval (keyless, per-regime) ---
        retrieval_stats = await run_retrieval_eval(pool, rows, id_map, row_to_regime)

        # --- Generation eval (needs key) ---
        if has_key:
            gen_buckets = await run_generation_eval(pool, rows, row_to_regime)
        else:
            gen_buckets = None
            print(
                f"\n[set {'GEMINI_API_KEY' if provider == 'gemini' else 'ANTHROPIC_API_KEY'}"
                " to populate the generation delta table]",
                flush=True,
            )

    finally:
        await close_pool(pool)

    # Build summary.json with per-bucket structure
    summary: dict = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "provider":        provider,
        "model":           settings.generation_model,
        "golden_set_size": len(rows),
        "rag_top_k":       _RAG_TOP_K,
        "per_family_counts": dict(family_counts.most_common()),
        "buckets": {
            "recurring": {
                "rows":     retrieval_stats["recurring"]["count"],
                "families": recurring_families,
                "retrieval": {
                    "mean_context_precision": retrieval_stats["recurring"]["mean_context_precision"],
                    "threshold":              0.7,
                },
                "generation": (
                    gen_buckets["recurring"] if gen_buckets else "pending-key"
                ),
            },
            "one_off": {
                "rows":     retrieval_stats["one_off"]["count"],
                "retrieval": {
                    "mean_context_precision": retrieval_stats["one_off"]["mean_context_precision"],
                    "threshold":              0.7,
                },
                "generation": (
                    gen_buckets["one_off"] if gen_buckets else "pending-key"
                ),
            },
            "overall": {
                "rows":     retrieval_stats["overall"]["count"],
                "retrieval": {
                    "mean_context_precision": retrieval_stats["overall"]["mean_context_precision"],
                    "threshold":              0.7,
                },
                "generation": (
                    gen_buckets["overall"] if gen_buckets else "pending-key"
                ),
            },
        },
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {SUMMARY_PATH}", flush=True)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gfix-cloud eval harness")
    parser.add_argument(
        "--with-gfix-baseline",
        action="store_true",
        help="Add gfix BYOK column (needs GITFIX_BYOK=1 + generation key)",
    )
    parsed = parser.parse_args()
    sys.exit(asyncio.run(main(parsed)))
