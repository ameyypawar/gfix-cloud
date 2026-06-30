"""
Hybrid retrieval: HNSW inner-product (<#>) + BM25 tsvector, fused via RRF in SQL.

RRF k=60; weights: vector=0.6, bm25=0.4.  Language is a hard filter on both arms.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Neighbor:
    resolution_id: str
    file_path: str
    language: str
    base_code: str
    ours_code: str
    theirs_code: str
    resolved_content: str
    resolution_kind: str
    ai_rationale: Optional[str]
    rrf_score: float


_RRF_SQL = """
WITH vec AS (
    SELECT resolution_id,
           row_number() OVER (ORDER BY embedding <#> $1) AS rank
    FROM past_resolutions
    WHERE language = $2
      AND ($3::text IS NULL OR resolution_id::text != $3)
    ORDER BY embedding <#> $1
    LIMIT 20
),
bm25 AS (
    SELECT resolution_id,
           row_number() OVER (
               ORDER BY ts_rank(conflict_tsv, plainto_tsquery('english', $4)) DESC
           ) AS rank
    FROM past_resolutions
    WHERE language = $2
      AND ($3::text IS NULL OR resolution_id::text != $3)
      AND conflict_tsv @@ plainto_tsquery('english', $4)
    ORDER BY ts_rank(conflict_tsv, plainto_tsquery('english', $4)) DESC
    LIMIT 20
),
fused AS (
    SELECT id, SUM(weight / (60.0 + rank)) AS rrf
    FROM (
        SELECT resolution_id AS id, rank, 0.6 AS weight FROM vec
        UNION ALL
        SELECT resolution_id AS id, rank, 0.4 AS weight FROM bm25
    ) combined
    GROUP BY id
    ORDER BY rrf DESC
    LIMIT $5
)
SELECT
    pr.resolution_id, pr.file_path, pr.language,
    pr.base_code, pr.ours_code, pr.theirs_code,
    pr.resolved_content, pr.resolution_kind, pr.ai_rationale,
    f.rrf
FROM fused f
JOIN past_resolutions pr ON pr.resolution_id = f.id
ORDER BY f.rrf DESC
"""


async def hybrid_search(
    pool,
    query_text: str,
    query_embedding: list[float],
    language: str,
    k: int,
    exclude_id: Optional[str] = None,
) -> list[Neighbor]:
    """
    Hybrid retrieval using a single SQL RRF CTE.

    Both vector and BM25 arms apply the language hard-filter.
    exclude_id: skip a specific resolution_id (leave-one-out eval support).
    """
    emb_arr = np.array(query_embedding, dtype=np.float32)
    exclude = str(exclude_id) if exclude_id else None

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            _RRF_SQL,
            emb_arr,
            language,
            exclude,
            query_text,
            k,
        )

    return [
        Neighbor(
            resolution_id=str(r["resolution_id"]),
            file_path=r["file_path"],
            language=r["language"],
            base_code=r["base_code"] or "",
            ours_code=r["ours_code"] or "",
            theirs_code=r["theirs_code"] or "",
            resolved_content=r["resolved_content"] or "",
            resolution_kind=r["resolution_kind"] or "",
            ai_rationale=r["ai_rationale"],
            rrf_score=float(r["rrf"]),
        )
        for r in rows
    ]
