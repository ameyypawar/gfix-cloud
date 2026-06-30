"""
Persistence helpers: insert resolved conflicts, query similar ones.

rerere_hash mirrors gfix's own rerere scheme so our records can be
cross-referenced with gfix's native rerere refs.
"""
import logging
from uuid import uuid4

import blake3
import asyncpg
import numpy as np

logger = logging.getLogger(__name__)


def rerere_hash(file_path: str, base_oid: str, ours_oid: str, theirs_oid: str) -> str:
    """Compute gfix-compatible rerere content-key.

    Mirrors:  blake3(b"v1\\0" + path + b"\\0" + base_oid + b"\\0" + ours_oid + b"\\0" + theirs_oid)
    """
    data = (
        b"v1\x00"
        + file_path.encode()
        + b"\x00"
        + base_oid.encode()
        + b"\x00"
        + ours_oid.encode()
        + b"\x00"
        + theirs_oid.encode()
    )
    return blake3.blake3(data).hexdigest()


async def persist_resolution(pool: asyncpg.Pool, record: dict) -> str:
    """INSERT a resolved conflict into past_resolutions.

    Returns the new resolution_id (uuid str).

    Expects record keys:
        merge_id, file_path, language, conflict_kind, resolution_kind,
        base_code, ours_code, theirs_code, resolved_content,
        ai_model, ai_confidence, ai_rationale, used_rag,
        base_oid, ours_oid, theirs_oid, rerere_hash,
        embedding (list[float], 384-dim),
        conflict_text (str, for tsvector indexing)
    """
    resolution_id = str(uuid4())
    conflict_text = record.get("conflict_text", "")
    embedding = np.array(record["embedding"], dtype=np.float32)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO past_resolutions (
                resolution_id, merge_id, file_path, language, conflict_kind,
                resolution_kind, base_code, ours_code, theirs_code, resolved_content,
                ai_model, ai_confidence, ai_rationale, used_rag,
                base_oid, ours_oid, theirs_oid, rerere_hash,
                embedding, conflict_tsv
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9, $10,
                $11, $12, $13, $14,
                $15, $16, $17, $18,
                $19, to_tsvector('english', $20)
            )
            """,
            resolution_id,
            record.get("merge_id"),
            record["file_path"],
            record.get("language"),
            record.get("conflict_kind"),
            record.get("resolution_kind"),
            record.get("base_code"),
            record.get("ours_code"),
            record.get("theirs_code"),
            record.get("resolved_content"),
            record.get("ai_model"),
            record.get("ai_confidence"),
            record.get("ai_rationale"),
            bool(record.get("used_rag", False)),
            record.get("base_oid"),
            record.get("ours_oid"),
            record.get("theirs_oid"),
            record.get("rerere_hash"),
            embedding,
            conflict_text,
        )
    return resolution_id


async def find_similar(
    pool: asyncpg.Pool,
    embedding: list[float],
    language: str,
    k: int = 5,
) -> list[dict]:
    """Return up to k nearest neighbors ordered by inner product (descending similarity).

    <#> returns negative inner product — ORDER BY ASC gives the most similar first.
    Filters by language first (HNSW + btree composite lookup).
    """
    embedding_arr = np.array(embedding, dtype=np.float32)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                resolution_id, merge_id, file_path, language, conflict_kind,
                resolution_kind, base_code, ours_code, theirs_code, resolved_content,
                ai_model, ai_confidence, ai_rationale, used_rag,
                base_oid, ours_oid, theirs_oid, rerere_hash,
                created_at,
                (embedding <#> $1) AS score
            FROM past_resolutions
            WHERE language = $2
            ORDER BY embedding <#> $1
            LIMIT $3
            """,
            embedding_arr,
            language,
            k,
        )
    return [dict(r) for r in rows]
