"""
Idempotent seed loader for gfix-cloud.

Reads /eval/golden_set.jsonl, embeds each conflict with bge-small-en-v1.5,
and inserts into past_resolutions.  Rows already present (matched by
rerere_hash) are skipped.  Safe to run multiple times.

Usage (from WORKDIR /app inside the api container):
    python3 -m app.seed
"""
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/gfixcloud",
)

# Absolute path inside the container (api/Dockerfile: COPY eval/ /eval/)
GOLDEN_SET_PATH = Path("/eval/golden_set.jsonl")


def _pseudo_oid(content: str) -> str:
    """40-char hex from content — proxy for a git blob OID."""
    return hashlib.sha1(content.encode()).hexdigest()


def _rerere_hash(file_path: str, base_oid: str, ours_oid: str, theirs_oid: str) -> str:
    """Mirrors the gfix rerere content key scheme."""
    import blake3  # imported here so top-level import doesn't fail if missing

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


async def main() -> None:
    import asyncpg
    import numpy as np
    from pgvector.asyncpg import register_vector
    from sentence_transformers import SentenceTransformer

    logger.info("seed: loading embedding model (BAAI/bge-small-en-v1.5)…")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    db_host = DATABASE_URL.split("@")[-1]
    logger.info("seed: connecting to %s", db_host)
    conn = await asyncpg.connect(DATABASE_URL)
    await register_vector(conn)

    raw_rows = GOLDEN_SET_PATH.read_text().splitlines()
    logger.info("seed: %d golden-set entries to process", len(raw_rows))

    inserted = 0
    skipped = 0

    for raw in raw_rows:
        entry = json.loads(raw)

        file_path = entry["file_path"]
        base = entry.get("base", "")
        ours = entry.get("ours", "")
        theirs = entry.get("theirs", "")
        expected = entry.get("expected_resolution", "")
        language = entry.get("language", "")
        merge_id = entry["id"]

        base_oid = _pseudo_oid(base)
        ours_oid = _pseudo_oid(ours)
        theirs_oid = _pseudo_oid(theirs)
        rh = _rerere_hash(file_path, base_oid, ours_oid, theirs_oid)

        # Idempotency: skip rows already present by content key
        existing = await conn.fetchval(
            "SELECT resolution_id FROM past_resolutions WHERE rerere_hash = $1 LIMIT 1",
            rh,
        )
        if existing:
            skipped += 1
            continue

        # Conflict text for both embedding and tsvector
        conflict_text = f"{file_path} {language} {base} {ours} {theirs}"
        # Truncate to avoid tsvector limits (max 1MB, practical cap ~200k chars)
        conflict_text_trunc = conflict_text[:200_000]

        embedding = model.encode(conflict_text, normalize_embeddings=True).astype(np.float32)

        resolution_id = str(uuid4())
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
            merge_id,
            file_path,
            language,
            "semantic",
            "golden_set",
            base,
            ours,
            theirs,
            expected,
            None,          # ai_model
            None,          # ai_confidence
            f"golden-set seed: {entry.get('provenance', '')}",
            False,         # used_rag
            base_oid,
            ours_oid,
            theirs_oid,
            rh,
            embedding,
            conflict_text_trunc,
        )
        inserted += 1

    await conn.close()
    logger.info("seed complete: inserted=%d  skipped=%d", inserted, skipped)


if __name__ == "__main__":
    asyncio.run(main())
