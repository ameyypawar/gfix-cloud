-- Phase 2: pgvector schema + HNSW index
-- Idempotent: safe to re-run on every startup.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS past_resolutions (
    resolution_id   uuid          PRIMARY KEY,
    merge_id        text,
    file_path       text          NOT NULL,
    language        text,
    conflict_kind   text,
    resolution_kind text,
    base_code       text,
    ours_code       text,
    theirs_code     text,
    resolved_content text,
    ai_model        text,
    ai_confidence   real,
    ai_rationale    text,
    used_rag        boolean       NOT NULL DEFAULT false,
    base_oid        text,
    ours_oid        text,
    theirs_oid      text,
    rerere_hash     text,
    created_at      timestamptz   NOT NULL DEFAULT now(),
    embedding       vector(384),
    conflict_tsv    tsvector
);

-- HNSW index for inner-product ANN (bge-small-en-v1.5 is L2-normalized → use IP)
CREATE INDEX IF NOT EXISTS past_resolutions_embedding_hnsw_idx
    ON past_resolutions
    USING hnsw (embedding vector_ip_ops)
    WITH (m = 16, ef_construction = 128);

-- Full-text search on conflict text
CREATE INDEX IF NOT EXISTS past_resolutions_tsv_gin_idx
    ON past_resolutions
    USING gin (conflict_tsv);

-- Language hard-filter (used in every retrieval query)
CREATE INDEX IF NOT EXISTS past_resolutions_language_btree_idx
    ON past_resolutions (language);
