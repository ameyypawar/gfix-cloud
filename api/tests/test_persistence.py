"""
Phase 2 tests: persistence, embeddings, rerere_hash.

Unit tests: always run (no external deps).
Integration tests: gated on docker CLI; spin a real pgvector container on
port 5433 to avoid clashing with any compose db on 5432.
"""
import math
import os
import shutil
import subprocess
import time
from typing import Generator

import pytest

from app.embeddings import (
    build_conflict_text,
    embed_conflict,
    language_from_path,
)
from app.persistence import rerere_hash


# ─── Unit: rerere_hash ────────────────────────────────────────────────────────

def test_rerere_hash_known_value():
    """Pin against a pre-computed blake3 hex so algorithm regressions are caught."""
    # blake3(b"v1\x00path/to/file.py\x00abc123\x00def456\x00ghi789")
    expected = "59e59af99c302da2c347dbcf5b45c6c05dfa8a9ebb359415a55bd10cf595afdd"
    result = rerere_hash("path/to/file.py", "abc123", "def456", "ghi789")
    assert result == expected


def test_rerere_hash_is_64_hex_chars():
    h = rerere_hash("src/main.rs", "a" * 40, "b" * 40, "c" * 40)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_rerere_hash_differs_with_different_oids():
    h1 = rerere_hash("file.py", "aaa", "bbb", "ccc")
    h2 = rerere_hash("file.py", "aaa", "bbb", "ddd")
    assert h1 != h2


# ─── Unit: language_from_path ─────────────────────────────────────────────────

@pytest.mark.parametrize("path,expected", [
    ("src/lib.rs", "rust"),
    ("app/main.py", "python"),
    ("frontend/index.ts", "typescript"),
    ("frontend/App.tsx", "typescript"),
    ("scripts/run.js", "javascript"),
    ("handler.go", "go"),
    ("Main.java", "java"),
    ("util.c", "c"),
    ("header.h", "c"),
    ("module.cpp", "cpp"),
    ("module.hpp", "cpp"),
    ("app.rb", "ruby"),
    ("README.md", "markdown"),
    ("data.json", "text"),
    ("Makefile", "text"),
])
def test_language_from_path(path, expected):
    assert language_from_path(path) == expected


# ─── Unit: embed_conflict + build_conflict_text ───────────────────────────────

def test_build_conflict_text_contains_markers():
    text = build_conflict_text(
        file_path="a.py",
        language="python",
        conflict_kind="ast",
        base="base code",
        ours="ours code",
        theirs="theirs code",
        context="extra",
    )
    assert "[CONFLICT]" in text
    assert "file_path=a.py" in text
    assert "lang=python" in text
    assert "kind=ast" in text
    assert "[BASE]" in text
    assert "[OURS]" in text
    assert "[THEIRS]" in text
    assert "[CONTEXT]" in text
    assert "base code" in text
    assert "ours code" in text
    assert "theirs code" in text
    assert "extra" in text


def test_embed_conflict_dim():
    vec = embed_conflict(
        file_path="main.py",
        language="python",
        conflict_kind="ast",
        base="def foo(): pass",
        ours="def foo(): return 1",
        theirs="def foo(): return 2",
    )
    assert len(vec) == 384


def test_embed_conflict_normalized():
    vec = embed_conflict(
        file_path="main.py",
        language="python",
        conflict_kind="ast",
        base="def foo(): pass",
        ours="def foo(): return 1",
        theirs="def foo(): return 2",
    )
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-5, f"expected L2-norm ≈ 1.0, got {norm}"


# ─── Integration: real pgvector on Docker ────────────────────────────────────

_docker_present = pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="docker not on PATH",
)

TEST_CONTAINER_NAME = "gfixcloud-test-pg"
TEST_HOST_PORT = 5433
TEST_DB_URL = f"postgresql://postgres:postgres@localhost:{TEST_HOST_PORT}/gfixcloud"


def _wait_pg_ready(timeout: int = 30) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = subprocess.run(
            [
                "docker", "exec", TEST_CONTAINER_NAME,
                "pg_isready", "-U", "postgres", "-d", "gfixcloud",
            ],
            capture_output=True,
        )
        if r.returncode == 0:
            return True
        time.sleep(1)
    return False


@pytest.fixture(scope="module")
def pg_container():
    """Spin up a pgvector container for the module, tear it down after."""
    # Clean up any leftover container from a previous failed run
    subprocess.run(
        ["docker", "rm", "-f", TEST_CONTAINER_NAME],
        capture_output=True,
    )

    subprocess.run(
        [
            "docker", "run", "-d", "--rm",
            "--name", TEST_CONTAINER_NAME,
            "-e", "POSTGRES_PASSWORD=postgres",
            "-e", "POSTGRES_DB=gfixcloud",
            "-p", f"{TEST_HOST_PORT}:5432",
            "pgvector/pgvector:pg16",
        ],
        check=True,
        capture_output=True,
    )

    if not _wait_pg_ready(timeout=60):
        subprocess.run(["docker", "rm", "-f", TEST_CONTAINER_NAME], capture_output=True)
        pytest.fail("pgvector container did not become ready within 60s")

    yield TEST_DB_URL

    subprocess.run(["docker", "rm", "-f", TEST_CONTAINER_NAME], capture_output=True)


@_docker_present
@pytest.mark.asyncio
async def test_integration_persist_and_find_similar(pg_container):
    """
    Insert 3 past_resolutions with very different conflict texts.
    Query with a vector close to record[0] → assert record[0] ranks first.
    Also verify dim=384 via the score field being a float.
    """
    import asyncio
    import asyncpg
    import numpy as np
    from pgvector.asyncpg import register_vector

    from app import db as _db
    from app import persistence as _pers
    from app.config import settings

    # Override DATABASE_URL for this test
    original_url = settings.database_url
    settings.database_url = pg_container

    try:
        await _db.run_migrations(pg_container)
        pool = await _db.open_pool(pg_container)

        # --- Three semantically distinct Python conflicts ---

        # Record A: retry with exponential backoff
        rec_a_base = "def retry(fn):\n    return fn()\n"
        rec_a_ours = "def retry(fn, attempts=3):\n    for i in range(attempts):\n        try:\n            return fn()\n        except Exception:\n            pass\n"
        rec_a_theirs = "def retry(fn, delay=1.0):\n    time.sleep(delay)\n    return fn()\n"

        # Record B: database connection pool
        rec_b_base = "def connect():\n    return psycopg2.connect(DSN)\n"
        rec_b_ours = "def connect():\n    pool = ConnectionPool(DSN, min=2, max=10)\n    return pool\n"
        rec_b_theirs = "def connect():\n    return psycopg2.connect(DSN, connect_timeout=5)\n"

        # Record C: JSON schema validation
        rec_c_base = "def parse(data):\n    return json.loads(data)\n"
        rec_c_ours = "def parse(data):\n    obj = json.loads(data)\n    validate(obj, schema=SCHEMA)\n    return obj\n"
        rec_c_theirs = "def parse(data):\n    return json.loads(data, strict=False)\n"

        from app.embeddings import embed_conflict, build_conflict_text, language_from_path

        def make_record(file_path, base, ours, theirs, resolution_kind):
            lang = language_from_path(file_path)
            emb = embed_conflict(file_path, lang, "ast", base, ours, theirs)
            ctxt = build_conflict_text(file_path, lang, "ast", base, ours, theirs)
            rh = _pers.rerere_hash(file_path, "aaa", "bbb", "ccc")
            return {
                "merge_id": "test-merge",
                "file_path": file_path,
                "language": lang,
                "conflict_kind": "ast",
                "resolution_kind": resolution_kind,
                "base_code": base,
                "ours_code": ours,
                "theirs_code": theirs,
                "resolved_content": ours,
                "ai_model": None,
                "ai_confidence": None,
                "ai_rationale": None,
                "used_rag": False,
                "base_oid": "aaa",
                "ours_oid": "bbb",
                "theirs_oid": "ccc",
                "rerere_hash": rh,
                "embedding": emb,
                "conflict_text": ctxt,
            }

        id_a = await _pers.persist_resolution(pool, make_record("retry.py", rec_a_base, rec_a_ours, rec_a_theirs, "ours_chosen"))
        id_b = await _pers.persist_resolution(pool, make_record("db.py", rec_b_base, rec_b_ours, rec_b_theirs, "ours_chosen"))
        id_c = await _pers.persist_resolution(pool, make_record("parser.py", rec_c_base, rec_c_ours, rec_c_theirs, "ours_chosen"))

        assert id_a and id_b and id_c  # valid uuid strings

        # Query with the retry conflict embedding → should rank A first
        query_emb = embed_conflict("retry.py", "python", "ast", rec_a_base, rec_a_ours, rec_a_theirs)

        neighbors = await _pers.find_similar(pool, query_emb, "python", k=3)

        assert len(neighbors) == 3, f"expected 3 neighbors, got {len(neighbors)}"

        # Top result must be the retry conflict (record A)
        # asyncpg returns UUID columns as uuid.UUID objects; compare via str
        top = neighbors[0]
        assert str(top["resolution_id"]) == id_a, (
            f"expected retry record ({id_a}) to rank first, got {top['resolution_id']}"
        )

        # score is a float (inner product, should be negative and close to -1 for self-similarity)
        assert isinstance(top["score"], float), "score should be float"
        # Self-score for normalized vectors: <#> returns negative inner product
        # For nearly-identical embeddings, score ≈ -1.0
        assert top["score"] < -0.9, f"self-similarity score should be close to -1.0, got {top['score']}"

        # Verify embedding dimension indirectly: embed returned 384 floats
        assert len(query_emb) == 384

        await _db.close_pool(pool)
    finally:
        settings.database_url = original_url
