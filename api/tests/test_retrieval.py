"""
Phase 3 retrieval tests: hybrid RRF search (HNSW + BM25).

Integration tests require docker.  Uses a separate container (port 5434) so
it can run concurrently with test_persistence.py's container (port 5433).

Assertions:
  (a) RRF returns top-k in descending rrf_score order
  (b) language hard-filter excludes rows from the other language
  (c) BM25 recall: a query naming a specific identifier (rebase_apply_commit)
      retrieves that row even when vector-only search ranks it below other rows
      that are semantically similar to the query embedding.
"""
import shutil
import subprocess
import time
import uuid

import pytest

from app.embeddings import embed_conflict, build_conflict_text, language_from_path
from app.persistence import rerere_hash

_docker_present = pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="docker not on PATH",
)

TEST_CONTAINER_NAME = "gfixcloud-test-pg-retrieval"
TEST_HOST_PORT = 5434
TEST_DB_URL = f"postgresql://postgres:postgres@localhost:{TEST_HOST_PORT}/gfixcloud"


def _wait_pg_ready(timeout: int = 60) -> bool:
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
    subprocess.run(["docker", "rm", "-f", TEST_CONTAINER_NAME], capture_output=True)
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
    if not _wait_pg_ready():
        subprocess.run(["docker", "rm", "-f", TEST_CONTAINER_NAME], capture_output=True)
        pytest.fail("pgvector container did not become ready in time")
    yield TEST_DB_URL
    subprocess.run(["docker", "rm", "-f", TEST_CONTAINER_NAME], capture_output=True)


def _make_record(file_path: str, base: str, ours: str, theirs: str, kind: str = "ours") -> dict:
    lang = language_from_path(file_path)
    emb = embed_conflict(file_path, lang, "ast", base, ours, theirs)
    ctxt = build_conflict_text(file_path, lang, "ast", base, ours, theirs)
    rh = rerere_hash(file_path, "aaa", "bbb", "ccc")
    return {
        "merge_id": "test-merge-retrieval",
        "file_path": file_path,
        "language": lang,
        "conflict_kind": "ast",
        "resolution_kind": kind,
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


@_docker_present
async def test_hybrid_search_rrf_order_and_language_filter(pg_container):
    """
    (a) RRF results are ordered descending by rrf_score.
    (b) Language hard-filter: rust query returns only rust rows.
    """
    from app import db as _db
    from app import persistence as _pers
    from app.retrieval import hybrid_search
    from app.config import settings

    orig = settings.database_url
    settings.database_url = pg_container
    try:
        await _db.run_migrations(pg_container)
        pool = await _db.open_pool(pg_container)

        # Seed 3 rust rows + 3 python rows
        rust_base = "fn authenticate(token: &str) -> bool { token == \"secret\" }"
        rust_ours = "fn authenticate(token: &str) -> bool { validate_jwt(token) }"
        rust_theirs = "fn authenticate(token: &str) -> bool { check_db(token) }"

        rust2_base = "fn authorize(user: &User) -> bool { user.role == Role::Admin }"
        rust2_ours = "fn authorize(user: &User) -> bool { user.permissions.contains(&\"write\") }"
        rust2_theirs = "fn authorize(user: &User) -> bool { user.role == Role::Admin || user.is_superuser }"

        rust3_base = "fn login(creds: Credentials) -> Session { Session::new(creds.username) }"
        rust3_ours = "fn login(creds: Credentials) -> Session { Session::new_with_mfa(creds) }"
        rust3_theirs = "fn login(creds: Credentials) -> Session { Session::from_oauth(creds) }"

        py_base = "def connect(): return psycopg2.connect(DSN)"
        py_ours = "def connect(): return Pool(DSN, min=2)"
        py_theirs = "def connect(): return psycopg2.connect(DSN, timeout=5)"

        id_r1 = await _pers.persist_resolution(pool, _make_record("auth.rs", rust_base, rust_ours, rust_theirs))
        id_r2 = await _pers.persist_resolution(pool, _make_record("authz.rs", rust2_base, rust2_ours, rust2_theirs))
        id_r3 = await _pers.persist_resolution(pool, _make_record("login.rs", rust3_base, rust3_ours, rust3_theirs))
        id_p1 = await _pers.persist_resolution(pool, _make_record("db.py", py_base, py_ours, py_theirs))
        id_p2 = await _pers.persist_resolution(pool, _make_record("conn.py", py_base, py_ours, py_theirs))
        id_p3 = await _pers.persist_resolution(pool, _make_record("pool.py", py_base, py_ours, py_theirs))

        # Query: authentication-related rust embedding
        query_emb = embed_conflict("auth.rs", "rust", "ast", rust_base, rust_ours, rust_theirs)
        query_text = "authenticate user token jwt session"

        # (a) RRF order: results must be descending by rrf_score
        results = await hybrid_search(pool, query_text, query_emb, "rust", k=3)
        assert len(results) == 3
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True), \
            f"RRF scores must be descending: {scores}"

        # (b) Language filter: all results are rust, none are python
        langs = {r.language for r in results}
        assert langs == {"rust"}, f"expected only rust rows, got: {langs}"

        await _db.close_pool(pool)
    finally:
        settings.database_url = orig


@_docker_present
async def test_bm25_recall_beats_vector_only(pg_container):
    """
    (c) BM25 recall: a query naming `rebase_apply_commit` retrieves that row
    even when the vector embedding is tuned toward semantically different rows.

    Setup:
    - auth_row: conflict about authenticate_user, session, login (rust)
    - rebase_row: conflict containing rebase_apply_commit identifier (rust)
    - Vector query embedding: "authenticate user session login" → ranks auth_row 1st
    - BM25 query text: "rebase_apply_commit" → rebase_row is the only match
    - Pure-vector (find_similar): auth_row should rank above rebase_row
    - Hybrid (hybrid_search): rebase_row should rank 1st due to BM25 boost
    """
    from app import db as _db
    from app import persistence as _pers
    from app.retrieval import hybrid_search
    from app.config import settings

    orig = settings.database_url
    settings.database_url = pg_container
    try:
        await _db.run_migrations(pg_container)
        pool = await _db.open_pool(pg_container)

        # Row A: auth — semantically matches "authenticate user session login"
        auth_base = "fn authenticate_user(user: &User, session: &Session) -> bool { false }"
        auth_ours = "fn authenticate_user(user: &User, session: &Session) -> bool { session.is_valid() && user.is_active() }"
        auth_theirs = "fn authenticate_user(user: &User, session: &Session) -> bool { validate_session_token(session.token) }"

        # Row B: rebase — contains the specific identifier rebase_apply_commit
        rebase_base = "fn rebase_apply_commit(commit: &Commit, target: &Branch) -> Result<()> { Err(Error::NotImplemented) }"
        rebase_ours = "fn rebase_apply_commit(commit: &Commit, target: &Branch) -> Result<()> { target.apply_patch(commit.diff()) }"
        rebase_theirs = "fn rebase_apply_commit(commit: &Commit, target: &Branch) -> Result<()> { cherry_pick(commit, target) }"

        id_auth = await _pers.persist_resolution(pool, _make_record(
            "user_auth.rs", auth_base, auth_ours, auth_theirs
        ))
        id_rebase = await _pers.persist_resolution(pool, _make_record(
            "rebase.rs", rebase_base, rebase_ours, rebase_theirs
        ))

        # Vector query embedding: semantically about "user authentication session"
        # (not about git rebase) → vector-only will rank auth_row first
        auth_query_emb = embed_conflict(
            "user_auth.rs", "rust", "ast",
            auth_base, auth_ours, auth_theirs
        )

        # --- Vector-only (baseline): auth_row should rank above rebase_row ---
        vector_results = await _pers.find_similar(pool, auth_query_emb, "rust", k=2)
        assert len(vector_results) >= 1
        top_vector_id = str(vector_results[0]["resolution_id"])
        assert top_vector_id == id_auth, (
            f"vector-only should rank auth_row first "
            f"(got {top_vector_id}, expected {id_auth})"
        )

        # --- Hybrid search: rebase_apply_commit BM25 match should pull rebase_row to #1 ---
        hybrid_results = await hybrid_search(
            pool=pool,
            query_text="rebase_apply_commit",
            query_embedding=auth_query_emb,  # same auth embedding — vector still favors auth_row
            language="rust",
            k=2,
        )
        assert len(hybrid_results) >= 1
        top_hybrid_id = hybrid_results[0].resolution_id
        assert top_hybrid_id == id_rebase, (
            f"hybrid search should rank rebase_row first due to BM25 recall "
            f"(got {top_hybrid_id}, expected {id_rebase})"
        )
        assert hybrid_results[0].rrf_score > 0

        await _db.close_pool(pool)
    finally:
        settings.database_url = orig


@_docker_present
async def test_exclude_id_filters_self(pg_container):
    """exclude_id removes a specific row from results."""
    from app import db as _db
    from app import persistence as _pers
    from app.retrieval import hybrid_search
    from app.config import settings

    orig = settings.database_url
    settings.database_url = pg_container
    try:
        await _db.run_migrations(pg_container)
        pool = await _db.open_pool(pg_container)

        base = "fn foo() -> i32 { 0 }"
        ours = "fn foo() -> i32 { 1 }"
        theirs = "fn foo() -> i32 { 2 }"
        id_foo = await _pers.persist_resolution(pool, _make_record("foo.rs", base, ours, theirs))

        emb = embed_conflict("foo.rs", "rust", "ast", base, ours, theirs)
        results = await hybrid_search(pool, "foo function return", emb, "rust", k=5)
        ids_present = {r.resolution_id for r in results}
        assert id_foo in ids_present, "foo row should appear without exclusion"

        results_excl = await hybrid_search(
            pool, "foo function return", emb, "rust", k=5, exclude_id=id_foo
        )
        ids_excl = {r.resolution_id for r in results_excl}
        assert id_foo not in ids_excl, "foo row must be excluded when exclude_id is set"

        await _db.close_pool(pool)
    finally:
        settings.database_url = orig
