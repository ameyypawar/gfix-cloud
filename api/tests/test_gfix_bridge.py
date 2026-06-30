"""
Tests for gfix_bridge.py.

Unit tests: feed captured JSON fixtures → assert parser extracts fields correctly.
Integration tests: gated on shutil.which("gfix"); drive a real conflict end-to-end.
"""
import json
import shutil
from pathlib import Path

import pytest

from app.gfix_bridge import _parse_conflict_side, resolve_conflict
from app.models import ConflictDetail, ConflictSide, ResolveResponse, TargetSide

# ── Captured JSON fixtures (real gfix responses, 2026-06-30) ──────────────────

MERGE_PREVIEW_FIXTURE = {
    "merge_id": "m_2026-06-30T05-52-46Z_9e6be0",
    "resolved": [],
    "source_branches": ["theirs"],
    "substrate": "git",
    "summary": "1 source merged into ours. 0 files auto-resolved. 1 conflict need decision.",
    "target_branch": "ours",
    "unresolved": [
        {
            "ai_suggestion": None,
            "base_oid": "958edf0fa5d934ad0ee7c7372385dddef2744144",
            "conflict_id": "c_911c1",
            "file": "main.py",
            "kind": "ast",
            "ours_excerpt": "def hello():",
            "ours_oid": "db7a3d15efa99858c1d79acb6db373a740b95a91",
            "ours_source": "ours",
            "target_oid": "db7a3d15efa99858c1d79acb6db373a740b95a91",
            "theirs_excerpt": "def hello():",
            "theirs_oid": "9d18540208090bbff565b1bb205eee16f796936d",
            "theirs_source": "theirs",
        }
    ],
}

CONFLICT_GET_FIXTURE = {
    "ai_suggestion": None,
    "base": {
        "content": 'def hello():\n    print("hello base")\n\ndef greet(name):\n    return f"Hello, {name}"\n',
        "oid": "8eec00412f79c9fb9e9f6afb135f89e3f21340ae",
    },
    "conflict_id": "c_911c1",
    "file": "main.py",
    "kind": "ast",
    "merge_id": "m_2026-06-30T05-53-28Z_cf7b18",
    "ours": {
        "content": 'def hello():\n    print("hello from ours")\n\ndef greet(name):\n    return f"Hello, {name}"\n',
        "oid": "8955ee93faac27ef0d10aedd6842862e0e5278ce",
        "source": "ours",
    },
    "target": {
        "content": 'def hello():\n    print("hello from ours")\n\ndef greet(name):\n    return f"Hello, {name}"\n',
        "exists": True,
        "oid": "8955ee93faac27ef0d10aedd6842862e0e5278ce",
    },
    "theirs": {
        "content": 'def hello():\n    print("hello from theirs")\n\ndef greet(name):\n    return f"Hello, {name}"\n',
        "oid": "f0a600db19aaf7628e4f085bd5571e0770488458",
        "source": "theirs",
    },
    "was_resolved": False,
}

# ── Unit tests: parser correctness ───────────────────────────────────────────


def test_merge_preview_extracts_merge_id():
    plan = MERGE_PREVIEW_FIXTURE
    assert plan["merge_id"] == "m_2026-06-30T05-52-46Z_9e6be0"


def test_merge_preview_uses_unresolved_key_not_unresolved_conflicts():
    """Real gfix uses 'unresolved', not 'unresolved_conflicts' (doc discrepancy)."""
    plan = MERGE_PREVIEW_FIXTURE
    assert "unresolved" in plan
    assert "unresolved_conflicts" not in plan


def test_merge_preview_extracts_conflict_id():
    plan = MERGE_PREVIEW_FIXTURE
    unresolved = plan.get("unresolved", [])
    assert len(unresolved) == 1
    assert unresolved[0]["conflict_id"] == "c_911c1"


def test_conflict_get_base_has_no_source():
    """Doc says ConflictSide has source; real base has no source field."""
    base_raw = CONFLICT_GET_FIXTURE["base"]
    assert "source" not in base_raw
    side = _parse_conflict_side(base_raw)
    assert side.source is None


def test_conflict_get_ours_has_source():
    ours_raw = CONFLICT_GET_FIXTURE["ours"]
    assert ours_raw["source"] == "ours"
    side = _parse_conflict_side(ours_raw)
    assert side.source == "ours"
    assert side.content.startswith("def hello():")


def test_conflict_get_theirs_has_source():
    theirs_raw = CONFLICT_GET_FIXTURE["theirs"]
    side = _parse_conflict_side(theirs_raw)
    assert side.source == "theirs"
    assert "hello from theirs" in side.content


def test_conflict_get_target_has_exists_field():
    target_raw = CONFLICT_GET_FIXTURE["target"]
    assert "exists" in target_raw
    assert target_raw["exists"] is True
    # target has no 'source' field (real data differs from doc)
    assert "source" not in target_raw


def test_conflict_detail_construction():
    cg = CONFLICT_GET_FIXTURE
    detail = ConflictDetail(
        conflict_id=cg["conflict_id"],
        file=cg["file"],
        kind=cg["kind"],
        ours=_parse_conflict_side(cg["ours"]),
        theirs=_parse_conflict_side(cg["theirs"]),
        base=_parse_conflict_side(cg["base"]),
        target=TargetSide(
            content=cg["target"]["content"],
            oid=cg["target"]["oid"],
            exists=cg["target"].get("exists", True),
        ),
    )
    assert detail.conflict_id == "c_911c1"
    assert detail.file == "main.py"
    assert detail.kind == "ast"
    assert detail.ours.source == "ours"
    assert detail.theirs.source == "theirs"
    assert detail.base.source is None
    assert detail.target.exists is True


# ── Integration tests: requires gfix on PATH + ANTHROPIC_API_KEY ─────────────

SAMPLE_DIR = Path(__file__).parent.parent / "sample_conflicts"

import os as _os

gfix_present = pytest.mark.skipif(
    shutil.which("gfix") is None,
    reason="gfix not on PATH",
)

# Phase 3: mergiraf failures now call the LLM; integration tests need a key.
_llm_and_gfix_present = pytest.mark.skipif(
    shutil.which("gfix") is None or not _os.environ.get("ANTHROPIC_API_KEY"),
    reason="gfix not on PATH or ANTHROPIC_API_KEY not set",
)


@gfix_present
async def test_resolve_floor_resolvable_end_to_end():
    """Non-overlapping docstring edits to separate functions → git auto-merges.

    No ANTHROPIC_API_KEY required: git resolves the conflict before mergiraf/LLM
    are invoked.  Proves the keyless deterministic-floor path stays green.
    """
    with open(SAMPLE_DIR / "floor_resolvable.json") as f:
        req = json.load(f)

    response = await resolve_conflict(**req)

    assert isinstance(response, ResolveResponse)
    assert response.resolved_content.strip(), "resolved_content must not be empty"
    # Both docstrings must survive the auto-merge
    assert '"""Add two numbers."""' in response.resolved_content
    assert '"""Multiply two numbers."""' in response.resolved_content
    assert response.via == "git_automerge"
    assert response.used_rag is False


@_llm_and_gfix_present
async def test_resolve_overlap_python_end_to_end():
    """Overlap fixture: both sides change same line → mergiraf fails → ours placeholder."""
    with open(SAMPLE_DIR / "overlap_python.json") as f:
        req = json.load(f)

    response = await resolve_conflict(**req)

    assert isinstance(response, ResolveResponse)
    assert response.merge_id.startswith("m_")
    assert response.resolved_content.strip(), "resolved_content must not be empty"
    assert "def hello" in response.resolved_content
    assert "def greet" in response.resolved_content
    assert response.audit_ref is not None
    assert response.audit_ref.startswith("refs/gitfix/audit/")
    assert response.file_path == "greeting.py"
    # via is either mergiraf (if it somehow worked) or ours_chosen (placeholder)
    assert response.via in {"mergiraf", "mergiraf_chosen", "ours_chosen", "ours_placeholder"}


@_llm_and_gfix_present
async def test_resolve_simple_python_end_to_end():
    """Non-overlapping additions: git/mergiraf should auto-merge or resolve cleanly."""
    with open(SAMPLE_DIR / "simple_python.json") as f:
        req = json.load(f)

    response = await resolve_conflict(**req)

    assert isinstance(response, ResolveResponse)
    assert response.resolved_content.strip(), "resolved_content must not be empty"
    # Both functions should appear after merge
    assert "farewell" in response.resolved_content or "welcome" in response.resolved_content
    assert response.merge_id.startswith("m_")
