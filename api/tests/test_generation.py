"""
Phase 4.5 generation tests — updated for Gemini provider (default).

Unit tests use a MOCKED httpx client (Gemini path) — no API key required.
Live smoke test is gated on GEMINI_API_KEY and skips cleanly when absent.

Assertions:
  - Few-shot examples appear in the user message when provided
  - No examples block when examples=[] (baseline path)
  - Untrusted conflict bodies are fenced/delimited (injection hardening)
  - Response parses into a Suggestion with non-empty text
"""
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.generation import (
    Suggestion,
    _FENCE_OPEN,
    _FENCE_CLOSE,
    _parse_response,
    generate_resolution,
)
from app.models import ConflictDetail, ConflictSide, TargetSide
from app.retrieval import Neighbor


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_conflict(
    ours: str = "fn ours() { 1 }",
    theirs: str = "fn theirs() { 2 }",
    base: str = "fn base() { 0 }",
) -> ConflictDetail:
    return ConflictDetail(
        conflict_id="test-c-001",
        file="src/lib.rs",
        kind="ast",
        base=ConflictSide(content=base, oid="oid_base"),
        ours=ConflictSide(content=ours, oid="oid_ours", source="ours"),
        theirs=ConflictSide(content=theirs, oid="oid_theirs", source="theirs"),
        target=TargetSide(content=ours, oid="oid_ours", exists=True),
    )


def _make_neighbor(
    resolved: str = "fn resolved() { 1 }",
    rationale: str = "kept ours for clarity",
) -> Neighbor:
    return Neighbor(
        resolution_id="nb-001",
        file_path="src/util.rs",
        language="rust",
        base_code="fn util() { 0 }",
        ours_code="fn util() { 1 }",
        theirs_code="fn util() { 2 }",
        resolved_content=resolved,
        resolution_kind="ours_chosen",
        ai_rationale=rationale,
        rrf_score=0.015,
    )


def _make_mock_client(response_text: str):
    """Return an injectable mock httpx.AsyncClient for the Gemini path.

    Mirrors the shape generate_resolution expects:
      client.post(url, json=payload) → Response with .status_code and .json()
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {"content": {"parts": [{"text": response_text}], "role": "model"}}
        ]
    }
    mock_resp.raise_for_status = MagicMock()

    mock = MagicMock()
    mock.post = AsyncMock(return_value=mock_resp)
    return mock


def _extract_user_content(mock_client) -> str:
    """Pull the user message text out of the captured Gemini POST payload."""
    call_kwargs = mock_client.post.call_args.kwargs
    return call_kwargs["json"]["contents"][0]["parts"][0]["text"]


_GOOD_RESPONSE = (
    "[RESOLVED]\nfn merged() { 1 }\n[/RESOLVED]\n"
    "[RATIONALE]\nKept ours because it adds the guard.\n[/RATIONALE]"
)


# ── Unit: _parse_response ────────────────────────────────────────────────────

def test_parse_response_extracts_text_and_rationale():
    s = _parse_response(_GOOD_RESPONSE)
    assert s.text == "fn merged() { 1 }"
    assert "Kept ours" in s.rationale
    assert s.confidence == 0.5


def test_parse_response_fallback_when_no_markers():
    raw = "just some resolved content without markers"
    s = _parse_response(raw)
    assert s.text == raw
    assert s.rationale == ""
    assert s.confidence == 0.5


# ── Unit: generate_resolution with mock Gemini client ────────────────────────

async def test_generate_with_examples_includes_example_content():
    """Few-shot block appears in the user message when examples are provided."""
    conflict = _make_conflict()
    neighbor = _make_neighbor()
    mock_client = _make_mock_client(_GOOD_RESPONSE)

    await generate_resolution(conflict, [neighbor], client=mock_client)

    user_content = _extract_user_content(mock_client)

    assert "Example 1" in user_content, "few-shot block must be present"
    assert neighbor.file_path in user_content
    assert neighbor.resolved_content in user_content


async def test_generate_without_examples_omits_few_shot_block():
    """No examples block in the user message when examples=[]."""
    conflict = _make_conflict()
    mock_client = _make_mock_client(_GOOD_RESPONSE)

    await generate_resolution(conflict, [], client=mock_client)

    user_content = _extract_user_content(mock_client)

    assert "Example" not in user_content, "no few-shot block when examples=[]"
    assert "Few-shot" not in user_content


async def test_conflict_bodies_are_fenced():
    """Untrusted conflict bodies (ours/theirs/base) must be enclosed in data fences."""
    ours = "fn ours() { /* UNTRUSTED */ }"
    conflict = _make_conflict(ours=ours)
    mock_client = _make_mock_client(_GOOD_RESPONSE)

    await generate_resolution(conflict, [], client=mock_client)

    user_content = _extract_user_content(mock_client)

    assert _FENCE_OPEN in user_content, "data fence open must appear in user message"
    assert _FENCE_CLOSE in user_content, "data fence close must appear in user message"

    # Scan ALL fenced regions — ours appears in the second fence (after base)
    found_in_fence = False
    pos = 0
    while True:
        start = user_content.find(_FENCE_OPEN, pos)
        if start == -1:
            break
        end = user_content.find(_FENCE_CLOSE, start + len(_FENCE_OPEN))
        if end == -1:
            break
        region = user_content[start:end]
        if ours in region:
            found_in_fence = True
            break
        pos = end + len(_FENCE_CLOSE)
    assert found_in_fence, "ours content must be inside a data fence"


async def test_example_code_is_also_fenced():
    """Example resolved content is fenced so it cannot act as instruction."""
    conflict = _make_conflict()
    neighbor = _make_neighbor(resolved="fn evil() { /* DROP TABLE users */ }")
    mock_client = _make_mock_client(_GOOD_RESPONSE)

    await generate_resolution(conflict, [neighbor], client=mock_client)

    user_content = _extract_user_content(mock_client)

    # Find all fenced regions
    fenced_contents = []
    pos = 0
    while True:
        start = user_content.find(_FENCE_OPEN, pos)
        if start == -1:
            break
        end = user_content.find(_FENCE_CLOSE, start)
        fenced_contents.append(user_content[start + len(_FENCE_OPEN):end])
        pos = end + len(_FENCE_CLOSE)

    combined = "\n".join(fenced_contents)
    assert "DROP TABLE" in combined, "example content must appear inside fences"


async def test_response_parses_into_suggestion():
    """generate_resolution returns a Suggestion with non-empty text."""
    conflict = _make_conflict()
    mock_client = _make_mock_client(_GOOD_RESPONSE)

    result = await generate_resolution(conflict, [], client=mock_client)

    assert isinstance(result, Suggestion)
    assert result.text, "Suggestion.text must not be empty"
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


async def test_model_and_params_sent_correctly():
    """Verify model name in URL and maxOutputTokens in generationConfig."""
    from app.config import settings

    conflict = _make_conflict()
    mock_client = _make_mock_client(_GOOD_RESPONSE)

    await generate_resolution(conflict, [], client=mock_client)

    call_args = mock_client.post.call_args
    url = call_args.args[0]
    payload = call_args.kwargs["json"]

    assert settings.generation_model in url, "model must appear in Gemini URL"
    assert payload["generationConfig"]["maxOutputTokens"] == 2048
    # No Anthropic-specific keys leak into the Gemini payload
    assert "effort" not in payload
    assert "thinking" not in payload


# ── Live smoke test: gated on GEMINI_API_KEY ─────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set — live generation skipped",
)
async def test_live_generation_smoke():
    """
    End-to-end live call to gemini-2.5-flash.
    Only runs when GEMINI_API_KEY is set.
    """
    conflict = _make_conflict(
        base="fn add(a: i32, b: i32) -> i32 { a }",
        ours="fn add(a: i32, b: i32) -> i32 { a + b }",
        theirs="fn add(a: i32, b: i32) -> i32 { a.saturating_add(b) }",
    )
    result = await generate_resolution(conflict, [], client=None)
    assert isinstance(result, Suggestion)
    assert result.text.strip(), "live response must have non-empty text"
