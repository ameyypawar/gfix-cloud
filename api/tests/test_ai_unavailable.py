"""
Tests for the graceful no-key path.

When a hard conflict triggers the AI path but no API key is configured,
/resolve must return HTTP 200 with ai_unavailable=True — not a 500.
Retrieval is keyless; neighbors can be populated even without a key.
"""
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.gfix_bridge import resolve_conflict

SAMPLE_DIR = Path(__file__).parent.parent / "sample_conflicts"

gfix_present = pytest.mark.skipif(
    shutil.which("gfix") is None,
    reason="gfix not on PATH",
)


@gfix_present
async def test_no_key_hard_conflict_returns_partial_200():
    """Patch _call_gemini to raise the no-key RuntimeError.

    Asserts: resolved=False, ai_unavailable=True, no exception propagated (no 500).
    neighbors is always a list (empty if no DB, populated with live DB — both OK).
    """
    with open(SAMPLE_DIR / "overlap_python.json") as f:
        req = json.load(f)

    with patch(
        "app.generation._call_gemini",
        new=AsyncMock(
            side_effect=RuntimeError(
                "GEMINI_API_KEY not set — cannot call generation LLM"
            )
        ),
    ):
        response = await resolve_conflict(**req)

    assert response.resolved is False, "resolved must be False when AI is unavailable"
    assert response.ai_unavailable is True
    assert response.ai_unavailable_reason is not None
    assert "not configured" in response.ai_unavailable_reason or "No API key" in response.ai_unavailable_reason
    assert isinstance(response.neighbors, list), "neighbors must be a list (keyless retrieval)"
    assert response.resolved_content == "", "no resolved content when AI unavailable"
    assert response.via == "ai"
