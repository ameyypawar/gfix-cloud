"""
Pure scoring functions for the gfix-cloud eval harness.

No network calls.  All functions operate on plain strings.
"""
import difflib

# Similarity threshold for context_precision.
# 0.7: conservative enough to skip trivially-similar empty/whitespace blobs;
# lenient enough to credit paraphrased but semantically equivalent resolutions.
_CONTEXT_PRECISION_THRESHOLD = 0.7


def exact_match(pred: str, gold: str) -> bool:
    """True iff pred matches gold after stripping trailing whitespace/newlines.

    Trailing-only differences (extra newlines, trailing spaces) are normalized
    away.  Internal whitespace and leading whitespace are NOT normalized.
    """
    return pred.rstrip() == gold.rstrip()


def edit_distance_ratio(pred: str, gold: str) -> float:
    """Normalized Levenshtein similarity in [0, 1].

    Uses difflib.SequenceMatcher (character-level) which is O(n²) but avoids
    adding a new runtime dependency.  For the eval corpus (files ≤ 12 KB) this
    is fast enough.

    1.0 = identical strings, 0.0 = no characters in common.
    Both-empty corner case returns 1.0 (degenerate, nothing to distinguish).
    """
    if not pred and not gold:
        return 1.0
    return difflib.SequenceMatcher(None, pred, gold).ratio()


def context_precision(retrieved_resolutions: list[str], gold: str) -> float:
    """Fraction of retrieved neighbors whose resolution meaningfully overlaps gold.

    A neighbor is considered a "hit" when
        SequenceMatcher(neighbor_resolution, gold).ratio() >= _CONTEXT_PRECISION_THRESHOLD

    The threshold (0.7) is documented above.

    Returns 0.0 when retrieved_resolutions is empty (no retrieval → no signal).
    """
    if not retrieved_resolutions:
        return 0.0
    hits = sum(
        1
        for r in retrieved_resolutions
        if difflib.SequenceMatcher(None, r, gold).ratio() >= _CONTEXT_PRECISION_THRESHOLD
    )
    return hits / len(retrieved_resolutions)
