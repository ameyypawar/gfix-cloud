"""
Unit tests for eval/metrics.py — pure functions, no network, no DB.
"""
import sys
from pathlib import Path

# Locate the eval/ directory from api/tests/ (two levels up → project root → eval/)
_EVAL_DIR = Path(__file__).parent.parent.parent / "eval"
sys.path.insert(0, str(_EVAL_DIR))

from metrics import context_precision, edit_distance_ratio, exact_match  # noqa: E402


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------


class TestExactMatch:
    def test_identical(self):
        assert exact_match("hello\n", "hello\n") is True

    def test_trailing_newline_normalized(self):
        """Extra trailing newlines on pred are stripped before comparison."""
        assert exact_match("hello\n\n\n", "hello\n") is True

    def test_trailing_spaces_normalized(self):
        assert exact_match("hello   ", "hello") is True

    def test_different_content(self):
        assert exact_match("hello", "world") is False

    def test_empty_both(self):
        assert exact_match("", "") is True

    def test_only_whitespace_vs_empty(self):
        # rstrip turns "   " into "" which equals ""
        assert exact_match("   ", "") is True

    def test_content_different_ignores_trailing_only(self):
        assert exact_match("foo\n", "foobar\n") is False

    def test_multiline_identical(self):
        code = "def foo():\n    return 1\n"
        assert exact_match(code, code) is True

    def test_multiline_differs_in_middle(self):
        pred = "def foo():\n    return 1\n"
        gold = "def foo():\n    return 2\n"
        assert exact_match(pred, gold) is False


# ---------------------------------------------------------------------------
# edit_distance_ratio
# ---------------------------------------------------------------------------


class TestEditDistanceRatio:
    def test_identical(self):
        assert edit_distance_ratio("abc", "abc") == 1.0

    def test_empty_both(self):
        assert edit_distance_ratio("", "") == 1.0

    def test_empty_vs_nonempty(self):
        assert edit_distance_ratio("", "nonempty") == 0.0

    def test_bounds_in_range(self):
        r = edit_distance_ratio("hello world", "completely different text")
        assert 0.0 <= r <= 1.0

    def test_high_similarity(self):
        r = edit_distance_ratio("abc", "abX")
        # One char changed in a 3-char string → ratio should be > 0.5
        assert r > 0.5

    def test_prefix_similarity(self):
        r = edit_distance_ratio("hello", "helloworld")
        assert 0.0 < r < 1.0

    def test_completely_different(self):
        r = edit_distance_ratio("aaaa", "bbbb")
        assert r == 0.0

    def test_long_strings_identical(self):
        code = "def " + "x" * 200 + "():\n    pass\n"
        assert edit_distance_ratio(code, code) == 1.0

    def test_long_strings_one_char_diff(self):
        code_a = "def " + "x" * 200 + "():\n    pass\n"
        code_b = "def " + "x" * 199 + "y():\n    pass\n"
        r = edit_distance_ratio(code_a, code_b)
        # One substitution in ~210 chars — ratio should be very high
        assert r > 0.99


# ---------------------------------------------------------------------------
# context_precision
# ---------------------------------------------------------------------------


class TestContextPrecision:
    def test_empty_retrieved(self):
        assert context_precision([], "gold content") == 0.0

    def test_all_identical_to_gold(self):
        gold = "def foo():\n    return 42\n"
        retrieved = [gold, gold, gold]
        assert context_precision(retrieved, gold) == 1.0

    def test_none_similar_to_gold(self):
        gold = "def foo():\n    return 42\n"
        retrieved = [
            "completely different text xyz",
            "nothing alike at all",
            "random string 12345",
        ]
        assert context_precision(retrieved, gold) == 0.0

    def test_partial_match_half(self):
        gold = "def foo():\n    return 42\n"
        very_similar = "def foo():\n    return 42\n  # resolved"   # ratio ≥ 0.7
        nothing_alike = "SELECT * FROM users WHERE id = 99;"        # ratio < 0.7
        retrieved = [very_similar, nothing_alike]
        cp = context_precision(retrieved, gold)
        assert cp == 0.5

    def test_threshold_99pct_similar_passes(self):
        """String that is 99%+ similar to gold must count as a hit."""
        gold = "abc" * 100                           # 300 chars
        slightly_different = gold[:-3] + "xyz"       # last 3 chars differ → 99% ratio
        retrieved = [slightly_different]
        assert context_precision(retrieved, gold) == 1.0

    def test_single_neighbor_identical(self):
        gold = "fn main() {}\n"
        assert context_precision([gold], gold) == 1.0

    def test_single_neighbor_unrelated(self):
        gold = "fn main() {}\n"
        assert context_precision(["totally unrelated"], gold) == 0.0

    def test_result_bounded_between_0_and_1(self):
        gold = "some content"
        retrieved = ["similar content", "different", "also similar content"]
        cp = context_precision(retrieved, gold)
        assert 0.0 <= cp <= 1.0
