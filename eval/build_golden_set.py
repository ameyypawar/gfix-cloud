#!/usr/bin/env python3
"""
Mine real (conflict → resolution) pairs from local OSS repos non-destructively.

Strategy
--------
For each merge commit M with parents P1 P2:
  git merge-tree --write-tree --name-only P1 P2  (exit 1 = conflicts found)

For each conflicted text file F:
  base               = git show (merge-base P1 P2):F
  ours               = git show P1:F       (first-parent / target branch)
  theirs             = git show P2:F       (feature branch)
  expected_resolution = git show M:F       (maintainers' actual merge result)

The working trees of the source repos are NEVER touched.
git merge-tree and git show are purely read-side; no index or worktree mutations.

Repos
-----
  gitbutler  /Users/amey/Projects/gitbutler   (TypeScript / Rust / Svelte)
  gitoxide   /Users/amey/Projects/gitoxide    (Rust)
  mio        /Users/amey/Projects/mio         (Rust)

Output
------
  eval/golden_set.jsonl  — one JSON object per line:
    id, repo, merge_sha, file_path, language,
    base, ours, theirs, expected_resolution, provenance
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPOS = [
    {"path": "/Users/amey/Projects/gitbutler", "name": "gitbutler"},
    {"path": "/Users/amey/Projects/gitoxide",  "name": "gitoxide"},
    {"path": "/Users/amey/Projects/mio",       "name": "mio"},
]

# scan_cap: max merge commits per repo; widened past 300 for gitoxide (1287 total)
# to raise the total yield above the 30-entry floor.
SCAN_CAP: dict[str, int] = {
    "gitbutler": 356,   # all
    "gitoxide":  1287,  # all — only ~1 conflict per 300 merges, need full scan
    "mio":       30,    # all
}

TOTAL_CAP = 50          # stop after this many golden-set entries collected
MAX_BYTES = 12 * 1024   # 12 KB per side

# Filenames that are lock / generated files — skip
LOCKFILE_NAMES: set[str] = {
    "Cargo.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Gemfile.lock", "composer.lock", "bun.lockb",
}

# Binary / non-text extensions — skip
BINARY_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pdf", ".zip", ".tar", ".gz", ".bz2",
    ".exe", ".dll", ".so", ".dylib",
}

# Canonical language names by extension
_EXT_TO_LANG: dict[str, str] = {
    ".rs":     "rust",
    ".ts":     "typescript",
    ".tsx":    "typescript",
    ".js":     "javascript",
    ".jsx":    "javascript",
    ".py":     "python",
    ".svelte": "svelte",
    ".md":     "markdown",
    ".toml":   "toml",
    ".yaml":   "yaml",
    ".yml":    "yaml",
    ".json":   "json",
    ".css":    "css",
    ".scss":   "scss",
    ".html":   "html",
    ".sh":     "shell",
    ".c":      "c",
    ".h":      "c",
    ".cpp":    "cpp",
    ".hpp":    "cpp",
    ".go":     "go",
    ".rb":     "ruby",
    ".java":   "java",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git(*args: str, cwd: str) -> tuple[int, str]:
    r = subprocess.run(
        ["git"] + list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",   # non-UTF-8 bytes → replacement char (won't crash)
    )
    return r.returncode, r.stdout


def _lang(path: str) -> str:
    return _EXT_TO_LANG.get(Path(path).suffix.lower(), "text")


def _is_text_file(path: str) -> bool:
    name = Path(path).name
    if name in LOCKFILE_NAMES:
        return False
    ext = Path(path).suffix.lower()
    if ext in BINARY_EXTENSIONS:
        return False
    if ".min." in path or ".generated." in path:
        return False
    # Accept if the extension is in our known-text set; reject everything else
    # (avoids accidentally embedding binary blobs with unknown extensions).
    return ext in _EXT_TO_LANG


def _parse_conflict_files(stdout: str) -> list[str]:
    """Extract conflicted file paths from git merge-tree --write-tree --name-only stdout.

    Output format (when conflicts exist):
      line 1  : result tree SHA  (40 hex chars)
      lines 2+: conflicted file paths (--name-only)
      then    : informational messages ("Auto-merging …", "CONFLICT …")

    We collect lines after the tree SHA that look like file paths and stop at
    the first "Auto-merging" or "CONFLICT" message.
    """
    lines = stdout.strip().splitlines()
    files: list[str] = []
    for line in lines[1:]:         # skip tree SHA on line 1
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Auto-merging") or stripped.startswith("CONFLICT"):
            break                   # informational section starts here
        # Keep lines that look like paths (contain / or .)
        if "/" in stripped or "." in stripped:
            files.append(stripped)
    return files


def _get_content(sha: str, path: str, cwd: str) -> str | None:
    rc, out = _git("show", f"{sha}:{path}", cwd=cwd)
    if rc != 0:
        return None
    return out


# ---------------------------------------------------------------------------
# Main mining loop
# ---------------------------------------------------------------------------

def mine_repo(
    cfg: dict,
    collected: list[dict],
    out_file,
) -> int:
    repo_path = cfg["path"]
    repo_name = cfg["name"]
    cap = SCAN_CAP.get(repo_name, 300)

    rc, merges_stdout = _git(
        "log", "--merges", "--format=%H", f"-{cap}",
        cwd=repo_path,
    )
    merges = [s.strip() for s in merges_stdout.strip().splitlines() if s.strip()]
    print(f"\n{repo_name}: scanning {len(merges)} merge commits (cap={cap})", flush=True)

    repo_count = 0

    for M in merges:
        if len(collected) >= TOTAL_CAP:
            print(f"  [cap] total cap {TOTAL_CAP} reached", flush=True)
            break

        # Parents
        rc, parents_out = _git("log", "--format=%P", "-1", M, cwd=repo_path)
        parts = parents_out.strip().split()
        if len(parts) < 2:
            continue
        P1, P2 = parts[0], parts[1]

        # Non-destructive conflict detection
        rc, mt_out = _git(
            "merge-tree", "--write-tree", "--name-only", P1, P2,
            cwd=repo_path,
        )
        if rc == 0:
            continue        # clean merge — nothing to mine

        conflict_files = _parse_conflict_files(mt_out)
        if not conflict_files:
            continue        # binary / delete conflicts with no text paths

        # Merge base
        rc, base_out = _git("merge-base", P1, P2, cwd=repo_path)
        if rc != 0:
            continue
        BASE = base_out.strip()

        for F in conflict_files:
            if len(collected) >= TOTAL_CAP:
                break

            if not _is_text_file(F):
                print(f"  skip {F} (not text/known-ext)", flush=True)
                continue

            base_content     = _get_content(BASE, F, repo_path)
            ours_content     = _get_content(P1,   F, repo_path)
            theirs_content   = _get_content(P2,   F, repo_path)
            resolution       = _get_content(M,    F, repo_path)

            if any(c is None for c in [base_content, ours_content, theirs_content, resolution]):
                print(f"  skip {F} (missing side)", flush=True)
                continue

            # Size filter (bytes, after UTF-8 encoding)
            oversized = False
            for label, content in [
                ("base", base_content), ("ours", ours_content),
                ("theirs", theirs_content), ("resolution", resolution),
            ]:
                if len(content.encode("utf-8", errors="replace")) > MAX_BYTES:
                    print(
                        f"  skip {F} ({label} too large: "
                        f"{len(content.encode('utf-8', errors='replace'))} B)",
                        flush=True,
                    )
                    oversized = True
                    break
            if oversized:
                continue

            # Use full-path slug to avoid collisions (e.g. multiple Cargo.toml
            # files in the same merge commit).
            path_slug = F.replace("/", "_")
            row_id = f"{repo_name}-{M[:8]}-{path_slug}"
            row: dict = {
                "id":                  row_id,
                "repo":                repo_name,
                "merge_sha":           M,
                "file_path":           F,
                "language":            _lang(F),
                "base":                base_content,
                "ours":                ours_content,
                "theirs":              theirs_content,
                "expected_resolution": resolution,
                "provenance": (
                    f"git merge-tree --write-tree --name-only {P1[:8]} {P2[:8]}; "
                    f"base=git show {BASE[:8]}:{F}; "
                    f"ours=git show {P1[:8]}:{F}; "
                    f"theirs=git show {P2[:8]}:{F}; "
                    f"resolution=git show {M[:8]}:{F}"
                ),
            }

            collected.append(row)
            out_file.write(json.dumps(row) + "\n")
            out_file.flush()
            repo_count += 1

            lang = _lang(F)
            size_ours = len(ours_content.encode("utf-8", errors="replace"))
            print(f"  + {row_id}  ({lang}, {size_ours} B ours)", flush=True)

    return repo_count


def main() -> int:
    output_path = Path(__file__).parent / "golden_set.jsonl"
    collected: list[dict] = []

    print("gfix-cloud golden set miner", flush=True)
    print(f"Output → {output_path}", flush=True)
    print(f"Caps: {SCAN_CAP}  total={TOTAL_CAP}", flush=True)

    with open(output_path, "w") as out:
        for repo_cfg in REPOS:
            mine_repo(repo_cfg, collected, out)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"Total entries: {len(collected)}", flush=True)
    if not collected:
        print("WARNING: no conflicts mined — check repo paths and git version", flush=True)
        return 1

    repo_counts = Counter(r["repo"]      for r in collected)
    lang_counts = Counter(r["language"]  for r in collected)

    print("Per repo:", flush=True)
    for repo, n in sorted(repo_counts.items()):
        print(f"  {repo}: {n}", flush=True)

    print("Per language:", flush=True)
    for lang, n in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {n}", flush=True)

    if len(collected) < 30:
        print(
            f"\nWARN: only {len(collected)} entries (target ≥ 30). "
            "Repos merged cleanly — this is the full yield from these three repos.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
