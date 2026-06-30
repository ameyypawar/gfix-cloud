# gfix-cloud Eval

Real-conflict golden set and leave-one-out RAG eval harness.

---

## Golden set provenance

`golden_set.jsonl` contains **50 real (conflict → resolution) pairs** mined
non-destructively from three local OSS repos using `git merge-tree`.

| Repo | Merges scanned | Entries | Languages |
|---|---|---|---|
| gitbutler | 356 (all) | 9 | TypeScript, Rust, CSS, Markdown |
| gitoxide  | 1287 (all) | 41 | Rust, TOML |
| mio       | 30 (all, cap hit before) | 0 | — |

Language breakdown: `rust` 21, `toml` 21, `typescript` 4, `css` 3, `markdown` 1.

### How ground truth is extracted

```
For each merge commit M with parents P1, P2:

  1.  git merge-tree --write-tree --name-only P1 P2
      Exit 1 → conflicted files listed in stdout (after tree SHA, before
      "Auto-merging …" messages).

  2.  base               = git show $(git merge-base P1 P2):FILE
      ours               = git show P1:FILE        # first-parent / target branch
      theirs             = git show P2:FILE        # feature branch
      expected_resolution = git show M:FILE        # maintainers' actual merge result

  3.  Filters applied:
      - Text extension only (no .png, .woff, .zip …)
      - No lockfiles (Cargo.lock, package-lock.json …)
      - No .min.* or .generated.* paths
      - All four sides (base/ours/theirs/M) must exist in the object DB
      - Each side ≤ 12 KB
```

The working trees of gitbutler, gitoxide, and mio are **never touched**.
`git merge-tree` and `git show` are purely read operations that operate on
the object database without modifying the index or worktree.

Each row carries a `provenance` field recording the exact SHAs used:

```json
{
  "provenance": "git merge-tree --write-tree --name-only <P1> <P2>; base=git show <BASE>:<FILE>; ..."
}
```

---

## Regenerating the golden set

```bash
# From the project root, with the api venv activated
source api/.venv/bin/activate
python eval/build_golden_set.py
```

The script scans up to `SCAN_CAP` merge commits per repo (see top of
`build_golden_set.py`) and stops once `TOTAL_CAP` (50) entries are collected.
Repos: `/Users/amey/Projects/gitbutler`, `/Users/amey/Projects/gitoxide`,
`/Users/amey/Projects/mio`.  Edit the `REPOS` list to add targets.

---

## Running the eval

### Keyless (no API key required)

```bash
source api/.venv/bin/activate
docker compose up -d db      # pgvector on localhost:5432
python eval/run_eval.py
```

What happens:
1. Golden-set rows are inserted into `past_resolutions` (idempotent).
2. For each row, `hybrid_search` retrieves top-k neighbors with
   `exclude_id=<self>` so the conflict never retrieves its own resolution.
3. **Context Precision** is computed and printed.
4. `eval/summary.json` is written with the retrieval section populated.

### With ANTHROPIC_API_KEY (generation delta)

```bash
ANTHROPIC_API_KEY=sk-... python eval/run_eval.py
```

Additional steps:
5. Per row: `produce_suggestion(rag=True)` and `produce_suggestion(rag=False)`.
6. Both suggestions are scored against `expected_resolution`.
7. A delta table is printed:

```
Metric                    no-RAG        RAG  Δ (RAG−no-RAG)
-----------------------------------------------------------------
Exact Match (%)              0.0        0.0            +0.0
Edit Distance Ratio         0.32       0.41           +0.09
```

8. `eval/summary.json` is updated with the `generation` section.

### Optional: gfix BYOK baseline

```bash
ANTHROPIC_API_KEY=sk-... GITFIX_BYOK=1 python eval/run_eval.py --with-gfix-baseline
```

Adds a third column from gfix's own BYOK resolution path.

---

## Metrics

### `exact_match(pred, gold) → bool`

Strips trailing whitespace/newlines from both sides before comparing.
A score of 1.0 means the model reproduced the maintainer's exact resolution.
Hard to achieve on multi-line source files; useful as a "perfect recall" ceiling.

### `edit_distance_ratio(pred, gold) → float ∈ [0, 1]`

`difflib.SequenceMatcher(pred, gold).ratio()` — normalized Levenshtein
similarity.  1.0 = identical, 0.0 = nothing in common.  Implemented with
stdlib `difflib` to avoid an extra runtime dependency.

### `context_precision(retrieved, gold) → float ∈ [0, 1]`

Fraction of retrieved neighbors whose `resolved_content` is ≥ 0.7 similar to
`gold` (by `SequenceMatcher.ratio()`).

**Threshold 0.7**: conservative enough to reject trivially-empty blobs and
cosmetic snippets; lenient enough to credit semantically equivalent resolutions
that differ only in formatting or comments.

Returns 0.0 when no neighbors are retrieved (empty-corpus edge case).

This metric runs **keyless** — it only requires the local embedding model
(BAAI/bge-small-en-v1.5) and pgvector, not the generation LLM.

---

## Output: `eval/summary.json`

```json
{
  "golden_set_size": 50,
  "rag_top_k": 3,
  "retrieval": {
    "mean_context_precision": 0.XXXX,
    "threshold": 0.7,
    "per_row": [...]
  },
  "generation": null   // null until ANTHROPIC_API_KEY is set
}
```

`summary.json` is consumed by the Phase-5 dashboard (`/eval` page) and
pinned in the top-level README once numbers are produced by a Chief-run pass.
