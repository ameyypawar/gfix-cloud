"use client";

import { useState } from "react";
import { resolveConflict, type ResolveResponse } from "@/lib/api";
import ResolvePanel from "./components/ResolvePanel";
import NeighborsPanel from "./components/NeighborsPanel";

// ── Inline sample conflicts (from api/sample_conflicts/*.json) ────────────────
const SAMPLES = {
  floor_resolvable: {
    label: "floor_resolvable",
    description: "non-overlapping edits to separate functions — git auto-merge",
    base: 'def add(x, y):\n    return x + y\n\ndef multiply(x, y):\n    return x * y\n',
    ours: 'def add(x, y):\n    """Add two numbers."""\n    return x + y\n\ndef multiply(x, y):\n    return x * y\n',
    theirs:
      'def add(x, y):\n    return x + y\n\ndef multiply(x, y):\n    """Multiply two numbers."""\n    return x * y\n',
    file_path: "math_ops.py",
  },
  simple_python: {
    label: "simple_python",
    description: "non-overlapping function additions — floor resolution",
    base: 'def hello():\n    print("hello")\n\n\ndef greet(name: str) -> str:\n    return f"Hello, {name}"\n',
    ours: 'def hello():\n    print("hello")\n\n\ndef greet(name: str) -> str:\n    return f"Hello, {name}"\n\n\ndef farewell(name: str) -> str:\n    return f"Goodbye, {name}"\n',
    theirs:
      'def hello():\n    print("hello")\n\n\ndef greet(name: str) -> str:\n    return f"Hello, {name}"\n\n\ndef welcome(name: str) -> str:\n    return f"Welcome, {name}!"\n',
    file_path: "greeting.py",
  },
  overlap_python: {
    label: "overlap_python",
    description: "hard overlap — both sides change same line, triggers AI path",
    base: 'def hello():\n    print("hello base")\n\n\ndef greet(name: str) -> str:\n    return f"Hello, {name}"\n',
    ours: 'def hello():\n    print("hello from ours")\n\n\ndef greet(name: str) -> str:\n    return f"Hello, {name}"\n',
    theirs:
      'def hello():\n    print("hello from theirs")\n\n\ndef greet(name: str) -> str:\n    return f"Hello, {name}"\n',
    file_path: "greeting.py",
  },
} as const;

type SampleKey = keyof typeof SAMPLES;

export default function Home() {
  const [base, setBase] = useState("");
  const [ours, setOurs] = useState("");
  const [theirs, setTheirs] = useState("");
  const [filePath, setFilePath] = useState("");
  const [useRag, setUseRag] = useState(true);
  const [result, setResult] = useState<ResolveResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function loadSample(key: SampleKey) {
    const s = SAMPLES[key];
    setBase(s.base);
    setOurs(s.ours);
    setTheirs(s.theirs);
    setFilePath(s.file_path);
    setResult(null);
    setError(null);
  }

  async function handleResolve() {
    if (!base.trim() || !ours.trim() || !theirs.trim() || !filePath.trim()) {
      setError("All four fields are required. Use a quick-fill sample to start.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await resolveConflict({
        base,
        ours,
        theirs,
        file_path: filePath,
        rag: useRag,
      });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-content">
      {/* Page header */}
      <div className="mb-24">
        <h1>Conflict Resolver</h1>
        <p className="mt-4">
          Submit a 3-way merge conflict. See retrieved past resolutions (RAG), the
          AI suggestion, and the gfix audit ref.
        </p>
      </div>

      {/* Input card */}
      <div className="card">
        {/* Quick-fill */}
        <div className="flex-row flex-wrap mb-16">
          <span
            className="text-muted text-small"
            style={{ marginRight: 4, flexShrink: 0 }}
          >
            Quick-fill:
          </span>
          {(Object.keys(SAMPLES) as SampleKey[]).map((key) => (
            <button
              key={key}
              className="btn btn-ghost"
              style={{ marginRight: 4, marginBottom: 4 }}
              onClick={() => loadSample(key)}
              title={SAMPLES[key].description}
            >
              {SAMPLES[key].label}
            </button>
          ))}
        </div>

        {/* File path */}
        <div className="mb-16">
          <label>File path</label>
          <input
            type="text"
            value={filePath}
            onChange={(e) => setFilePath(e.target.value)}
            placeholder="e.g. src/lib/utils.ts"
            style={{ maxWidth: 340 }}
          />
        </div>

        {/* Conflict sides */}
        <div className="conflict-grid">
          <div>
            <label>Base (common ancestor)</label>
            <textarea
              rows={12}
              value={base}
              onChange={(e) => setBase(e.target.value)}
              placeholder="base content"
              spellCheck={false}
            />
          </div>
          <div>
            <label>Ours (target branch)</label>
            <textarea
              rows={12}
              value={ours}
              onChange={(e) => setOurs(e.target.value)}
              placeholder="ours content"
              spellCheck={false}
            />
          </div>
          <div>
            <label>Theirs (source branch)</label>
            <textarea
              rows={12}
              value={theirs}
              onChange={(e) => setTheirs(e.target.value)}
              placeholder="theirs content"
              spellCheck={false}
            />
          </div>
        </div>

        {/* Controls */}
        <div className="flex-between mt-16 flex-wrap" style={{ gap: 12 }}>
          <label className="toggle-row" style={{ cursor: "pointer", userSelect: "none" }}>
            <span className="toggle">
              <input
                type="checkbox"
                checked={useRag}
                onChange={(e) => setUseRag(e.target.checked)}
              />
              <span className="toggle-track" />
              <span className="toggle-thumb" />
            </span>
            <span
              className="text-muted text-small"
              style={{ marginBottom: 0, textTransform: "none", letterSpacing: 0 }}
            >
              RAG retrieval{" "}
              <span className="text-dim">
                (off = baseline, no few-shot examples)
              </span>
            </span>
          </label>

          <button
            className="btn btn-primary"
            onClick={handleResolve}
            disabled={loading}
            style={{ minWidth: 100 }}
          >
            {loading ? (
              <>
                <span className="spinner" />
                Resolving
              </>
            ) : (
              "Resolve"
            )}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="alert alert-error mt-16">
          <strong>Error</strong> — {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <>
          <hr className="section-divider" />
          <div className="output-grid">
            <NeighborsPanel neighbors={result.neighbors} usedRag={result.used_rag} />
            <ResolvePanel result={result} />
          </div>
        </>
      )}
    </div>
  );
}
