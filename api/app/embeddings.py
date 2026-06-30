"""
In-process embedding via sentence-transformers (BAAI/bge-small-en-v1.5, 384-dim).

The model is loaded once as a module-level singleton; subsequent calls reuse it.
Always embed the CONFLICT representation — never the resolution.
"""
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None

# File-extension → language name
_EXT_MAP: dict[str, str] = {
    ".py": "python",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".md": "markdown",
}


def get_model() -> SentenceTransformer:
    """Return the singleton model, loading it on first call."""
    global _model
    if _model is None:
        logger.info("loading sentence-transformers model: %s", settings.embed_model)
        _model = SentenceTransformer(settings.embed_model)
        logger.info("model loaded")
    return _model


def language_from_path(file_path: str) -> str:
    """Derive a canonical language name from a file extension."""
    ext = Path(file_path).suffix.lower()
    return _EXT_MAP.get(ext, "text")


def build_conflict_text(
    file_path: str,
    language: str,
    conflict_kind: str,
    base: str,
    ours: str,
    theirs: str,
    context: str = "",
) -> str:
    """Build the structured conflict representation that is embedded.

    The CONFLICT (problem) is embedded, never the resolution.
    """
    return (
        f"[CONFLICT] file_path={file_path} lang={language} kind={conflict_kind}\n"
        f"[BASE]\n{base}\n"
        f"[OURS]\n{ours}\n"
        f"[THEIRS]\n{theirs}\n"
        f"[CONTEXT]\n{context}"
    )


def embed_conflict(
    file_path: str,
    language: str,
    conflict_kind: str,
    base: str,
    ours: str,
    theirs: str,
    context: str = "",
) -> list[float]:
    """Return a 384-dim L2-normalized embedding for the given conflict."""
    text = build_conflict_text(file_path, language, conflict_kind, base, ours, theirs, context)
    model = get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()
