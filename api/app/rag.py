"""
Orchestrates: embed → hybrid_search → generate for one conflict.

produce_suggestion is the single entry point called by the bridge.
"""
from __future__ import annotations

import logging
from typing import Optional

from app import embeddings as _emb
from app import retrieval as _ret
from app import generation as _gen
from app.config import settings
from app.generation import Suggestion
from app.retrieval import Neighbor

logger = logging.getLogger(__name__)


async def produce_suggestion(
    pool,
    conflict,
    use_rag: bool,
    client=None,
) -> tuple[Suggestion, list[Neighbor]]:
    """
    Produce an LLM-generated resolution for conflict.

    use_rag=True  → embed → hybrid_search → few-shot generation
    use_rag=False → generation with empty examples list (baseline for eval delta)

    Returns (suggestion, neighbors_used).  neighbors_used is [] when use_rag=False
    or when retrieval finds nothing.
    """
    neighbors: list[Neighbor] = []

    if use_rag and pool is not None:
        language = _emb.language_from_path(conflict.file)
        conflict_text = _emb.build_conflict_text(
            file_path=conflict.file,
            language=language,
            conflict_kind=conflict.kind,
            base=conflict.base.content,
            ours=conflict.ours.content,
            theirs=conflict.theirs.content,
        )
        query_embedding = _emb.embed_conflict(
            file_path=conflict.file,
            language=language,
            conflict_kind=conflict.kind,
            base=conflict.base.content,
            ours=conflict.ours.content,
            theirs=conflict.theirs.content,
        )
        try:
            neighbors = await _ret.hybrid_search(
                pool=pool,
                query_text=conflict_text,
                query_embedding=query_embedding,
                language=language,
                k=settings.rag_top_k,
            )
            logger.info("hybrid_search returned %d neighbors", len(neighbors))
        except Exception as exc:
            logger.warning("retrieval failed, degrading to baseline: %s", exc)
            neighbors = []

    suggestion = await _gen.generate_resolution(conflict, neighbors, client=client)
    return suggestion, neighbors
