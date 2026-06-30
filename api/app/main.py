import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app import db as _db
from app import gfix_bridge
from app import embeddings as _emb
from app import persistence as _pers
from app.models import ResolveRequest, ResolveResponse

logger = logging.getLogger(__name__)

# Module-level pool reference — set during lifespan
_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    # Migrations run first (plain connection, no vector codec) so the
    # extension exists before the pool registers the pgvector type codec.
    await _db.run_migrations(settings.database_url)
    _pool = await _db.open_pool(settings.database_url)
    # Pre-warm the embedding model so the first /resolve is not slow
    _emb.get_model()
    yield
    await _db.close_pool(_pool)
    _pool = None


app = FastAPI(title="gfix-cloud", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/resolve", response_model=ResolveResponse)
async def resolve(req: ResolveRequest) -> ResolveResponse:
    try:
        response = await gfix_bridge.resolve_conflict(
            base=req.base,
            ours=req.ours,
            theirs=req.theirs,
            file_path=req.file_path,
            pool=_pool,
            use_rag=req.rag,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Persist with embedding — failure must NOT break the resolved response.
    if _pool is not None:
        try:
            language = _emb.language_from_path(req.file_path)
            conflict_kind = response.conflict.kind if response.conflict.kind != "none" else "text"
            embedding = _emb.embed_conflict(
                file_path=req.file_path,
                language=language,
                conflict_kind=conflict_kind,
                base=response.conflict.base.content,
                ours=response.conflict.ours.content,
                theirs=response.conflict.theirs.content,
            )
            conflict_text = _emb.build_conflict_text(
                file_path=req.file_path,
                language=language,
                conflict_kind=conflict_kind,
                base=response.conflict.base.content,
                ours=response.conflict.ours.content,
                theirs=response.conflict.theirs.content,
            )
            rh = _pers.rerere_hash(
                file_path=req.file_path,
                base_oid=response.conflict.base.oid,
                ours_oid=response.conflict.ours.oid,
                theirs_oid=response.conflict.theirs.oid,
            )
            ai_model = settings.generation_model if response.ai_rationale else None
            record = {
                "merge_id": response.merge_id,
                "file_path": req.file_path,
                "language": language,
                "conflict_kind": conflict_kind,
                "resolution_kind": response.via,
                "base_code": response.conflict.base.content,
                "ours_code": response.conflict.ours.content,
                "theirs_code": response.conflict.theirs.content,
                "resolved_content": response.resolved_content,
                "ai_model": ai_model,
                "ai_confidence": response.ai_confidence,
                "ai_rationale": response.ai_rationale,
                "used_rag": response.used_rag,
                "base_oid": response.conflict.base.oid,
                "ours_oid": response.conflict.ours.oid,
                "theirs_oid": response.conflict.theirs.oid,
                "rerere_hash": rh,
                "embedding": embedding,
                "conflict_text": conflict_text,
            }
            rid = await _pers.persist_resolution(_pool, record)
            logger.info("persisted resolution_id=%s", rid)
        except Exception as persist_exc:
            logger.exception("persistence failed (non-fatal): %s", persist_exc)

    return response


# ── Debug endpoint: retrieve similar past conflicts ───────────────────────────

class SimilarRequest(BaseModel):
    base: str
    ours: str
    theirs: str
    file_path: str
    k: Optional[int] = 5


@app.post("/similar")
async def similar(req: SimilarRequest) -> list[dict]:
    """Return top-k similar past resolutions ordered by inner-product similarity."""
    if _pool is None:
        raise HTTPException(status_code=503, detail="database not available")
    language = _emb.language_from_path(req.file_path)
    embedding = _emb.embed_conflict(
        file_path=req.file_path,
        language=language,
        conflict_kind="text",
        base=req.base,
        ours=req.ours,
        theirs=req.theirs,
    )
    neighbors = await _pers.find_similar(_pool, embedding, language, k=req.k or 5)
    # Convert non-serializable types
    for n in neighbors:
        if "created_at" in n and n["created_at"] is not None:
            n["created_at"] = n["created_at"].isoformat()
        if "score" in n and n["score"] is not None:
            n["score"] = float(n["score"])
    return neighbors
