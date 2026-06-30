from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app import gfix_bridge
from app.models import ResolveRequest, ResolveResponse

app = FastAPI(title="gfix-cloud", version="0.1.0")

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
        return await gfix_bridge.resolve_conflict(
            base=req.base,
            ours=req.ours,
            theirs=req.theirs,
            file_path=req.file_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
