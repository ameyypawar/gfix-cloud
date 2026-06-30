from typing import Optional

from pydantic import BaseModel


class ResolveRequest(BaseModel):
    base: str
    ours: str
    theirs: str
    file_path: str
    rag: bool = True


class ConflictSide(BaseModel):
    """One side of a conflict (base, ours, or theirs).

    Real conflict_get shape (captured 2026-06-30):
    - base:   {content, oid}           — no source field
    - ours:   {content, oid, source}
    - theirs: {content, oid, source}
    Deviation from engine-interface.md: 'encoding_error' field does not exist in
    any side; 'source' is absent on base.
    """

    content: str
    oid: str
    source: Optional[str] = None


class TargetSide(BaseModel):
    """Target (destination branch) state of the conflicted file.

    Real shape: {content, oid, exists}.  No 'source' field.
    """

    content: str
    oid: str
    exists: bool = True


class ConflictDetail(BaseModel):
    """Structured conflict data from gitfix_conflict_get."""

    conflict_id: str
    file: str
    kind: str
    ours: ConflictSide
    theirs: ConflictSide
    base: ConflictSide
    target: TargetSide


class RetrievedNeighbor(BaseModel):
    file_path: str
    language: str
    resolution_kind: str
    similarity: float
    resolved_content_preview: str


class ResolveResponse(BaseModel):
    merge_id: str
    file_path: str
    resolved_content: str
    via: str
    audit_ref: Optional[str] = None
    conflict: ConflictDetail
    used_rag: bool = False
    neighbors: list[RetrievedNeighbor] = []
    ai_rationale: Optional[str] = None
    ai_confidence: Optional[float] = None
