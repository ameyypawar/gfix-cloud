"""
gfix MCP bridge.

Flow per request:
  materialize scratch repo → spawn gfix mcp → merge_preview → conflict_get
  → deterministic floor (mergiraf) → if mergiraf fails: RAG-augmented LLM suggestion
  → merge_apply → read resolved file
"""
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

from app.models import ConflictDetail, ConflictSide, ResolveResponse, RetrievedNeighbor, TargetSide

logger = logging.getLogger(__name__)


def _materialize_scratch_repo(
    tmpdir: str, file_path: str, base: str, ours: str, theirs: str
) -> None:
    """Build a 3-commit scratch git repo.

    Layout:
      main  (initial): file = base content
      ours  (branch):  file = ours content  ← HEAD (target branch)
      theirs(branch):  file = theirs content, forked from base commit

    Target branch = 'ours' (not main/master/develop) to dodge gfix's
    protected-branch guard without needing auto_approve on preview.
    """
    env = {**os.environ}

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git"] + list(args),
            cwd=tmpdir,
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git {list(args)} failed (rc={result.returncode}): {result.stderr.strip()}"
            )
        return result.stdout.strip()

    fp = Path(tmpdir) / file_path
    fp.parent.mkdir(parents=True, exist_ok=True)

    git("init")
    git("config", "user.email", "gfix-cloud@scratch.local")
    git("config", "user.name", "gfix-cloud")

    fp.write_text(base)
    git("add", file_path)
    git("commit", "-m", "base")
    base_sha = git("rev-parse", "HEAD")

    git("checkout", "-b", "ours")
    fp.write_text(ours)
    git("add", file_path)
    git("commit", "-m", "ours")

    git("checkout", "-b", "theirs", base_sha)
    fp.write_text(theirs)
    git("add", file_path)
    git("commit", "-m", "theirs")

    git("checkout", "ours")


def _parse_conflict_side(d: dict) -> ConflictSide:
    return ConflictSide(
        content=d["content"],
        oid=d["oid"],
        source=d.get("source"),
    )


async def resolve_conflict(
    base: str,
    ours: str,
    theirs: str,
    file_path: str,
    pool=None,
    use_rag: bool = True,
) -> ResolveResponse:
    """Drive the real gfix MCP server end-to-end and return a ResolveResponse.

    GITFIX_ALLOW_ANY_REPO=1 is mandatory: scratch repos live in the system
    tempdir, outside the server's startup CWD, and gfix's workspace fence
    rejects them otherwise.

    When mergiraf fails for a conflict, falls back to RAG-augmented LLM
    generation (produce_suggestion).  use_rag=False sends no examples to
    the LLM, giving the baseline for eval.
    """
    # Deferred import to avoid circular issues and keep top-level import clean
    from app.rag import produce_suggestion

    with tempfile.TemporaryDirectory() as tmpdir:
        _materialize_scratch_repo(tmpdir, file_path, base, ours, theirs)

        server_params = StdioServerParameters(
            command="gfix",
            args=["mcp"],
            cwd=tmpdir,
            env={**os.environ, "GITFIX_ALLOW_ANY_REPO": "1"},
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                tools = await session.list_tools()
                logger.info("gfix tools available: %s", [t.name for t in tools.tools])

                # ── 1. merge_preview ──────────────────────────────────────
                preview_result = await session.call_tool(
                    "gitfix_merge_preview",
                    {
                        "repo_path": tmpdir,
                        "target": "ours",
                        "sources": ["theirs"],
                    },
                )
                plan: dict = json.loads(preview_result.content[0].text)
                merge_id: str = plan["merge_id"]
                unresolved: list = plan.get("unresolved", [])
                logger.info(
                    "merge_preview: merge_id=%s unresolved=%d",
                    merge_id,
                    len(unresolved),
                )

                conflict_detail: Optional[ConflictDetail] = None
                via: str = "git_automerge"
                ai_rationale: Optional[str] = None
                ai_confidence: Optional[float] = None
                suggestion_neighbors: list = []
                used_llm = False

                # ── 2. Per-conflict: get → resolve ────────────────────────
                for entry in unresolved:
                    conflict_id: str = entry["conflict_id"]

                    cg_result = await session.call_tool(
                        "gitfix_conflict_get",
                        {
                            "repo_path": tmpdir,
                            "merge_id": merge_id,
                            "conflict_id": conflict_id,
                            "include_ai_suggestion": False,
                        },
                    )
                    cg: dict = json.loads(cg_result.content[0].text)

                    if conflict_detail is None:
                        target_raw = cg["target"]
                        conflict_detail = ConflictDetail(
                            conflict_id=cg["conflict_id"],
                            file=cg["file"],
                            kind=cg["kind"],
                            ours=_parse_conflict_side(cg["ours"]),
                            theirs=_parse_conflict_side(cg["theirs"]),
                            base=_parse_conflict_side(cg["base"]),
                            target=TargetSide(
                                content=target_raw["content"],
                                oid=target_raw["oid"],
                                exists=target_raw.get("exists", True),
                            ),
                        )

                    # Deterministic floor: try mergiraf first
                    try:
                        res = await session.call_tool(
                            "gitfix_conflict_resolve",
                            {
                                "repo_path": tmpdir,
                                "merge_id": merge_id,
                                "conflict_id": conflict_id,
                                "resolution": {"kind": "mergiraf"},
                            },
                        )
                        res_j: dict = json.loads(res.content[0].text)
                        if not res_j.get("resolved", False):
                            raise ValueError("mergiraf returned resolved=false")
                        via = res_j.get("via", "mergiraf")
                        logger.info("conflict %s resolved via mergiraf", conflict_id)
                    except (McpError, ValueError):
                        # Mergiraf cannot resolve this conflict — call the LLM
                        logger.info(
                            "mergiraf failed for %s — calling RAG+LLM (use_rag=%s)",
                            conflict_id,
                            use_rag,
                        )
                        suggestion, neighbors = await produce_suggestion(
                            pool=pool,
                            conflict=conflict_detail,
                            use_rag=use_rag,
                        )
                        ai_rationale = suggestion.rationale
                        ai_confidence = suggestion.confidence
                        suggestion_neighbors = neighbors
                        used_llm = True

                        res2 = await session.call_tool(
                            "gitfix_conflict_resolve",
                            {
                                "repo_path": tmpdir,
                                "merge_id": merge_id,
                                "conflict_id": conflict_id,
                                "resolution": {
                                    "kind": "manual",
                                    "text": suggestion.text,
                                },
                            },
                        )
                        res2_j: dict = json.loads(res2.content[0].text)
                        via = res2_j.get("via", "manual")

                # ── 3. merge_apply ────────────────────────────────────────
                apply_result = await session.call_tool(
                    "gitfix_merge_apply",
                    {
                        "repo_path": tmpdir,
                        "merge_id": merge_id,
                        "commit": True,
                        "auto_approve": True,
                    },
                )
                apply: dict = json.loads(apply_result.content[0].text)
                audit_ref: Optional[str] = apply.get("audit_ref")
                logger.info(
                    "merge_apply: committed=%s audit_ref=%s",
                    apply.get("committed"),
                    audit_ref,
                )

                # ── 4. Read resolved file from working tree ───────────────
                resolved_content = (Path(tmpdir) / file_path).read_text()

                # Stub conflict_detail when git auto-merged with zero conflicts
                if conflict_detail is None:
                    conflict_detail = ConflictDetail(
                        conflict_id="",
                        file=file_path,
                        kind="none",
                        ours=ConflictSide(content=ours, oid="", source="ours"),
                        theirs=ConflictSide(content=theirs, oid="", source="theirs"),
                        base=ConflictSide(content=base, oid="", source=None),
                        target=TargetSide(content=ours, oid="", exists=True),
                    )

                # Map internal Neighbor list → public RetrievedNeighbor list
                public_neighbors = [
                    RetrievedNeighbor(
                        file_path=n.file_path,
                        language=n.language,
                        resolution_kind=n.resolution_kind,
                        similarity=n.rrf_score,
                        resolved_content_preview=n.resolved_content[:200],
                    )
                    for n in suggestion_neighbors
                ]

                return ResolveResponse(
                    merge_id=merge_id,
                    file_path=file_path,
                    resolved_content=resolved_content,
                    via=via,
                    audit_ref=audit_ref,
                    conflict=conflict_detail,
                    used_rag=used_llm and use_rag,
                    neighbors=public_neighbors,
                    ai_rationale=ai_rationale,
                    ai_confidence=ai_confidence,
                )
