"""
asyncpg connection pool + migration runner for gfix-cloud.

Usage (FastAPI lifespan):
    await db.run_migrations(settings.database_url)   # plain connection, no codec
    pool = await db.open_pool(settings.database_url) # codec registered after ext exists
    ...
    await db.close_pool(pool)
"""
import logging
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

from app.config import settings

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"


async def _init_conn(conn: asyncpg.Connection) -> None:
    """Per-connection init: register pgvector codec + set HNSW ef_search.

    Called only AFTER run_migrations() has ensured the vector extension exists.
    """
    await register_vector(conn)
    await conn.execute(f"SET hnsw.ef_search = {settings.hnsw_ef_search}")


async def run_migrations(dsn: str) -> None:
    """Execute all migration SQL files in order via a plain connection.

    Uses a bare connection (no vector codec) so that CREATE EXTENSION IF NOT
    EXISTS vector can run before the codec is registered.  Safe to call on every
    startup — all DDL is guarded by IF NOT EXISTS.
    """
    migration_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    conn = await asyncpg.connect(dsn)
    try:
        for mf in migration_files:
            sql = mf.read_text()
            logger.info("running migration: %s", mf.name)
            await conn.execute(sql)
    finally:
        await conn.close()
    logger.info("migrations complete")


async def open_pool(dsn: str) -> asyncpg.Pool:
    """Create an asyncpg pool with the pgvector codec registered per connection.

    Must be called AFTER run_migrations() so the vector extension exists.
    """
    pool = await asyncpg.create_pool(
        dsn,
        init=_init_conn,
        min_size=1,
        max_size=10,
    )
    logger.info("asyncpg pool opened")
    return pool


async def close_pool(pool: asyncpg.Pool) -> None:
    await pool.close()
    logger.info("asyncpg pool closed")
