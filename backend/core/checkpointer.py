"""
Aegis – LangGraph PostgresSaver Checkpointer (Phase 4).

Manages persistent graph state in PostgreSQL so the HITL interrupt
can be resumed across HTTP requests.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import structlog

log = structlog.get_logger()


async def create_checkpointer(dsn: str):
    """
    Create a LangGraph PostgresSaver backed by psycopg_pool.

    Returns the saver and pool so they can be managed as app lifespan resources.
    """
    try:
        from psycopg_pool import AsyncConnectionPool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        pool = AsyncConnectionPool(
            conninfo=dsn,
            max_size=10,
            open=False,
        )
        await pool.open()
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        log.info("checkpointer.ready", dsn=dsn[:40])
        return saver, pool

    except ImportError as exc:
        log.warning("checkpointer.missing_deps", error=str(exc))
        # Fallback to in-memory for local dev without psycopg3
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver(), None

    except Exception as exc:
        log.error("checkpointer.failed", error=str(exc))
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver(), None


def build_thread_config(thread_id: str) -> dict:
    """LangGraph thread config for graph invocation and resume."""
    return {"configurable": {"thread_id": thread_id}}
