"""
Aegis – FastAPI Entry Point (Phase 1-6).

Starts the application, initializes DB, mounts all routers,
and sets up LangGraph checkpointer on lifespan.
"""
from __future__ import annotations

import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.database.models import init_db
from backend.api.crud import router as crud_router
from backend.api.ingestion import router as plaid_router
from backend.api.hitl_queue import router as agent_router

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init DB and LangGraph checkpointer."""
    log.info("aegis.startup", app=settings.APP_NAME)
    try:
        init_db(settings.DATABASE_URL)
        log.info("aegis.db.initialized")
    except Exception as exc:
        log.error("aegis.db.init_failed", error=str(exc))

    yield
    log.info("aegis.shutdown")


app = FastAPI(
    title="Aegis – Financial Risk Orchestrator",
    description="""
## Aegis API

AI-powered personal finance and financial risk agent with:
- **Deterministic Math Engine** (Zero LLM Math)
- **Human-In-The-Loop (HITL)** action approval
- **GRPO Reinforcement Learning** for portfolio optimization
- **Graceful Degradation** with circuit breakers
- **Ghost Ledger** cash flow simulation
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────────────────────────
app.include_router(crud_router)
app.include_router(plaid_router)
app.include_router(agent_router)


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok", "app": settings.APP_NAME, "version": "1.0.0"}


@app.get("/", tags=["system"])
def root():
    return {
        "message": "AEGIS Financial Risk Orchestrator",
        "docs": "/docs",
        "health": "/health",
    }
