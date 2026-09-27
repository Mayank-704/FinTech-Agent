"""
Aegis – Agent Invocation & HITL Queue API (Phase 4).

/api/agent/chat    POST  → Invoke LangGraph agent
/api/agent/resume  POST  → Resume interrupted graph (APPROVE/REJECT)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.crud import get_db
from backend.core.config import settings
from backend.database.models import (
    Account, ActionQueue, ActionStatus, ActionType, Transaction
)

log = structlog.get_logger()

router = APIRouter(prefix="/api/agent", tags=["agent"])

# Global graph (initialized on app startup)
_aegis_graph = None
_checkpointer = None


def get_graph():
    global _aegis_graph
    if _aegis_graph is None:
        from backend.agent.orchestrator import build_aegis_graph
        _aegis_graph = build_aegis_graph()
    return _aegis_graph


# ─── Request/Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    message: str
    thread_id: Optional[str] = None   # Resume existing thread or start new


class ResumeRequest(BaseModel):
    action_id: str
    decision: Literal["APPROVE", "REJECT"]
    thread_id: str
    user_id: str


# ─── Chat Endpoint ────────────────────────────────────────────────────────────

@router.post("/chat", summary="Send message to Aegis agent")
def agent_chat(body: ChatRequest, db: Session = Depends(get_db)):
    """
    Invoke the LangGraph orchestrator.

    If the agent reaches a HITL gate, it pauses, saves the ActionQueue entry,
    and returns a pending_action_id for the frontend to display in the command queue.
    """
    thread_id = body.thread_id or str(uuid.uuid4())
    user_id = body.user_id

    # Fetch context from PostgreSQL
    try:
        accounts = db.query(Account).filter(Account.user_id == user_id).all()
        acct_ids = [a.id for a in accounts]
        transactions = []
        for aid in acct_ids:
            txns = (
                db.query(Transaction)
                .filter(Transaction.account_id == aid)
                .order_by(Transaction.date.desc())
                .limit(30)
                .all()
            )
            transactions.extend(txns)

        pending = (
            db.query(ActionQueue)
            .filter(ActionQueue.user_id == user_id, ActionQueue.status == ActionStatus.PENDING)
            .all()
        )

        accounts_data = [
            {"id": str(a.id), "name": a.name, "balance": a.balance, "type": a.type.value}
            for a in accounts
        ]
        txns_data = [
            {
                "amount": t.amount,
                "date": str(t.date),
                "raw_description": t.raw_description,
                "clean_category": t.clean_category,
                "is_recurring": t.is_recurring,
            }
            for t in sorted(transactions, key=lambda x: x.date, reverse=True)
        ]
        pending_data = [
            {"id": str(a.id), "type": a.proposed_action_type.value, "payload": a.payload}
            for a in pending
        ]
    except Exception as exc:
        log.error("agent.chat.db_error", error=str(exc))
        accounts_data, txns_data, pending_data = [], [], []

    # Build initial state
    initial_state = {
        "user_input": body.message,
        "user_id": body.user_id,
        "thread_id": thread_id,
        "accounts": accounts_data,
        "transactions": txns_data,
        "pending_actions": pending_data,
        "semantic_preferences": [],
        "extracted_action": None,
        "math_results": None,
        "grpo_result": None,
        "proposed_action": None,
        "pending_action_id": None,
        "response": None,
        "error": None,
        "degraded_mode": False,
    }

    # Run the graph
    try:
        graph = get_graph()
        config = {"configurable": {"thread_id": thread_id}}

        result = graph.invoke(initial_state, config=config)

        # Check if we hit a HITL interrupt
        proposed = result.get("proposed_action")
        if proposed:
            # Save to ActionQueue for the frontend command panel
            action_id = str(uuid.uuid4())
            action_type_map = {
                "PROPOSE_TRANSFER": ActionType.PROPOSE_TRANSFER,
                "REALLOCATE": ActionType.REALLOCATE,
                "CATEGORIZE": ActionType.CATEGORIZE,
                "SIMULATE_SHOCK": ActionType.SIMULATE_SHOCK,
            }
            action_type = action_type_map.get(proposed.get("type", ""), ActionType.PROPOSE_TRANSFER)

            queue_entry = ActionQueue(
                id=action_id,
                user_id=user_id,
                langgraph_thread_id=thread_id,
                proposed_action_type=action_type,
                payload=proposed,
                simulation_result=result.get("math_results"),
                status=ActionStatus.PENDING,
            )
            db.add(queue_entry)
            db.commit()

            return {
                "status": "pending_approval",
                "thread_id": thread_id,
                "action_id": action_id,
                "message": "Action requires your approval. Check the Command Queue.",
                "proposed_action": proposed,
                "simulation_preview": result.get("math_results"),
            }

        return {
            "status": "success",
            "thread_id": thread_id,
            "response": result.get("response"),
            "math_results": result.get("math_results"),
            "degraded_mode": result.get("degraded_mode", False),
        }

    except Exception as exc:
        log.error("agent.chat.error", error=str(exc))
        return {
            "status": "error",
            "thread_id": thread_id,
            "error": str(exc),
            "degraded_mode": True,
        }


# ─── Resume Endpoint (HITL) ───────────────────────────────────────────────────

@router.post("/resume", summary="Resume paused graph after HITL decision")
def agent_resume(body: ResumeRequest, db: Session = Depends(get_db)):
    """
    Resume a LangGraph execution that was interrupted at the HITL gate.

    Decision: APPROVE → execute_db_commit → return success
              REJECT  → return aborted, no state change
    """
    action_id_str = body.action_id
    action = db.query(ActionQueue).filter(ActionQueue.id == action_id_str).first()
    if not action:
        raise HTTPException(404, f"Action {body.action_id} not found")

    if action.status != ActionStatus.PENDING:
        raise HTTPException(400, f"Action is already {action.status.value}")

    # Update DB status
    action.status = ActionStatus.APPROVED if body.decision == "APPROVE" else ActionStatus.REJECTED
    action.decided_at = datetime.now(timezone.utc)
    db.commit()

    # Resume the LangGraph execution with the decision
    try:
        graph = get_graph()
        config = {"configurable": {"thread_id": body.thread_id}}
        result = graph.invoke(body.decision, config=config)

        return {
            "status": "resumed",
            "decision": body.decision,
            "action_id": body.action_id,
            "result": result.get("response") if result else {"status": body.decision.lower()},
        }

    except Exception as exc:
        # Even if graph resume fails, the DB decision is recorded
        log.error("agent.resume.graph_error", error=str(exc))
        return {
            "status": "decision_recorded",
            "decision": body.decision,
            "action_id": body.action_id,
            "note": "DB updated. Graph resume encountered an error (may need restart).",
        }
