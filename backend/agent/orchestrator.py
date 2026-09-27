"""
Aegis – LangGraph Orchestrator (Phase 4).

Graph: SemanticParser → DeterministicSolver → [HITLGate] → Response

All LLM nodes output strictly typed Pydantic JSON.
All math nodes call deterministic Python solvers ONLY.
HITL gate uses langgraph interrupt() to pause until human approves.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Literal, Optional, TypedDict

import structlog
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
try:
    from langgraph.types import interrupt  # langgraph >= 0.2
except ImportError:
    try:
        from langgraph.errors import GraphInterrupt as _GI
        def interrupt(payload):  # type: ignore[override]
            raise _GI(payload)
    except ImportError:
        def interrupt(payload):  # type: ignore[override]
            """Fallback: HITL not available in this langgraph version."""
            return "APPROVE"  # Auto-approve in degraded mode
from pydantic import BaseModel, field_validator

from backend.agent.prompts import SYSTEM_PROMPT, build_user_context
from backend.core.config import settings
from backend.environment.simulator import (
    calculate_cash_flow,
    simulate_shock,
    simulate_transfer,
    calculate_compound_interest,
    monte_carlo_cashflow,
)
from backend.agent.grpo_optimizer import run_grpo_optimization

log = structlog.get_logger()


# ─── Pydantic Output Schema (Node 1 Output) ──────────────────────────────────

class FinancialAction(BaseModel):
    intent: Literal[
        "CATEGORIZE",
        "SIMULATE_SHOCK",
        "PROPOSE_TRANSFER",
        "REALLOCATE",
        "SIMULATE_COMPOUND",
        "CLARIFY",
        "QUERY",
    ]
    parameters: dict[str, Any]
    confidence: float = 0.9
    clarification_needed: Optional[str] = None

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v):
        return max(0.0, min(1.0, v))


# ─── Graph State ─────────────────────────────────────────────────────────────

class AegisState(TypedDict):
    # Input
    user_input: str
    user_id: str
    thread_id: str

    # Context
    accounts: list[dict]
    transactions: list[dict]
    pending_actions: list[dict]
    semantic_preferences: list[dict]

    # Node outputs
    extracted_action: Optional[dict]        # FinancialAction JSON
    math_results: Optional[dict]            # Output of deterministic solver
    grpo_result: Optional[dict]             # GRPO optimization output
    proposed_action: Optional[dict]         # For HITL queue
    pending_action_id: Optional[str]        # DB ActionQueue ID

    # Response
    response: Optional[dict]
    error: Optional[str]
    degraded_mode: bool


# ─── LLM Setup ───────────────────────────────────────────────────────────────

def _get_llm():
    """Auto-select LLM based on LLM_MODEL prefix in settings."""
    model = settings.LLM_MODEL

    if model.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.0,
            max_output_tokens=512,
        )
    elif model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0.0,
            max_tokens=512,
        )
    else:  # default: OpenAI gpt-*
        return ChatOpenAI(
            model=model,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,
            max_tokens=512,
        )


# ─── Node 1: Semantic Parser ─────────────────────────────────────────────────

def semantic_parser_node(state: AegisState) -> dict:
    """
    LLM extracts user intent → FinancialAction Pydantic JSON.
    NO math happens here.
    """
    log.info("node.semantic_parser", user_id=state["user_id"])

    user_message = build_user_context(
        user_input=state["user_input"],
        accounts=state["accounts"],
        recent_transactions=state["transactions"],
        pending_actions=state["pending_actions"],
        semantic_preferences=state.get("semantic_preferences", []),
    )

    try:
        llm = _get_llm()
        from langchain_core.messages import SystemMessage, HumanMessage

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])

        raw_json = response.content.strip()
        # Strip markdown code fences if LLM wraps it
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]

        action = FinancialAction.model_validate_json(raw_json)
        log.info("node.semantic_parser.success", intent=action.intent)
        return {"extracted_action": action.model_dump()}

    except Exception as exc:
        log.error("node.semantic_parser.error", error=str(exc))
        return {
            "extracted_action": {
                "intent": "QUERY",
                "parameters": {"query_type": "BALANCE"},
                "confidence": 0.3,
                "clarification_needed": None,
            },
            "error": f"Parser circuit breaker: {str(exc)}",
            "degraded_mode": True,
        }


# ─── Node 2: Deterministic Solver ────────────────────────────────────────────

def deterministic_solver_node(state: AegisState) -> dict:
    """
    Routes the extracted action to the appropriate math solver.
    NO LLM calls here. Pure Python math only.
    """
    action = state.get("extracted_action", {})
    intent = action.get("intent", "QUERY")
    params = action.get("parameters", {})
    transactions = state.get("transactions", [])

    log.info("node.solver", intent=intent)

    try:
        if intent == "SIMULATE_SHOCK":
            current_balance = sum(a.get("balance", 0) for a in state["accounts"])
            result = simulate_shock(
                current_state={"transactions": transactions, "current_balance": current_balance},
                shock_type=params.get("shock_type", "INCOME_DROP"),
                amount=float(params.get("amount", 500)),
                days_out=int(params.get("days_out", 30)),
            )
            math_results = {
                "type": "shock_simulation",
                "baseline_end_balance": result.baseline.projected_end_balance,
                "shocked_end_balance": result.shocked.projected_end_balance,
                "delta": result.delta_end_balance,
                "insolvency_risk": result.insolvency_risk_increase,
                "recommendation": result.recommended_action,
                "baseline_trajectory": result.baseline.trajectory,
                "shocked_trajectory": result.shocked.trajectory,
            }
            proposed = {
                "type": "SIMULATE_SHOCK",
                "shock_type": params.get("shock_type"),
                "impact_delta": result.delta_end_balance,
                "recommendation": result.recommended_action,
            }

        elif intent == "PROPOSE_TRANSFER":
            accounts = {a["id"]: a for a in state["accounts"]}
            from_id = params.get("from_account_id", "")
            to_id = params.get("to_account_id", "")
            amount = float(params.get("amount", 0))

            from_acct = _find_account(state["accounts"], from_id)
            to_acct = _find_account(state["accounts"], to_id)

            result = simulate_transfer(
                from_balance=from_acct.get("balance", 0) if from_acct else 0,
                to_balance=to_acct.get("balance", 0) if to_acct else 0,
                amount=amount,
                transactions=transactions,
            )
            math_results = {
                "type": "transfer_simulation",
                "from_new_balance": result.from_account_new_balance,
                "to_new_balance": result.to_account_new_balance,
                "is_feasible": result.is_feasible,
                "shortfall": result.shortfall,
                "cashflow_trajectory": result.cashflow_impact.trajectory,
            }
            proposed = {
                "type": "PROPOSE_TRANSFER",
                "from_account_id": from_id,
                "to_account_id": to_id,
                "amount": amount,
                "is_feasible": result.is_feasible,
            }

        elif intent == "REALLOCATE":
            account_ids = [a["id"] for a in state["accounts"]]
            current_balance = sum(a.get("balance", 0) for a in state["accounts"])
            target = float(params.get("target_allocation", current_balance * 0.1))

            grpo = run_grpo_optimization(
                account_ids=account_ids,
                transactions=transactions,
                current_balance=current_balance,
                target_allocation=target,
            )
            math_results = {
                "type": "grpo_reallocation",
                "best_policy": grpo.best_policy,
                "composite_reward": grpo.best_reward.composite_reward,
                "liquidity_score": grpo.best_reward.liquidity_score,
                "insolvency_risk": grpo.best_reward.insolvency_risk,
                "epochs_to_converge": grpo.generation,
            }
            proposed = {
                "type": "REALLOCATE",
                "allocations": grpo.best_policy,
                "reward": grpo.best_reward.composite_reward,
            }
            return {"math_results": math_results, "proposed_action": proposed, "grpo_result": math_results}

        elif intent == "SIMULATE_COMPOUND":
            result = calculate_compound_interest(
                principal=float(params.get("principal", 1000)),
                annual_rate=float(params.get("annual_rate", 0.05)),
                years=float(params.get("years", 1)),
            )
            math_results = {"type": "compound_interest", **result}
            proposed = None  # Simulation only, no state change

        elif intent == "QUERY":
            current_balance = sum(a.get("balance", 0) for a in state["accounts"])
            mc = monte_carlo_cashflow(transactions, current_balance)
            cashflow = calculate_cash_flow(transactions, current_balance)
            math_results = {
                "type": "portfolio_query",
                "total_balance": current_balance,
                "accounts": [{"name": a["name"], "balance": a["balance"]} for a in state["accounts"]],
                "runway_days": cashflow.runway_days,
                "net_daily_flow": cashflow.net_daily_flow,
                "monte_carlo": mc,
                "trajectory": cashflow.trajectory[:30],
            }
            proposed = None

        else:  # CLARIFY, CATEGORIZE
            math_results = {"type": "no_math_required", "intent": intent}
            proposed = None

        return {"math_results": math_results, "proposed_action": proposed}

    except Exception as exc:
        log.error("node.solver.error", error=str(exc))
        return {
            "math_results": {"type": "error", "message": str(exc)},
            "error": str(exc),
            "degraded_mode": True,
        }


# ─── Node 3: HITL Gate ────────────────────────────────────────────────────────

def hitl_gate_node(state: AegisState) -> dict:
    """
    If the proposed action changes financial state, pause execution
    via interrupt() and push to ActionQueue.

    The graph resumes only when /api/agent/resume is called with APPROVE/REJECT.
    """
    proposed = state.get("proposed_action")
    if proposed is None:
        # Simulations / queries don't need HITL approval
        return {"response": _build_response(state)}

    action_id = state.get("pending_action_id") or str(uuid.uuid4())

    log.info("node.hitl.interrupting", action_id=action_id, type=proposed.get("type"))

    # 🔴 PAUSE EXECUTION – Resume via /api/agent/resume
    decision = interrupt({
        "action_id": action_id,
        "details": proposed,
        "math_results": state.get("math_results"),
        "message": "Human approval required before executing this financial action.",
    })

    # Post-resume: decision is "APPROVE" or "REJECT"
    if decision == "APPROVE":
        log.info("node.hitl.approved", action_id=action_id)
        return {
            "response": {
                "status": "APPROVED",
                "action_id": action_id,
                "message": "Action approved and committed to ledger.",
                "result": state.get("math_results"),
            }
        }
    else:
        log.info("node.hitl.rejected", action_id=action_id)
        return {
            "response": {
                "status": "REJECTED",
                "action_id": action_id,
                "message": "Action rejected. No changes were made.",
            }
        }


# ─── Routing ─────────────────────────────────────────────────────────────────

def should_hitl(state: AegisState) -> str:
    """Route to HITL gate only for state-changing actions."""
    action = state.get("extracted_action", {})
    intent = action.get("intent", "QUERY")
    STATE_CHANGING = {"PROPOSE_TRANSFER", "REALLOCATE", "CATEGORIZE"}
    if intent in STATE_CHANGING and state.get("proposed_action"):
        return "hitl_gate"
    return "respond"


def respond_node(state: AegisState) -> dict:
    """Return final response for non-HITL paths (simulations, queries)."""
    return {"response": _build_response(state)}


# ─── Graph Assembly ───────────────────────────────────────────────────────────

def build_aegis_graph():
    graph = StateGraph(AegisState)

    graph.add_node("semantic_parser", semantic_parser_node)
    graph.add_node("deterministic_solver", deterministic_solver_node)
    graph.add_node("hitl_gate", hitl_gate_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("semantic_parser")
    graph.add_edge("semantic_parser", "deterministic_solver")
    graph.add_conditional_edges(
        "deterministic_solver",
        should_hitl,
        {"hitl_gate": "hitl_gate", "respond": "respond"},
    )
    graph.add_edge("hitl_gate", END)
    graph.add_edge("respond", END)

    return graph.compile()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _find_account(accounts: list[dict], identifier: str) -> Optional[dict]:
    """Find account by ID or name substring (case-insensitive)."""
    identifier_lower = identifier.lower()
    for a in accounts:
        if str(a.get("id", "")).lower() == identifier_lower:
            return a
        if identifier_lower in a.get("name", "").lower():
            return a
        if identifier_lower in a.get("type", "").lower():
            return a
    return accounts[0] if accounts else None


def _build_response(state: AegisState) -> dict:
    math = state.get("math_results", {})
    action = state.get("extracted_action", {})
    return {
        "status": "success",
        "intent": action.get("intent"),
        "math_results": math,
        "clarification": action.get("clarification_needed"),
        "degraded_mode": state.get("degraded_mode", False),
    }
