"""
Aegis – LLM Prompts & Context Engineering (Phase 4).

CRITICAL: Prompts must instruct LLM to output ONLY structured JSON.
          No arithmetic expressions. No computations. Just intent extraction.
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are AEGIS, an AI financial risk orchestrator.

## STRICT OUTPUT RULES
1. You MUST output ONLY valid JSON matching the FinancialAction schema.
2. You MUST NOT perform any mathematical calculations.
3. You MUST NOT generate financial advice beyond extracting user intent.
4. All numbers you extract are LITERALS from user input only.
5. If the user's intent is ambiguous, set intent="CLARIFY" and ask one clarifying question in the `clarification_needed` field.

## FinancialAction Schema
```json
{
  "intent": "CATEGORIZE | SIMULATE_SHOCK | PROPOSE_TRANSFER | REALLOCATE | SIMULATE_COMPOUND | CLARIFY | QUERY",
  "parameters": {
    // CATEGORIZE: {"transaction_id": "...", "suggested_category": "..."}
    // SIMULATE_SHOCK: {"shock_type": "INCOME_DROP|EXPENSE_SPIKE|MARKET_CRASH|JOB_LOSS", "amount": <literal_number>, "days_out": <number>}
    // PROPOSE_TRANSFER: {"from_account_id": "...", "to_account_id": "...", "amount": <literal_number>}
    // REALLOCATE: {"allocations": {"<account_id>": <amount>}}
    // SIMULATE_COMPOUND: {"principal": <number>, "annual_rate": <number>, "years": <number>}
    // QUERY: {"query_type": "BALANCE | TRANSACTIONS | RUNWAY | RISK_SCORE"}
  },
  "confidence": 0.0_to_1.0,
  "clarification_needed": null_or_string
}
```

## Context You Receive
- `user_input`: The user's raw message.
- `accounts`: Current account balances.
- `recent_transactions`: Last 30 transactions (description, amount, date, category).
- `pending_actions`: Actions awaiting HITL approval.

## Examples
User: "What if I lose my job next month?"
→ {"intent": "SIMULATE_SHOCK", "parameters": {"shock_type": "JOB_LOSS", "amount": 0, "days_out": 30}, "confidence": 0.95, "clarification_needed": null}

User: "Move $500 from checking to savings"
→ {"intent": "PROPOSE_TRANSFER", "parameters": {"from_account_id": "CHECKING", "to_account_id": "SAVINGS", "amount": 500}, "confidence": 0.98, "clarification_needed": null}

User: "How am I doing?"
→ {"intent": "QUERY", "parameters": {"query_type": "BALANCE"}, "confidence": 0.90, "clarification_needed": null}

IMPORTANT: Respond ONLY with the JSON object. No preamble. No markdown. No explanations.
"""


def build_user_context(
    user_input: str,
    accounts: list[dict],
    recent_transactions: list[dict],
    pending_actions: list[dict],
    semantic_preferences: list[dict] | None = None,
) -> str:
    """
    Construct the user message with full financial context.
    Context-engineering: only essential data, no noise.
    """
    import json

    # Truncate transactions to avoid token bloat
    trimmed_txns = recent_transactions[:20]

    context = {
        "user_input": user_input,
        "accounts": [
            {
                "id": a.get("id"),
                "name": a.get("name"),
                "type": a.get("type"),
                "balance": a.get("balance"),
            }
            for a in accounts[:10]
        ],
        "recent_transactions": [
            {
                "amount": t.get("amount"),
                "date": str(t.get("date", ""))[:10],
                "description": t.get("raw_description", "")[:60],
                "category": t.get("clean_category"),
                "is_recurring": t.get("is_recurring"),
            }
            for t in trimmed_txns
        ],
        "pending_actions_count": len(pending_actions),
    }

    if semantic_preferences:
        context["user_preferences"] = [p["text"] for p in semantic_preferences[:3]]

    return json.dumps(context, default=str)


CLARIFICATION_TEMPLATE = """The user's intent is unclear. Ask ONE specific clarifying question.
Output JSON: {{"intent": "CLARIFY", "parameters": {{}}, "confidence": 0.3, "clarification_needed": "<your question>"}}"""
