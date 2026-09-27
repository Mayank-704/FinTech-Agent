"""
Aegis – Plaid Ingestion Endpoint (Phase 2).

/api/sync-plaid  POST  → Fetch Plaid Sandbox data → Upsert to PostgreSQL

Circuit breaker: Any Plaid failure returns {degraded: true, data: <last_pg_state>}
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.crud import get_db
from backend.core.config import settings
from backend.database.models import Account, AccountType, Transaction

log = structlog.get_logger()

router = APIRouter(prefix="/api", tags=["plaid"])


class PlaidSyncRequest(BaseModel):
    user_id: str
    access_token: Optional[str] = None  # From Plaid Link (sandbox or real)


class PlaidLinkRequest(BaseModel):
    user_id: str


# ─── Plaid Client Factory (with circuit breaker) ─────────────────────────────

def _get_plaid_client():
    """Returns a Plaid API client or raises PlaidUnavailableError."""
    try:
        import plaid
        from plaid.api import plaid_api
        from plaid.model.products import Products
        from plaid.model.country_code import CountryCode
        from plaid.configuration import Configuration
        from plaid.api_client import ApiClient

        env_map = {
            "sandbox": plaid.Environment.Sandbox,
            "development": plaid.Environment.Development,
            "production": plaid.Environment.Production,
        }

        configuration = Configuration(
            host=env_map.get(settings.PLAID_ENV, plaid.Environment.Sandbox),
            api_key={
                "clientId": settings.PLAID_CLIENT_ID,
                "secret": settings.PLAID_SECRET,
            },
        )
        api_client = ApiClient(configuration)
        return plaid_api.PlaidApi(api_client)

    except ImportError:
        raise RuntimeError("plaid-python not installed")


# ─── Sandbox Token Helper ─────────────────────────────────────────────────────

@router.post("/plaid/sandbox-token", summary="Create a Plaid Sandbox access token")
def create_sandbox_token(body: PlaidLinkRequest):
    """
    Creates a Plaid Sandbox public token and exchanges it for an access token.
    Use this to get a token to pass to /api/sync-plaid.
    """
    try:
        from plaid.model.sandbox_public_token_create_request import SandboxPublicTokenCreateRequest
        from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
        from plaid.model.products import Products
        from plaid.model.country_code import CountryCode

        client = _get_plaid_client()

        # Create sandbox public token (simulates Plaid Link flow)
        sandbox_req = SandboxPublicTokenCreateRequest(
            institution_id="ins_109508",  # Chase Bank sandbox
            initial_products=[Products("transactions")],
        )
        sandbox_response = client.sandbox_public_token_create(sandbox_req)
        public_token = sandbox_response["public_token"]

        # Exchange for access token
        exchange_req = ItemPublicTokenExchangeRequest(public_token=public_token)
        exchange_response = client.item_public_token_exchange(exchange_req)
        access_token = exchange_response["access_token"]

        return {
            "status": "ok",
            "access_token": access_token,
            "message": "Use this access_token in /api/sync-plaid",
        }

    except Exception as exc:
        log.error("plaid.sandbox_token.failed", error=str(exc))
        return {"status": "degraded", "error": str(exc), "access_token": None}


# ─── Main Sync Endpoint ───────────────────────────────────────────────────────

@router.post("/sync-plaid", summary="Sync Plaid Sandbox accounts & transactions")
def sync_plaid(body: PlaidSyncRequest, db: Session = Depends(get_db)):
    """
    Phase 2 Core Endpoint.

    Circuit Breaker: If Plaid fails at any point, returns the last known
    PostgreSQL state with degraded_mode: true. Never crashes.
    """
    user_id = body.user_id

    # ── Attempt Plaid fetch ────────────────────────────────────────────────
    try:
        if not body.access_token:
            raise ValueError("No Plaid access_token provided. Use /api/plaid/sandbox-token first.")

        plaid_data = _fetch_plaid_data(body.access_token)
        _upsert_plaid_data(db, user_id, plaid_data)
        log.info("plaid.sync.success", user_id=str(user_id))
        return {
            "status": "ok",
            "degraded_mode": False,
            "accounts_synced": len(plaid_data["accounts"]),
            "transactions_synced": len(plaid_data["transactions"]),
        }

    except Exception as exc:
        # ── Circuit breaker: Return last known PostgreSQL state ───────────
        log.error("plaid.sync.circuit_breaker", error=str(exc))
        last_accounts = db.query(Account).filter(Account.user_id == user_id).all()
        last_txns = (
            db.query(Transaction)
            .join(Account)
            .filter(Account.user_id == user_id)
            .order_by(Transaction.date.desc())
            .limit(50)
            .all()
        )
        return {
            "status": "degraded",
            "degraded_mode": True,
            "error": str(exc),
            "data": {
                "accounts": [
                    {"id": str(a.id), "name": a.name, "balance": a.balance, "type": a.type.value}
                    for a in last_accounts
                ],
                "transactions": [
                    {
                        "id": str(t.id),
                        "amount": t.amount,
                        "date": str(t.date),
                        "description": t.raw_description,
                        "category": t.clean_category,
                    }
                    for t in last_txns
                ],
            },
        }


# ─── Plaid Data Fetching ──────────────────────────────────────────────────────

def _fetch_plaid_data(access_token: str) -> dict:
    """Fetch accounts and transactions from Plaid (v44+ cursor-based sync)."""
    from plaid.model.accounts_get_request import AccountsGetRequest
    from plaid.model.transactions_sync_request import TransactionsSyncRequest

    client = _get_plaid_client()

    # Fetch accounts
    acct_response = client.accounts_get(AccountsGetRequest(access_token=access_token))
    accounts = acct_response["accounts"]

    # Fetch transactions via cursor-based sync (v44+)
    all_transactions = []
    cursor = None
    while True:
        kwargs = {"access_token": access_token}
        if cursor:
            kwargs["cursor"] = cursor
        txn_response = client.transactions_sync(TransactionsSyncRequest(**kwargs))
        all_transactions.extend(txn_response["added"])
        if not txn_response["has_more"]:
            break
        cursor = txn_response["next_cursor"]

    return {"accounts": accounts, "transactions": all_transactions}



def _upsert_plaid_data(db: Session, user_id: uuid.UUID, plaid_data: dict):
    """Upsert Plaid accounts and transactions into PostgreSQL."""
    type_map = {
        "depository": AccountType.DEPOSITORY,
        "credit": AccountType.CREDIT,
        "investment": AccountType.INVESTMENT,
        "loan": AccountType.LOAN,
    }

    for pa in plaid_data["accounts"]:
        existing = db.query(Account).filter(Account.plaid_account_id == pa["account_id"]).first()
        if existing:
            existing.balance = float(pa["balances"]["current"] or 0)
            existing.available_balance = float(pa["balances"].get("available") or 0)
            existing.last_synced_at = datetime.now(timezone.utc)
        else:
            acct = Account(
                user_id=user_id,
                plaid_account_id=pa["account_id"],
                name=pa["name"],
                balance=float(pa["balances"]["current"] or 0),
                available_balance=float(pa["balances"].get("available") or 0),
                type=type_map.get(pa.get("type", "depository"), AccountType.DEPOSITORY),
                subtype=pa.get("subtype"),
                last_synced_at=datetime.now(timezone.utc),
            )
            db.add(acct)

    db.flush()

    # Upsert transactions (immutable ledger: skip if already exists)
    for pt in plaid_data["transactions"]:
        existing = db.query(Transaction).filter(
            Transaction.plaid_transaction_id == pt["transaction_id"]
        ).first()
        if existing:
            continue

        acct = db.query(Account).filter(Account.plaid_account_id == pt["account_id"]).first()
        if not acct:
            continue

        txn_date = datetime.combine(pt["date"], datetime.min.time()).replace(tzinfo=timezone.utc)
        txn = Transaction(
            account_id=acct.id,
            plaid_transaction_id=pt["transaction_id"],
            amount=float(pt["amount"]),
            date=txn_date,
            raw_description=pt.get("name", "Unknown"),
            merchant_name=pt.get("merchant_name"),
            clean_category=pt.get("personal_finance_category", {}).get("primary") if pt.get("personal_finance_category") else None,
        )
        db.add(txn)

    db.commit()
