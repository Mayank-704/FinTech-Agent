"""
Aegis – FastAPI Route: CRUD endpoints (Phase 1).

/api/users     GET, POST
/api/accounts  GET, POST
/api/transactions GET, POST
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from backend.database.models import (
    Account, AccountType, ActionQueue, ActionStatus,
    Transaction, User, get_engine,
)
from backend.core.config import settings

router = APIRouter(prefix="/api", tags=["crud"])


# ─── DB Dependency ───────────────────────────────────────────────────────────

def get_db():
    from sqlalchemy.orm import sessionmaker
    engine = get_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Schemas (Request / Response) ────────────────────────────────────────────

class UserCreate(BaseModel):
    name: str
    email: str
    baseline_risk_tolerance: float = 0.5

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    baseline_risk_tolerance: float
    created_at: datetime

    class Config:
        from_attributes = True

class AccountCreate(BaseModel):
    user_id: str
    name: str
    balance: float = 0.0
    type: str = "depository"

class TransactionCreate(BaseModel):
    account_id: str
    amount: float
    date: datetime
    raw_description: str
    clean_category: Optional[str] = None
    is_recurring: bool = False


# ─── Users ───────────────────────────────────────────────────────────────────

@router.post("/users", summary="Create a new user")
def create_user(body: UserCreate, db: Session = Depends(get_db)):
    user = User(
        name=body.name,
        email=body.email,
        baseline_risk_tolerance=body.baseline_risk_tolerance,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": str(user.id), "name": user.name, "email": user.email}


@router.get("/users/{user_id}", summary="Get user by ID")
def get_user(user_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    return {"id": str(user.id), "name": user.name, "email": user.email,
            "baseline_risk_tolerance": user.baseline_risk_tolerance}


# ─── Accounts ────────────────────────────────────────────────────────────────

@router.post("/accounts", summary="Create account")
def create_account(body: AccountCreate, db: Session = Depends(get_db)):
    acct = Account(
        user_id=body.user_id,
        name=body.name,
        balance=body.balance,
        type=AccountType(body.type),
    )
    db.add(acct)
    db.commit()
    db.refresh(acct)
    return {"id": str(acct.id), "name": acct.name, "balance": acct.balance}


@router.get("/accounts", summary="List accounts for a user")
def list_accounts(user_id: str = Query(...), db: Session = Depends(get_db)):
    accounts = db.query(Account).filter(Account.user_id == user_id).all()
    return [
        {
            "id": str(a.id),
            "name": a.name,
            "balance": a.balance,
            "available_balance": a.available_balance,
            "type": a.type.value,
            "last_synced_at": a.last_synced_at,
        }
        for a in accounts
    ]


# ─── Transactions ─────────────────────────────────────────────────────────────

@router.post("/transactions", summary="Ingest a transaction")
def create_transaction(body: TransactionCreate, db: Session = Depends(get_db)):
    txn = Transaction(
        account_id=body.account_id,
        amount=body.amount,
        date=body.date,
        raw_description=body.raw_description,
        clean_category=body.clean_category,
        is_recurring=body.is_recurring,
    )
    db.add(txn)
    db.commit()
    db.refresh(txn)
    return {"id": str(txn.id), "amount": txn.amount}


@router.get("/transactions", summary="List transactions for an account")
def list_transactions(
    account_id: str = Query(...),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
):
    txns = (
        db.query(Transaction)
        .filter(Transaction.account_id == account_id)
        .order_by(Transaction.date.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": str(t.id),
            "amount": t.amount,
            "date": t.date,
            "raw_description": t.raw_description,
            "clean_category": t.clean_category,
            "is_recurring": t.is_recurring,
        }
        for t in txns
    ]


# ─── Action Queue ─────────────────────────────────────────────────────────────

@router.get("/action-queue", summary="List PENDING actions for HITL review")
def list_pending_actions(user_id: str = Query(...), db: Session = Depends(get_db)):
    actions = (
        db.query(ActionQueue)
        .filter(
            ActionQueue.user_id == user_id,
            ActionQueue.status == ActionStatus.PENDING,
        )
        .order_by(ActionQueue.created_at.desc())
        .all()
    )
    return [
        {
            "id": str(a.id),
            "proposed_action_type": a.proposed_action_type.value,
            "payload": a.payload,
            "simulation_result": a.simulation_result,
            "status": a.status.value,
            "created_at": a.created_at,
        }
        for a in actions
    ]


@router.get("/dashboard", summary="Aggregate dashboard data for a user")
def get_dashboard(user_id: str = Query(...), db: Session = Depends(get_db)):
    """Returns all accounts + recent transactions + pending HITL actions."""
    try:
        accounts = db.query(Account).filter(Account.user_id == user_id).all()
        acct_ids = [a.id for a in accounts]

        transactions = []
        for aid in acct_ids:
            txns = (
                db.query(Transaction)
                .filter(Transaction.account_id == aid)
                .order_by(Transaction.date.desc())
                .limit(20)
                .all()
            )
            transactions.extend(txns)

        pending = (
            db.query(ActionQueue)
            .filter(ActionQueue.user_id == user_id, ActionQueue.status == ActionStatus.PENDING)
            .order_by(ActionQueue.created_at.desc())
            .limit(10)
            .all()
        )

        total_balance = sum(a.balance for a in accounts if a.type == AccountType.DEPOSITORY)

        return {
            "status": "ok",
            "degraded_mode": False,
            "total_balance": total_balance,
            "accounts": [
                {"id": str(a.id), "name": a.name, "balance": a.balance, "type": a.type.value}
                for a in accounts
            ],
            "recent_transactions": [
                {
                    "id": str(t.id),
                    "amount": t.amount,
                    "date": str(t.date),
                    "description": t.raw_description,
                    "category": t.clean_category,
                    "is_recurring": t.is_recurring,
                }
                for t in sorted(transactions, key=lambda x: x.date, reverse=True)[:30]
            ],
            "pending_actions": [
                {
                    "id": str(a.id),
                    "type": a.proposed_action_type.value,
                    "payload": a.payload,
                    "simulation_result": a.simulation_result,
                    "created_at": str(a.created_at),
                }
                for a in pending
            ],
        }
    except Exception as exc:
        # Circuit breaker: return degraded state
        return {
            "status": "degraded",
            "degraded_mode": True,
            "error": str(exc),
            "total_balance": 0,
            "accounts": [],
            "recent_transactions": [],
            "pending_actions": [],
        }
