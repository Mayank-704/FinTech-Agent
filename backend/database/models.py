"""
Aegis – SQLAlchemy ORM models (Phase 1).

Immutable ledger design: transactions are APPEND-ONLY, never updated.
The ActionQueue holds HITL pending actions that require human approval.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, Session


# ─── Base ────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─── Enums ───────────────────────────────────────────────────────────────────

class AccountType(str, enum.Enum):
    DEPOSITORY = "depository"
    CREDIT = "credit"
    INVESTMENT = "investment"
    LOAN = "loan"


class ActionStatus(str, enum.Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class ActionType(str, enum.Enum):
    CATEGORIZE = "CATEGORIZE"
    SIMULATE_SHOCK = "SIMULATE_SHOCK"
    PROPOSE_TRANSFER = "PROPOSE_TRANSFER"
    REALLOCATE = "REALLOCATE"


# ─── Models ──────────────────────────────────────────────────────────────────

def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    baseline_risk_tolerance = Column(Float, default=0.5)  # 0.0 (conservative) → 1.0 (aggressive)
    created_at = Column(DateTime, default=utcnow)

    accounts = relationship("Account", back_populates="user", cascade="all, delete-orphan")
    action_queue = relationship("ActionQueue", back_populates="user", cascade="all, delete-orphan")


class Account(Base):
    __tablename__ = "accounts"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    plaid_account_id = Column(String(255), unique=True, nullable=True)
    name = Column(String(255), nullable=False)
    balance = Column(Float, default=0.0)
    available_balance = Column(Float, nullable=True)
    type = Column(Enum(AccountType), default=AccountType.DEPOSITORY)
    subtype = Column(String(80), nullable=True)
    currency_code = Column(String(10), default="USD")
    last_synced_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="accounts")
    transactions = relationship("Transaction", back_populates="account", cascade="all, delete-orphan")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String(36), primary_key=True, default=_uuid)
    account_id = Column(String(36), ForeignKey("accounts.id"), nullable=False)
    plaid_transaction_id = Column(String(255), unique=True, nullable=True)
    amount = Column(Float, nullable=False)           # Positive = debit, Negative = credit
    date = Column(DateTime, nullable=False)
    raw_description = Column(Text, nullable=False)
    clean_category = Column(String(120), nullable=True)
    merchant_name = Column(String(255), nullable=True)
    is_recurring = Column(Boolean, default=False)
    confidence_score = Column(Float, nullable=True)  # AI categorization confidence
    created_at = Column(DateTime, default=utcnow)

    account = relationship("Account", back_populates="transactions")


class ActionQueue(Base):
    """
    HITL (Human-In-The-Loop) action queue.
    Every proposed state-changing action is stored here and PAUSED
    until a human approves or rejects it via the /api/agent/resume endpoint.
    """
    __tablename__ = "action_queue"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    langgraph_thread_id = Column(String(255), nullable=True)  # For resuming the graph
    proposed_action_type = Column(Enum(ActionType), nullable=False)
    payload = Column(JSON, nullable=False)            # Full action payload (Pydantic JSON)
    simulation_result = Column(JSON, nullable=True)  # Math solver output for display
    status = Column(Enum(ActionStatus), default=ActionStatus.PENDING)
    decided_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="action_queue")


# ─── DB Setup ────────────────────────────────────────────────────────────────

def get_engine(url: str):
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, pool_pre_ping=True, echo=False, **kwargs)


def init_db(url: str):
    """Create all tables (idempotent)."""
    engine = get_engine(url)
    Base.metadata.create_all(engine)
    return engine

