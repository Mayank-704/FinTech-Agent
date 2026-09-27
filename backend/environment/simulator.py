"""
Aegis – Deterministic Math Solvers (Phase 3).

CRITICAL RULE: No LLM computations here. All arithmetic via NumPy/Pandas/stdlib.
These functions are the single source of truth for financial math.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import numpy as np
import pandas as pd


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class CashFlowResult:
    projected_end_balance: float
    avg_daily_income: float
    avg_daily_expense: float
    net_daily_flow: float
    runway_days: Optional[int]          # Days until balance < 0 (None if positive)
    trajectory: list[dict]              # [{date, balance}] for charting
    degraded_mode: bool = False

@dataclass
class ShockResult:
    baseline: CashFlowResult
    shocked: CashFlowResult
    delta_end_balance: float
    insolvency_risk_increase: float     # 0.0 – 1.0
    recommended_action: str


@dataclass
class TransferSimulationResult:
    from_account_new_balance: float
    to_account_new_balance: float
    cashflow_impact: CashFlowResult
    is_feasible: bool
    shortfall: float                    # 0 if feasible


@dataclass
class PortfolioAllocation:
    allocations: dict[str, float]       # {account_id: amount}
    expected_monthly_cashflow: float
    risk_score: float                   # 0.0 – 1.0
    sharpe_ratio: float


# ─── Core Calculators ────────────────────────────────────────────────────────

def calculate_cash_flow(
    transactions: list[dict],
    current_balance: float,
    days_out: int = 30,
) -> CashFlowResult:
    """
    Project end-of-period balance from transaction history.

    Args:
        transactions: List of {amount, date, is_recurring, clean_category}
                      Positive amount = expense, negative = income (Plaid convention).
        current_balance: Current account balance in USD.
        days_out: Forecast horizon in days.

    Returns:
        CashFlowResult with trajectory for Ghost Ledger chart.
    """
    if not transactions:
        trajectory = _flat_trajectory(current_balance, days_out)
        return CashFlowResult(
            projected_end_balance=current_balance,
            avg_daily_income=0.0,
            avg_daily_expense=0.0,
            net_daily_flow=0.0,
            runway_days=None,
            trajectory=trajectory,
        )

    df = pd.DataFrame(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)

    # Separate income (negative amounts in Plaid = money coming in) from expenses
    income_df = df[df["amount"] < 0].copy()
    expense_df = df[df["amount"] > 0].copy()

    # Compute daily averages over the history window
    n_days = max((df["date"].max() - df["date"].min()).days, 1)
    avg_daily_income = float(abs(income_df["amount"].sum()) / n_days)
    avg_daily_expense = float(expense_df["amount"].sum() / n_days)
    net_daily_flow = avg_daily_income - avg_daily_expense

    # Build trajectory
    trajectory = []
    balance = current_balance
    today = datetime.now(timezone.utc).date()

    for day in range(days_out + 1):
        date = today + timedelta(days=day)
        trajectory.append({"date": date.isoformat(), "balance": round(balance, 2)})
        balance += net_daily_flow  # Linear projection

    projected_end = trajectory[-1]["balance"]

    # Runway calculation
    runway_days: Optional[int] = None
    if net_daily_flow < 0 and current_balance > 0:
        runway_days = int(current_balance / abs(net_daily_flow))

    return CashFlowResult(
        projected_end_balance=round(projected_end, 2),
        avg_daily_income=round(avg_daily_income, 4),
        avg_daily_expense=round(avg_daily_expense, 4),
        net_daily_flow=round(net_daily_flow, 4),
        runway_days=runway_days,
        trajectory=trajectory,
    )


def simulate_shock(
    current_state: dict,
    shock_type: Literal["INCOME_DROP", "EXPENSE_SPIKE", "MARKET_CRASH", "JOB_LOSS"],
    amount: float,
    days_out: int = 30,
) -> ShockResult:
    """
    Apply an economic shock to the current cash flow state and measure impact.

    Args:
        current_state: {transactions, current_balance, days_out}
        shock_type: Type of shock to apply.
        amount: Magnitude of shock (USD for dollar shocks, % for MARKET_CRASH).
        days_out: Simulation horizon.
    """
    transactions = current_state.get("transactions", [])
    current_balance = float(current_state.get("current_balance", 0))

    baseline = calculate_cash_flow(transactions, current_balance, days_out)

    # Apply shock to transaction stream
    shocked_transactions = _apply_shock(transactions, shock_type, amount)
    shocked_balance_start = current_balance
    if shock_type == "MARKET_CRASH":
        shocked_balance_start = current_balance * (1 - amount / 100)

    shocked = calculate_cash_flow(shocked_transactions, shocked_balance_start, days_out)

    delta = shocked.projected_end_balance - baseline.projected_end_balance

    # Insolvency risk: probability end balance < 0 in shocked scenario
    insolvency_risk = float(np.clip(
        1.0 - (shocked.projected_end_balance / max(abs(baseline.projected_end_balance), 1)),
        0.0, 1.0
    ))

    recommendation = _generate_recommendation(shock_type, insolvency_risk, shocked)

    return ShockResult(
        baseline=baseline,
        shocked=shocked,
        delta_end_balance=round(delta, 2),
        insolvency_risk_increase=round(insolvency_risk, 4),
        recommended_action=recommendation,
    )


def simulate_transfer(
    from_balance: float,
    to_balance: float,
    amount: float,
    transactions: list[dict],
    days_out: int = 30,
) -> TransferSimulationResult:
    """Simulate a fund transfer between two accounts and project cash flow impact."""
    is_feasible = amount <= from_balance
    shortfall = max(0.0, amount - from_balance)

    new_from = from_balance - amount if is_feasible else from_balance
    new_to = to_balance + amount if is_feasible else to_balance

    cashflow = calculate_cash_flow(transactions, new_from, days_out)

    return TransferSimulationResult(
        from_account_new_balance=round(new_from, 2),
        to_account_new_balance=round(new_to, 2),
        cashflow_impact=cashflow,
        is_feasible=is_feasible,
        shortfall=round(shortfall, 2),
    )


def calculate_compound_interest(
    principal: float,
    annual_rate: float,
    periods_per_year: int = 12,
    years: float = 1.0,
) -> dict:
    """
    Calculate compound interest trajectory.

    Returns detailed monthly schedule for Ghost Ledger visualization.
    """
    n = periods_per_year
    r = annual_rate / n
    total_periods = int(years * n)

    schedule = []
    balance = principal
    for period in range(1, total_periods + 1):
        interest = balance * r
        balance += interest
        schedule.append({
            "period": period,
            "interest": round(interest, 4),
            "balance": round(balance, 2),
        })

    return {
        "principal": principal,
        "annual_rate": annual_rate,
        "final_balance": round(balance, 2),
        "total_interest_earned": round(balance - principal, 2),
        "schedule": schedule,
    }


# ─── Monte Carlo ─────────────────────────────────────────────────────────────

def monte_carlo_cashflow(
    transactions: list[dict],
    current_balance: float,
    days_out: int = 30,
    simulations: int = 1000,
    volatility_factor: float = 0.15,
) -> dict:
    """
    Run N Monte Carlo simulations of future cash flow with random noise.

    Used by the GRPO harness for reward computation.
    """
    if not transactions:
        return {"p10": current_balance, "p50": current_balance, "p90": current_balance, "trajectories": []}

    base = calculate_cash_flow(transactions, current_balance, days_out)
    net_flow = base.net_daily_flow

    rng = np.random.default_rng(42)
    end_balances = []

    for _ in range(simulations):
        daily_noise = rng.normal(0, abs(net_flow) * volatility_factor, days_out)
        sim_flow = net_flow + daily_noise
        end = current_balance + float(sim_flow.sum())
        end_balances.append(end)

    arr = np.array(end_balances)
    return {
        "p10": round(float(np.percentile(arr, 10)), 2),
        "p25": round(float(np.percentile(arr, 25)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p75": round(float(np.percentile(arr, 75)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "insolvency_probability": round(float(np.mean(arr < 0)), 4),
        "mean": round(float(arr.mean()), 2),
        "std": round(float(arr.std()), 2),
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _flat_trajectory(balance: float, days: int) -> list[dict]:
    today = datetime.now(timezone.utc).date()
    return [{"date": (today + timedelta(days=i)).isoformat(), "balance": round(balance, 2)} for i in range(days + 1)]


def _apply_shock(transactions: list[dict], shock_type: str, amount: float) -> list[dict]:
    """Mutate a copy of transactions to reflect the shock."""
    import copy
    txns = copy.deepcopy(transactions)

    if shock_type == "INCOME_DROP":
        # Reduce all negative (income) transactions by `amount` USD
        for t in txns:
            if t["amount"] < 0:
                t["amount"] = t["amount"] + amount  # Less negative = less income

    elif shock_type == "EXPENSE_SPIKE":
        # Add a recurring extra expense of `amount` per period
        if txns:
            from_date = txns[0]["date"]
            txns.append({
                "amount": amount,
                "date": from_date,
                "is_recurring": True,
                "clean_category": "SHOCK_EXPENSE",
            })

    elif shock_type == "JOB_LOSS":
        # Zero out all income
        for t in txns:
            if t["amount"] < 0:
                t["amount"] = 0.0

    # MARKET_CRASH handled via balance adjustment in simulate_shock
    return txns


def _generate_recommendation(shock_type: str, insolvency_risk: float, shocked: CashFlowResult) -> str:
    if insolvency_risk > 0.7:
        return "CRITICAL: Liquidate non-essential subscriptions immediately. Activate emergency fund."
    elif insolvency_risk > 0.4:
        return f"HIGH RISK after {shock_type}: Reduce discretionary spending by 30%. Consider bridge income."
    elif insolvency_risk > 0.2:
        return f"MODERATE RISK: Build 3-month emergency buffer within {shocked.runway_days or 90} days."
    else:
        return "LOW RISK: Current financial resilience is adequate. Maintain savings rate."
