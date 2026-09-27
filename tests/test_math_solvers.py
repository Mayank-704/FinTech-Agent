"""Tests for Aegis deterministic math solvers."""
import pytest
from backend.environment.simulator import (
    calculate_cash_flow,
    simulate_shock,
    simulate_transfer,
    calculate_compound_interest,
    monte_carlo_cashflow,
)


SAMPLE_TRANSACTIONS = [
    {"amount": -6500.0, "date": "2026-09-01", "is_recurring": True, "clean_category": "INCOME"},
    {"amount": 1850.0, "date": "2026-09-01", "is_recurring": True, "clean_category": "HOUSING"},
    {"amount": 420.0, "date": "2026-09-03", "is_recurring": False, "clean_category": "GROCERIES"},
    {"amount": 85.0, "date": "2026-09-04", "is_recurring": True, "clean_category": "ENTERTAINMENT"},
    {"amount": 245.0, "date": "2026-09-05", "is_recurring": True, "clean_category": "UTILITIES"},
]


def test_cash_flow_basic():
    result = calculate_cash_flow(SAMPLE_TRANSACTIONS, current_balance=10000, days_out=30)
    assert result.avg_daily_income > 0
    assert result.avg_daily_expense > 0
    assert len(result.trajectory) == 31
    assert result.trajectory[0]["balance"] == 10000.0


def test_cash_flow_empty():
    result = calculate_cash_flow([], current_balance=5000, days_out=30)
    assert result.projected_end_balance == 5000.0
    assert result.net_daily_flow == 0.0


def test_simulate_shock_income_drop():
    state = {"transactions": SAMPLE_TRANSACTIONS, "current_balance": 10000}
    result = simulate_shock(state, "INCOME_DROP", amount=2000, days_out=30)
    assert result.delta_end_balance < 0  # Shock must reduce projected balance
    assert 0.0 <= result.insolvency_risk_increase <= 1.0


def test_simulate_shock_job_loss():
    state = {"transactions": SAMPLE_TRANSACTIONS, "current_balance": 10000}
    result = simulate_shock(state, "JOB_LOSS", amount=0, days_out=30)
    # Job loss should severely impact cash flow
    assert result.shocked.avg_daily_income <= result.baseline.avg_daily_income


def test_simulate_transfer_feasible():
    result = simulate_transfer(
        from_balance=5000,
        to_balance=1000,
        amount=2000,
        transactions=SAMPLE_TRANSACTIONS,
    )
    assert result.is_feasible is True
    assert result.from_account_new_balance == 3000.0
    assert result.to_account_new_balance == 3000.0
    assert result.shortfall == 0.0


def test_simulate_transfer_infeasible():
    result = simulate_transfer(
        from_balance=500,
        to_balance=1000,
        amount=2000,
        transactions=[],
    )
    assert result.is_feasible is False
    assert result.shortfall == 1500.0


def test_compound_interest():
    result = calculate_compound_interest(principal=10000, annual_rate=0.05, years=1)
    assert result["final_balance"] > 10000
    assert result["total_interest_earned"] == pytest.approx(result["final_balance"] - 10000, rel=1e-4)
    assert len(result["schedule"]) == 12


def test_monte_carlo():
    result = monte_carlo_cashflow(SAMPLE_TRANSACTIONS, current_balance=10000, simulations=200)
    assert "p10" in result
    assert "p50" in result
    assert "p90" in result
    assert 0.0 <= result["insolvency_probability"] <= 1.0
    assert result["p10"] <= result["p50"] <= result["p90"]
