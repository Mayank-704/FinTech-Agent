"""
Aegis – Monte Carlo Economic Shock Generator (Phase 3).

Generates structured economic shock scenarios for stress-testing
user portfolios. Used by GRPO reward harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ShockType = Literal["INCOME_DROP", "EXPENSE_SPIKE", "MARKET_CRASH", "JOB_LOSS", "MEDICAL_EMERGENCY"]


@dataclass
class EconomicShock:
    shock_type: ShockType
    severity: float           # 0.0 (mild) → 1.0 (catastrophic)
    amount: float             # USD impact magnitude
    probability: float        # 0.0 – 1.0 historical probability
    description: str


# Historical shock probabilities (annualized, US consumer data)
SHOCK_PROFILES: dict[ShockType, dict] = {
    "INCOME_DROP": {
        "probability": 0.12,
        "severity_range": (0.15, 0.45),
        "description": "Reduction in employment income",
    },
    "EXPENSE_SPIKE": {
        "probability": 0.20,
        "severity_range": (0.05, 0.25),
        "description": "Unexpected large expense",
    },
    "MARKET_CRASH": {
        "probability": 0.07,
        "severity_range": (0.20, 0.50),
        "description": "Portfolio/investment value decline",
    },
    "JOB_LOSS": {
        "probability": 0.04,
        "severity_range": (0.80, 1.00),
        "description": "Complete loss of primary income source",
    },
    "MEDICAL_EMERGENCY": {
        "probability": 0.08,
        "severity_range": (0.10, 0.40),
        "description": "Unexpected medical or dental expense",
    },
}


def generate_shock_scenarios(
    current_income: float,
    current_balance: float,
    n_scenarios: int = 20,
    seed: int = 42,
) -> list[EconomicShock]:
    """
    Generate N random shock scenarios weighted by historical probabilities.

    Args:
        current_income: Monthly income in USD.
        current_balance: Current account balance in USD.
        n_scenarios: Number of scenarios to generate.
        seed: RNG seed for reproducibility.

    Returns:
        List of EconomicShock objects sorted by severity desc.
    """
    rng = np.random.default_rng(seed)
    shocks: list[EconomicShock] = []

    shock_types = list(SHOCK_PROFILES.keys())
    weights = [SHOCK_PROFILES[s]["probability"] for s in shock_types]
    weight_arr = np.array(weights)
    weight_arr = weight_arr / weight_arr.sum()

    for _ in range(n_scenarios):
        shock_type: ShockType = rng.choice(shock_types, p=weight_arr)  # type: ignore
        profile = SHOCK_PROFILES[shock_type]
        lo, hi = profile["severity_range"]
        severity = float(rng.uniform(lo, hi))

        # Compute USD impact
        if shock_type in ("INCOME_DROP", "JOB_LOSS"):
            amount = current_income * severity
        elif shock_type == "MARKET_CRASH":
            amount = current_balance * severity  # Treat as % of balance
        else:
            amount = current_income * severity * 2  # Spike = 2x income fraction

        shocks.append(EconomicShock(
            shock_type=shock_type,
            severity=round(severity, 4),
            amount=round(amount, 2),
            probability=profile["probability"],
            description=profile["description"],
        ))

    shocks.sort(key=lambda s: s.severity, reverse=True)
    return shocks


def worst_case_shock(
    current_income: float,
    current_balance: float,
) -> EconomicShock:
    """Return the single worst-case shock for a given financial state."""
    scenarios = generate_shock_scenarios(current_income, current_balance, n_scenarios=100)
    return scenarios[0]
