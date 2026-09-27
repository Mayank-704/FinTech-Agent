"""
Aegis – GRPO Verifiable Reward Function (Phase 4 / GRPO).

Computes reward signals for candidate allocation policies generated
by the GRPO optimizer. Rewards are fully deterministic (no LLM).

Reward = α·liquidity_score - β·insolvency_risk - γ·volatility_penalty
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.environment.simulator import calculate_cash_flow, monte_carlo_cashflow


@dataclass
class PolicyReward:
    policy_id: int
    allocations: dict[str, float]       # {account_id: allocated_amount}
    liquidity_score: float              # 0.0 – 1.0
    insolvency_risk: float              # 0.0 – 1.0
    volatility_penalty: float           # 0.0 – 1.0
    composite_reward: float             # Final scalar reward
    monte_carlo_p50: float              # Median projected balance
    is_feasible: bool


# Reward weights (tunable via config)
ALPHA = 0.5   # Liquidity weight
BETA = 0.3    # Insolvency risk weight
GAMMA = 0.2   # Volatility penalty weight


def compute_reward(
    policy_id: int,
    allocations: dict[str, float],
    transactions: list[dict],
    current_balance: float,
    days_out: int = 30,
    simulations: int = 500,
) -> PolicyReward:
    """
    Compute a verifiable reward for a single allocation policy.

    Args:
        policy_id: Index of this candidate in the GRPO group.
        allocations: Proposed allocation {account_id: amount}.
        transactions: Historical transaction list.
        current_balance: Starting balance.
        days_out: Forecast horizon.
        simulations: Monte Carlo sample count.

    Returns:
        PolicyReward with composite scalar reward.
    """
    total_allocated = sum(allocations.values())
    effective_balance = max(0.0, current_balance - total_allocated)
    is_feasible = effective_balance >= 0

    mc = monte_carlo_cashflow(transactions, effective_balance, days_out, simulations)

    # Liquidity score: probability of staying positive
    liquidity_score = 1.0 - mc["insolvency_probability"]

    # Insolvency risk from simulation
    insolvency_risk = mc["insolvency_probability"]

    # Volatility penalty: coefficient of variation of end balances
    mean = mc["mean"]
    std = mc["std"]
    cv = (std / max(abs(mean), 1.0))
    volatility_penalty = float(np.clip(cv, 0.0, 1.0))

    # Composite reward (higher = better)
    reward = (
        ALPHA * liquidity_score
        - BETA * insolvency_risk
        - GAMMA * volatility_penalty
    )
    if not is_feasible:
        reward -= 1.0  # Hard penalty for infeasible allocations

    return PolicyReward(
        policy_id=policy_id,
        allocations=allocations,
        liquidity_score=round(liquidity_score, 4),
        insolvency_risk=round(insolvency_risk, 4),
        volatility_penalty=round(volatility_penalty, 4),
        composite_reward=round(reward, 6),
        monte_carlo_p50=mc["p50"],
        is_feasible=is_feasible,
    )


def compute_group_rewards(
    candidate_policies: list[dict[str, float]],
    transactions: list[dict],
    current_balance: float,
    days_out: int = 30,
) -> list[PolicyReward]:
    """
    Evaluate all G candidate policies and return sorted rewards.
    Used by grpo_optimizer.py to compute relative advantages.
    """
    rewards = [
        compute_reward(
            policy_id=i,
            allocations=policy,
            transactions=transactions,
            current_balance=current_balance,
            days_out=days_out,
        )
        for i, policy in enumerate(candidate_policies)
    ]
    rewards.sort(key=lambda r: r.composite_reward, reverse=True)
    return rewards


def compute_relative_advantages(rewards: list[PolicyReward]) -> list[float]:
    """
    Compute GRPO relative advantages: A_i = (r_i - mean(r)) / std(r).
    Returns normalized advantage per policy (same order as input).
    """
    scores = np.array([r.composite_reward for r in rewards])
    mean = scores.mean()
    std = scores.std() + 1e-8  # Avoid division by zero
    return [(float((s - mean) / std)) for s in scores]
