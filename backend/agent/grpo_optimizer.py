"""
Aegis – GRPO (Group Relative Policy Optimization) Engine (Phase 4).

Generates G candidate allocation policies, evaluates them through
the deterministic harness, and computes relative advantages.

Architecture:
  User Intent (Pydantic) → GRPO generates G policies → Harness scores each
  → Best policy proposed → HITL gate → Human approves/rejects
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.environment.reward import (
    PolicyReward,
    compute_group_rewards,
    compute_relative_advantages,
)
from backend.core.config import settings


@dataclass
class GRPOResult:
    best_policy: dict[str, float]
    best_reward: PolicyReward
    all_rewards: list[PolicyReward]
    relative_advantages: list[float]
    generation: int
    converged: bool


def generate_candidate_policies(
    current_balance: float,
    account_ids: list[str],
    group_size: int,
    target_allocation: float,
    seed: int = 0,
) -> list[dict[str, float]]:
    """
    Generate G candidate allocation policies using random perturbation.

    Each policy distributes `target_allocation` across accounts
    with different weighting strategies (conservative → aggressive).
    """
    rng = np.random.default_rng(seed)
    policies: list[dict[str, float]] = []

    if not account_ids:
        return [{}] * group_size

    n = len(account_ids)
    base = target_allocation / max(n, 1)

    for g in range(group_size):
        # Generate Dirichlet-distributed weights (explores diverse allocations)
        alpha = rng.uniform(0.5, 5.0, size=n)
        weights = rng.dirichlet(alpha)
        amounts = {
            acc_id: round(float(weights[i] * target_allocation), 2)
            for i, acc_id in enumerate(account_ids)
        }
        policies.append(amounts)

    return policies


def run_grpo_optimization(
    account_ids: list[str],
    transactions: list[dict],
    current_balance: float,
    target_allocation: float,
    days_out: int = 30,
    epochs: int = None,
    group_size: int = None,
) -> GRPOResult:
    """
    Main GRPO optimization loop.

    Args:
        account_ids: Available accounts to allocate funds across.
        transactions: Historical transaction data for simulation.
        current_balance: Current total balance.
        target_allocation: Total USD amount to allocate.
        days_out: Simulation horizon.
        epochs: Training epochs (defaults to settings.GRPO_EPOCHS).
        group_size: Candidate policies per generation (defaults to settings.GRPO_GROUP_SIZE).

    Returns:
        GRPOResult with the best policy and all evaluation metrics.
    """
    epochs = epochs or settings.GRPO_EPOCHS
    group_size = group_size or settings.GRPO_GROUP_SIZE

    best_result: Optional[GRPOResult] = None
    prev_best_reward = -math.inf

    for epoch in range(epochs):
        # 1. Generate G candidate policies
        candidates = generate_candidate_policies(
            current_balance=current_balance,
            account_ids=account_ids,
            group_size=group_size,
            target_allocation=target_allocation,
            seed=epoch * 137,  # Deterministic but varied per epoch
        )

        # 2. Evaluate all candidates through deterministic harness
        rewards = compute_group_rewards(
            candidate_policies=candidates,
            transactions=transactions,
            current_balance=current_balance,
            days_out=days_out,
        )

        # 3. Compute relative advantages
        advantages = compute_relative_advantages(rewards)

        # 4. Select best policy
        best_reward = rewards[0]  # Already sorted desc
        converged = abs(best_reward.composite_reward - prev_best_reward) < 1e-4

        best_result = GRPOResult(
            best_policy=best_reward.allocations,
            best_reward=best_reward,
            all_rewards=rewards,
            relative_advantages=advantages,
            generation=epoch + 1,
            converged=converged,
        )

        prev_best_reward = best_reward.composite_reward
        if converged:
            break

    return best_result  # type: ignore
