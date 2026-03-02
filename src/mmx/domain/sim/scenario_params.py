from __future__ import annotations

"""Scenario-level parameters for synthetic backtest data."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SeasonalityParams:
    yearly_amp: float
    weekly_amp: float


@dataclass(frozen=True)
class AllocationRuleParams:
    """Human-like spend allocation rule used to generate historical spend."""

    type: str
    trailing_days: int
    smoothing_gamma: float
    min_share: float
    max_share: float


@dataclass(frozen=True)
class ScenarioParams:
    base_daily_budget: float
    seasonality: SeasonalityParams
    allocation_rule: AllocationRuleParams
