from __future__ import annotations

"""Simulation parameter schemas for backtest synthetic data.

These dataclasses define the *ground-truth* generative process used to create
raw events that are later aggregated into the mart (daily_channel_fact).

Notes
-----
- These parameters are *not* the SEM posterior parameters.
- They are used only by the simulator (scripts/generate_backtest_data.py).
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SpendToLeadsParams:
    """Spend -> Leads response parameters."""

    scale: float
    adstock_theta: float
    hill_alpha: float
    hill_k: float
    noise_sigma: float


@dataclass(frozen=True)
class FunnelParams:
    """Funnel conversion parameters."""

    p_attempt: float
    p_connect: float
    p_contract: float
    lag_kernel: Sequence[float]


@dataclass(frozen=True)
class PremiumParams:
    """Premium distribution parameters per contract."""

    per_contract_mean: float
    per_contract_logn_sigma: float


@dataclass(frozen=True)
class ChannelParams:
    """All parameters for a channel."""

    channel: str
    spend_to_leads: SpendToLeadsParams
    funnel: FunnelParams
    premium: PremiumParams
