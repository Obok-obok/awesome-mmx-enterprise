from __future__ import annotations

"""Backtest result schemas.

These objects are used for in-memory transport. On disk, artifacts are stored as
JSON/CSV under artifacts/backtests/{run_id}/.
"""

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass(frozen=True)
class OverallMetrics:
    premium_wape: float
    premium_mae: float
    premium_rmse: float
    coverage_p10_p90: float
    coverage_p05_p95: float
    pred_ra_mean: float
    pred_ra_std: float
    pred_ra_value: float


@dataclass(frozen=True)
class BacktestResult:
    run_id: str
    config: Dict[str, Any]
    splits: Dict[str, Any]
    overall: OverallMetrics
    timeseries_daily: pd.DataFrame
    metrics_by_channel: pd.DataFrame
    plan_compare_by_channel: pd.DataFrame
    plan_compare_totals: Dict[str, Any]
    plan_compare_monthly_by_channel: pd.DataFrame
    plan_compare_monthly_totals: Dict[str, Any]
    period_summary: pd.DataFrame
    lineage: Dict[str, Any]
