from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from mmx.domain.types import RolloutMode


@dataclass(frozen=True)
class Decision:
    """A persisted decision artifact for auditability and dashboarding."""

    decision_id: str
    run_id: str
    period_start: str
    period_end: str
    total_budget: float

    # Primary recommendation
    recommended_budget: Dict[str, float]
    expected_premium: float
    premium_std: float
    ci_low: float
    ci_high: float
    risk_adjusted_premium: float

    # Baseline scenario (Do Nothing)
    baseline_budget: Dict[str, float]
    baseline_expected_premium: float
    baseline_premium_std: float
    baseline_ci_low: float
    baseline_ci_high: float
    baseline_risk_adjusted_premium: float
    p_ai_better_vs_baseline: float

    # Governance
    rollout_mode: RolloutMode
    rollout_reason_codes: List[str]
    # Explainability
    top_reasons_by_channel: Dict[str, List[str]]

    # Lineage
    model_version: str
    policy_hash: str
    constraints_hash: str

    # Data quality
    data_coverage: Optional[float] = None

    experiment_id: Optional[str] = None

    # Reproducibility / audit (optional for backward compatibility)
    objective_mode: Optional[str] = None
    policy_lambda: Optional[float] = None
    policy_delta: Optional[float] = None
    n_days: Optional[int] = None
    warm_start_enabled: Optional[bool] = None
    warm_start_days: Optional[int] = None
    seed_main: Optional[int] = None
    seed_baseline: Optional[int] = None

    # Validation / integrity report (constraint satisfaction)
    validation_report: Optional[Dict[str, float]] = None