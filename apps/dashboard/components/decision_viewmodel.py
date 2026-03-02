from __future__ import annotations

"""Dashboard ViewModel utilities.

The dashboard should never render raw artifacts directly.
Instead, we convert artifacts (decision.json, mart, optional human plan)
into a UI-friendly ViewModel with sane defaults.

Key guarantees:
- "Do Nothing" budget is never all zeros (fallback from mart).
- Plan comparison metrics are always present (missing -> 0 with flags).
- Channel-level insight carries both numbers and explanation bullets.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import json

import pandas as pd


@dataclass(frozen=True)
class PlanSummary:
    """Plan-level summary for executive comparison."""

    budget_by_channel: dict[str, float]
    expected_premium: float
    q10_premium: float | None
    ra_premium: float
    p_win: float


@dataclass(frozen=True)
class ChannelInsight:
    """Channel-level insight for recommendation/explainability."""

    channel: str
    spend_ai: float
    spend_do_nothing: float
    delta_spend: float
    mroi: float | None
    saturation_ratio: float | None
    ec50: float | None
    half_life: float | None
    downside_risk: float | None
    reason_bullets: list[str]
    recommended_action: str


@dataclass(frozen=True)
class DecisionViewModel:
    """Normalized decision payload for UI."""

    decision_id: str
    period_start: str
    period_end: str
    rollout_mode: str
    plans: dict[str, PlanSummary]
    channels: list[ChannelInsight]


def latest_decision_path(decision_dir: Path) -> Path | None:
    """Return the latest dec_*.json path."""
    if not decision_dir.exists():
        return None
    files = sorted(decision_dir.glob("dec_*.json"))
    return files[-1] if files else None


def latest_human_plan_path(artifacts_root: Path) -> Path | None:
    """Best-effort lookup for the latest human plan artifact.

    The ingestion API may store human plans under different folders depending on mode.
    We search a small set of known locations to avoid expensive recursive scans.

    Returns:
        Path to a JSON file if found, else None.
    """
    candidates: list[Path] = []
    known_dirs = [
        artifacts_root / "plans" / "human",
        artifacts_root / "recommendations" / "human_plans",
        artifacts_root / "ingestion" / "human_plans",
        artifacts_root / "human_plans",
    ]
    patterns = ["*.json", "human*.json", "*human*.json", "plan*.json"]
    for d in known_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            candidates.extend(sorted(d.glob(pat)))
    if not candidates:
        return None
    # Prefer newest by filename sort (timestamps embedded) then mtime
    candidates = sorted(candidates, key=lambda p: (p.name, p.stat().st_mtime))
    return candidates[-1]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _parse_iso_date(s: Any) -> date | None:
    try:
        if not s:
            return None
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def _fallback_do_nothing_budget(
    mart: pd.DataFrame,
    *,
    total_budget: float,
    period_start: str,
    days_fallback: int = 28,
) -> dict[str, float]:
    """Compute a reproducible 'Do Nothing' budget from recent mart spend.

    Policy:
    - Use the most recent `days_fallback` days *before* period_start.
    - If period_start is missing/unparseable, use the last N days in mart.
    - Compute average daily spend per channel.
    - Scale to match total_budget (so AI vs DN is comparable).
    """
    if mart.empty:
        return {}
    df = mart.copy()
    required = {"date", "channel", "spend"}
    if not required.issubset(df.columns):
        return {}
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])  # type: ignore[arg-type]

    ps = _parse_iso_date(period_start)
    if ps is not None:
        end_dt = pd.Timestamp(ps)
        start_dt = end_dt - pd.Timedelta(days=days_fallback)
        hist = df[(df["date"] < end_dt) & (df["date"] >= start_dt)]
    else:
        max_dt = df["date"].max()
        start_dt = max_dt - pd.Timedelta(days=days_fallback)
        hist = df[df["date"] >= start_dt]

    if hist.empty:
        hist = df

    daily = hist.groupby([hist["date"].dt.date, "channel"], as_index=False)["spend"].sum()
    by_ch = daily.groupby("channel", as_index=False)["spend"].mean()
    budgets = {str(r["channel"]): float(r["spend"]) for _, r in by_ch.iterrows()}

    s = sum(budgets.values())
    if s <= 0:
        return {k: 0.0 for k in budgets}

    scale = float(total_budget) / float(s) if total_budget > 0 else 1.0
    return {k: float(v) * scale for k, v in budgets.items()}


def build_decision_viewmodel(
    *,
    mart: pd.DataFrame,
    decision_path: Path,
    human_plan_path: Path | None = None,
) -> DecisionViewModel:
    """Build a DecisionViewModel from artifacts."""
    dec = json.loads(decision_path.read_text(encoding="utf-8"))

    decision_id = str(dec.get("decision_id", "-"))
    period_start = str(dec.get("period_start", ""))
    period_end = str(dec.get("period_end", ""))
    rollout_mode = str(dec.get("rollout_mode", "-"))

    ai_budget: dict[str, float] = {
        str(k): _safe_float(v) for k, v in (dec.get("recommended_budget", {}) or {}).items()
    }
    total_budget = float(sum(ai_budget.values()))

    dn_raw: Mapping[str, Any] = dec.get("do_nothing_budget", {}) or {}
    dn_budget: dict[str, float] = {str(k): _safe_float(v) for k, v in dn_raw.items()}
    if (not dn_budget) or (sum(dn_budget.values()) <= 0.0):
        dn_budget = _fallback_do_nothing_budget(mart, total_budget=total_budget, period_start=period_start)

    human_budget: dict[str, float] = {}
    if human_plan_path is not None and human_plan_path.exists():
        try:
            hp = json.loads(human_plan_path.read_text(encoding="utf-8"))
            human_budget = {str(k): _safe_float(v) for k, v in (hp.get("budget", {}) or {}).items()}
        except Exception:
            human_budget = {}

    # Plan-level metrics
    ai_ra = _safe_float(dec.get("risk_adjusted_premium", 0.0))
    dn_ra = _safe_float(dec.get("baseline_risk_adjusted_premium", 0.0))
    p_win = _safe_float(dec.get("p_ai_better_vs_baseline", 0.0))

    # Optional extras (may be absent).
    # For q10 we prefer None over 0 when the key is absent.
    ai_exp = _safe_float(dec.get("expected_premium", dec.get("premium_mean", 0.0)))
    ai_q10 = _safe_float(dec.get("premium_q10")) if ("premium_q10" in dec) else None
    dn_exp = _safe_float(dec.get("baseline_expected_premium", 0.0))
    dn_q10 = _safe_float(dec.get("baseline_premium_q10")) if ("baseline_premium_q10" in dec) else None

    plans: dict[str, PlanSummary] = {
        "ai": PlanSummary(
            budget_by_channel=ai_budget,
            expected_premium=ai_exp,
            q10_premium=ai_q10,
            ra_premium=ai_ra,
            p_win=p_win,
        ),
        "do_nothing": PlanSummary(
            budget_by_channel=dn_budget,
            expected_premium=dn_exp,
            q10_premium=dn_q10,
            ra_premium=dn_ra,
            p_win=1.0 - p_win,
        ),
    }
    if human_budget:
        plans["human"] = PlanSummary(
            budget_by_channel=human_budget,
            expected_premium=_safe_float(dec.get("human_expected_premium", 0.0)),
            q10_premium=_safe_float(dec.get("human_premium_q10")) if ("human_premium_q10" in dec) else None,
            ra_premium=_safe_float(dec.get("human_risk_adjusted_premium", 0.0)),
            p_win=_safe_float(dec.get("p_ai_better_vs_human", 0.0)),
        )

    reasons_by_channel: Mapping[str, Any] = dec.get("top_reasons_by_channel", {}) or {}
    explain: Mapping[str, Any] = dec.get("explainability", {}) or {}
    metrics_by_channel: Mapping[str, Any] = explain.get("channel_metrics", {}) or {}

    all_channels = sorted(
        set(ai_budget.keys())
        | set(dn_budget.keys())
        | set(metrics_by_channel.keys())
        | set(reasons_by_channel.keys())
    )

    channel_insights: list[ChannelInsight] = []
    for ch in all_channels:
        spend_ai = float(ai_budget.get(ch, 0.0))
        spend_dn = float(dn_budget.get(ch, 0.0))
        delta = spend_ai - spend_dn

        m = metrics_by_channel.get(ch, {}) if isinstance(metrics_by_channel.get(ch, {}), dict) else {}
        msgs = reasons_by_channel.get(ch, [])
        bullets: list[str] = []
        if isinstance(msgs, list):
            bullets = [str(x) for x in msgs if str(x).strip()][:3]
        elif msgs:
            bullets = [str(msgs)][:1]

        if abs(delta) < 1e-6:
            action = "유지"
        elif delta > 0:
            action = "증액"
        else:
            action = "감액"

        channel_insights.append(
            ChannelInsight(
                channel=ch,
                spend_ai=spend_ai,
                spend_do_nothing=spend_dn,
                delta_spend=delta,
                mroi=_safe_float(m.get("mroi"), default=float("nan")) if m else None,
                saturation_ratio=_safe_float(m.get("saturation_ratio"), default=float("nan")) if m else None,
                ec50=_safe_float(m.get("ec50"), default=float("nan")) if m else None,
                half_life=_safe_float(m.get("half_life"), default=float("nan")) if m else None,
                downside_risk=_safe_float(m.get("downside_risk"), default=float("nan")) if m else None,
                reason_bullets=bullets,
                recommended_action=action,
            )
        )

    return DecisionViewModel(
        decision_id=decision_id,
        period_start=period_start,
        period_end=period_end,
        rollout_mode=rollout_mode,
        plans=plans,
        channels=channel_insights,
    )
