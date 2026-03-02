from __future__ import annotations

"""Business tables for recommendation UI."""

import pandas as pd


def build_budget_comparison_table(
    *,
    ai_budget: dict[str, float],
    dn_budget: dict[str, float],
) -> pd.DataFrame:
    rows = []
    all_ch = sorted(set(ai_budget.keys()) | set(dn_budget.keys()))
    for ch in all_ch:
        ai = float(ai_budget.get(ch, 0.0))
        dn = float(dn_budget.get(ch, 0.0))
        rows.append(
            {
                "매체": ch,
                "현상유지 예산": dn,
                "AI 추천 예산": ai,
                "Δ 예산(AI-DN)": ai - dn,
            }
        )
    return pd.DataFrame(rows)


def build_channel_decision_table(
    channels: list["object"],
) -> pd.DataFrame:
    """Build a decision table including reasons.

    Expected columns from ChannelInsight:
    - channel, spend_do_nothing, spend_ai, delta_spend
    - mroi, saturation_ratio, half_life, downside_risk
    - reason_bullets
    """
    rows = []
    for ch in channels:
        bullets = list(getattr(ch, "reason_bullets", []) or [])
        rows.append(
            {
                "매체": getattr(ch, "channel", "-"),
                "현상유지 예산": float(getattr(ch, "spend_do_nothing", 0.0)),
                "AI 추천 예산": float(getattr(ch, "spend_ai", 0.0)),
                "Δ 예산(AI-DN)": float(getattr(ch, "delta_spend", 0.0)),
                "mROI": getattr(ch, "mroi", None),
                "포화도": getattr(ch, "saturation_ratio", None),
                "half-life": getattr(ch, "half_life", None),
                "Downside": getattr(ch, "downside_risk", None),
                "핵심 이유(요약)": " / ".join([str(b) for b in bullets[:3]]) if bullets else "",
            }
        )
    return pd.DataFrame(rows)
