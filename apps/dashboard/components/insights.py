from __future__ import annotations

"""Insight rendering components.

Standardizes "Conclusion → Evidence(3) → Action".
"""

from typing import Iterable

import streamlit as st

from .ui import badge


def render_insight_card(
    *,
    title: str,
    conclusion: str,
    evidence: Iterable[str],
    action: str,
    kind: str = "info",
) -> None:
    """Render a single insight card."""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    cols = st.columns([6, 2])
    with cols[0]:
        st.markdown(f"**{title}**")
    with cols[1]:
        badge(kind.upper(), kind=kind)

    st.markdown(f"**결론:** {conclusion}")
    ev = [e for e in evidence if str(e).strip()]
    if ev:
        st.markdown("**근거:**")
        for e in ev[:3]:
            st.markdown(f"- {e}")
    st.markdown(f"**권장 액션:** {action}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_top_insights_from_channels(
    channels: list["object"],
    *,
    top_k: int = 3,
) -> None:
    """Render top-K channel insights.

    Selection heuristic:
    - largest absolute spend delta first.
    """
    if not channels:
        st.info("표시할 인사이트가 없습니다.")
        return
    ordered = sorted(channels, key=lambda c: abs(float(getattr(c, "delta_spend", 0.0))), reverse=True)
    for ch in ordered[:top_k]:
        delta = float(getattr(ch, "delta_spend", 0.0))
        kind = "ok" if delta > 0 else ("warn" if delta < 0 else "info")
        title = f"{getattr(ch, 'channel', '-')}: {getattr(ch, 'recommended_action', '-') }"
        conclusion = (
            f"AI 추천 예산은 Do Nothing 대비 {('+' if delta >= 0 else '')}{delta:,.0f}원"
        )
        evidence = list(getattr(ch, "reason_bullets", []) or [])
        action = f"{getattr(ch, 'channel', '-') } 채널 {getattr(ch, 'recommended_action', '-') } 검토"
        render_insight_card(
            title=title,
            conclusion=conclusion,
            evidence=evidence,
            action=action,
            kind=kind,
        )
