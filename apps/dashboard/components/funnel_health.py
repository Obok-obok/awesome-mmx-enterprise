from __future__ import annotations

import streamlit as st

from apps.dashboard.components.ui import kpi_row, section
from apps.dashboard.viewmodels.funnel_health_vm import FunnelHealthVM


def render_funnel_health(vm: FunnelHealthVM) -> None:
    """Render Funnel Health section.

    Funnel Health is **rate-first** by design. Amounts are shown only as subtitles.
    """

    section("Funnel Health")

    st.caption(f"비교 기준: **{vm.compare_label}** · 비교 구간 {vm.prev_start} ~ {vm.prev_end}")

    kpi_row(
        [(c.title, c.value, c.comp) for c in vm.cards]
    )
    # Subtitles below (volumes)
    with st.expander("카드 상세(분모/분자 볼륨)", expanded=False):
        for c in vm.cards:
            st.caption(f"• **{c.title}** — {c.subtitle}")
