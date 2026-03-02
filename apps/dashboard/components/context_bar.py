from __future__ import annotations

"""Context bar shown at the top of every page."""

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class ContextBar:
    as_of: str
    data_coverage: str
    model_version: str
    objective_mode: str
    policy_lambda: float
    policy_delta: float
    reporting_delay: str


def render_context_bar(ctx: ContextBar) -> None:
    st.markdown(
        (
            "<div class='small-muted'>"
            f"기준시점(as-of): <b>{ctx.as_of}</b> · "
            f"데이터 커버리지: <b>{ctx.data_coverage}</b> · "
            f"모델: <b>{ctx.model_version}</b> · "
            f"Objective: <b>{ctx.objective_mode}</b> (λ={ctx.policy_lambda}, δ={ctx.policy_delta}) · "
            f"Reporting delay: <b>{ctx.reporting_delay}</b>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
