from __future__ import annotations

"""Business charts for executive decision-making.

We avoid "pretty" charts that don't answer a question.
Primary usage is showing target gap (Actual vs Target) and variance.
"""

import matplotlib.pyplot as plt

import pandas as pd
import streamlit as st

from .plot_style import configure_matplotlib_korean
from .plots import ChartSpec, DEFAULT_CHART_SPEC


configure_matplotlib_korean()


def plot_target_variance_bar(
    df: pd.DataFrame,
    *,
    month_col: str = "month",
    actual_col: str = "premium",
    target_col: str = "target_premium",
    title: str = "목표 대비 실적(갭)",
    y_label: str = "Premium(원)",
    spec: ChartSpec = DEFAULT_CHART_SPEC,
) -> None:
    """Plot bar chart: Actual + Target and annotate the gap.

    Args:
        df: DataFrame with month, actual, target.
    """
    if df.empty or month_col not in df.columns:
        st.info("표시할 데이터가 없습니다.")
        return

    d = df[[month_col, actual_col, target_col]].copy()
    d[actual_col] = pd.to_numeric(d[actual_col], errors="coerce").fillna(0.0)
    d[target_col] = pd.to_numeric(d[target_col], errors="coerce").fillna(0.0)
    d["gap"] = d[actual_col] - d[target_col]

    x = list(range(len(d)))
    width = 0.35
    fig, ax = plt.subplots(figsize=spec.figsize)
    ax.bar([i - width / 2 for i in x], d[target_col].values, width=width, label="목표")
    ax.bar([i + width / 2 for i in x], d[actual_col].values, width=width, label="실적")

    ax.set_title(title, fontsize=spec.title_size)
    ax.set_ylabel(y_label, fontsize=spec.label_size)
    ax.set_xticks(x)
    ax.set_xticklabels(d[month_col].astype(str).tolist(), fontsize=spec.tick_size)
    ax.tick_params(axis="y", labelsize=spec.tick_size)
    ax.legend(loc="best", fontsize=spec.tick_size)

    # Gap annotations
    for i, g in enumerate(d["gap"].values):
        # annotate above actual bar
        y = float(d[actual_col].iloc[i])
        ax.annotate(
            f"{g:+,.0f}",
            xy=(i + width / 2, y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=spec.tick_size,
        )

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
