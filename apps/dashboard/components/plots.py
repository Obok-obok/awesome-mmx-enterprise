from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import matplotlib.pyplot as plt

from .plot_style import configure_matplotlib_korean
configure_matplotlib_korean()

import pandas as pd
import streamlit as st



@dataclass(frozen=True)
class ChartSpec:
    """Standard chart spec to enforce consistent sizing and typography."""

    figsize: tuple[float, float] = (7.2, 3.2)
    title_size: int = 14
    label_size: int = 11
    tick_size: int = 10

DEFAULT_CHART_SPEC = ChartSpec()

@dataclass(frozen=True)
class CurveMarkers:
    current_spend: Optional[float] = None
    baseline_spend: Optional[float] = None
    human_spend: Optional[float] = None
    ec50: Optional[float] = None


def plot_curve(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    x_label: str,
    y_label: str,
    markers: CurveMarkers,
    subtitle_lines: Optional[Sequence[str]] = None,
    spec: ChartSpec = DEFAULT_CHART_SPEC,
) -> None:
    """Render a business-grade matplotlib curve with vertical markers.

    Notes:
      - We intentionally avoid explicit colors per platform instruction.
      - Marker lines use default styling; emphasis via linestyle/annotation.
    """
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return

    fig, ax = plt.subplots(figsize=spec.figsize)
    ax.plot(df[x].values, df[y].values)
    ax.set_title(title, fontsize=spec.title_size)
    ax.set_xlabel(x_label, fontsize=spec.label_size)
    ax.set_ylabel(y_label, fontsize=spec.label_size)
    ax.tick_params(axis='both', labelsize=spec.tick_size)

    # Markers
    def _vline(val: float, label: str, ls: str) -> None:
        ax.axvline(val, linestyle=ls)
        ymax = ax.get_ylim()[1]
        ax.annotate(label, xy=(val, ymax), xytext=(5, -18), textcoords="offset points", rotation=90, va="top")

    if markers.ec50 is not None:
        _vline(float(markers.ec50), "EC50", "--")
    if markers.baseline_spend is not None:
        _vline(float(markers.baseline_spend), "Do Nothing", ":")
    if markers.human_spend is not None:
        _vline(float(markers.human_spend), "Human", (0, (3, 1, 1, 1)))
    if markers.current_spend is not None:
        _vline(float(markers.current_spend), "AI 추천", "-.")

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    if subtitle_lines:
        st.caption(" · " + " | ".join([str(s) for s in subtitle_lines if s]))


def plot_timeseries(
    df: pd.DataFrame,
    x: str,
    ys: Sequence[str],
    title: str,
    x_label: str,
    y_label: str,
    spec: ChartSpec = DEFAULT_CHART_SPEC,
) -> None:
    """Render a standardized line chart for multiple series."""
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    fig, ax = plt.subplots(figsize=spec.figsize)
    for col in ys:
        if col in df.columns:
            ax.plot(df[x].values, df[col].values, label=col)
    ax.set_title(title, fontsize=spec.title_size)
    ax.set_xlabel(x_label, fontsize=spec.label_size)
    ax.set_ylabel(y_label, fontsize=spec.label_size)
    ax.tick_params(axis='both', labelsize=spec.tick_size)
    if len(ys) > 1:
        ax.legend(loc="best", fontsize=spec.tick_size)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
