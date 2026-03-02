from __future__ import annotations

"""Dashboard state: global filters, targets input, and safe persistence.

Design requirements:
- A single source of truth for global filters (date range + channel selection)
- Visible and consistent across all multipage views
- Provide a 'targets' input (monthly premium target) for Executive narrative
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st


FILTER_KEY = "mmx_filters"
TARGETS_REL = Path("targets") / "monthly_targets.csv"


@dataclass(frozen=True)
class Filters:
    start: pd.Timestamp
    end: pd.Timestamp
    channels: List[str]


def _ensure_filter_state(default_start: pd.Timestamp, default_end: pd.Timestamp, channels: List[str]) -> None:
    if FILTER_KEY not in st.session_state:
        st.session_state[FILTER_KEY] = {
            "start": default_start,
            "end": default_end,
            "channels": channels,
        }


def sidebar_global_filters(paths: "object") -> Filters:
    """Render global filters in the sidebar.

    Args:
        paths: Resolved data paths.

    Returns:
        Filters (start, end, channels)
    """
    mart_path = Path(paths.mart) / "daily_channel_fact.csv"
    df = pd.read_csv(mart_path) if mart_path.exists() else pd.DataFrame(columns=["date", "channel"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    all_channels = sorted(df["channel"].dropna().unique().tolist()) if "channel" in df.columns else []

    if df.empty:
        today = pd.Timestamp.today()
        _ensure_filter_state(today - pd.Timedelta(days=29), today, all_channels)
        st.info("Mart 데이터가 없습니다. 먼저 Build Mart를 실행하세요.")
        return Filters(start=today - pd.Timedelta(days=29), end=today, channels=all_channels)

    min_d = df["date"].min()
    max_d = df["date"].max()
    _ensure_filter_state(min_d, max_d, all_channels)

    s0: pd.Timestamp = st.session_state[FILTER_KEY]["start"]
    e0: pd.Timestamp = st.session_state[FILTER_KEY]["end"]
    c0: List[str] = st.session_state[FILTER_KEY]["channels"]

    start_date, end_date = st.date_input(
        "조회 기간",
        value=(s0.date(), e0.date()),
        min_value=min_d.date(),
        max_value=max_d.date(),
    )
    channels = st.multiselect("매체 선택", options=all_channels, default=c0 or all_channels)

    st.session_state[FILTER_KEY] = {
        "start": pd.Timestamp(start_date),
        "end": pd.Timestamp(end_date),
        "channels": channels,
    }

    return Filters(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), channels=channels)


def get_filters() -> Filters:
    d = st.session_state.get(FILTER_KEY)
    if not d:
        today = pd.Timestamp.today()
        return Filters(start=today - pd.Timedelta(days=29), end=today, channels=[])
    return Filters(start=pd.Timestamp(d["start"]), end=pd.Timestamp(d["end"]), channels=list(d["channels"]))


def filter_mart(df: pd.DataFrame) -> pd.DataFrame:
    """Filter mart by global sidebar filters."""
    if df.empty:
        return df
    f = get_filters()
    m = (df["date"] >= f.start) & (df["date"] <= f.end)
    if f.channels:
        m &= df["channel"].isin(f.channels)
    return df.loc[m].copy()


def sidebar_targets(paths: "object") -> None:
    """Targets uploader + quick view.

    Target file schema:
    - month: YYYY-MM
    - target_premium: numeric
    """
    target_path = Path(paths.data_curated) / TARGETS_REL
    target_path.parent.mkdir(parents=True, exist_ok=True)

    up = st.file_uploader("월별 Premium 목표치 업로드 (CSV)", type=["csv"], help="columns: month(YYYY-MM), target_premium")
    if up is not None:
        df = pd.read_csv(up)
        df = df[["month", "target_premium"]].copy()
        # Normalize month to YYYY-MM as a contract (Excel exports are often messy).
        s = df["month"].astype(str).str.strip()
        # Accept: YYYY-MM, YYYY-MM-DD, YYYYMM, YYYY/MM
        s = s.str.replace("/", "-", regex=False)
        # Try to extract YYYY-MM first.
        extracted = s.str.extract(r"(\d{4}-\d{2})", expand=False)
        # Fallback: YYYYMM -> YYYY-MM
        yyyymm = s.str.extract(r"(\d{4})(\d{2})", expand=True)
        fallback = yyyymm[0].fillna("") + "-" + yyyymm[1].fillna("")
        df["month"] = extracted.fillna(fallback).str.slice(0, 7)
        df["target_premium"] = pd.to_numeric(df["target_premium"], errors="coerce").fillna(0.0)
        tmp = target_path.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(target_path)
        st.success("Targets 저장 완료")

    if target_path.exists():
        df = pd.read_csv(target_path)
        st.caption("현재 등록된 목표치(최근 6개월)")
        view = df.tail(6).copy()
        # Business-friendly column names & hide index
        rename_map = {
            "month": "월",
            "target_premium": "목표 프리미엄",
        }
        view = view.rename(columns={k: v for k, v in rename_map.items() if k in view.columns})
        st.dataframe(view, use_container_width=True, height=210, hide_index=True)
    else:
        st.caption("등록된 목표치가 없습니다.")
