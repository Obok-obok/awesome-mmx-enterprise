from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


def _safe_div(n: float, d: float) -> float | None:
    if d <= 0:
        return None
    return n / d


def _pct_change(cur: float | None, prev: float | None) -> float | None:
    if cur is None or prev is None:
        return None
    if prev <= 0:
        return None
    return (cur - prev) / prev


@dataclass(frozen=True)
class FunnelRateCardVM:
    """Rate-first funnel health KPI card.

    The main value is always a rate/efficiency. Volumes are provided only as subtitle.
    """

    title: str
    value: str
    comp: str
    subtitle: str


@dataclass(frozen=True)
class FunnelHealthVM:
    """View-model for Funnel Health section."""

    period_start: str
    period_end: str
    compare_mode: str  # 'MoM' | 'YoY'
    compare_label: str
    prev_start: str
    prev_end: str
    cards: list[FunnelRateCardVM]


def build_funnel_health_vm(*, mart_full: pd.DataFrame, mart_filtered: pd.DataFrame) -> FunnelHealthVM:
    """Build FunnelHealthVM from mart.

    Args:
        mart_full: Unfiltered mart. Must include the full data coverage.
        mart_filtered: Filtered mart for the currently selected period/channels.

    Returns:
        FunnelHealthVM with 5 rate-first cards.
    """

    if mart_filtered.empty:
        # Provide an empty-but-safe VM.
        na_cards = [
            FunnelRateCardVM(title=t, value="-", comp="전월대비 N/A", subtitle="")
            for t in [
                "Lead Efficiency",
                "Attempt Rate",
                "Connect Rate",
                "Contract Rate (CRR)",
                "Premium / Contract",
            ]
        ]
        return FunnelHealthVM(
            period_start="-",
            period_end="-",
            compare_mode="MoM",
            compare_label="전월대비",
            prev_start="-",
            prev_end="-",
            cards=na_cards,
        )

    # Normalize date dtype.
    mf = mart_full.copy()
    if not mf.empty and not pd.api.types.is_datetime64_any_dtype(mf["date"]):
        mf["date"] = pd.to_datetime(mf["date"])
    mc = mart_filtered.copy()
    if not pd.api.types.is_datetime64_any_dtype(mc["date"]):
        mc["date"] = pd.to_datetime(mc["date"])

    cur_start = pd.to_datetime(mc["date"].min())
    cur_end = pd.to_datetime(mc["date"].max())

    # Comparison policy (monthly ops):
    # - If the selected range is within a single calendar month -> MoM (previous month)
    # - If the selected range spans multiple months -> YoY (same period last year)
    months = pd.period_range(cur_start, cur_end, freq="M")
    is_single_month = len(months) == 1

    def _shift_month_range(start: pd.Timestamp, end: pd.Timestamp, months_delta: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Shift a date range by N months, clamping to end-of-month if needed."""
        s = (start + pd.DateOffset(months=months_delta)).normalize()
        e = (end + pd.DateOffset(months=months_delta)).normalize()
        # Clamp days that overflow the target month (e.g., Mar 31 -> Feb 28/29)
        s_last = (s + pd.offsets.MonthEnd(0)).normalize()
        e_last = (e + pd.offsets.MonthEnd(0)).normalize()
        if s.day > s_last.day:
            s = s_last
        if e.day > e_last.day:
            e = e_last
        return s, e

    if is_single_month:
        compare_mode = "MoM"
        compare_label = "전월대비"
        prev_start, prev_end = _shift_month_range(cur_start, cur_end, -1)
    else:
        compare_mode = "YoY"
        compare_label = "전년동기"
        prev_start = (cur_start - pd.DateOffset(years=1)).normalize()
        prev_end = (cur_end - pd.DateOffset(years=1)).normalize()

    def _sum(df: pd.DataFrame, col: str) -> float:
        if df.empty or col not in df.columns:
            return 0.0
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())

    def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if df.empty:
            return df
        m = (df["date"] >= start) & (df["date"] <= end)
        return df.loc[m]

    cur = mc
    prev = _slice(mf, prev_start, prev_end)

    # Volumes (current)
    spend = _sum(cur, "spend")
    leads = _sum(cur, "leads")
    attempts = _sum(cur, "call_attempt") if "call_attempt" in cur.columns else _sum(cur, "attempts")
    connected = _sum(cur, "call_connected") if "call_connected" in cur.columns else _sum(cur, "connected")
    contracts = _sum(cur, "contracts")
    premium = _sum(cur, "premium")

    # Volumes (previous)
    p_spend = _sum(prev, "spend")
    p_leads = _sum(prev, "leads")
    p_attempts = _sum(prev, "call_attempt") if "call_attempt" in prev.columns else _sum(prev, "attempts")
    p_connected = _sum(prev, "call_connected") if "call_connected" in prev.columns else _sum(prev, "connected")
    p_contracts = _sum(prev, "contracts")
    p_premium = _sum(prev, "premium")

    # Rates
    leads_per_1m = None if spend <= 0 else (leads / spend) * 1_000_000.0
    p_leads_per_1m = None if p_spend <= 0 else (p_leads / p_spend) * 1_000_000.0

    attempt_rate = _safe_div(attempts, leads)
    connect_rate = _safe_div(connected, attempts)
    contract_rate = _safe_div(contracts, connected)
    ppc = _safe_div(premium, contracts)

    p_attempt_rate = _safe_div(p_attempts, p_leads)
    p_connect_rate = _safe_div(p_connected, p_attempts)
    p_contract_rate = _safe_div(p_contracts, p_connected)
    p_ppc = _safe_div(p_premium, p_contracts)

    def _fmt_pct(x: float | None) -> str:
        return "-" if x is None else f"{x*100:.1f}%"

    def _fmt_leads_per_1m(x: float | None) -> str:
        return "-" if x is None else f"{x:,.1f} / 1M"

    def _fmt_money(x: float | None) -> str:
        if x is None:
            return "-"
        # KRW integer formatting
        return f"{x:,.0f}"

    def _fmt_comp(p: float | None) -> str:
        if p is None or pd.isna(p):
            return f"{compare_label} N/A"
        sign = "+" if p >= 0 else ""
        return f"{compare_label} {sign}{p*100:.1f}%"

    cards = [
        FunnelRateCardVM(
            title="Lead Efficiency",
            value=_fmt_leads_per_1m(leads_per_1m),
            comp=_fmt_comp(_pct_change(leads_per_1m, p_leads_per_1m)),
            subtitle=f"Leads {int(leads):,} / Spend {spend:,.0f}",
        ),
        FunnelRateCardVM(
            title="Attempt Rate",
            value=_fmt_pct(attempt_rate),
            comp=_fmt_comp(_pct_change(attempt_rate, p_attempt_rate)),
            subtitle=f"Attempts {int(attempts):,} / Leads {int(leads):,}",
        ),
        FunnelRateCardVM(
            title="Connect Rate",
            value=_fmt_pct(connect_rate),
            comp=_fmt_comp(_pct_change(connect_rate, p_connect_rate)),
            subtitle=f"Connected {int(connected):,} / Attempts {int(attempts):,}",
        ),
        FunnelRateCardVM(
            title="Contract Rate (CRR)",
            value=_fmt_pct(contract_rate),
            comp=_fmt_comp(_pct_change(contract_rate, p_contract_rate)),
            subtitle=f"Contracts {int(contracts):,} / Connected {int(connected):,}",
        ),
        FunnelRateCardVM(
            title="Premium / Contract",
            value=_fmt_money(ppc),
            comp=_fmt_comp(_pct_change(ppc, p_ppc)),
            subtitle=f"Premium {premium:,.0f} / Contracts {int(contracts):,}",
        ),
    ]

    return FunnelHealthVM(
        period_start=str(cur_start.date()),
        period_end=str(cur_end.date()),
        compare_mode=compare_mode,
        compare_label=compare_label,
        prev_start=str(prev_start.date()),
        prev_end=str(prev_end.date()),
        cards=cards,
    )
