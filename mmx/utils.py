
from __future__ import annotations

import numpy as np
import pandas as pd

def fmt_int(x: float | int) -> str:
    """Format integer-like values with thousands separators (no unit)."""
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return "-"

def won(x: float | int, unit: str = "원") -> str:
    """Format currency values in KRW with unit."""
    v = fmt_int(x)
    if v == "-":
        return v
    return f"{v}{unit}"

# Backward-compat alias used across the app for count-style values
def money(x: float | int) -> str:
    return fmt_int(x)


def parse_number(x, default=np.nan) -> float:
    """Parse numbers that may contain commas/whitespace.

    We use text_input for business-facing inputs to show thousand separators.
    """
    if x is None:
        return default
    if isinstance(x, (int, float, np.number)):
        try:
            return float(x)
        except Exception:
            return default
    s = str(x).strip()
    if s == "":
        return default
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return default


import re

def masked_number_input(
    label: str,
    *,
    key: str,
    default: float = 0.0,
    help: str | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Business-friendly numeric input with thousands separators.

    Streamlit disallows mutating `st.session_state[widget_key]` for an already-instantiated
    widget during the same script run. To avoid `StreamlitAPIException`, this function only
    reformats the widget value inside an `on_change` callback (which Streamlit allows).

    - Displays comma formatting (e.g., 23000000 -> 23,000,000)
    - Accepts values with/without commas
    - Returns float (rounded) or np.nan if empty
    """
    import streamlit as st

    widget_key = f"{key}__masked"

    # Initialize once (BEFORE widget instantiation)
    if widget_key not in st.session_state:
        try:
            st.session_state[widget_key] = f"{float(default):,.0f}"
        except Exception:
            st.session_state[widget_key] = str(default)

    def _format_inplace():
        raw = str(st.session_state.get(widget_key, "") or "").strip()
        if raw == "":
            return
        cleaned = re.sub(r"[^0-9\-\.,]", "", raw)
        num = parse_number(cleaned, default=np.nan)
        if np.isfinite(num):
            if min_value is not None:
                num = max(min_value, num)
            if max_value is not None:
                num = min(max_value, num)
            st.session_state[widget_key] = f"{float(num):,.0f}"
        else:
            try:
                st.session_state[widget_key] = f"{float(default):,.0f}"
            except Exception:
                st.session_state[widget_key] = str(default)

    raw = st.text_input(label, key=widget_key, help=help, on_change=_format_inplace)

    raw = (raw or "").strip()
    if raw == "":
        return np.nan

    cleaned = re.sub(r"[^0-9\-\.,]", "", raw)
    num = parse_number(cleaned, default=np.nan)

    if np.isfinite(num):
        if min_value is not None:
            num = max(min_value, num)
        if max_value is not None:
            num = min(max_value, num)
        return float(round(num))

    return float(round(default))


def pct(x: float) -> str:
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "-"

def safe_div(a: float, b: float) -> float:
    try:
        a=float(a); b=float(b)
        return a/b if b!=0 else 0.0
    except Exception:
        return 0.0

def month_range_calendar(today: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (month_start, month_end) for calendar month."""
    t = (today or pd.Timestamp.today()).normalize()
    start = t.replace(day=1)
    end = (start + pd.offsets.MonthEnd(0)).normalize()
    return start, end

def add_total_row_breakdown(df: pd.DataFrame, *, total_rr: float, total_premium_per_sale: float) -> pd.DataFrame:
    """Append total row named '합계' assuming columns: channel, 예산, Lead, Sales, 총보험료, RR, 건당 보험료."""
    if df is None or df.empty:
        return df
    out = df.copy()
    total = {
        "channel": "합계",
        "예산": float(pd.to_numeric(out["예산"], errors="coerce").fillna(0).sum()),
        "Lead": float(pd.to_numeric(out["Lead"], errors="coerce").fillna(0).sum()),
        "Sales": float(pd.to_numeric(out["Sales"], errors="coerce").fillna(0).sum()),
        "총보험료": float(pd.to_numeric(out["총보험료"], errors="coerce").fillna(0).sum()),
        "RR": float(total_rr),
        "건당 보험료": float(total_premium_per_sale),
    }
    out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)
    return out
