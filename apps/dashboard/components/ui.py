from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import streamlit as st



def load_css(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    st.markdown(f"<style>{p.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def fmt_money(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.0f}"


def fmt_count(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.0f}"


def fmt_ratio(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def fmt_percent(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "—"
    return f"{x*100:.{digits}f}%"


def fmt_float(x: float, digits: int = 2) -> str:
    """Format a float with a fixed number of digits.

    Returns an em dash for NaN/None.
    """
    if pd.isna(x):
        return "—"
    return f"{float(x):.{digits}f}"



def fmt_ci(low: float, high: float, money: bool = True) -> str:
    if money:
        return f"[{fmt_money(low)} ~ {fmt_money(high)}]"
    return f"[{low:.3f} ~ {high:.3f}]"


def kpi_card(label: str, value: str, sub: Optional[str] = None) -> None:
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    st.markdown(
        f"""
<div class='card'>
  <div class='kpi-label'>{label}</div>
  <div class='kpi-value'>{value}</div>
  {sub_html}
</div>
""",
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: Optional[str] = None) -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


KpiItem = Union[
    Mapping[str, str],
    Tuple[str, str],
    Tuple[str, str, str],
]


def kpi_row(items: Sequence[KpiItem]) -> None:
    """Render a standardized KPI row.

    This component is intentionally tolerant of input shapes because upstream
    code may build KPI lists as dicts or tuples.

    Accepted item shapes:
      - {"label": str, "value": str, "sub": Optional[str]}
      - (label, value)
      - (label, value, sub)

    Args:
        items: KPI items.
    """
    if not items:
        return

    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        with c:
            if isinstance(it, Mapping):
                kpi_card(it.get("label", ""), it.get("value", ""), it.get("sub"))
            elif isinstance(it, tuple):
                label = it[0] if len(it) > 0 else ""
                value = it[1] if len(it) > 1 else ""
                sub = it[2] if len(it) > 2 else None
                kpi_card(label, value, sub)
            else:
                # Fallback: render best-effort string.
                kpi_card("KPI", str(it), None)



def badge_html(text: str, kind: str = "info") -> str:
    """Return HTML for a small badge (for composing markup strings)."""
    kind = kind.lower().strip()
    cls = {
        "info": "badge",
        "muted": "badge badge-muted",
        "ok": "badge badge-ok",
        "warn": "badge badge-warn",
        "warning": "badge badge-warn",
        "danger": "badge badge-danger",
        "error": "badge badge-danger",
    }.get(kind, "badge")
    return f"<span class='{cls}'>{text}</span>"

def badge(text: str, kind: str = "info") -> None:
    """Render a small badge.

    Args:
        text: Badge text.
        kind: info|ok|warn|danger.
    """
    kind = kind.lower().strip()
    cls = {
        "info": "badge",
        "ok": "badge badge-ok",
        "warn": "badge badge-warn",
        "danger": "badge badge-danger",
    }.get(kind, "badge")
    st.markdown(f"<span class='{cls}'>{text}</span>", unsafe_allow_html=True)


def style_table(
    df: pd.DataFrame,
    money_cols: Iterable[str] | None = None,
    count_cols: Iterable[str] | None = None,
    pct_cols: Iterable[str] | None = None,
    float_cols: Iterable[str] | None = None,
    digits: int = 3,
) -> "pd.io.formats.style.Styler":
    """Return a pandas Styler applying enterprise formatting.

    This is used instead of raw st.dataframe(df) to standardize:
    - number formats
    - alignment
    - zebra rows
    """
    money_cols = set(money_cols or [])
    count_cols = set(count_cols or [])
    pct_cols = set(pct_cols or [])
    float_cols = set(float_cols or [])

    def _coerce_float(x: object) -> float | None:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        try:
            return float(x)  # type: ignore[arg-type]
        except Exception:
            return None

    def _fmt_money(x: object) -> str:
        v = _coerce_float(x)
        if v is None:
            return "—" if pd.isna(x) else str(x)
        return f"{v:,.0f}"

    def _fmt_count(x: object) -> str:
        v = _coerce_float(x)
        if v is None:
            return "—" if pd.isna(x) else str(x)
        return f"{v:,.0f}"

    def _fmt_pct(x: object) -> str:
        v = _coerce_float(x)
        if v is None:
            return "—" if pd.isna(x) else str(x)
        return f"{v:.1%}"

    def _fmt_float(x: object) -> str:
        v = _coerce_float(x)
        if v is None:
            return "—" if pd.isna(x) else str(x)
        return f"{v:,.{digits}f}"

    fmt: Dict[str, object] = {}
    for c in df.columns:
        if c in money_cols:
            fmt[c] = _fmt_money
        elif c in count_cols:
            fmt[c] = _fmt_count
        elif c in pct_cols:
            fmt[c] = _fmt_pct
        elif c in float_cols:
            fmt[c] = _fmt_float

    sty = df.style.format(fmt, na_rep="—")
    sty = sty.set_table_styles(
        [
            {"selector": "th", "props": "text-align:left; font-weight:600;"},
            {"selector": "td", "props": "text-align:right;"},
            {"selector": "td:first-child", "props": "text-align:left;"},
        ]
    )
    sty = sty.apply(lambda _: ["background-color: #F9FAFB" if i % 2 else "" for i in range(len(df))], axis=0)
    return sty

