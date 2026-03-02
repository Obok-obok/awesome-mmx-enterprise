"""Formatting helpers (single source of truth).

The dashboard previously relied on ad-hoc helpers (e.g., `won()`) that were
sometimes not imported, causing runtime NameError in Streamlit pages.

This module centralizes all user-facing numeric formatting and guarantees:
- Safe handling of None/NaN/inf
- Consistent thousand separators
- Consistent KRW unit rendering
"""

from __future__ import annotations

import math
from typing import Any


def _is_finite_number(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def format_int(x: Any) -> str:
    """Format an integer-like number with thousand separators.

    Returns "-" when x is None/NaN/inf or cannot be converted.
    """
    if not _is_finite_number(x):
        return "-"
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return "-"


def format_won(x: Any, unit: str = "원") -> str:
    """Format currency values in KRW with unit suffix."""
    v = format_int(x)
    if v == "-":
        return "-"
    return f"{v}{unit}"


def format_ratio(x: Any, ndigits: int = 2) -> str:
    """Format ratio/ROI values."""
    if not _is_finite_number(x):
        return "-"
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "-"
