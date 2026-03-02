from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple
import pandas as pd

def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")

def assert_non_negative(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    for c in cols:
        if (df[c] < 0).any():
            raise ValueError(f"{name}: negative values in column '{c}'")

@dataclass(frozen=True)
class DateParseReport:
    column: str
    total_rows: int
    nat_rows: int
    nat_ratio: float

def parse_datetime_series(
    s: pd.Series,
    *,
    name: str,
    column: str,
    max_nat_ratio: float,
) -> Tuple[pd.Series, DateParseReport]:
    dt = pd.to_datetime(s, errors='coerce')
    nat_rows = int(dt.isna().sum())
    total = int(len(dt))
    ratio = float(nat_rows / total) if total > 0 else 0.0
    report = DateParseReport(column=column, total_rows=total, nat_rows=nat_rows, nat_ratio=ratio)
    if ratio > max_nat_ratio:
        raise ValueError(
            f"{name}: invalid timestamps in '{column}' (NaT ratio={ratio:.3%}, rows={nat_rows}/{total}) exceeds max {max_nat_ratio:.3%}"
        )
    return dt, report

def normalize_channels(
    s: pd.Series,
    *,
    normalize: bool,
) -> pd.Series:
    if not normalize:
        return s.astype(str)
    return s.astype(str).str.strip().str.lower()

@dataclass(frozen=True)
class ChannelReport:
    total_rows: int
    unique_channels: int
    unknown_rows: int
    unknown_ratio: float
    unknown_channels: Tuple[str, ...]

def validate_channels(
    df: pd.DataFrame,
    *,
    name: str,
    allowed_channels: Tuple[str, ...],
    enforce_allowlist: bool,
    max_unknown_ratio: float,
) -> ChannelReport:
    total = int(len(df))
    if total == 0:
        return ChannelReport(total_rows=0, unique_channels=0, unknown_rows=0, unknown_ratio=0.0, unknown_channels=tuple())
    uniq = sorted(set(df['channel'].astype(str)))
    allowed = set(allowed_channels)
    unknown = sorted([c for c in uniq if allowed and c not in allowed])
    unknown_mask = df['channel'].isin(unknown) if unknown else pd.Series([False]*len(df))
    unknown_rows = int(unknown_mask.sum())
    unknown_ratio = float(unknown_rows / total) if total > 0 else 0.0
    rep = ChannelReport(
        total_rows=total,
        unique_channels=len(uniq),
        unknown_rows=unknown_rows,
        unknown_ratio=unknown_ratio,
        unknown_channels=tuple(unknown),
    )
    if allowed and unknown_ratio > max_unknown_ratio:
        msg = f"{name}: unknown channels detected (unknown_ratio={unknown_ratio:.3%}, rows={unknown_rows}/{total}, unknown={unknown})"
        if enforce_allowlist:
            raise ValueError(msg)
    return rep

@dataclass(frozen=True)
class FunnelMonotonicityReport:
    total_rows: int
    violation_rows: int
    violation_ratio: float
    violations: Dict[str, int]

def validate_funnel_monotonicity(
    df_mart: pd.DataFrame,
    *,
    enforce: bool,
    max_violation_ratio: float,
    name: str = 'mart',
) -> FunnelMonotonicityReport:
    if df_mart.empty:
        return FunnelMonotonicityReport(total_rows=0, violation_rows=0, violation_ratio=0.0, violations={})
    total = int(len(df_mart))
    v_connected = (df_mart['call_connected'] > df_mart['call_attempt']).sum()
    v_contracts = (df_mart['contracts'] > df_mart['call_connected']).sum()
    v_premium = ((df_mart['premium'] > 0) & (df_mart['contracts'] == 0)).sum()
    any_violation = (df_mart['call_connected'] > df_mart['call_attempt']) | (df_mart['contracts'] > df_mart['call_connected']) | ((df_mart['premium'] > 0) & (df_mart['contracts'] == 0))
    violation_rows = int(any_violation.sum())
    ratio = float(violation_rows / total) if total > 0 else 0.0
    rep = FunnelMonotonicityReport(
        total_rows=total,
        violation_rows=violation_rows,
        violation_ratio=ratio,
        violations={
            'call_connected_gt_call_attempt': int(v_connected),
            'contracts_gt_call_connected': int(v_contracts),
            'premium_positive_but_no_contracts': int(v_premium),
        },
    )
    if ratio > max_violation_ratio and enforce:
        raise ValueError(
            f"{name}: funnel monotonicity violations ratio={ratio:.3%} (rows={violation_rows}/{total}) exceeds max {max_violation_ratio:.3%}; details={rep.violations}"
        )
    return rep
