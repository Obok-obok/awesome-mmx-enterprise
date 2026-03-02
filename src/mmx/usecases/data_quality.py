from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
from mmx.data.paths import Paths

@dataclass(frozen=True)
class ReportingDelaySignal:
    weekend_spike_ratio: float
    recommendation: str  # OFF | CONSIDER_ON

def detect_weekend_reporting_delay(paths: Paths) -> ReportingDelaySignal:
    mart = paths.mart / 'daily_channel_fact.csv'
    if not mart.exists():
        return ReportingDelaySignal(weekend_spike_ratio=float('nan'), recommendation='OFF')
    df = pd.read_csv(mart)
    df['date'] = pd.to_datetime(df['date'])
    df['dow'] = df['date'].dt.dayofweek
    mon = df[df['dow'] == 0]['leads'].mean() if (df['dow'] == 0).any() else 0.0
    mid = df[df['dow'].isin([1,2,3,4])]['leads'].mean() if (df['dow'].isin([1,2,3,4])).any() else 0.0
    ratio = float(mon / (mid + 1e-9)) if mid > 0 else float('nan')
    rec = 'CONSIDER_ON' if np.isfinite(ratio) and ratio > 1.3 else 'OFF'
    return ReportingDelaySignal(weekend_spike_ratio=ratio, recommendation=rec)
