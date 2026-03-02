from __future__ import annotations
import pandas as pd

STAGES = ['lead_per_spend','attempt_per_lead','connected_rate','contract_rate','premium_per_contract']

def detect_bottleneck(df_rates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, sub in df_rates.groupby('date'):
        means = {s: float(sub[s].mean()) for s in STAGES if s in sub.columns}
        if not means:
            continue
        worst = min(means, key=means.get)
        rows.append({'date': str(d), 'bottleneck_stage': worst, 'value': means[worst]})
    return pd.DataFrame(rows)
