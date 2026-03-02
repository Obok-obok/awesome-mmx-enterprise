from __future__ import annotations
import pandas as pd
import numpy as np

def compute_rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-9
    out['lead_per_spend'] = out['leads'] / (out['spend'] + eps)
    out['attempt_per_lead'] = out['call_attempt'] / (out['leads'] + eps)
    out['connected_rate'] = out['call_connected'] / (out['call_attempt'] + eps)
    out['contract_rate'] = out['contracts'] / (out['call_connected'] + eps)
    out['premium_per_contract'] = out['premium'] / (out['contracts'] + eps)
    return out
