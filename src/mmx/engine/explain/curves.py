from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from mmx.engine.sem.adstock import half_life
from mmx.engine.sem.saturation import saturation_ratio

def build_curves(posterior_channel_params: Dict[str, Dict[str, Any]], max_spend: float, n_days: int = 30) -> Dict[str, pd.DataFrame]:
    spend_grid = np.linspace(0, max_spend, 120)
    sat_rows, resp_rows, mroi_rows = [], [], []

    for ch, p in posterior_channel_params.items():
        ec50 = float(p['ec50'])
        alpha = float(p['alpha_lead'])
        h = float(p.get('hill', 1.0))
        decay = float(p['decay'])
        hl = half_life(decay)

        x = spend_grid / n_days
        resp = alpha * (np.power(np.clip(x, 0, None), h) / (np.power(np.clip(x, 0, None), h) + ec50**h + 1e-9))
        mroi = alpha * h * (ec50**h) * np.power(np.clip(x, 1e-9, None), h-1) / (np.power(x, h) + ec50**h + 1e-9)**2
        mroi = mroi / n_days

        sat_rows.append({
            'channel': ch,
            'ec50': ec50,
            'half_life_days': hl,
            'saturation_at_mid': saturation_ratio(float((max_spend/2)/n_days), ec50),
        })

        for s, y in zip(spend_grid, resp):
            resp_rows.append({'channel': ch, 'spend': float(s), 'response': float(y)})
        for s, y in zip(spend_grid, mroi):
            mroi_rows.append({'channel': ch, 'spend': float(s), 'mroi': float(y)})

    return {
        'saturation_metrics': pd.DataFrame(sat_rows),
        'response_curve': pd.DataFrame(resp_rows),
        'mroi_curve': pd.DataFrame(mroi_rows),
    }
