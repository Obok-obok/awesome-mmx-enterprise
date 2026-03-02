from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import json
import numpy as np
import pandas as pd

from mmx.data.paths import Paths
from mmx.engine.sem.inference import PosteriorSummary, posterior_predict_premium
from mmx.evaluation.shadow import shadow_eval

@dataclass(frozen=True)
class EvaluateShadowResult:
    delta_ra_mean: float
    p_ai_better: float
    ci_low: float
    ci_high: float

def _load_posterior(paths: Paths, model_version: str) -> PosteriorSummary:
    p = paths.artifacts / f'models/mmx_sem/{model_version}/posterior_summary.json'
    obj = json.loads(p.read_text(encoding='utf-8'))
    return PosteriorSummary(model_version=obj['model_version'], channel_params=obj['channel_params'], globals=obj['globals'])

def evaluate_shadow(paths: Paths, model_version: str, ai_plan: Dict[str, float], human_plan: Dict[str, float], policy_lambda: float) -> EvaluateShadowResult:
    mart = paths.mart / 'daily_channel_fact.csv'
    df = pd.read_csv(mart) if mart.exists() else pd.DataFrame(columns=['date','channel','spend','leads','call_attempt','call_connected','contracts','premium'])
    posterior = _load_posterior(paths, model_version)

    s_ai = posterior_predict_premium(df, posterior, ai_plan, n_days=30, n_samples=2000, seed=11)
    s_h  = posterior_predict_premium(df, posterior, human_plan, n_days=30, n_samples=2000, seed=22)

    ra_ai = s_ai - policy_lambda * np.std(s_ai, ddof=1)
    ra_h  = s_h  - policy_lambda * np.std(s_h, ddof=1)

    res = shadow_eval(ra_ai, ra_h)
    return EvaluateShadowResult(**res.__dict__)
