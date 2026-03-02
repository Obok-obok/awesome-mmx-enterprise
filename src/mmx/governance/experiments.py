from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from mmx.data.io_csv import append_csv, file_lock
from mmx.data.paths import Paths

@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    split_strategy: str
    ai_share: float
    is_locked: bool

def register_experiment(paths: Paths, experiment_id: str, split_strategy: str, ai_share: float) -> Experiment:
    lock = paths.logs / 'experiments/experiments.lock'
    with file_lock(lock):
        p = paths.logs / 'experiments/experiment_registry.csv'
        append_csv(p, [{
            'timestamp': pd.Timestamp.utcnow().isoformat(),
            'experiment_id': experiment_id,
            'split_strategy': split_strategy,
            'ai_share': ai_share,
            'is_locked': True,
        }], fieldnames=['timestamp','experiment_id','split_strategy','ai_share','is_locked'])
    return Experiment(experiment_id, split_strategy, ai_share, True)
