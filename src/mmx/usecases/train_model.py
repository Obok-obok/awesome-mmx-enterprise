from __future__ import annotations
from dataclasses import dataclass
import json
import pandas as pd
from mmx.data.paths import Paths
from mmx.data.io_csv import append_csv, file_lock
from mmx.engine.sem.inference import fit_posterior

@dataclass(frozen=True)
class TrainResult:
    model_version: str
    artifact_dir: str

def train_model(paths: Paths, backend: str, method: str, reporting_delay: str = "OFF", reporting_delay_max: int = 3) -> TrainResult:
    lock = paths.logs / 'training/training.lock'
    with file_lock(lock):
        mart = paths.mart / 'daily_channel_fact.csv'
        if not mart.exists():
            raise FileNotFoundError(f'Mart not found: {mart}')
        df = pd.read_csv(mart)
        posterior, _trace = fit_posterior(df, backend=backend, method=method, report_delay_max=(reporting_delay_max if reporting_delay.upper()=="ON" else 0))

        vdir = paths.artifacts / f'models/mmx_sem/{posterior.model_version}'
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir/'posterior_summary.json').write_text(json.dumps({
            'model_version': posterior.model_version,
            'channel_params': posterior.channel_params,
            'globals': posterior.globals,
            'backend': backend,
            'method': method,
        }, ensure_ascii=False, indent=2), encoding='utf-8')

        append_csv(paths.logs / 'training/training_log.csv', [{
            'timestamp': pd.Timestamp.utcnow().isoformat(),
            'model_version': posterior.model_version,
            'backend': backend,
            'method': method,
            'status': 'SUCCESS',
        }], fieldnames=['timestamp','model_version','backend','method','status'])

        return TrainResult(posterior.model_version, str(vdir))
