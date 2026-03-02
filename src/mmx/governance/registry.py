from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import json
import pandas as pd
from mmx.data.io_csv import append_csv, file_lock
from mmx.data.paths import Paths

REGISTRY_FILE = 'model_registry.json'

@dataclass(frozen=True)
class ModelRegistry:
    production_version: str
    staging_version: Optional[str] = None

def load_registry(paths: Paths) -> ModelRegistry:
    p = paths.artifacts / REGISTRY_FILE
    if not p.exists():
        return ModelRegistry(production_version='none')
    obj = json.loads(p.read_text(encoding='utf-8'))
    return ModelRegistry(production_version=obj.get('production_version','none'), staging_version=obj.get('staging_version'))

def save_registry(paths: Paths, reg: ModelRegistry) -> None:
    p = paths.artifacts / REGISTRY_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg.__dict__, ensure_ascii=False, indent=2), encoding='utf-8')

def promote_to_production(paths: Paths, model_version: str, reason: str) -> None:
    lock = paths.logs / 'training/registry.lock'
    with file_lock(lock):
        save_registry(paths, ModelRegistry(production_version=model_version))
        append_csv(paths.logs / 'training/model_promotion_log.csv', [{
            'timestamp': pd.Timestamp.utcnow().isoformat(),
            'new_production_version': model_version,
            'reason': reason,
        }], fieldnames=['timestamp','new_production_version','reason'])
