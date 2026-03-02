from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _to_list(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def save_adaptive_state(model: Any, path: str | Path) -> None:
    """Persist AdaptiveSCM state to JSON.

    - No pickle (ops-safe)
    - Works on GCP VM-free
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "config": asdict(model.cfg),
        "dlm": {"m": _to_list(model.state.dlm.m), "C": _to_list(model.state.dlm.C)},
        "beta_conn": {
            ch: {bucket: {"a": float(post.a), "b": float(post.b)} for bucket, post in buckets.items()}
            for ch, buckets in model.state.conn_rate.items()
        },
        "beta_close": {ch: {"a": float(post.a), "b": float(post.b)} for ch, post in model.state.close_rate.items()},
        "nig_prem": {ch: {"m": float(post.m), "k": float(post.k), "a": float(post.a), "b": float(post.b)} for ch, post in model.state.prem_per_contract.items()},
        "meta": {"last_date": model.state.last_date, "t_index": int(model.state.t_index)},
    }
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_adaptive_state(model_cls: Any, conf_cls: Any, path: str | Path) -> Any:
    """Load AdaptiveSCM state from JSON and return a model instance."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    conf = conf_cls(**raw["config"])
    model = model_cls(conf)

    model.state.dlm.m = np.array(raw["dlm"]["m"], dtype=float)
    model.state.dlm.C = np.array(raw["dlm"]["C"], dtype=float)

    for ch, buckets in raw["beta_conn"].items():
        for bucket, v in buckets.items():
            model.state.conn_rate[ch][bucket].a = float(v["a"])
            model.state.conn_rate[ch][bucket].b = float(v["b"])
    for ch, v in raw["beta_close"].items():
        model.state.close_rate[ch].a = float(v["a"])
        model.state.close_rate[ch].b = float(v["b"])
    for ch, v in raw["nig_prem"].items():
        model.state.prem_per_contract[ch].m = float(v["m"])
        model.state.prem_per_contract[ch].k = float(v["k"])
        model.state.prem_per_contract[ch].a = float(v["a"])
        model.state.prem_per_contract[ch].b = float(v["b"])

    model.state.last_date = raw.get("meta", {}).get("last_date")
    model.state.t_index = int(raw.get("meta", {}).get("t_index", 0))
    return model
