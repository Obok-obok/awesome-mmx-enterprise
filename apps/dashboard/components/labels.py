from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


@dataclass(frozen=True)
class Labels:
    metrics: Dict[str, str]
    derived: Dict[str, str]
    axes: Dict[str, str]
    scenarios: Dict[str, str]

    def name(self, key: str) -> str:
        return self.metrics.get(key) or self.derived.get(key) or self.axes.get(key) or self.scenarios.get(key) or key


def load_labels(yaml_path: Path) -> Labels:
    obj = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return Labels(
        metrics=obj.get("metrics", {}),
        derived=obj.get("derived", {}),
        axes=obj.get("axes", {}),
        scenarios=obj.get("scenarios", {}),
    )


def rename_columns_for_display(df, labels: Labels):
    mapping = {c: labels.name(c) for c in df.columns}
    return df.rename(columns=mapping)
