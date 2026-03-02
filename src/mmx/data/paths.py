from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from mmx.config.settings import Settings

@dataclass(frozen=True)
class Paths:
    base_dir: Path
    data_raw: Path
    data_curated: Path
    mart: Path
    logs: Path
    artifacts: Path
    docs_src: Path
    docs_build: Path

def resolve_paths(settings: Settings) -> Paths:
    base = settings.base_dir
    return Paths(
        base_dir=base,
        data_raw=base/'data/raw',
        data_curated=base/'data/curated',
        mart=base/'data/curated/mart',
        logs=base/'logs',
        artifacts=base/'artifacts',
        docs_src=base/'docs/src',
        docs_build=base/'docs/build',
    )
