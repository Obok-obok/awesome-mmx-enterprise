from __future__ import annotations
# MMX_SYS_PATH_GUARD: ensure repo root/src are importable when running via uvicorn or direct import
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / 'src'
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from functools import lru_cache
from mmx.config.settings import load_settings
from mmx.data.paths import resolve_paths

@lru_cache(maxsize=1)
def get_paths():
    s = load_settings()
    return resolve_paths(s)
