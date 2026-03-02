from __future__ import annotations
# MMX_SYS_PATH_GUARD: ensure repo root/src are importable when running this script directly
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / 'src'
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mmx.config.settings import load_settings
from mmx.data.paths import resolve_paths
from mmx.usecases.build_mart import build_daily_channel_fact

if __name__ == '__main__':
    s = load_settings()
    p = resolve_paths(s)
    res = build_daily_channel_fact(p, s)
    print(res)
