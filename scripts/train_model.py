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
from mmx.usecases.train_model import train_model

if __name__ == '__main__':
    import argparse
    s = load_settings()
    p = resolve_paths(s)

    parser = argparse.ArgumentParser(description="MMX 모델 학습 (VI/MCMC). 환경변수 설정을 CLI로 덮어쓸 수 있습니다.")
    parser.add_argument("--backend", type=str, default=s.inference_backend, help="추론 백엔드 (PYMC 또는 REFERENCE)")
    parser.add_argument("--method", type=str, default=s.inference_method, help="추론 방법 (VI 또는 MCMC)")
    parser.add_argument("--reporting-delay", type=str, default=s.reporting_delay, help="Reporting delay (ON/OFF)")
    parser.add_argument("--reporting-delay-max", type=int, default=s.reporting_delay_max, help="Reporting delay 최대 일수")
    args = parser.parse_args()

    res = train_model(
        p,
        backend=args.backend,
        method=args.method,
        reporting_delay=args.reporting_delay,
        reporting_delay_max=args.reporting_delay_max,
    )
    print(res)
