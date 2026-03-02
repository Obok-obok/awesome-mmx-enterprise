# MMX_SYS_PATH_GUARD: ensure repo root/src are importable when running this script directly
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / 'src'
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse

from mmx.config.settings import load_settings
from mmx.data.paths import resolve_paths
from mmx.governance.registry import promote_to_production

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / 'artifacts' / 'models' / 'mmx_sem'


def get_latest_model_version() -> str:
    versions = sorted([pp.name for pp in MODEL_DIR.iterdir() if pp.is_dir()])
    if not versions:
        raise RuntimeError(f'No model versions found under: {MODEL_DIR}')
    return versions[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-version', type=str, help='Model version to promote')
    parser.add_argument('--latest', action='store_true', help='Promote latest model version')
    parser.add_argument('--reason', type=str, default='manual promotion', help='Promotion reason (audit log)')
    args = parser.parse_args()

    if args.latest:
        model_version = get_latest_model_version()
    elif args.model_version:
        model_version = args.model_version
    else:
        raise SystemExit('Provide --model-version or --latest')

    target = MODEL_DIR / model_version
    if not target.exists():
        raise FileNotFoundError(f'Model version not found: {model_version}')

    # 1) Official registry used by recommend.py
    s = load_settings()
    paths = resolve_paths(s)
    promote_to_production(paths, model_version=model_version, reason=args.reason)

    # 2) Backward-compatible marker file (legacy)
    (MODEL_DIR / 'promoted.txt').write_text(model_version, encoding='utf-8')

    print(f'promoted {model_version}')


if __name__ == '__main__':
    main()
