from __future__ import annotations
# MMX_SYS_PATH_GUARD: ensure repo root/src are importable when running this script directly
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / 'src'
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import os
import json
import ast
from mmx.config.settings import load_settings
from mmx.data.paths import resolve_paths
from mmx.governance.registry import load_registry
from mmx.usecases.evaluate_shadow import evaluate_shadow

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="MMX Shadow Evaluation (AI vs Human) 실행 스크립트")
    ap.add_argument('--ai-plan-json', required=True, help="AI 플랜(JSON 문자열 또는 JSON 파일 경로)")
    ap.add_argument('--human-plan-json', required=True, help="Human 플랜(JSON 문자열 또는 JSON 파일 경로)")
    ap.add_argument('--model-version', default=None, help="평가에 사용할 모델 버전(미지정 시 production 사용)")
    args = ap.parse_args()

    def _load_json(v: str):
        """JSON 문자열 또는 JSON 파일 경로를 로드합니다.

        지원 입력 형태:
        - JSON 문자열(표준 JSON: double quotes)
        - JSON 파일 경로
        - (호환) 파이썬 dict/list repr 문자열(single quotes)  ※ 쉘 파이프라인 실수 방어
        """

        v = (v or "").strip()
        if not v:
            return {}

        # file path
        if (v.startswith("/") or v.startswith("./") or v.endswith(".json")) and os.path.exists(v):
            with open(v, "r", encoding="utf-8") as f:
                return json.load(f)

        # JSON string
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            # fallback: python literal (e.g., {'a': 1})
            try:
                return ast.literal_eval(v)
            except Exception as e:
                raise ValueError(f"Invalid json input: {e}") from e

    s = load_settings()
    p = resolve_paths(s)
    reg = load_registry(p)

    model_version = args.model_version or reg.production_version
    if model_version in (None, '', 'none'):
        raise SystemExit('No model version available. Promote a model first (or pass --model-version).')

    ai = _load_json(args.ai_plan_json)
    hu = _load_json(args.human_plan_json)

    res = evaluate_shadow(p, model_version, ai, hu, policy_lambda=s.policy_lambda)

    # Persist for dashboard
    out_dir = p.artifacts / 'evaluations/shadow'
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as _pd
    out_path = out_dir / f"shadow_{model_version}_{_pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(res.__dict__, ensure_ascii=False, indent=2), encoding='utf-8')

    print(out_path)
