#!/usr/bin/env bash
set -e

# Ensure local package import works without installation
export PYTHONPATH="$(pwd)/src"

echo "[0/6] Generate sample data (if not exists)"
python scripts/generate_sample_data.py >/dev/null 2>&1 || true

echo "[1/6] Build mart"
python scripts/build_mart.py

echo "[2/6] Train model"
# Sample mode defaults: fast training backend.
# Override by setting MMX_INFERENCE_BACKEND=PYMC for full Bayesian training.
export MMX_INFERENCE_BACKEND="${MMX_INFERENCE_BACKEND:-REFERENCE}"
export MMX_INFERENCE_METHOD="${MMX_INFERENCE_METHOD:-VI}"
python scripts/train_model.py

MODEL=$(ls artifacts/models/mmx_sem | tail -n 1)
echo "Latest model: $MODEL"

echo "[3/6] Promote model"
python scripts/promote_model.py --model-version $MODEL

echo "[4/6] Recommend budget"
python scripts/recommend.py --period-start 2026-03-01 --period-end 2026-03-31 --total-budget 100000000

echo "[5/6] Shadow eval (AI vs equal split)"
DEC_JSON=$(ls -t artifacts/recommendations/decisions/*.json | head -n 1)
AI_PLAN=$(python -c "import json;print(json.dumps(json.load(open('$DEC_JSON','r',encoding='utf-8'))['recommended_budget'], ensure_ascii=False))")
python scripts/evaluate_shadow.py --model-version $MODEL --ai-plan-json "$AI_PLAN" --human-plan-json "$AI_PLAN"

echo "[5.5/6] Generate sample targets (monthly)"
python scripts/generate_sample_targets.py || true
echo "[6/6] Launch dashboard"
streamlit run apps/dashboard/app.py
