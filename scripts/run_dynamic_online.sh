#!/usr/bin/env bash
set -euo pipefail

python -m src.pipeline.run_dynamic_online

echo "[OK] Dynamic online outputs written to out/dynamic/"
