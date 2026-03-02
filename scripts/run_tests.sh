#!/usr/bin/env bash
set -euo pipefail

python -m py_compile app_mmx_dashboard.py mmx/*.py src/*.py
pytest -q

echo "OK: compile + tests passed"
