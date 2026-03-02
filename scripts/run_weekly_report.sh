#!/usr/bin/env bash
set -euo pipefail

python -m src.pipeline.run_ab_weekly
python -m src.pipeline.run_geo_weekly
python -m src.pipeline.run_weekly_report
