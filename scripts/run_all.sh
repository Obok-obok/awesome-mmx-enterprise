#!/usr/bin/env bash
set -e
python -m src.cli run --config configs/mmx.yaml
echo "Outputs saved to ./out"
