from __future__ import annotations
# MMX_SYS_PATH_GUARD: ensure repo root/src are importable when running this script directly
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from mmx.config.settings import load_settings
from mmx.data.paths import resolve_paths


def main() -> None:
    """샘플용 월별 Premium 목표치를 생성합니다."""
    s = load_settings()
    p = resolve_paths(s)

    mart = p.mart / "daily_channel_fact.csv"
    if not mart.exists():
        raise SystemExit("Mart가 없습니다. 먼저 scripts/build_mart.py를 실행하세요.")

    df = pd.read_csv(mart)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby("month", as_index=False)["premium"].sum()
    # 샘플 목표: 실적의 1.1배(보수적)로 설정
    monthly["target_premium"] = (monthly["premium"] * 1.1).round(0)
    out = Path(p.data_curated) / "targets" / "monthly_targets.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    monthly[["month", "target_premium"]].to_csv(out, index=False)
    print(f"sample targets written: {out}")


if __name__ == "__main__":
    main()
