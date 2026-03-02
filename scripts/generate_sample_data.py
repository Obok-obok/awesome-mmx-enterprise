# MMX_SYS_PATH_GUARD: ensure repo root/src are importable when running this script directly
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / 'src'
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

"""Generate synthetic raw event data for demo.

This script intentionally generates **enough history** so that the dashboard can
compute:
  - 전월 대비 (previous period)
  - 전년 동기 대비 (YoY)

The dashboard's default demo period is 2026-03, so we generate from 2025-01.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

base = Path("data/raw/events")
channels = ["Search", "Social"]

# Ensure MoM + YoY are available for the demo period.
start = os.environ.get("MMX_SAMPLE_START", "2025-01-01")
end = os.environ.get("MMX_SAMPLE_END", "2026-03-31")
dates = pd.date_range(start, end, freq="D")

seed = int(os.environ.get("MMX_SAMPLE_SEED", "0"))
np.random.seed(seed)

for folder in ["spend","leads","call_attempt","call_connected","contracts","premium"]:
    (base/folder).mkdir(parents=True, exist_ok=True)

spend_rows = []
lead_rows = []
att_rows = []
con_rows = []
ctr_rows = []
prm_rows = []


def _seasonality_multiplier(ts: pd.Timestamp) -> float:
    """Simple annual seasonality (insurance-like) with mild monthly effect."""
    # Month seasonality: Q1 stronger, summer softer
    month = ts.month
    month_mult = {
        1: 1.10,
        2: 1.05,
        3: 1.08,
        4: 1.00,
        5: 0.98,
        6: 0.95,
        7: 0.92,
        8: 0.94,
        9: 0.99,
        10: 1.02,
        11: 1.06,
        12: 1.12,
    }[month]
    # Day-of-week effect: weekends lower
    dow = ts.dayofweek
    dow_mult = 0.85 if dow >= 5 else 1.0
    return float(month_mult * dow_mult)


for d in dates:
    base_mult = _seasonality_multiplier(d)
    for ch in channels:
        # Channel baseline differences
        ch_mult = 1.15 if ch == "Search" else 0.95

        # Spend is the driver; keep within a practical range but with seasonality.
        s = np.random.uniform(1_500, 6_500) * base_mult * ch_mult
        spend_rows.append({"event_id": f"{ch}_{d.date()}_s", "event_time": d, "channel": ch, "spend": s})

        leads = int(max(0, s / 220 + np.random.normal(0, 1)))
        for i in range(leads):
            lead_rows.append({"lead_id":f"{ch}_{d}_l{i}","occurred_at":d,"channel":ch})

        # Funnel rates with mild drift/noise so MoM/YoY differences are visible.
        att_rate = np.clip(0.78 + np.random.normal(0, 0.02), 0.60, 0.95)
        con_rate = np.clip(0.48 + np.random.normal(0, 0.03), 0.20, 0.80)
        ctr_rate = np.clip(0.22 + np.random.normal(0, 0.03), 0.05, 0.60)

        att = int(leads * att_rate)
        for i in range(att):
            att_rows.append({"attempt_id":f"{ch}_{d}_a{i}","attempt_at":d,"channel":ch})
        con = int(att * con_rate)
        for i in range(con):
            con_rows.append({"call_id":f"{ch}_{d}_c{i}","connected_at":d,"channel":ch})
        ctr = int(con * ctr_rate)
        for i in range(ctr):
            ctr_rows.append({"contract_id":f"{ch}_{d}_ctr{i}","contract_signed_at":d,"channel":ch})

        # Premium: per contract value varies by channel and season.
        premium_per_contract = int((1_000 + (200 if ch == "Search" else 0)) * base_mult)
        prm = int(ctr)
        for i in range(prm):
            prm_rows.append(
                {
                    "premium_event_id": f"{ch}_{d}_p{i}",
                    "premium_recognized_at": d,
                    "channel": ch,
                    "premium_amount": premium_per_contract,
                }
            )

pd.DataFrame(spend_rows).to_csv(base/"spend/sample.csv",index=False)
pd.DataFrame(lead_rows).to_csv(base/"leads/sample.csv",index=False)
pd.DataFrame(att_rows).to_csv(base/"call_attempt/sample.csv",index=False)
pd.DataFrame(con_rows).to_csv(base/"call_connected/sample.csv",index=False)
pd.DataFrame(ctr_rows).to_csv(base/"contracts/sample.csv",index=False)
pd.DataFrame(prm_rows).to_csv(base / "premium/sample.csv", index=False)
print(f"Sample data generated: {start} ~ {end} (seed={seed})")
