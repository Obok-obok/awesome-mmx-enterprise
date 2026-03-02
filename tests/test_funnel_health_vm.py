from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from apps.dashboard.viewmodels.funnel_health_vm import build_funnel_health_vm


SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _to_dict(vm) -> dict:
    return {
        "period": {
            "start": vm.period_start,
            "end": vm.period_end,
            "compare_mode": vm.compare_mode,
            "compare_label": vm.compare_label,
            "prev_start": vm.prev_start,
            "prev_end": vm.prev_end,
        },
        "cards": [
            {"title": c.title, "value": c.value, "comp": c.comp, "subtitle": c.subtitle}
            for c in vm.cards
        ],
    }


def test_funnel_health_is_rate_first_snapshot() -> None:
    """Regression guard: Funnel Health main values must be rates/efficiencies, not raw amounts."""

    # Build a full mart with 2 periods (current + previous) so comparison is computable.
    # Selected range is within a single month -> MoM policy.
    # Previous period: 2025-12-08~2025-12-14 (shifted by -1 month)
    # Current period:  2026-01-08~2026-01-14
    dates_prev = pd.date_range("2025-12-08", "2025-12-14", freq="D")
    dates_cur = pd.date_range("2026-01-08", "2026-01-14", freq="D")

    def _rows(dates, spend, leads, attempts, connected, contracts, premium):
        return pd.DataFrame(
            {
                "date": dates,
                "channel": ["구글"] * len(dates),
                "spend": [spend] * len(dates),
                "leads": [leads] * len(dates),
                "call_attempt": [attempts] * len(dates),
                "call_connected": [connected] * len(dates),
                "contracts": [contracts] * len(dates),
                "premium": [premium] * len(dates),
            }
        )

    prev = _rows(dates_prev, 10_000_000, 1000, 800, 240, 60, 60_000_000)
    cur = _rows(dates_cur, 12_000_000, 1200, 960, 300, 75, 82_500_000)
    mart_full = pd.concat([prev, cur], ignore_index=True)
    mart_filtered = cur.copy()

    vm = build_funnel_health_vm(mart_full=mart_full, mart_filtered=mart_filtered)

    payload = _to_dict(vm)

    # Quick semantic guards
    assert payload["cards"][0]["title"] == "Lead Efficiency"
    assert "/ 1M" in payload["cards"][0]["value"]  # not raw leads
    assert payload["cards"][1]["value"].endswith("%")
    assert payload["cards"][2]["value"].endswith("%")
    assert payload["cards"][3]["value"].endswith("%")

    snap_path = SNAPSHOT_DIR / "funnel_health_vm.json"
    if not snap_path.exists():
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snap_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise AssertionError("Snapshot file created. Re-run tests.")

    expected = json.loads(snap_path.read_text(encoding="utf-8"))
    assert payload == expected
