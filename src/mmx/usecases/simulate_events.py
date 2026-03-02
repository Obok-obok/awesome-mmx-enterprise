from __future__ import annotations

"""Synthetic raw event generator for backtesting.

This generator writes raw append-only event CSVs that conform to the existing
mart builder (mmx.usecases.build_mart). The produced events can be aggregated
into data/mart/daily_channel_fact.csv and used to train/test the SEM engine.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from mmx.domain.sim.channel_params import ChannelParams
from mmx.domain.sim.scenario_params import ScenarioParams
from mmx.domain.sim.transforms import adstock, hill_saturation


def _kst_iso(d: date, hh: int = 12) -> str:
    """Return an ISO date string for mart compatibility.

    Important:
    ---------
    pandas.to_datetime may infer a single strict format when the column contains
    many values. If we mix date-only strings (YYYY-MM-DD) with datetime strings
    (YYYY-MM-DDTHH:MM:SS), 일부 환경에서 후자가 NaT로 강제될 수 있습니다.

    Therefore we emit date-only strings for *all* event time columns.
    """

    _ = hh  # reserved for future, keep signature stable
    return d.isoformat()


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


@dataclass(frozen=True)
class SimulateEventsCommand:
    start_date: date
    end_date: date
    seed: int
    channels: Sequence[str]
    raw_root_dir: Path
    channel_params: Mapping[str, ChannelParams]
    scenario_params: ScenarioParams
    source_system: str = "simulator"


@dataclass(frozen=True)
class SimulateEventsResult:
    spend_events_written: int
    funnel_events_written: int
    premium_events_written: int
    min_occurred_at: str
    max_occurred_at: str


class SimulateEventsUsecase:
    """Generate raw event CSVs for spend and funnel metrics."""

    def execute(self, cmd: SimulateEventsCommand) -> SimulateEventsResult:
        root = Path(cmd.raw_root_dir)
        (root / "events/spend").mkdir(parents=True, exist_ok=True)
        (root / "events/leads").mkdir(parents=True, exist_ok=True)
        (root / "events/call_attempt").mkdir(parents=True, exist_ok=True)
        (root / "events/call_connected").mkdir(parents=True, exist_ok=True)
        (root / "events/contracts").mkdir(parents=True, exist_ok=True)
        (root / "events/premium").mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(int(cmd.seed))

        days: list[date] = []
        cur = cmd.start_date
        while cur <= cmd.end_date:
            days.append(cur)
            cur += timedelta(days=1)

        channels = list(cmd.channels)

        # Spend allocation state
        prev_spend: dict[str, float] = {c: cmd.scenario_params.base_daily_budget / max(1, len(channels)) for c in channels}
        trailing_leads: dict[str, list[int]] = {c: [] for c in channels}
        carry: dict[str, float] = {c: 0.0 for c in channels}

        def seasonality_multiplier(d: date) -> float:
            # Simple yearly + weekly seasonality.
            day_of_year = int(d.strftime("%j"))
            wday = d.weekday()  # 0=Mon
            ya = float(cmd.scenario_params.seasonality.yearly_amp)
            wa = float(cmd.scenario_params.seasonality.weekly_amp)
            yearly = 1.0 + ya * float(np.sin(2.0 * np.pi * day_of_year / 365.25))
            weekly = 1.0 + wa * float(np.sin(2.0 * np.pi * (wday + 1) / 7.0))
            return float(max(0.1, yearly * weekly))

        def total_budget(d: date) -> float:
            return float(cmd.scenario_params.base_daily_budget) * seasonality_multiplier(d)

        # Accumulators for writing event rows.
        spend_rows: list[dict] = []
        lead_rows: list[dict] = []
        attempt_rows: list[dict] = []
        connected_rows: list[dict] = []
        contract_rows: list[dict] = []
        premium_rows: list[dict] = []

        rule = cmd.scenario_params.allocation_rule

        for d in days:
            B = total_budget(d)

            # Human-like allocation based on trailing leads size.
            scores = {}
            for c in channels:
                hist = trailing_leads[c][-int(rule.trailing_days) :] if int(rule.trailing_days) > 0 else []
                scores[c] = float(sum(hist)) + 1.0
            ssum = float(sum(scores.values())) if scores else 1.0
            shares = {c: float(scores[c]) / ssum for c in channels}
            # min/max share clamp then renormalize
            for c in channels:
                shares[c] = _clamp(shares[c], float(rule.min_share), float(rule.max_share))
            ssum2 = float(sum(shares.values())) if shares else 1.0
            shares = {c: float(v) / ssum2 for c, v in shares.items()}

            gamma = float(rule.smoothing_gamma)
            for c in channels:
                target = float(B) * float(shares[c])
                prev = float(prev_spend[c])
                spend = (1.0 - gamma) * prev + gamma * target
                prev_spend[c] = float(max(0.0, spend))

            # Generate funnel counts per channel.
            for c in channels:
                p = cmd.channel_params[c]
                stl = p.spend_to_leads
                f = p.funnel
                prem = p.premium

                # Update adstock carry.
                carry[c] = float(prev_spend[c]) + float(stl.adstock_theta) * float(carry[c])
                sat = hill_saturation(np.asarray([carry[c]], dtype=float), alpha=float(stl.hill_alpha), k=float(stl.hill_k))[0]
                mult = seasonality_multiplier(d)
                noise = float(rng.lognormal(mean=0.0, sigma=float(stl.noise_sigma)))
                leads_mean = float(stl.scale) * float(sat) * float(mult) * float(noise)
                leads = int(rng.poisson(lam=max(0.0, leads_mean)))

                # Keep per-day funnel monotonicity to satisfy mart data-quality gate.
                attempts = int(rng.binomial(n=leads, p=_clamp(float(f.p_attempt), 0.0, 1.0)))
                connected = int(rng.binomial(n=attempts, p=_clamp(float(f.p_connect), 0.0, 1.0)))
                contracts = int(rng.binomial(n=connected, p=_clamp(float(f.p_contract), 0.0, 1.0)))

                # Premium per contract (lognormal); set mu so that mean matches per_contract_mean.
                sigma = float(max(1e-9, prem.per_contract_logn_sigma))
                mu = float(np.log(max(1.0, prem.per_contract_mean))) - 0.5 * sigma * sigma
                if contracts > 0:
                    ppc = rng.lognormal(mean=mu, sigma=sigma, size=int(contracts))
                    premium_amounts = [float(x) for x in ppc]
                else:
                    premium_amounts = []

                # Append spend event (one row/day/channel).
                spend_rows.append(
                    {
                        "event_id": str(uuid4()),
                        "event_time": _kst_iso(d, hh=9),
                        "channel": c,
                        "spend": float(prev_spend[c]),
                    }
                )

                # Append lead/attempt/connected/contract events as id-level rows.
                occ = _kst_iso(d, hh=10)
                for _ in range(leads):
                    lead_rows.append({"lead_id": str(uuid4()), "occurred_at": occ, "channel": c})
                for _ in range(attempts):
                    attempt_rows.append({"attempt_id": str(uuid4()), "attempt_at": _kst_iso(d, hh=11), "channel": c})
                for _ in range(connected):
                    connected_rows.append({"call_id": str(uuid4()), "connected_at": _kst_iso(d, hh=12), "channel": c})
                for _ in range(contracts):
                    contract_rows.append({"contract_id": str(uuid4()), "contract_signed_at": _kst_iso(d, hh=14), "channel": c})

                # Premium: one row per contract for realistic variance.
                for amt in premium_amounts:
                    premium_rows.append(
                        {
                            "premium_event_id": str(uuid4()),
                            "premium_recognized_at": _kst_iso(d, hh=16),
                            "channel": c,
                            "premium_amount": float(amt),
                        }
                    )

                trailing_leads[c].append(int(leads))

        # Write in daily chunks to keep file sizes manageable.
        # We write single CSV per event type (simple, deterministic).
        def write_csv(path: Path, rows: list[dict], columns: Sequence[str]) -> None:
            df = pd.DataFrame(rows)
            if df.empty:
                df = pd.DataFrame(columns=list(columns))
            else:
                df = df[list(columns)]
            df.to_csv(path, index=False)

        write_csv(root / "events/spend" / "spend_simulated.csv", spend_rows, ["event_id", "event_time", "channel", "spend"])
        write_csv(root / "events/leads" / "leads_simulated.csv", lead_rows, ["lead_id", "occurred_at", "channel"])
        write_csv(root / "events/call_attempt" / "call_attempt_simulated.csv", attempt_rows, ["attempt_id", "attempt_at", "channel"])
        write_csv(root / "events/call_connected" / "call_connected_simulated.csv", connected_rows, ["call_id", "connected_at", "channel"])
        write_csv(root / "events/contracts" / "contracts_simulated.csv", contract_rows, ["contract_id", "contract_signed_at", "channel"])
        write_csv(root / "events/premium" / "premium_simulated.csv", premium_rows, ["premium_event_id", "premium_recognized_at", "channel", "premium_amount"])

        return SimulateEventsResult(
            spend_events_written=int(len(spend_rows)),
            funnel_events_written=int(len(lead_rows) + len(attempt_rows) + len(connected_rows) + len(contract_rows)),
            premium_events_written=int(len(premium_rows)),
            min_occurred_at=_kst_iso(cmd.start_date, hh=0),
            max_occurred_at=_kst_iso(cmd.end_date, hh=23),
        )
