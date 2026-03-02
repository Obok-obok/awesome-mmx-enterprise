from __future__ import annotations
# MMX_SYS_PATH_GUARD: ensure repo root/src are importable when running this script directly
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
from datetime import date

import yaml

from mmx.config.settings import load_settings
from mmx.data.paths import resolve_paths
from mmx.domain.sim.channel_params import ChannelParams, FunnelParams, PremiumParams, SpendToLeadsParams
from mmx.domain.sim.scenario_params import AllocationRuleParams, ScenarioParams, SeasonalityParams
from mmx.usecases.run_backtest import RunBacktestCommand, RunBacktestUsecase


def _parse_date(s: str) -> date:
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def _load_channel_params(path: Path) -> dict[str, ChannelParams]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: dict[str, ChannelParams] = {}
    for ch, cfg in obj.get("channels", {}).items():
        stl = cfg["spend_to_leads"]
        fun = cfg["funnel"]
        pre = cfg["premium"]
        out[str(ch)] = ChannelParams(
            channel=str(ch),
            spend_to_leads=SpendToLeadsParams(
                scale=float(stl["scale"]),
                adstock_theta=float(stl["adstock_theta"]),
                hill_alpha=float(stl["hill_alpha"]),
                hill_k=float(stl["hill_k"]),
                noise_sigma=float(stl.get("noise_sigma", 0.15)),
            ),
            funnel=FunnelParams(
                p_attempt=float(fun["p_attempt"]),
                p_connect=float(fun["p_connect"]),
                p_contract=float(fun["p_contract"]),
                lag_kernel=list(fun.get("lag_kernel", [1.0])),
            ),
            premium=PremiumParams(
                per_contract_mean=float(pre["per_contract_mean"]),
                per_contract_logn_sigma=float(pre.get("per_contract_logn_sigma", 0.35)),
            ),
        )
    return out


def _load_scenario(path: Path) -> ScenarioParams:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    sc = obj.get("scenario", {})
    tb = sc.get("total_budget", {})
    seas = tb.get("seasonality", {})
    rule = sc.get("allocation_rule", {})
    return ScenarioParams(
        base_daily_budget=float(tb.get("base_daily", 12_000_000.0)),
        seasonality=SeasonalityParams(yearly_amp=float(seas.get("yearly_amp", 0.12)), weekly_amp=float(seas.get("weekly_amp", 0.08))),
        allocation_rule=AllocationRuleParams(
            type=str(rule.get("type", "proportional_to_trailing_leads")),
            trailing_days=int(rule.get("trailing_days", 7)),
            smoothing_gamma=float(rule.get("smoothing_gamma", 0.35)),
            min_share=float(rule.get("min_share", 0.06)),
            max_share=float(rule.get("max_share", 0.30)),
        ),
    )


if __name__ == "__main__":
    s = load_settings()
    p = resolve_paths(s)

    parser = argparse.ArgumentParser(description="MMX 백테스트 원커맨드 실행")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "prod"])
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default="2026-03-31")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channels", type=str, default="카카오,토스,구글,메타,네이버,쿠팡")
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--lambda", dest="policy_lambda", type=float, default=s.policy_lambda)
    parser.add_argument("--backend", type=str, default=s.inference_backend)
    parser.add_argument("--method", type=str, default=s.inference_method)
    parser.add_argument("--n-samples", type=int, default=1200)
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--raw-root", type=str, default=str(p.data_raw))
    parser.add_argument("--channel-config", type=str, default=str(_REPO_ROOT / "configs/sim/backtest_channels.yaml"))
    parser.add_argument("--scenario-config", type=str, default=str(_REPO_ROOT / "configs/sim/backtest_scenarios.yaml"))
    args = parser.parse_args()

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    generate = not bool(args.no_generate)

    ch_params = _load_channel_params(Path(args.channel_config)) if generate else None
    sc_params = _load_scenario(Path(args.scenario_config)) if generate else None

    uc = RunBacktestUsecase(paths=p, settings=s)
    res = uc.execute(
        RunBacktestCommand(
            mode=str(args.mode),
            start_date=_parse_date(args.start),
            end_date=_parse_date(args.end),
            seed=int(args.seed),
            channels=channels,
            test_days=int(args.test_days),
            n_posterior_samples=int(args.n_samples),
            policy_lambda=float(args.policy_lambda),
            backend=str(args.backend),
            method=str(args.method),
            generate_data=bool(generate),
            raw_root_dir=Path(args.raw_root),
            channel_params=ch_params,
            scenario_params=sc_params,
        )
    )
    print(f"Backtest completed: run_id={res.run_id}")
    print(f"Artifacts: {p.artifacts / 'backtests' / 'latest'}")
