from __future__ import annotations

"""End-to-end backtest runner.

Pipeline:
  (optional) simulate raw events
  -> build mart
  -> time split train/test
  -> fit posterior on train
  -> posterior predictive on test (using actual spend)
  -> evaluate & write artifacts
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from mmx.config.settings import Settings
from mmx.data.paths import Paths
from mmx.domain.backtest.schemas import BacktestResult, OverallMetrics
from mmx.domain.backtest.split import DateRange, HoldoutLastNDaysSplit, make_holdout_last_n_days
from mmx.domain.sim.channel_params import ChannelParams
from mmx.domain.sim.scenario_params import ScenarioParams
from mmx.engine.sem.inference import PosteriorSummary, fit_posterior, posterior_predict_premium, posterior_predict_funnel
from mmx.infra.artifacts.backtest_repo import BacktestArtifactRepository, BacktestRepoConfig
from mmx.optimization.constraints import default_constraints
from mmx.optimization.solver import solve_slsqp
from mmx.usecases.build_mart import build_daily_channel_fact


def _col(df: pd.DataFrame, *candidates: str) -> str:
    """Return the first existing column name from candidates.

    The mart schema historically used both canonical funnel names
    (attempts/connected) and event-derived names (call_attempt/call_connected).
    This helper makes backtests robust across both schemas.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Missing required column. Tried: {list(candidates)}. Available: {list(df.columns)}"
    )


def _canonicalize_mart(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize mart column names to the backtest/dashboard canonical schema."""
    if df.empty:
        return df
    df = df.copy()
    rename_map = {
        _col(df, 'call_attempt', 'attempts'): 'attempts',
        _col(df, 'call_connected', 'connected'): 'connected',
    }
    rename_map = {k: v for k, v in rename_map.items() if k != v}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df
from mmx.usecases.simulate_events import SimulateEventsCommand, SimulateEventsUsecase


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _percentile(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if x.size else 0.0


def _wape(pred: np.ndarray, actual: np.ndarray) -> float:
    denom = float(np.sum(np.abs(actual)))
    if denom <= 0:
        return 0.0
    return float(np.sum(np.abs(pred - actual)) / denom)


def _mae(pred: np.ndarray, actual: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float(np.mean(np.abs(pred - actual)))


def _rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def _load_posterior_summary(paths: Paths, model_version: str) -> PosteriorSummary:
    p = paths.artifacts / f"models/mmx_sem/{model_version}/posterior_summary.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    return PosteriorSummary(model_version=obj["model_version"], channel_params=obj["channel_params"], globals=obj["globals"])


def _save_posterior_summary(paths: Paths, posterior: PosteriorSummary, backend: str, method: str) -> str:
    vdir = paths.artifacts / f"models/mmx_sem/{posterior.model_version}"
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "posterior_summary.json").write_text(
        json.dumps(
            {
                "model_version": posterior.model_version,
                "channel_params": posterior.channel_params,
                "globals": posterior.globals,
                "backend": backend,
                "method": method,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(vdir)


@dataclass(frozen=True)
class RunBacktestCommand:
    mode: str  # demo|prod
    start_date: date
    end_date: date
    seed: int
    channels: Sequence[str]
    test_days: int
    n_posterior_samples: int
    policy_lambda: float
    backend: str
    method: str
    generate_data: bool
    raw_root_dir: Path
    channel_params: Mapping[str, ChannelParams] | None = None
    scenario_params: ScenarioParams | None = None


class RunBacktestUsecase:
    def __init__(self, *, paths: Paths, settings: Settings) -> None:
        self._paths = paths
        self._settings = settings

    def execute(self, cmd: RunBacktestCommand) -> BacktestResult:
        created_at = datetime.now().astimezone().isoformat()
        run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + f"_seed{cmd.seed}"

        if cmd.generate_data:
            if cmd.channel_params is None or cmd.scenario_params is None:
                raise ValueError("channel_params and scenario_params are required when generate_data=True")
            SimulateEventsUsecase().execute(
                SimulateEventsCommand(
                    start_date=cmd.start_date,
                    end_date=cmd.end_date,
                    seed=cmd.seed,
                    channels=cmd.channels,
                    raw_root_dir=cmd.raw_root_dir,
                    channel_params=cmd.channel_params,
                    scenario_params=cmd.scenario_params,
                )
            )

        # Build mart from raw events (single source of truth)
        mart_res = build_daily_channel_fact(self._paths, self._settings)
        mart_path = Path(mart_res.output_path)

        df = _canonicalize_mart(pd.read_csv(mart_path))
        if df.empty:
            raise ValueError("Mart is empty; cannot run backtest")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()
        df = df[(df["date"].dt.date >= cmd.start_date) & (df["date"].dt.date <= cmd.end_date)].copy()
        if df.empty:
            raise ValueError("Mart has no rows in requested date range")

        full = DateRange(cmd.start_date, cmd.end_date)
        split = make_holdout_last_n_days(full, HoldoutLastNDaysSplit(test_days=int(cmd.test_days)))

        train_df = df[(df["date"].dt.date >= split.train.start) & (df["date"].dt.date <= split.train.end)].copy()
        test_df = df[(df["date"].dt.date >= split.test.start) & (df["date"].dt.date <= split.test.end)].copy()
        if train_df.empty or test_df.empty:
            raise ValueError("Train or test split is empty")

        posterior, _trace = fit_posterior(train_df.assign(date=train_df["date"].dt.date.astype(str)), backend=cmd.backend, method=cmd.method)
        _save_posterior_summary(self._paths, posterior, backend=cmd.backend, method=cmd.method)
        posterior_loaded = _load_posterior_summary(self._paths, posterior.model_version)

        # Daily test predictions using actual daily spend as plan; n_days=1
        dates = pd.date_range(start=pd.Timestamp(split.test.start), end=pd.Timestamp(split.test.end), freq="D")
        rows = []
        lam = float(cmd.policy_lambda)

        # Deterministic premium scale for optimization.
        # Avoid Monte Carlo noise inside SLSQP by using closed-form LogNormal moments.
        mu = float(posterior_loaded.globals.get("mu_log_premium_per_contract", 0.0))
        sig = float(posterior_loaded.globals.get("sigma_log_premium_per_contract", 0.0))
        # E[X] for LogNormal(mu, sig)
        ppc_mean = float(np.exp(mu + 0.5 * sig * sig))
        # Std[X] for LogNormal(mu, sig)
        ppc_std = float(np.sqrt((np.exp(sig * sig) - 1.0) * np.exp(2.0 * mu + sig * sig)))
        # Risk-adjusted multiplier (>=0 in most realistic cases; clamp to small positive)
        ppc_ra = float(max(1e-9, ppc_mean - lam * ppc_std))
        for d0 in dates:
            day = d0.date()
            sub = test_df[test_df["date"].dt.date == day]
            if sub.empty:
                continue
            plan = {str(r["channel"]): float(r["spend"]) for _, r in sub.iterrows()}
            samples = posterior_predict_premium(
                df_daily=train_df.assign(date=train_df["date"].dt.date.astype(str)),
                posterior=posterior_loaded,
                budget_plan=plan,
                period_start=pd.Timestamp(day),
                n_days=1,
                warm_start=True,
                warm_start_days=None,
                n_samples=int(cmd.n_posterior_samples),
                seed=int(cmd.seed) + 11,
            )
            actual_premium = float(sub["premium"].sum())
            actual_spend = float(sub["spend"].sum())
            pred_mean = float(np.mean(samples))
            pred_std = float(np.std(samples))
            p10 = _percentile(samples, 10)
            p50 = _percentile(samples, 50)
            p90 = _percentile(samples, 90)
            p05 = _percentile(samples, 5)
            p95 = _percentile(samples, 95)
            rows.append(
                {
                    "date": day.isoformat(),
                    "actual_spend": actual_spend,
                    "actual_premium": actual_premium,
                    "pred_mean": pred_mean,
                    "pred_std": pred_std,
                    "pred_p10": p10,
                    "pred_p50": p50,
                    "pred_p90": p90,
                    "pred_p05": p05,
                    "pred_p95": p95,
                    "pred_ra": float(pred_mean - lam * pred_std),
                }
            )

        ts = pd.DataFrame(rows)
        if ts.empty:
            raise ValueError("No daily predictions produced")

        pred = ts["pred_mean"].to_numpy(dtype=float)
        actual = ts["actual_premium"].to_numpy(dtype=float)

        wape = _wape(pred, actual)
        mae = _mae(pred, actual)
        rmse = _rmse(pred, actual)
        cov_10_90 = float(np.mean((actual >= ts["pred_p10"].to_numpy()) & (actual <= ts["pred_p90"].to_numpy())))
        cov_05_95 = float(np.mean((actual >= ts["pred_p05"].to_numpy()) & (actual <= ts["pred_p95"].to_numpy())))

        # Per-channel aggregated prediction over test window
        by_rows = []
        n_days = int((pd.Timestamp(split.test.end) - pd.Timestamp(split.test.start)).days) + 1
        train_daily = train_df.assign(date=train_df["date"].dt.date.astype(str))
        for ch in sorted(set(cmd.channels)):
            act_sp = float(test_df.loc[test_df["channel"] == ch, "spend"].sum())
            act_pr = float(test_df.loc[test_df["channel"] == ch, "premium"].sum())
            plan = {ch: act_sp}
            samples = posterior_predict_premium(
                df_daily=train_daily,
                posterior=posterior_loaded,
                budget_plan=plan,
                period_start=pd.Timestamp(split.test.start),
                n_days=n_days,
                warm_start=True,
                n_samples=int(max(500, cmd.n_posterior_samples // 2)),
                seed=int(cmd.seed) + 31,
            )
            pm = float(np.mean(samples))
            ps = float(np.std(samples))
            ch_wape = float(abs(pm - act_pr) / act_pr) if act_pr > 0 else 0.0
            bias = float(pm - act_pr)
            cov = 1.0 if (_percentile(samples, 10) <= act_pr <= _percentile(samples, 90)) else 0.0
            by_rows.append(
                {
                    "channel": ch,
                    "actual_spend": act_sp,
                    "actual_premium": act_pr,
                    "pred_mean": pm,
                    "pred_std": ps,
                    "wape": ch_wape,
                    "bias": bias,
                    "coverage_p10_p90": cov,
                }
            )

        by = pd.DataFrame(by_rows)

        # ------------------------------
        # Plan comparison (Actual allocation vs Optimized allocation)
        # ------------------------------
        total_budget = float(test_df["spend"].sum())
        channels = sorted(set(cmd.channels))
        plan_actual: Dict[str, float] = {ch: float(test_df.loc[test_df["channel"] == ch, "spend"].sum()) for ch in channels}

        constraints = default_constraints(channels=channels, total_budget=total_budget)
        n_days_test = int((pd.Timestamp(split.test.end) - pd.Timestamp(split.test.start)).days) + 1
        train_daily = train_df.assign(date=train_df["date"].dt.date.astype(str))
        lam = float(cmd.policy_lambda)

        def objective(plan: Dict[str, float]) -> float:
            # Use funnel expected contracts (deterministic) * risk-adjusted PPC multiplier.
            # This is stable and differentiable enough for SLSQP.
            s = posterior_predict_funnel(
                posterior=posterior_loaded,
                budget_plan=plan,
                df_daily=train_daily,
                period_start=pd.Timestamp(split.test.start),
                n_days=n_days_test,
                warm_start=True,
                n_samples=600,
                seed=int(cmd.seed) + 777,
            )
            ctr = float(np.mean(s.get("contracts")))
            return float(ctr * ppc_ra)

        solve = solve_slsqp(channels=channels, constraints=constraints, objective_fn=objective)
        plan_opt = solve.budget

        actual_totals = {
            "spend": float(test_df["spend"].sum()),
            "leads": float(test_df["leads"].sum()),
            "attempts": float(test_df["attempts"].sum()),
            "connected": float(test_df["connected"].sum()),
            "contracts": float(test_df["contracts"].sum()),
            "premium": float(test_df["premium"].sum()),
        }

        def pred_funnel_mean_window(
            plan: Dict[str, float],
            *,
            seed_off: int,
            period_start: pd.Timestamp,
            n_days: int,
        ) -> Dict[str, float]:
            """Posterior predictive mean funnel totals for a specific window.

            NOTE: Monthly backtests MUST call this with the monthly window.
            If we accidentally reuse the full test window for every month,
            each monthly value becomes inflated and Σ(months) != 전체(Test).
            """
            s = posterior_predict_funnel(
                posterior=posterior_loaded,
                budget_plan=plan,
                df_daily=train_daily,
                period_start=pd.Timestamp(period_start),
                n_days=int(n_days),
                warm_start=True,
                n_samples=int(max(800, cmd.n_posterior_samples)),
                seed=int(cmd.seed) + int(seed_off),
            )
            return {k: float(np.mean(v)) for k, v in s.items()}

        def _sat_ratio_from_posterior(ch: str, spend_total: float, n_days: int) -> float:
            """Estimate saturation ratio in [0,1] for the channel at the given spend.

            We approximate steady-state adstock under constant daily spend and compute
            the Hill saturation component: a^h / (a^h + ec50^h).
            """
            p = posterior_loaded.channel_params.get(ch)
            if not p or spend_total <= 0 or n_days <= 0:
                return 0.0
            decay = float(p["decay"])
            ec50 = float(p["ec50"])
            hill_p = float(p.get("hill", 1.0))

            daily = float(spend_total) / float(max(1, int(n_days)))
            # steady-state geometric adstock for constant input: x / (1 - decay)
            a_ss = daily / max(1e-9, (1.0 - decay))
            # pure saturation component (0..1)
            num = a_ss ** hill_p
            den = num + (ec50 ** hill_p)
            if den <= 0:
                return 0.0
            return float(np.clip(num / den, 0.0, 1.0))

        def _pred_premium_mean_single_channel(ch: str, spend_total: float, *, period_start: pd.Timestamp, n_days: int, seed_off: int) -> float:
            if spend_total <= 0:
                return 0.0
            s = posterior_predict_funnel(
                posterior=posterior_loaded,
                budget_plan={ch: float(spend_total)},
                df_daily=train_daily,
                period_start=period_start,
                n_days=n_days,
                warm_start=True,
                n_samples=int(max(600, cmd.n_posterior_samples // 2)),
                seed=int(cmd.seed) + seed_off,
            )
            return float(np.mean(s["premium"]))

        pred_actual_plan = pred_funnel_mean_window(
            plan_actual,
            seed_off=121,
            period_start=pd.Timestamp(split.test.start),
            n_days=n_days_test,
        )
        pred_opt_plan = pred_funnel_mean_window(
            plan_opt,
            seed_off=131,
            period_start=pd.Timestamp(split.test.start),
            n_days=n_days_test,
        )

        # Per-channel diagnostics: ROI + saturation estimate (for backtest comparability)
        actual_premium_by_channel = {ch: float(test_df.loc[test_df["channel"] == ch, "premium"].sum()) for ch in channels}
        pred_premium_by_channel_on_actual: Dict[str, float] = {}
        pred_premium_by_channel_on_opt: Dict[str, float] = {}
        sat_ratio_actual: Dict[str, float] = {}
        sat_ratio_opt: Dict[str, float] = {}

        for i, ch in enumerate(channels):
            s_act = float(plan_actual.get(ch, 0.0))
            s_opt = float(plan_opt.get(ch, 0.0))
            pred_premium_by_channel_on_actual[ch] = _pred_premium_mean_single_channel(
                ch, s_act, period_start=pd.Timestamp(split.test.start), n_days=n_days_test, seed_off=411 + i
            )
            pred_premium_by_channel_on_opt[ch] = _pred_premium_mean_single_channel(
                ch, s_opt, period_start=pd.Timestamp(split.test.start), n_days=n_days_test, seed_off=511 + i
            )
            sat_ratio_actual[ch] = _sat_ratio_from_posterior(ch, s_act, n_days=n_days_test)
            sat_ratio_opt[ch] = _sat_ratio_from_posterior(ch, s_opt, n_days=n_days_test)

        plan_compare_by_channel = pd.DataFrame(
            [
                {
                    "channel": ch,
                    "actual_spend": float(plan_actual.get(ch, 0.0)),
                    "actual_share": float(plan_actual.get(ch, 0.0) / total_budget) if total_budget > 0 else 0.0,
                    "actual_premium": float(actual_premium_by_channel.get(ch, 0.0)),
                    "actual_roi": float(actual_premium_by_channel.get(ch, 0.0) / plan_actual.get(ch, 1.0)) if float(plan_actual.get(ch, 0.0)) > 0 else 0.0,
                    "pred_premium_on_actual": float(pred_premium_by_channel_on_actual.get(ch, 0.0)),
                    "pred_roi_on_actual": float(pred_premium_by_channel_on_actual.get(ch, 0.0) / plan_actual.get(ch, 1.0)) if float(plan_actual.get(ch, 0.0)) > 0 else 0.0,
                    "sat_ratio_est_actual": float(sat_ratio_actual.get(ch, 0.0)),
                    "opt_spend": float(plan_opt.get(ch, 0.0)),
                    "opt_share": float(plan_opt.get(ch, 0.0) / total_budget) if total_budget > 0 else 0.0,
                    "pred_premium_on_opt": float(pred_premium_by_channel_on_opt.get(ch, 0.0)),
                    "pred_roi_on_opt": float(pred_premium_by_channel_on_opt.get(ch, 0.0) / plan_opt.get(ch, 1.0)) if float(plan_opt.get(ch, 0.0)) > 0 else 0.0,
                    "sat_ratio_est_opt": float(sat_ratio_opt.get(ch, 0.0)),
                }
                for ch in channels
            ]
        )

        plan_compare_totals: Dict[str, Any] = {
            "objective": {"lambda": lam, "type": "risk_adjusted"},
            "window": {"start": split.test.start.isoformat(), "end": split.test.end.isoformat(), "n_days": n_days_test},
            "solver": {"success": bool(solve.success), "message": str(solve.message)},
            "actual": actual_totals,
            "pred_on_actual_plan": pred_actual_plan,
            "pred_on_opt_plan": pred_opt_plan,
        }

        # ------------------------------
        # Monthly optimization (monthly budget operations)
        # ------------------------------
        test_daily = test_df.copy()
        test_daily["date"] = pd.to_datetime(test_daily["date"], errors="coerce")
        test_daily = test_daily.dropna(subset=["date"]).sort_values("date")
        test_daily["period"] = test_daily["date"].dt.to_period("M").astype(str)

        monthly_rows: List[Dict[str, Any]] = []
        monthly_totals: Dict[str, Any] = {"objective": {"lambda": lam, "type": "risk_adjusted"}, "periods": {}}
        period_summary_rows: List[Dict[str, Any]] = []

        for period, df_p in test_daily.groupby("period"):
            total_budget_p = float(df_p["spend"].sum())
            if total_budget_p <= 0:
                continue

            # Period boundaries
            p_start = pd.Period(period, freq="M").start_time
            p_end = pd.Period(period, freq="M").end_time
            # Clip to actual test range
            p_start = max(p_start, pd.Timestamp(split.test.start))
            p_end = min(p_end, pd.Timestamp(split.test.end) + pd.Timedelta(days=0))
            n_days_p = int((p_end.normalize() - p_start.normalize()).days) + 1

            plan_actual_p: Dict[str, float] = {ch: float(df_p.loc[df_p["channel"] == ch, "spend"].sum()) for ch in channels}
            constraints_p = default_constraints(channels=channels, total_budget=total_budget_p)

            def objective_p(plan: Dict[str, float]) -> float:
                samples_p = posterior_predict_premium(
                    df_daily=train_daily,
                    posterior=posterior_loaded,
                    budget_plan=plan,
                    period_start=p_start,
                    n_days=n_days_p,
                    warm_start=True,
                    n_samples=int(max(500, cmd.n_posterior_samples // 2)),
                    seed=int(cmd.seed) + 201 + int(p_start.month),
                )
                return float(np.mean(samples_p) - lam * np.std(samples_p))

            solve_p = solve_slsqp(channels=channels, constraints=constraints_p, objective_fn=objective_p)
            plan_opt_p = solve_p.budget

            actual_totals_p = {
                "spend": float(df_p["spend"].sum()),
                "leads": float(df_p["leads"].sum()),
                "attempts": float(df_p["attempts"].sum()),
                "connected": float(df_p["connected"].sum()),
                "contracts": float(df_p["contracts"].sum()),
                "premium": float(df_p["premium"].sum()),
            }

            pred_actual_p = pred_funnel_mean_window(
                plan_actual_p,
                seed_off=221 + int(p_start.month),
                period_start=p_start,
                n_days=n_days_p,
            )
            pred_opt_p = pred_funnel_mean_window(
                plan_opt_p,
                seed_off=231 + int(p_start.month),
                period_start=p_start,
                n_days=n_days_p,
            )

            for ch in channels:
                actual_premium_ch = float(df_p.loc[df_p["channel"] == ch, "premium"].sum())
                spend_act_ch = float(plan_actual_p.get(ch, 0.0))
                spend_opt_ch = float(plan_opt_p.get(ch, 0.0))
                pred_prem_act_ch = _pred_premium_mean_single_channel(
                    ch,
                    spend_act_ch,
                    period_start=p_start,
                    n_days=n_days_p,
                    seed_off=621 + int(p_start.month) * 10 + channels.index(ch),
                )
                pred_prem_opt_ch = _pred_premium_mean_single_channel(
                    ch,
                    spend_opt_ch,
                    period_start=p_start,
                    n_days=n_days_p,
                    seed_off=721 + int(p_start.month) * 10 + channels.index(ch),
                )
                monthly_rows.append(
                    {
                        "period": period,
                        "channel": ch,
                        "actual_spend": float(plan_actual_p.get(ch, 0.0)),
                        "actual_share": float(plan_actual_p.get(ch, 0.0) / total_budget_p) if total_budget_p > 0 else 0.0,
                        "actual_premium": actual_premium_ch,
                        "actual_roi": float(actual_premium_ch / spend_act_ch) if spend_act_ch > 0 else 0.0,
                        "pred_premium_on_actual": float(pred_prem_act_ch),
                        "pred_roi_on_actual": float(pred_prem_act_ch / spend_act_ch) if spend_act_ch > 0 else 0.0,
                        "sat_ratio_est_actual": float(_sat_ratio_from_posterior(ch, spend_act_ch, n_days=n_days_p)),
                        "opt_spend": float(plan_opt_p.get(ch, 0.0)),
                        "opt_share": float(plan_opt_p.get(ch, 0.0) / total_budget_p) if total_budget_p > 0 else 0.0,
                        "pred_premium_on_opt": float(pred_prem_opt_ch),
                        "pred_roi_on_opt": float(pred_prem_opt_ch / spend_opt_ch) if spend_opt_ch > 0 else 0.0,
                        "sat_ratio_est_opt": float(_sat_ratio_from_posterior(ch, spend_opt_ch, n_days=n_days_p)),
                    }
                )

            monthly_totals["periods"][period] = {
                "window": {"start": str(p_start.date()), "end": str(p_end.date()), "n_days": int(n_days_p)},
                "solver": {"success": bool(solve_p.success), "message": str(solve_p.message)},
                "budget": {"total": float(total_budget_p)},
                "actual": actual_totals_p,
                "pred_on_actual_plan": pred_actual_p,
                "pred_on_opt_plan": pred_opt_p,
            }

            period_summary_rows.append(
                {
                    "period": period,
                    "actual_spend": float(actual_totals_p["spend"]),
                    "actual_premium": float(actual_totals_p["premium"]),
                    "pred_premium_on_actual_plan": float(pred_actual_p.get("premium", 0.0)),
                    "pred_premium_on_opt_plan": float(pred_opt_p.get("premium", 0.0)),
                }
            )

        plan_compare_monthly_by_channel = pd.DataFrame(monthly_rows)
        plan_compare_monthly_totals = monthly_totals
        period_summary = pd.DataFrame(period_summary_rows).sort_values("period") if period_summary_rows else pd.DataFrame()

        overall = OverallMetrics(
            premium_wape=float(wape),
            premium_mae=float(mae),
            premium_rmse=float(rmse),
            coverage_p10_p90=float(cov_10_90),
            coverage_p05_p95=float(cov_05_95),
            pred_ra_mean=float(ts["pred_mean"].sum()),
            pred_ra_std=float(np.sqrt(float(np.sum(ts["pred_std"].to_numpy(dtype=float) ** 2)))),
            pred_ra_value=float(ts["pred_ra"].sum()),
        )

        config: Dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "seed": int(cmd.seed),
            "channels": list(cmd.channels),
            "date_range": {"start": cmd.start_date.isoformat(), "end": cmd.end_date.isoformat()},
            "split": {"type": "holdout_last_n_days", "test_days": int(cmd.test_days)},
            "model": {
                "backend": cmd.backend,
                "inference": cmd.method,
                "model_version": posterior_loaded.model_version,
                "n_posterior_samples": int(cmd.n_posterior_samples),
            },
            "policy": {"lambda": float(cmd.policy_lambda)},
            # Persist SEM / latent-quality guardrail parameters for auditability & UI debugging.
            "sem_globals": dict(getattr(posterior_loaded, "globals", {}) or {}),
            "paths": {
                "mart_path": str(mart_path),
                "raw_root": str(cmd.raw_root_dir),
            },
        }
        splits = {
            "train": {"start": split.train.start.isoformat(), "end": split.train.end.isoformat()},
            "test": {"start": split.test.start.isoformat(), "end": split.test.end.isoformat()},
        }

        lineage = {
            "mart_file": {
                "path": str(mart_path),
                "sha256": _sha256_file(mart_path) if mart_path.exists() else "",
                "rows": int(len(df)),
                "min_date": str(df["date"].min().date()),
                "max_date": str(df["date"].max().date()),
            },
            "model_version": posterior_loaded.model_version,
        }

        result = BacktestResult(
            run_id=run_id,
            config=config,
            splits=splits,
            overall=overall,
            timeseries_daily=ts,
            metrics_by_channel=by,
            plan_compare_by_channel=plan_compare_by_channel,
            plan_compare_totals=plan_compare_totals,
            plan_compare_monthly_by_channel=plan_compare_monthly_by_channel,
            plan_compare_monthly_totals=plan_compare_monthly_totals,
            period_summary=period_summary,
            lineage=lineage,
        )

        repo = BacktestArtifactRepository(BacktestRepoConfig(root_dir=self._paths.artifacts / "backtests"))
        repo.write(result)
        repo.promote_latest(run_id)
        return result
