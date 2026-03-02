from __future__ import annotations

"""Backtest artifact repository.

Persist and load backtest runs under artifacts/backtests.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from mmx.domain.backtest.schemas import BacktestResult, OverallMetrics


@dataclass(frozen=True)
class BacktestRepoConfig:
    root_dir: Path


class BacktestArtifactRepository:
    def __init__(self, cfg: BacktestRepoConfig) -> None:
        self._root = Path(cfg.root_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return self._root / run_id

    def latest_dir(self) -> Path:
        return self._root / "latest"

    def write(self, result: BacktestResult) -> Path:
        d = self.run_dir(result.run_id)
        d.mkdir(parents=True, exist_ok=True)

        (d / "config.json").write_text(json.dumps(result.config, ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "splits.json").write_text(json.dumps(result.splits, ensure_ascii=False, indent=2), encoding="utf-8")

        overall = {
            "premium": {
                "wape": result.overall.premium_wape,
                "mae": result.overall.premium_mae,
                "rmse": result.overall.premium_rmse,
            },
            "coverage": {
                "p10_p90": result.overall.coverage_p10_p90,
                "p05_p95": result.overall.coverage_p05_p95,
            },
            "ra_premium": {
                "lambda": float(result.config.get("policy", {}).get("lambda", 0.0)),
                "pred_mean": result.overall.pred_ra_mean,
                "pred_std": result.overall.pred_ra_std,
                "pred_ra": result.overall.pred_ra_value,
            },
        }
        (d / "metrics_overall.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")

        result.timeseries_daily.to_csv(d / "timeseries_daily.csv", index=False)
        result.metrics_by_channel.to_csv(d / "metrics_by_channel.csv", index=False)
        # Optional but recommended dashboard artifacts
        result.plan_compare_by_channel.to_csv(d / "plan_compare_by_channel.csv", index=False)
        (d / "plan_compare_totals.json").write_text(
            json.dumps(result.plan_compare_totals, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Monthly optimization artifacts (for monthly budget operations)
        result.plan_compare_monthly_by_channel.to_csv(d / "plan_compare_monthly_by_channel.csv", index=False)
        (d / "plan_compare_monthly_totals.json").write_text(
            json.dumps(result.plan_compare_monthly_totals, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result.period_summary.to_csv(d / "period_summary.csv", index=False)
        (d / "lineage.json").write_text(json.dumps(result.lineage, ensure_ascii=False, indent=2), encoding="utf-8")
        return d

    def promote_latest(self, run_id: str) -> None:
        src = self.run_dir(run_id)
        if not src.exists():
            raise FileNotFoundError(str(src))
        dst = self.latest_dir()
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    def load(self, run_id: str) -> BacktestResult:
        d = self.run_dir(run_id)
        return self._load_dir(d)

    def load_latest(self) -> BacktestResult:
        d = self.latest_dir()
        return self._load_dir(d)

    def _load_dir(self, d: Path) -> BacktestResult:
        if not d.exists():
            raise FileNotFoundError(str(d))
        config = json.loads((d / "config.json").read_text(encoding="utf-8"))
        splits = json.loads((d / "splits.json").read_text(encoding="utf-8"))
        overall_obj = json.loads((d / "metrics_overall.json").read_text(encoding="utf-8"))
        prem = overall_obj.get("premium", {})
        cov = overall_obj.get("coverage", {})
        ra = overall_obj.get("ra_premium", {})
        overall = OverallMetrics(
            premium_wape=float(prem.get("wape", 0.0)),
            premium_mae=float(prem.get("mae", 0.0)),
            premium_rmse=float(prem.get("rmse", 0.0)),
            coverage_p10_p90=float(cov.get("p10_p90", 0.0)),
            coverage_p05_p95=float(cov.get("p05_p95", 0.0)),
            pred_ra_mean=float(ra.get("pred_mean", 0.0)),
            pred_ra_std=float(ra.get("pred_std", 0.0)),
            pred_ra_value=float(ra.get("pred_ra", 0.0)),
        )
        ts = pd.read_csv(d / "timeseries_daily.csv")
        by = pd.read_csv(d / "metrics_by_channel.csv")
        plan_by = pd.read_csv(d / "plan_compare_by_channel.csv") if (d / "plan_compare_by_channel.csv").exists() else pd.DataFrame()
        plan_totals = json.loads((d / "plan_compare_totals.json").read_text(encoding="utf-8")) if (d / "plan_compare_totals.json").exists() else {}
        plan_by_m = (
            pd.read_csv(d / "plan_compare_monthly_by_channel.csv")
            if (d / "plan_compare_monthly_by_channel.csv").exists()
            else pd.DataFrame()
        )
        plan_totals_m = (
            json.loads((d / "plan_compare_monthly_totals.json").read_text(encoding="utf-8"))
            if (d / "plan_compare_monthly_totals.json").exists()
            else {}
        )
        period_summary = pd.read_csv(d / "period_summary.csv") if (d / "period_summary.csv").exists() else pd.DataFrame()
        lineage = json.loads((d / "lineage.json").read_text(encoding="utf-8"))
        return BacktestResult(
            run_id=str(config.get("run_id", d.name)),
            config=config,
            splits=splits,
            overall=overall,
            timeseries_daily=ts,
            metrics_by_channel=by,
            plan_compare_by_channel=plan_by,
            plan_compare_totals=plan_totals,
            plan_compare_monthly_by_channel=plan_by_m,
            plan_compare_monthly_totals=plan_totals_m,
            period_summary=period_summary,
            lineage=lineage,
        )
