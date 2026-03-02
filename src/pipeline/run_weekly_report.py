from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from src.ops.reporting import WeeklyReportConfig, generate_weekly_pdf
from src.ops.emailer import send_email_with_attachment

logger = logging.getLogger(__name__)


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/mmx.yaml") -> None:
    cfg = _load_cfg(config_path)
    out_root = Path(cfg["data"]["out_dir"]) / "ops"

    perf_path = out_root / "performance_daily.csv"
    reco_hist_path = out_root / "reco_history.csv"
    ab_results_path = out_root / "ab_results_weekly.csv"
    geo_mm_results_path = out_root / "geo_mm_results_weekly.csv"

    # Determine week_end as last available date
    if not perf_path.exists():
        raise FileNotFoundError(f"Missing perf log: {perf_path}")

    perf = pd.read_csv(perf_path)
    perf["date"] = pd.to_datetime(perf["date"])
    week_end = str(perf["date"].max().date())

    report_dir = Path(cfg["data"]["out_dir"]) / "reports"
    report_path = report_dir / f"weekly_report_{week_end}.pdf"

    generate_weekly_pdf(
        report_path,
        perf_path=perf_path,
        reco_hist_path=reco_hist_path,
        ab_results_path=ab_results_path if ab_results_path.exists() else None,
        geo_mm_results_path=geo_mm_results_path if geo_mm_results_path.exists() else None,
        cfg=WeeklyReportConfig(days=int(cfg.get("reporting", {}).get("weekly_days", 7))),
    )

    # optional email
    mail_cfg = cfg.get("reporting", {}).get("email", {})
    if bool(mail_cfg.get("enabled", False)):
        subject = mail_cfg.get("subject", "MMX Ops Weekly Report") + f" ({week_end})"
        body = mail_cfg.get(
            "body",
            "Attached is the weekly Ops Monitoring report (actual vs predicted, drift, budget recommendation, A/B holdout if enabled).",
        )
        send_email_with_attachment(subject, body, report_path)
        logger.info("Weekly report emailed: %s", report_path)
    else:
        logger.info("Weekly report generated (email disabled): %s", report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
