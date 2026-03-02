from __future__ import annotations

import argparse

from src.data.generate_sample import make_sample_inputs, SampleSpec
from src.data.raw_to_inputs import build_inputs_from_raw
from src.pipeline.run_all import run_all


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=False)

    p_run = sub.add_parser("run", help="Run end-to-end pipeline")
    p_run.add_argument("--config", default="configs/mmx.yaml")

    p_sample = sub.add_parser("make-sample", help="Generate demo inputs")
    p_sample.add_argument("--out", default="data")
    p_sample.add_argument("--days", type=int, default=180)
    p_sample.add_argument("--seed", type=int, default=42)

    p_build = sub.add_parser(
        "build-inputs",
        help="Build daily modeling inputs from raw lead/TM/policy tables (customer_id mapping with 30-day window)",
    )
    p_build.add_argument("--raw-dir", required=True, help="Directory containing raw_leads.csv, raw_tm_calls.csv, raw_policies.csv")
    p_build.add_argument(
        "--spend-campaign",
        required=False,
        help="Optional campaign spend file with columns: date,channel,campaign_id,message_type,spend",
    )
    p_build.add_argument("--out", default="data/inputs", help="Output directory")
    p_build.add_argument("--window-days", type=int, default=30, help="Attribution window in days")
    p_build.add_argument("--dedupe-hours", type=int, default=24, help="Deduplicate leads within N hours per customer+campaign+message")

    args = p.parse_args()
    if args.cmd in (None, "run"):
        run_all(getattr(args, "config", "configs/mmx.yaml"))
        return

    if args.cmd == "make-sample":
        spec = SampleSpec(n_days=args.days, seed=args.seed)
        make_sample_inputs(args.out, spec)
        return

    if args.cmd == "build-inputs":
        build_inputs_from_raw(
            raw_dir=args.raw_dir,
            out_dir=args.out,
            spend_campaign_path=args.spend_campaign,
            window_days=args.window_days,
            dedupe_hours=args.dedupe_hours,
        )
        return


if __name__ == "__main__":
    main()
