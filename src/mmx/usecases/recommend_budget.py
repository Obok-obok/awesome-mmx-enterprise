from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import json, hashlib
import numpy as np
import pandas as pd

from mmx.data.paths import Paths
from mmx.data.io_csv import append_csv, file_lock
from mmx.domain.types import Policy, ObjectiveMode
from mmx.domain.entities import Decision
from mmx.engine.sem.inference import PosteriorSummary, posterior_predict_premium, posterior_predict_funnel
from mmx.engine.explain.curves import build_curves
from mmx.engine.explain.reasons import reasons_for_channel
from mmx.engine.sem.saturation import saturation_ratio
from mmx.engine.sem.adstock import half_life
from mmx.optimization.constraints import Constraints, default_constraints
from mmx.optimization.solver import solve_slsqp
from mmx.optimization.objective import compute
from mmx.governance.rollout import apply_rollout_policy


def _hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode('utf-8')).hexdigest()[:12]

def _load_posterior(paths: Paths, model_version: str) -> PosteriorSummary:
    p = paths.artifacts / f'models/mmx_sem/{model_version}/posterior_summary.json'
    obj = json.loads(p.read_text(encoding='utf-8'))
    return PosteriorSummary(model_version=obj['model_version'], channel_params=obj['channel_params'], globals=obj['globals'])

@dataclass(frozen=True)
class RecommendResult:
    decision: Decision
    decision_json_path: str

def recommend_budget(
    paths: Paths,
    period_start: str,
    period_end: str,
    total_budget: float,
    model_version: str,
    policy: Policy,
    experiment_id: Optional[str] = None,
    prev_budget_by_channel: Optional[Dict[str, float]] = None,
    ramp_up_cap_share: float = 0.0,
    ramp_up_cap_abs: float = 0.0,
) -> RecommendResult:
    lock = paths.logs / 'decisions/decision.lock'
    with file_lock(lock):
        posterior = _load_posterior(paths, model_version)
        channels = sorted(list(posterior.channel_params.keys()))

        cons = default_constraints(channels, total_budget=total_budget)
        if prev_budget_by_channel is not None and policy.policy_delta > 0:
            cons = Constraints(
                total_budget=float(total_budget),
                min_by_channel=cons.min_by_channel,
                max_by_channel=cons.max_by_channel,
                prev_by_channel=prev_budget_by_channel,
                delta=float(policy.policy_delta),
                ramp_up_cap_share=float(ramp_up_cap_share),
                ramp_up_cap_abs=float(ramp_up_cap_abs),
            )

        policy_hash = _hash({'mode': policy.objective_mode.value, 'lambda': policy.policy_lambda, 'delta': policy.policy_delta})
        constraints_hash = _hash({'total': total_budget, 'min': cons.min_by_channel, 'max': cons.max_by_channel, 'delta': cons.delta})

        mart = paths.mart / 'daily_channel_fact.csv'
        df = pd.read_csv(mart) if mart.exists() else pd.DataFrame(columns=['date','channel','spend','leads','call_attempt','call_connected','contracts','premium'])

        ps = pd.to_datetime(period_start)
        pe = pd.to_datetime(period_end)
        n_days = int((pe - ps).days) + 1
        n_days = max(1, n_days)

        # Resolve a single warm-start window (days) for auditability and reproducibility.
        # We use the maximum warmup across channels: max(max_lag, adstock_warmup_days(decay)).
        warm_days_resolved = 0
        for ch in channels:
            p = posterior.channel_params[ch]
            try:
                w_la = np.array(json.loads(p['w_lead_to_attempt']), dtype=float)
                w_cc = np.array(json.loads(p['w_connected_to_contract']), dtype=float)
                max_lag = max(len(w_la) - 1, len(w_cc) - 1)
            except Exception:
                max_lag = 0
            decay = float(p.get('decay', 0.8))
            # Approx warmup: 4 * half-life (in days). Fallback to lag if ill-defined.
            ad_warm = int(np.ceil(4.0 * (np.log(0.5) / np.log(decay)))) if (0 < decay < 1) else 0
            warm_days_resolved = int(max(warm_days_resolved, max_lag, max(0, ad_warm)))

        def objective_fn(plan: Dict[str, float]) -> float:
            """Deterministic objective (v5.4.1): include Latent Quality effect on PPC.

            We avoid Monte Carlo noise inside SLSQP by:
            - computing deterministic contracts totals (from SEM mean path)
            - computing PPC moments in closed-form, with mu shifted by latent quality
              (period-average q_bar) via gamma_ppc_lq.
            """
            lam = float(policy.policy_lambda) if policy.objective_mode == ObjectiveMode.RISK_ADJUSTED else 0.0
            mu0 = float(posterior.globals.get("mu_log_premium_per_contract", 0.0))
            sig = float(posterior.globals.get("sigma_log_premium_per_contract", 0.0))
            gamma_ppc = float(posterior.globals.get("gamma_ppc_lq", 0.0))
            lq_clip_low = float(posterior.globals.get("lq_clip_low", -1.0))
            lq_clip_high = float(posterior.globals.get("lq_clip_high", 1.0))
            lq_alpha = float(posterior.globals.get("lq_alpha_rate", 0.5))

            # Closed-form PPC moments given mu_eff.
            def _ppc_moments(mu_eff: float) -> tuple[float, float]:
                mean = float(np.exp(mu_eff + 0.5 * sig * sig))
                var = float((np.exp(sig * sig) - 1.0) * np.exp(2.0 * mu_eff + sig * sig))
                return mean, float(np.sqrt(max(0.0, var)))

            # Deterministic contracts + quality average per channel.
            from mmx.engine.sem.adstock import adstock_geometric
            from mmx.engine.sem.saturation import hill
            from mmx.engine.sem.lag import apply_lag_kernel
            from mmx.engine.sem.inference import _build_spend_series, _adstock_warmup_days, _sigmoid  # type: ignore
            import json as _json

            exp_premium = 0.0
            var_premium = 0.0

            for ch in channels:
                spend_total = float(plan.get(ch, 0.0))
                p = posterior.channel_params.get(ch)
                if not p:
                    continue
                decay = float(p.get("decay", 0.8))
                ec50 = float(p.get("ec50", max(1.0, spend_total)))
                hill_p = float(p.get("hill", 1.0))
                alpha_lead = float(p.get("alpha_lead", 1.0))

                rate_att = float(p.get("rate_attempt_per_lead", 0.7))
                rate_con = float(p.get("rate_connected_per_attempt", 0.35))
                rate_ctr = float(p.get("rate_contract_per_connected", 0.2))

                lq_rho = float(p.get("lq_rho", 0.0))
                lq_beta = float(p.get("lq_beta_log1p_spend", 0.0))
                lq_last = float(p.get("lq_last_state", 0.0))
                lq_logit_att = float(p.get("lq_logit_base_attempt", float(np.log(rate_att / max(1e-9, 1-rate_att)))))
                lq_logit_con = float(p.get("lq_logit_base_connected", float(np.log(rate_con / max(1e-9, 1-rate_con)))))
                lq_logit_ctr = float(p.get("lq_logit_base_contract", float(np.log(rate_ctr / max(1e-9, 1-rate_ctr)))))

                w_la = np.array(_json.loads(p.get("w_lead_to_attempt", "[]")), dtype=float)
                w_cc = np.array(_json.loads(p.get("w_connected_to_contract", "[]")), dtype=float)
                if len(w_la) == 0:
                    w_la = np.array([1.0], dtype=float)
                if len(w_cc) == 0:
                    w_cc = np.array([1.0], dtype=float)
                max_lag = max(len(w_la) - 1, len(w_cc) - 1)
                pre = int(max(max_lag, _adstock_warmup_days(decay)))

                x_full = _build_spend_series(df, ch, ps, pre_days=pre, n_days=int(n_days), spend_total=float(spend_total))
                a_full = adstock_geometric(x_full, decay=decay)
                leads_full = hill(a_full, alpha=alpha_lead, ec50=ec50, hill=hill_p)

                q = np.zeros_like(x_full, dtype=float)
                q_prev = lq_last
                for i in range(len(x_full)):
                    q_prev = lq_rho * q_prev + lq_beta * float(np.log1p(max(0.0, x_full[i])))
                    q[i] = q_prev
                q = np.clip(q, lq_clip_low, lq_clip_high)
                q_win = q[-int(n_days):]
                q_bar = float(np.mean(q_win)) if len(q_win) else 0.0

                p_att = _sigmoid(lq_logit_att + lq_alpha * q)
                p_con = _sigmoid(lq_logit_con + lq_alpha * q)
                p_ctr = _sigmoid(lq_logit_ctr + lq_alpha * q)

                lag_leads = apply_lag_kernel(leads_full, w_la, max_lag=len(w_la) - 1)
                att_full = p_att * lag_leads
                con_full = p_con * att_full
                lag_con = apply_lag_kernel(con_full, w_cc, max_lag=len(w_cc) - 1)
                ctr_full = p_ctr * lag_con
                ctr_total = float(np.sum(ctr_full[-int(n_days):]))

                # Enforce causal monotonicity at aggregate level.
                ctr_total = float(max(0.0, ctr_total))

                mu_eff = float(mu0 + gamma_ppc * q_bar)
                m_ppc, s_ppc = _ppc_moments(mu_eff)

                exp_premium += ctr_total * m_ppc
                var_premium += ctr_total * (s_ppc ** 2)

            std_premium = float(np.sqrt(max(0.0, var_premium)))
            return float(exp_premium - lam * std_premium)

        sol = solve_slsqp(channels, cons, objective_fn)

        # Post-solve validation for auditability: constraint satisfaction / violations.
        sum_budget = float(sum(sol.budget.values()))
        sum_error = float(sum_budget - float(total_budget))
        min_violation = 0.0
        max_violation = 0.0
        for ch in channels:
            v = float(sol.budget.get(ch, 0.0))
            lo = float(cons.min_by_channel.get(ch, 0.0))
            hi = float(cons.max_by_channel.get(ch, float(total_budget)))
            min_violation = float(max(min_violation, lo - v))
            max_violation = float(max(max_violation, v - hi))

        delta_violation = 0.0
        ramp_violation = 0.0
        if cons.prev_by_channel and cons.delta > 0:
            prev = cons.prev_by_channel
            for ch in channels:
                p = float(prev.get(ch, 0.0))
                v = float(sol.budget.get(ch, 0.0))
                if p > 0:
                    lim = float(cons.delta) * p
                    delta_violation = float(max(delta_violation, v - (p + lim), (p - lim) - v))
                else:
                    cap_share = float(cons.ramp_up_cap_share) * float(total_budget)
                    cap_abs = float(cons.ramp_up_cap_abs)
                    cap = 0.0
                    if cap_abs > 0:
                        cap = cap_abs
                    if cap_share > 0:
                        cap = cap if cap > 0 else cap_share
                        cap = min(cap, cap_share)
                    if cap > 0:
                        ramp_violation = float(max(ramp_violation, v - cap))

        validation_report = {
            "sum_budget": sum_budget,
            "sum_error": sum_error,
            "min_violation": float(min_violation),
            "max_violation": float(max_violation),
            "delta_violation": float(max(0.0, delta_violation)),
            "ramp_violation": float(max(0.0, ramp_violation)),
        }

        # Use ONE posterior predictive draw set for BOTH decision metrics and funnel metrics.
        # This guarantees Decision.premium and FunnelForecast.premium match exactly.
        seed_main = 777
        funnel_samples = posterior_predict_funnel(
            posterior,
            sol.budget,
            df_daily=df,
            period_start=ps,
            n_days=n_days,
            warm_start=True,
            warm_start_days=warm_days_resolved,
            n_samples=2000,
            seed=seed_main,
        )
        premium_samples = funnel_samples["premium"]
        obj = compute(premium_samples, policy.policy_lambda if policy.objective_mode == ObjectiveMode.RISK_ADJUSTED else 0.0)

        # Explainability artifacts
        expl_dir = paths.artifacts / f'explainability/{model_version}'
        expl_dir.mkdir(parents=True, exist_ok=True)
        tables = build_curves(posterior.channel_params, max_spend=total_budget, n_days=n_days)
        tables['saturation_metrics'].to_csv(expl_dir/'saturation_metrics.csv', index=False)
        tables['response_curve'].to_csv(expl_dir/'response_curve.csv', index=False)
        try:
            tables['response_curve'].to_parquet(expl_dir/'response_curve.parquet', index=False)
        except Exception:
            pass
        tables['mroi_curve'].to_csv(expl_dir/'mroi_curve.csv', index=False)
        try:
            tables['mroi_curve'].to_parquet(expl_dir/'mroi_curve.parquet', index=False)
        except Exception:
            pass

        reasons: Dict[str, List[str]] = {}
        mroi_df = tables['mroi_curve']
        for ch in channels:
            spend = float(sol.budget[ch])
            p = posterior.channel_params[ch]
            ec50 = float(p['ec50'])
            sat = saturation_ratio(spend/float(n_days), ec50)
            sub = mroi_df[mroi_df['channel'] == ch]
            if len(sub):
                idx = (sub['spend'] - spend).abs().idxmin()
                mroi = float(sub.loc[idx, 'mroi'])
            else:
                mroi = 0.0
            hl = half_life(float(p['decay']))
            reasons[ch] = reasons_for_channel(ch, sat, mroi, hl)

        # Baseline (do nothing): equal allocation
        dn = {ch: total_budget/max(1, len(channels)) for ch in channels}
        seed_baseline = 2
        funnel_dn = posterior_predict_funnel(
            posterior,
            dn,
            df_daily=df,
            period_start=ps,
            n_days=n_days,
            warm_start=True,
            warm_start_days=warm_days_resolved,
            n_samples=2000,
            seed=seed_baseline,
        )
        prem_dn = funnel_dn["premium"]
        obj_dn = compute(prem_dn, policy.policy_lambda if policy.objective_mode == ObjectiveMode.RISK_ADJUSTED else 0.0)
        p_ai_better = float(np.mean(premium_samples > prem_dn))
        # Data coverage (0~1): proportion of expected (date,channel) rows present with any observed metric.
        data_coverage = 1.0
        try:
            dff_cov = df.copy()
            dff_cov['date'] = pd.to_datetime(dff_cov['date'])
            mm = dff_cov[(dff_cov['date'] >= ps) & (dff_cov['date'] <= pe) & (dff_cov['channel'].isin(channels))]
            expected_rows = float(n_days * max(1, len(channels)))
            if expected_rows > 0 and not mm.empty:
                metric_cols = [c for c in ['spend','leads','call_attempt','call_connected','contracts','premium'] if c in mm.columns]
                present = mm[metric_cols].notna().any(axis=1) if metric_cols else pd.Series([True]*len(mm))
                actual_rows = float(mm.loc[present, ['date','channel']].drop_duplicates().shape[0])
                data_coverage = float(max(0.0, min(1.0, actual_rows / expected_rows)))
            elif expected_rows > 0 and mm.empty:
                data_coverage = 0.0
        except Exception:
            data_coverage = 0.85  # fallback

        rollout = apply_rollout_policy(p_ai_better, data_coverage=float(data_coverage))

        decision_id = 'dec_' + pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')
        run_id = 'run_' + pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')

        dec = Decision(
            decision_id=decision_id,
            run_id=run_id,
            period_start=period_start,
            period_end=period_end,
            total_budget=float(total_budget),
            recommended_budget=sol.budget,
            expected_premium=obj.expected,
            premium_std=obj.std,
            ci_low=obj.ci_low,
            ci_high=obj.ci_high,
            risk_adjusted_premium=obj.risk_adjusted,

            baseline_budget=dn,
            baseline_expected_premium=obj_dn.expected,
            baseline_premium_std=obj_dn.std,
            baseline_ci_low=obj_dn.ci_low,
            baseline_ci_high=obj_dn.ci_high,
            baseline_risk_adjusted_premium=obj_dn.risk_adjusted,
            p_ai_better_vs_baseline=p_ai_better,

            rollout_mode=rollout.mode,
            rollout_reason_codes=rollout.reason_codes,
            data_coverage=float(data_coverage),

            top_reasons_by_channel=reasons,
            model_version=model_version,
            policy_hash=policy_hash,
            constraints_hash=constraints_hash,
            experiment_id=experiment_id,

            # Reproducibility / audit fields
            objective_mode=policy.objective_mode.value,
            policy_lambda=float(policy.policy_lambda),
            policy_delta=float(policy.policy_delta),
            n_days=int(n_days),
            warm_start_enabled=True,
            warm_start_days=int(warm_days_resolved),
            seed_main=int(seed_main),
            seed_baseline=int(seed_baseline),

            validation_report=validation_report,
        )

        dec_dir = paths.artifacts / 'recommendations/decisions'
        dec_dir.mkdir(parents=True, exist_ok=True)
        dec_path = dec_dir / f'{decision_id}.json'
        dec_path.write_text(json.dumps(dec.__dict__, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

        # Funnel stage forecast vs actual artifact (for dashboard & exec narrative)
        # NOTE: Reuse the exact same sample set used for Decision metrics above.

        def _summ(arr: np.ndarray) -> Dict[str, float]:
            return {
                "expected": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)),
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
            }

        forecast_rows = []
        for k, arr in funnel_samples.items():
            s = _summ(arr)
            forecast_rows.append({"stage": k, **s})

        actual = {k: None for k in ["leads", "call_attempt", "call_connected", "contracts", "premium"]}
        try:
            dff = df.copy()
            dff["date"] = pd.to_datetime(dff["date"])
            m = dff[(dff["date"] >= ps) & (dff["date"] <= pe)]
            if not m.empty:
                actual = {
                    "leads": float(m["leads"].sum()),
                    "call_attempt": float(m["call_attempt"].sum()),
                    "call_connected": float(m["call_connected"].sum()),
                    "contracts": float(m["contracts"].sum()),
                    "premium": float(m["premium"].sum()),
                }
        except Exception:
            pass

        for r in forecast_rows:
            r["actual"] = actual.get(r["stage"])
            r["gap"] = float(r["actual"] - r["expected"]) if r["actual"] is not None else None

        ff_dir = paths.artifacts / "recommendations/funnel_forecast"
        ff_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(forecast_rows).to_csv(ff_dir / f"{decision_id}.csv", index=False)


        append_csv(paths.logs / 'decisions/decision_log.csv', [{
            'timestamp': pd.Timestamp.utcnow().isoformat(),
            'decision_id': decision_id,
            'run_id': run_id,
            'period_start': period_start,
            'period_end': period_end,
            'total_budget': float(total_budget),
            'model_version': model_version,
            'expected_premium': obj.expected,
            'premium_std': obj.std,
            'ci_low': obj.ci_low,
            'ci_high': obj.ci_high,
            'risk_adjusted_premium': obj.risk_adjusted,
            'rollout_mode': rollout.mode.value,
            'rollout_reason_codes': '|'.join(rollout.reason_codes),
            'experiment_id': experiment_id or '',
        }], fieldnames=['timestamp','decision_id','run_id','period_start','period_end','total_budget','model_version','expected_premium','premium_std','ci_low','ci_high','risk_adjusted_premium','rollout_mode','rollout_reason_codes','experiment_id'])

        return RecommendResult(decision=dec, decision_json_path=str(dec_path))