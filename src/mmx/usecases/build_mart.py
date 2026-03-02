from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
from mmx.config.settings import Settings
from mmx.data.paths import Paths
from mmx.data.io_csv import atomic_write_csv, file_lock
from mmx.data.validators import (
    require_columns,
    assert_non_negative,
    normalize_channels,
    parse_datetime_series,
    validate_channels,
    validate_funnel_monotonicity,
)

@dataclass(frozen=True)
class BuildMartResult:
    output_path: str
    rows_written: int
    validation_report_path: str

def _read_glob_csv(dir_path: Path) -> pd.DataFrame:
    files = sorted(dir_path.glob('*.csv')) if dir_path.exists() else []
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def build_daily_channel_fact(paths: Paths, settings: Settings) -> BuildMartResult:
    lock = paths.logs / 'pipeline/mart_build.lock'
    with file_lock(lock):
        spend = _read_glob_csv(paths.data_raw / 'events/spend')
        leads = _read_glob_csv(paths.data_raw / 'events/leads')
        attempt = _read_glob_csv(paths.data_raw / 'events/call_attempt')
        connected = _read_glob_csv(paths.data_raw / 'events/call_connected')
        contracts = _read_glob_csv(paths.data_raw / 'events/contracts')
        premium = _read_glob_csv(paths.data_raw / 'events/premium')

        validation: dict = {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'max_nat_ratio': settings.max_nat_ratio,
            'allowed_channels': list(settings.allowed_channels),
            'enforce_channel_allowlist': settings.enforce_channel_allowlist,
            'max_unknown_channel_ratio': settings.max_unknown_channel_ratio,
            'enforce_funnel_monotonicity': settings.enforce_monotonicity,
            'max_monotonic_violation_ratio': settings.max_monotonic_violation_ratio,
            'sources': {},
        }

        def _prep(df: pd.DataFrame, name: str) -> pd.DataFrame:
            if df.empty:
                validation['sources'][name] = {'rows': 0}
                return df
            require_columns(df, ['channel'], name)
            df = df.copy()
            df['channel'] = normalize_channels(df['channel'], normalize=settings.channel_normalize)
            rep = validate_channels(
                df,
                name=name,
                allowed_channels=settings.allowed_channels,
                enforce_allowlist=settings.enforce_channel_allowlist,
                max_unknown_ratio=settings.max_unknown_channel_ratio,
            )
            validation['sources'][name] = {
                'rows': int(len(df)),
                'unique_channels': rep.unique_channels,
                'unknown_rows': rep.unknown_rows,
                'unknown_ratio': rep.unknown_ratio,
                'unknown_channels': list(rep.unknown_channels),
            }
            # If allowlist provided and not enforced, drop unknown channels to avoid contaminating mart.
            if settings.allowed_channels and (not settings.enforce_channel_allowlist) and rep.unknown_rows > 0:
                df = df[~df['channel'].isin(rep.unknown_channels)].reset_index(drop=True)
                validation['sources'][name]['rows_after_drop_unknown'] = int(len(df))
            return df

        spend = _prep(spend, 'spend')
        leads = _prep(leads, 'leads')
        attempt = _prep(attempt, 'call_attempt')
        connected = _prep(connected, 'call_connected')
        contracts = _prep(contracts, 'contracts')
        premium = _prep(premium, 'premium')

        def agg_sum(df: pd.DataFrame, id_col: str, time_col: str, val_col: str, out_col: str, name: str) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame(columns=['date','channel',out_col])
            require_columns(df, [id_col, time_col, 'channel', val_col], name)
            df = df.drop_duplicates(id_col).copy()
            dt, rep = parse_datetime_series(df[time_col], name=name, column=time_col, max_nat_ratio=settings.max_nat_ratio)
            df['date'] = dt.dt.date.astype(str)
            validation['sources'].setdefault(name, {})['date_parse'] = rep.__dict__
            assert_non_negative(df, [val_col], name)
            return df.groupby(['date','channel'], as_index=False)[val_col].sum().rename(columns={val_col: out_col})

        def agg_nunique(df: pd.DataFrame, id_col: str, time_col: str, out_col: str, name: str) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame(columns=['date','channel',out_col])
            require_columns(df, [id_col, time_col, 'channel'], name)
            df = df.drop_duplicates(id_col).copy()
            dt, rep = parse_datetime_series(df[time_col], name=name, column=time_col, max_nat_ratio=settings.max_nat_ratio)
            df['date'] = dt.dt.date.astype(str)
            validation['sources'].setdefault(name, {})['date_parse'] = rep.__dict__
            return df.groupby(['date','channel'], as_index=False)[id_col].nunique().rename(columns={id_col: out_col})

        spend_agg = agg_sum(spend,'event_id','event_time','spend','spend','spend')
        leads_agg = agg_nunique(leads,'lead_id','occurred_at','leads','leads')
        attempt_agg = agg_nunique(attempt,'attempt_id','attempt_at','call_attempt','call_attempt')
        connected_agg = agg_nunique(connected,'call_id','connected_at','call_connected','call_connected')
        contracts_agg = agg_nunique(contracts,'contract_id','contract_signed_at','contracts','contracts')
        premium_agg = agg_sum(premium,'premium_event_id','premium_recognized_at','premium_amount','premium','premium')

        dfs = [spend_agg, leads_agg, attempt_agg, connected_agg, contracts_agg, premium_agg]
        keys = pd.concat([d[['date','channel']] for d in dfs if not d.empty], ignore_index=True).drop_duplicates()

        if keys.empty:
            out = pd.DataFrame(columns=['date','channel','spend','leads','call_attempt','call_connected','contracts','premium'])
        else:
            out = keys
            for d in dfs:
                out = out.merge(d, on=['date','channel'], how='left')
            for c in ['spend','leads','call_attempt','call_connected','contracts','premium']:
                out[c] = out[c].fillna(0.0)
            out = out.sort_values(['date','channel']).reset_index(drop=True)

        # Funnel monotonicity validation (data quality gate)
        mono_rep = validate_funnel_monotonicity(
            out,
            enforce=settings.enforce_monotonicity,
            max_violation_ratio=settings.max_monotonic_violation_ratio,
            name='daily_channel_fact',
        )
        validation['mart'] = {
            'rows': int(len(out)),
            'monotonicity': {
                'total_rows': mono_rep.total_rows,
                'violation_rows': mono_rep.violation_rows,
                'violation_ratio': mono_rep.violation_ratio,
                'violations': mono_rep.violations,
            },
        }

        out_path = paths.mart / 'daily_channel_fact.csv'
        atomic_write_csv(
            out_path,
            out.to_dict('records'),
            fieldnames=['date','channel','spend','leads','call_attempt','call_connected','contracts','premium'],
        )

        # Persist validation report (latest + timestamped)
        vdir = paths.logs / 'pipeline'
        vdir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        report_path = vdir / f'mart_validation_{ts}.json'
        report_latest = vdir / 'mart_validation_latest.json'
        report_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2))
        report_latest.write_text(json.dumps(validation, ensure_ascii=False, indent=2))

        return BuildMartResult(str(out_path), int(len(out)), str(report_path))
