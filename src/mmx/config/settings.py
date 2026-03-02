from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

def _parse_csv_list(value: str) -> Tuple[str, ...]:
    items = [v.strip() for v in value.split(',') if v.strip()]
    return tuple(items)

def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ('1','true','yes','y','on')

@dataclass(frozen=True)
class Settings:
    base_dir: Path
    timezone: str
    premium_date_basis: str
    reporting_delay: str
    reporting_delay_max: int
    objective_mode: str
    policy_lambda: float
    policy_delta: float
    inference_backend: str
    inference_method: str
    n_samples: int
    ramp_up_cap_share: float
    ramp_up_cap_abs: float

    # Data validation / normalization
    channel_normalize: bool
    allowed_channels: Tuple[str, ...]
    enforce_channel_allowlist: bool
    max_nat_ratio: float
    max_unknown_channel_ratio: float
    enforce_monotonicity: bool
    max_monotonic_violation_ratio: float

def load_settings() -> Settings:
    base_dir = Path(os.getenv('MMX_BASE_DIR', '.')).resolve()
    allowed_channels = _parse_csv_list(os.getenv('MMX_ALLOWED_CHANNELS', ''))
    return Settings(
        base_dir=base_dir,
        timezone=os.getenv('MMX_TIMEZONE', 'Asia/Seoul'),
        premium_date_basis=os.getenv('MMX_PREMIUM_DATE_BASIS', 'RECOGNIZED_AT'),
        reporting_delay=os.getenv('MMX_REPORTING_DELAY', 'OFF'),
        reporting_delay_max=int(os.getenv('MMX_REPORTING_DELAY_MAX', '3')),
        objective_mode=os.getenv('MMX_OBJECTIVE_MODE', 'RISK_ADJUSTED'),
        policy_lambda=float(os.getenv('MMX_POLICY_LAMBDA', '0.8')),
        policy_delta=float(os.getenv('MMX_POLICY_DELTA', '0.15')),
        inference_backend=os.getenv('MMX_INFERENCE_BACKEND', 'PYMC'),
        inference_method=os.getenv('MMX_INFERENCE_METHOD', 'VI'),
        n_samples=int(os.getenv('MMX_N_SAMPLES', '2000')),
        ramp_up_cap_share=float(os.getenv('MMX_RAMP_UP_CAP_SHARE', '0.05')),
        ramp_up_cap_abs=float(os.getenv('MMX_RAMP_UP_CAP_ABS', '0.0')),

        channel_normalize=_parse_bool(os.getenv('MMX_CHANNEL_NORMALIZE', 'true')),
        allowed_channels=allowed_channels,
        enforce_channel_allowlist=_parse_bool(os.getenv('MMX_ENFORCE_CHANNEL_ALLOWLIST', 'false')),
        max_nat_ratio=float(os.getenv('MMX_MAX_NAT_RATIO', '0.001')),
        max_unknown_channel_ratio=float(os.getenv('MMX_MAX_UNKNOWN_CHANNEL_RATIO', '0.0')),
        enforce_monotonicity=_parse_bool(os.getenv('MMX_ENFORCE_FUNNEL_MONOTONICITY', 'false')),
        max_monotonic_violation_ratio=float(os.getenv('MMX_MAX_MONOTONIC_VIOLATION_RATIO', '0.0')),
    )
