# MMX_SYS_PATH_GUARD: ensure repo root is importable when running via `streamlit run`
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


"""Dashboard bootstrap utilities.

Streamlit multipage apps execute each page module independently.
This module ensures that *every page* applies the same:
- Page config
- CSS / typography
- Global filters (date range, channels)
- Context bar (as-of, coverage, model version, objective, etc.)

This is required to satisfy the design standardization requirements.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mmx.config.settings import load_settings
from mmx.data.paths import resolve_paths
from mmx.governance.registry import load_registry

from .labels import Labels, load_labels
from .ui import load_css
from .state import sidebar_global_filters, filter_mart, sidebar_targets
from .context_bar import ContextBar, render_context_bar

import sys
from pathlib import Path


def ensure_project_sys_path() -> None:
    """Ensure project root and src/ are on sys.path.

    Streamlit can be launched from arbitrary working directories. Many dashboard modules
    import via the `apps.*` and `mmx.*` namespaces, which require the project root and
    `src/` to be importable.
    """

    this_file = Path(__file__).resolve()
    project_root = this_file.parents[3]  # .../apps/dashboard/components/bootstrap.py -> project root
    src_dir = project_root / "src"

    for path in (str(project_root), str(src_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)



@dataclass(frozen=True)
class DashContext:
    """Resolved dashboard runtime context."""

    settings: "object"
    paths: "object"
    labels: Labels
    registry: "object"
    mart: pd.DataFrame
    mart_full: pd.DataFrame
    coverage: str


def bootstrap(page_title: str) -> DashContext:
    """Apply global UI standards and return the dashboard context.

    Args:
        page_title: Browser tab title.

    Returns:
        DashContext with settings/paths/labels/registry and filtered mart.
    """

    settings = load_settings()
    paths = resolve_paths(settings)
    labels = load_labels(ROOT / "src/mmx/config/labels.yaml")
    reg = load_registry(paths)

    # CSS must be loaded in every page module.
    load_css(ROOT / "apps/dashboard/assets/styles.css")

    # Sidebar: global filters + targets uploader.
    with st.sidebar:
        st.markdown("### Filters")
        sidebar_global_filters(paths)
        st.divider()
        st.markdown("### Targets")
        sidebar_targets(paths)

    mart_path = paths.mart / "daily_channel_fact.csv"
    df = pd.read_csv(mart_path) if mart_path.exists() else pd.DataFrame(
        columns=["date", "channel", "spend", "leads", "call_attempt", "call_connected", "contracts", "premium"]
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    df_f = filter_mart(df)
    coverage = "-"
    if not df.empty:
        coverage = f"{df['date'].min().date()} ~ {df['date'].max().date()}"

    # Header context bar.
    render_context_bar(
        ContextBar(
            as_of=pd.Timestamp.now(tz=settings.timezone).strftime("%Y-%m-%d %H:%M"),
            data_coverage=coverage,
            model_version=reg.production_version,
            objective_mode=settings.objective_mode,
            policy_lambda=settings.policy_lambda,
            policy_delta=settings.policy_delta,
            reporting_delay=settings.reporting_delay,
        )
    )

    return DashContext(settings=settings, paths=paths, labels=labels, registry=reg, mart=df_f, mart_full=df, coverage=coverage)
