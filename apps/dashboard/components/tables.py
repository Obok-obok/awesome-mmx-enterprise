from __future__ import annotations
import pandas as pd

def add_totals(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    # Use missing values (not empty strings) for non-aggregate cells to keep
    # dataframe Arrow-compatible and avoid noisy serialization tracebacks.
    total: dict[str, object] = {c: pd.NA for c in df.columns}
    total[df.columns[0]] = '합계'
    for c in numeric_cols:
        if c in df.columns:
            total[c] = float(df[c].sum())
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)
