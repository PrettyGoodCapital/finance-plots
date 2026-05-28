"""Alpha-analysis Great Tables helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["table_information", "table_returns_by_quantile", "table_turnover", "table_quantile_statistics"]


def _frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    return pd.DataFrame(data)


def _series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.copy()
    if hasattr(values, "to_numpy"):
        return pd.Series(values.to_numpy())
    return pd.Series(values)


def table_information(ic: Any):
    """Build a Great Tables information-coefficient summary."""
    import polars as pl
    from great_tables import GT, md

    series = pd.to_numeric(_series(ic), errors="coerce").dropna()
    mean = float(series.mean()) if len(series) else np.nan
    std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
    ir = mean / std if std and np.isfinite(std) else np.nan
    rows = [
        {"metric": "Mean IC", "value": mean},
        {"metric": "IC volatility", "value": std},
        {"metric": "ICIR", "value": ir},
        {"metric": "t-stat", "value": ir * np.sqrt(len(series)) if np.isfinite(ir) else np.nan},
        {"metric": "Positive IC", "value": float((series > 0).mean()) if len(series) else np.nan},
        {"metric": "Observations", "value": float(len(series))},
    ]
    return GT(pl.DataFrame(rows)).tab_header(title=md("**Information coefficient**")).fmt_number(columns=["value"], decimals=3)


def table_returns_by_quantile(data: Any, *, quantile_col: str = "quantile", return_col: str = "return"):
    """Build a Great Tables mean-return-by-quantile table."""
    import polars as pl
    from great_tables import GT, md

    frame = _frame(data)
    grouped = frame.groupby(quantile_col, dropna=False)[return_col].agg(["count", "mean", "std"]).reset_index()
    grouped = grouped.rename(columns={quantile_col: "quantile", "mean": "mean_return", "std": "volatility"})
    df = pl.from_pandas(grouped.sort_values("quantile"))
    return GT(df).tab_header(title=md("**Returns by quantile**")).fmt_percent(columns=["mean_return", "volatility"], decimals=2)


def table_turnover(data: Any, *, quantile_col: str = "quantile", turnover_col: str = "turnover"):
    """Build a Great Tables quantile-turnover table."""
    import polars as pl
    from great_tables import GT, md

    frame = _frame(data)
    grouped = (
        frame.groupby(quantile_col, dropna=False)[turnover_col]
        .mean()
        .reset_index()
        .rename(columns={quantile_col: "quantile", turnover_col: "turnover"})
    )
    df = pl.from_pandas(grouped.sort_values("quantile"))
    return GT(df).tab_header(title=md("**Quantile turnover**")).fmt_percent(columns=["turnover"], decimals=2)


def table_quantile_statistics(data: Any, *, quantile_col: str = "quantile", signal_col: str = "signal_mean", count_col: str = "count"):
    """Build a Great Tables quantile signal-statistics table."""
    import polars as pl
    from great_tables import GT, md

    frame = _frame(data)
    aggregations: dict[str, list[str] | str] = {}
    if count_col in frame.columns:
        aggregations[count_col] = "sum"
    if signal_col in frame.columns:
        aggregations[signal_col] = ["mean", "std"]
    grouped = frame.groupby(quantile_col, dropna=False).agg(aggregations).reset_index()
    columns = ["quantile"]
    if count_col in frame.columns:
        columns.append("count")
    if signal_col in frame.columns:
        columns.extend(["signal_mean", "signal_std"])
    grouped.columns = columns
    df = pl.from_pandas(grouped.sort_values("quantile"))
    return GT(df).tab_header(title=md("**Quantile statistics**")).fmt_number(columns=[c for c in df.columns if c != "quantile"], decimals=3)
