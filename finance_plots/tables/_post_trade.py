"""Post-trade Great Tables helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["table_cost_breakdown", "table_round_trip_stats", "table_execution_quality"]


def _frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    return pd.DataFrame(data)


def table_cost_breakdown(costs: Any, *, component_col: str = "component", value_col: str = "total"):
    """Build a Great Tables cost-breakdown table."""
    import polars as pl
    from great_tables import GT, md

    frame = _frame(costs)
    grouped = (
        frame.groupby(component_col, dropna=False)[value_col].sum().reset_index().rename(columns={component_col: "component", value_col: "total"})
    )
    total = grouped["total"].sum()
    grouped["pct_total"] = grouped["total"] / total if total else np.nan
    df = pl.from_pandas(grouped.sort_values("total", ascending=False))
    return GT(df).tab_header(title=md("**Cost breakdown**")).fmt_number(columns=["total"], decimals=2).fmt_percent(columns=["pct_total"], decimals=2)


def table_round_trip_stats(trades: Any, *, pnl_col: str = "pnl"):
    """Build a Great Tables round-trip statistics table."""
    import polars as pl
    from great_tables import GT, md

    frame = _frame(trades)
    pnl = pd.to_numeric(frame[pnl_col], errors="coerce").dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_loss = -losses.sum()
    rows = [
        {"metric": "Trades", "value": float(len(pnl))},
        {"metric": "Win rate", "value": float((pnl > 0).mean()) if len(pnl) else np.nan},
        {"metric": "Average PnL", "value": float(pnl.mean()) if len(pnl) else np.nan},
        {"metric": "Total PnL", "value": float(pnl.sum()) if len(pnl) else 0.0},
        {"metric": "Profit factor", "value": float(wins.sum() / gross_loss) if gross_loss else np.nan},
        {"metric": "Payoff ratio", "value": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else np.nan},
    ]
    return GT(pl.DataFrame(rows)).tab_header(title=md("**Round-trip statistics**")).fmt_number(columns=["value"], decimals=2)


def table_execution_quality(executions: Any, *, slippage_col: str = "implementation_shortfall_bps"):
    """Build a Great Tables execution-quality summary."""
    import polars as pl
    from great_tables import GT, md

    values = pd.to_numeric(_frame(executions)[slippage_col], errors="coerce").dropna()
    rows = [
        {"metric": "Count", "value": float(len(values))},
        {"metric": "Mean bps", "value": float(values.mean()) if len(values) else np.nan},
        {"metric": "Median bps", "value": float(values.median()) if len(values) else np.nan},
        {"metric": "Worst bps", "value": float(values.max()) if len(values) else np.nan},
        {"metric": "Best bps", "value": float(values.min()) if len(values) else np.nan},
    ]
    return GT(pl.DataFrame(rows)).tab_header(title=md("**Execution quality**")).fmt_number(columns=["value"], decimals=2)
