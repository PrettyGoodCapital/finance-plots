"""Performance summary tables."""

from __future__ import annotations

from typing import Any

import finance_calcs as fc
import polars as pl
from finance_enums import Frequency

from .._util import to_returns_and_index
from ..plots._returns import _period_key, _period_returns

__all__ = ["performance_statistics", "table_drawdowns", "table_performance_statistics", "table_period_returns"]

_PERF_STAT_LABELS = {
    "cumulative_return": "Cumulative return",
    "annualized_return": "Annualized return",
    "annualized_volatility": "Annualized volatility",
    "sharpe": "Sharpe ratio",
    "sortino": "Sortino ratio",
    "max_drawdown": "Max drawdown",
    "calmar": "Calmar ratio",
}

_PERF_STAT_PERCENT_KEYS = {
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
}


def performance_statistics(
    returns: Any,
    *,
    frequency: Frequency | str | float = Frequency.Day,
) -> dict[str, float]:
    """Compute summary performance statistics.

    Args:
        returns: 1-D series of periodic returns (narwhals-compatible).
        frequency: Observation frequency alias, enum, or observations per year.

    Returns:
        Dict keyed by ``cumulative_return``, ``annualized_return``, ``annualized_volatility``,
        ``sharpe``, ``sortino``, ``max_drawdown``, ``calmar``.
    """
    values, _ = to_returns_and_index(returns)
    if values.size == 0:
        return dict.fromkeys(_PERF_STAT_LABELS, float("nan"))
    frame = pl.DataFrame({"returns": values})
    row = frame.select(
        fc.cumulative_return(pl.col("returns")).alias("cumulative_return"),
        fc.annualized_return(pl.col("returns"), frequency=frequency).alias("annualized_return"),
        fc.annualized_volatility(pl.col("returns"), frequency=frequency).alias("annualized_volatility"),
        fc.sharpe(pl.col("returns"), frequency=frequency).alias("sharpe"),
        fc.sortino(pl.col("returns"), frequency=frequency).alias("sortino"),
        fc.max_drawdown(pl.col("returns")).alias("max_drawdown"),
        fc.calmar(pl.col("returns"), frequency=frequency).alias("calmar"),
    ).row(0, named=True)
    return {key: float(value) for key, value in row.items()}


def table_performance_statistics(
    returns: Any,
    benchmark: Any | None = None,
    *,
    frequency: Frequency | str | float = Frequency.Day,
):
    """Build a ``great_tables.GT`` performance-stats table.

    Args:
        returns: 1-D series of periodic returns.
        benchmark: Optional benchmark return series. When provided, a
            second value column is added to the table.
        frequency: Observation frequency alias, enum, or observations per year.

    Returns:
        A ``great_tables.GT`` table with one column per series and one
        row per metric.
    """
    from great_tables import GT, md

    strat = performance_statistics(returns, frequency=frequency)
    rows = {"metric": list(strat.keys()), "strategy": list(strat.values())}
    if benchmark is not None:
        bench = performance_statistics(benchmark, frequency=frequency)
        rows["benchmark"] = [bench[k] for k in strat]

    df = pl.DataFrame(rows)
    df = df.with_columns(pl.col("metric").replace_strict(_PERF_STAT_LABELS))
    gt = GT(df).tab_header(title=md("**Performance summary**"))
    value_cols = [c for c in df.columns if c != "metric"]
    pct_rows = [
        i
        for i, m in enumerate(df["metric"])
        if m
        in {
            "Cumulative return",
            "Annualized return",
            "Annualized volatility",
            "Max drawdown",
        }
    ]
    num_rows = [
        i
        for i, m in enumerate(df["metric"])
        if m
        in {
            "Sharpe ratio",
            "Sortino ratio",
            "Calmar ratio",
        }
    ]
    return gt.fmt_percent(columns=value_cols, rows=pct_rows, decimals=2).fmt_number(columns=value_cols, rows=num_rows, decimals=2)


def _period_label(value: Any, period: Any) -> str:
    key = _period_key(period)
    if key == "year":
        return str(value.year)
    if key == "quarter":
        return f"{value.year} Q{value.quarter}"
    if key == "month":
        return value.strftime("%Y-%m")
    if key == "week":
        iso = value.isocalendar()
        return f"{iso.year} W{iso.week:02d}"
    return str(value.date()) if hasattr(value, "date") else str(value)


def table_period_returns(
    returns: Any,
    *,
    period: Any = "year",
):
    """Build a ``great_tables.GT`` table of compounded period returns.

    Args:
        returns: 1-D series of periodic returns.
        period: Calendar bucket: ``"day"``, ``"week"``, ``"month"``,
            ``"quarter"``, or ``"year"``.

    Returns:
        A ``great_tables.GT`` table with one row per period.
    """
    import polars as pl
    from great_tables import GT, md

    period_returns = _period_returns(returns, period).dropna()
    rows = {
        "period": [_period_label(index_value, period) for index_value in period_returns.index],
        "return": [float(value) for value in period_returns.to_numpy()],
    }
    df = pl.DataFrame(rows)
    return GT(df).tab_header(title=md("**Period returns**")).fmt_percent(columns=["return"], decimals=2)


def _drawdown_rows(returns: Any, top: int = 5) -> list[dict[str, Any]]:
    values, index = to_returns_and_index(returns)
    details = fc.drawdown_details(pl.Series("returns", values), date=pl.Series("date", index)).sort("max_drawdown").head(top)
    return [
        {
            "rank": rank,
            "start": row["start"],
            "trough": row["valley"],
            "recovery": row["end"] if row["recovered"] else "Unrecovered",
            "drawdown": row["max_drawdown"],
            "duration": row["duration"],
        }
        for rank, row in enumerate(details.to_dicts(), start=1)
    ]


def _display_index_value(value: Any) -> str:
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)


def table_drawdowns(
    returns: Any,
    *,
    top: int = 5,
):
    """Build a ``great_tables.GT`` table of the largest drawdown periods.

    Args:
        returns: 1-D series of periodic returns.
        top: Maximum number of drawdown periods to include.

    Returns:
        A ``great_tables.GT`` table sorted by drawdown depth.
    """
    import polars as pl
    from great_tables import GT, md

    rows = _drawdown_rows(returns, top=top)
    display_rows = [
        {
            **row,
            "start": _display_index_value(row["start"]),
            "trough": _display_index_value(row["trough"]),
            "recovery": _display_index_value(row["recovery"]),
        }
        for row in rows
    ]
    df = (
        pl.DataFrame(display_rows)
        if display_rows
        else pl.DataFrame({"rank": [], "start": [], "trough": [], "recovery": [], "drawdown": [], "duration": []})
    )
    return GT(df).tab_header(title=md("**Drawdown periods**")).fmt_percent(columns=["drawdown"], decimals=2)
