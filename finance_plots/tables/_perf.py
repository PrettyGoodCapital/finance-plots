"""Performance summary tables."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._util import cumulative_returns, drawdown, to_returns_and_index
from ..plots._returns import _period_key, _period_returns

__all__ = ["perf_stats", "table_perf_stats", "table_period_returns", "table_drawdowns"]

_PERF_STAT_LABELS = {
    "cum_return": "Cumulative return",
    "ann_return": "Annualised return",
    "ann_vol": "Annualised volatility",
    "sharpe": "Sharpe ratio",
    "sortino": "Sortino ratio",
    "max_drawdown": "Max drawdown",
    "calmar": "Calmar ratio",
}

_PERF_STAT_PERCENT_KEYS = {
    "cum_return",
    "ann_return",
    "ann_vol",
    "max_drawdown",
}


def _annualised_return(values: np.ndarray, ppy: int) -> float:
    if values.size == 0:
        return float("nan")
    total = (1.0 + np.where(np.isfinite(values), values, 0.0)).prod()
    return float(total ** (ppy / values.size) - 1.0)


def _annualised_vol(values: np.ndarray, ppy: int) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.nanstd(values, ddof=1) * np.sqrt(ppy))


def _sharpe(values: np.ndarray, ppy: int) -> float:
    vol = _annualised_vol(values, ppy)
    if not np.isfinite(vol) or vol == 0.0:
        return float("nan")
    return float(np.nanmean(values) * ppy / vol)


def _sortino(values: np.ndarray, ppy: int) -> float:
    downside = np.where(values < 0, values, 0.0)
    dd = np.sqrt(np.nanmean(downside**2)) * np.sqrt(ppy)
    if not np.isfinite(dd) or dd == 0.0:
        return float("nan")
    return float(np.nanmean(values) * ppy / dd)


def perf_stats(
    returns: Any,
    *,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute summary performance statistics.

    Args:
        returns: 1-D series of periodic returns (narwhals-compatible).
        periods_per_year: Annualization factor.

    Returns:
        Dict keyed by ``cum_return``, ``ann_return``, ``ann_vol``,
        ``sharpe``, ``sortino``, ``max_drawdown``, ``calmar``.
    """
    values, _ = to_returns_and_index(returns)
    ann = _annualised_return(values, periods_per_year)
    mdd = float(drawdown(values).min()) if values.size else float("nan")
    calmar = ann / abs(mdd) if (np.isfinite(mdd) and mdd != 0.0) else float("nan")
    return {
        "cum_return": float(cumulative_returns(values)[-1]) if values.size else float("nan"),
        "ann_return": ann,
        "ann_vol": _annualised_vol(values, periods_per_year),
        "sharpe": _sharpe(values, periods_per_year),
        "sortino": _sortino(values, periods_per_year),
        "max_drawdown": mdd,
        "calmar": calmar,
    }


def table_perf_stats(
    returns: Any,
    benchmark: Any | None = None,
    *,
    periods_per_year: int = 252,
):
    """Build a ``great_tables.GT`` performance-stats table.

    Args:
        returns: 1-D series of periodic returns.
        benchmark: Optional benchmark return series. When provided, a
            second value column is added to the table.
        periods_per_year: Annualization factor.

    Returns:
        A ``great_tables.GT`` table with one column per series and one
        row per metric.
    """
    import polars as pl
    from great_tables import GT, md

    strat = perf_stats(returns, periods_per_year=periods_per_year)
    rows = {"metric": list(strat.keys()), "strategy": list(strat.values())}
    if benchmark is not None:
        bench = perf_stats(benchmark, periods_per_year=periods_per_year)
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
            "Annualised return",
            "Annualised volatility",
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
    if values.size == 0:
        return []

    dd = drawdown(values)
    rows: list[dict[str, Any]] = []
    in_drawdown = False
    start = 0
    trough = 0

    for i, value in enumerate(dd):
        if value < 0.0 and not in_drawdown:
            in_drawdown = True
            start = max(i - 1, 0)
            trough = i
        if in_drawdown and value < dd[trough]:
            trough = i
        if in_drawdown and (value == 0.0 or i == len(dd) - 1):
            recovery = i if value == 0.0 else None
            rows.append(
                {
                    "start": index[start],
                    "trough": index[trough],
                    "recovery": index[recovery] if recovery is not None else "Unrecovered",
                    "drawdown": float(dd[trough]),
                    "duration": int(i - start),
                }
            )
            in_drawdown = False

    rows.sort(key=lambda row: row["drawdown"])
    return [{"rank": i + 1, **row} for i, row in enumerate(rows[:top])]


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
