"""Return-series plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._util import cumulative_returns, drawdown, to_returns_and_index

try:
    from finance_enums import Frequency, to_frequency
except ImportError:  # pragma: no cover - optional dependency for now
    Frequency = None  # type: ignore[assignment]

    def to_frequency(value):  # type: ignore[no-redef]
        return str(value).lower()


__all__ = [
    "plot_returns",
    "plot_rolling_returns",
    "plot_rolling_volatility",
    "plot_rolling_sharpe",
    "plot_rolling_beta",
    "plot_rolling_correlation",
    "plot_return_scatter",
    "plot_drawdown_underwater",
    "plot_returns_heatmap",
    "plot_returns_bar",
    "plot_returns_dist",
    "plot_returns_timeseries",
]


def _new_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=(10, 4))
    return fig, ax


def _indexed_returns(returns: Any):
    import pandas as pd

    values, index = to_returns_and_index(returns)
    safe = np.where(np.isfinite(values), values, np.nan)
    return pd.Series(safe, index=index)


def _period_key(period: Any) -> str:
    freq = to_frequency(period) if Frequency is not None else str(period).lower()
    return freq.value if Frequency is not None else freq


def _datetime_index(index: np.ndarray, size: int):
    import pandas as pd

    if index.shape[0] != size or np.issubdtype(index.dtype, np.number):
        return pd.date_range("2000-01-01", periods=size, freq="B")
    try:
        dt_index = pd.to_datetime(index)
    except (TypeError, ValueError):
        return pd.date_range("2000-01-01", periods=size, freq="B")
    if dt_index.dtype.kind != "M":
        return pd.date_range("2000-01-01", periods=size, freq="B")
    return dt_index


_PERIOD_RETURN_SPEC = {
    "day": {"resample": None, "label": "daily"},
    "week": {"resample": "W", "label": "weekly"},
    "month": {"resample": "month_end", "label": "monthly"},
    "quarter": {"resample": "quarter_end", "label": "quarterly"},
    "year": {"resample": "year_end", "label": "annual"},
}


def _resample_rule(key: str):
    import pandas as pd

    resample = _PERIOD_RETURN_SPEC[key]["resample"]
    if resample == "month_end":
        return pd.offsets.MonthEnd()
    if resample == "quarter_end":
        return pd.offsets.QuarterEnd()
    if resample == "year_end":
        return pd.offsets.YearEnd()
    return resample


def _period_returns(returns: Any, period: Any):
    import pandas as pd

    key = _period_key(period)
    if key not in _PERIOD_RETURN_SPEC:
        raise ValueError(f"period={period!r} not supported (expected one of {sorted(_PERIOD_RETURN_SPEC)})")

    values, index = to_returns_and_index(returns)
    dt_index = _datetime_index(index, values.size)
    safe = np.where(np.isfinite(values), values, 0.0)
    series = pd.Series(safe, index=dt_index)
    resample = _resample_rule(key)
    if resample is None:
        return series
    return (1.0 + series).resample(resample).prod() - 1.0


def _plot_metric(series, title: str, ylabel: str, ax: Axes | None = None) -> Figure:
    fig, ax = _new_axes(ax)
    ax.plot(series.index, series.to_numpy(), color="#1f77b4", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def _paired_returns(returns: Any, benchmark: Any):
    import pandas as pd

    values, index = to_returns_and_index(returns)
    benchmark_values, _ = to_returns_and_index(benchmark)
    if benchmark_values.shape != values.shape:
        raise ValueError("benchmark length must match returns length")

    strategy = pd.Series(np.where(np.isfinite(values), values, np.nan), index=index)
    bench = pd.Series(np.where(np.isfinite(benchmark_values), benchmark_values, np.nan), index=index)
    return strategy, bench


def plot_returns(
    returns: Any,
    live_start: Any | None = None,
    *,
    log_scale: bool = False,
    ax: Axes | None = None,
) -> Figure:
    """Plot cumulative returns without requiring a benchmark argument.

    Args:
        returns: 1-D series of periodic returns.
        live_start: Optional index position or timestamp marking the
            in-/out-of-sample cutoff.
        log_scale: If True, the y-axis is symlog.
        ax: Existing matplotlib ``Axes`` to draw onto.

    Returns:
        The ``matplotlib.figure.Figure`` containing the plot.
    """
    return plot_rolling_returns(
        returns,
        live_start=live_start,
        log_scale=log_scale,
        ax=ax,
    )


def plot_rolling_returns(
    returns: Any,
    benchmark: Any | None = None,
    live_start: Any | None = None,
    *,
    log_scale: bool = False,
    ax: Axes | None = None,
) -> Figure:
    """Plot cumulative returns with an optional benchmark overlay.

    Args:
        returns: 1-D series of periodic returns (narwhals-compatible).
        benchmark: Optional benchmark return series; plotted on the
            same axes.
        live_start: Optional position in the index marking the
            in-/out-of-sample cutoff. The out-of-sample region is
            shaded.
        log_scale: If True, the y-axis is symlog.
        ax: Existing matplotlib ``Axes`` to draw onto.

    Returns:
        The ``matplotlib.figure.Figure`` containing the plot.
    """
    fig, ax = _new_axes(ax)

    values, index = to_returns_and_index(returns)
    cum = cumulative_returns(values)
    ax.plot(index, cum, label="strategy", color="#1f77b4", linewidth=1.5)

    if benchmark is not None:
        b_vals, b_idx = to_returns_and_index(benchmark)
        if b_vals.shape == values.shape:
            b_idx = index
        ax.plot(
            b_idx,
            cumulative_returns(b_vals),
            label="benchmark",
            color="#888",
            linewidth=1.0,
            linestyle="--",
        )

    if live_start is not None:
        try:
            cutoff = np.searchsorted(index, live_start)
        except TypeError:
            cutoff = int(live_start)
        ax.axvspan(
            index[min(cutoff, len(index) - 1)],
            index[-1],
            color="#ffaa00",
            alpha=0.08,
            label="out-of-sample",
        )

    ax.axhline(0.0, color="black", linewidth=0.5)
    if log_scale:
        ax.set_yscale("symlog", linthresh=0.05)
    ax.set_title("Cumulative returns")
    ax.set_ylabel("cumulative return")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_rolling_volatility(
    returns: Any,
    window: int = 63,
    *,
    periods_per_year: int = 252,
    ax: Axes | None = None,
) -> Figure:
    """Plot rolling annualized volatility.

    Args:
        returns: 1-D series of periodic returns.
        window: Rolling window length in observations.
        periods_per_year: Annualization factor.
        ax: Existing matplotlib ``Axes`` to draw onto.

    Returns:
        The ``matplotlib.figure.Figure`` containing the plot.
    """
    series = _indexed_returns(returns)
    vol = series.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    return _plot_metric(vol, f"Rolling {window}-period volatility", "annualized volatility", ax)


def plot_rolling_sharpe(
    returns: Any,
    window: int = 63,
    *,
    periods_per_year: int = 252,
    ax: Axes | None = None,
) -> Figure:
    """Plot rolling annualized Sharpe ratio.

    Args:
        returns: 1-D series of periodic returns.
        window: Rolling window length in observations.
        periods_per_year: Annualization factor.
        ax: Existing matplotlib ``Axes`` to draw onto.

    Returns:
        The ``matplotlib.figure.Figure`` containing the plot.
    """
    series = _indexed_returns(returns)
    rolling_mean = series.rolling(window).mean() * periods_per_year
    rolling_vol = series.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = rolling_mean / rolling_vol.replace(0.0, np.nan)
    return _plot_metric(sharpe, f"Rolling {window}-period Sharpe", "Sharpe ratio", ax)


def plot_rolling_beta(
    returns: Any,
    benchmark: Any,
    window: int = 63,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot rolling beta versus a benchmark return series.

    Args:
        returns: 1-D strategy return series.
        benchmark: 1-D benchmark return series.
        window: Rolling window length in observations.
        ax: Existing matplotlib ``Axes`` to draw onto.

    Returns:
        The ``matplotlib.figure.Figure`` containing the plot.
    """
    strategy, bench = _paired_returns(returns, benchmark)
    beta = strategy.rolling(window).cov(bench) / bench.rolling(window).var(ddof=1)
    return _plot_metric(beta, f"Rolling {window}-period beta", "beta", ax)


def plot_rolling_correlation(
    returns: Any,
    benchmark: Any,
    window: int = 63,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot rolling correlation versus a benchmark return series.

    Args:
        returns: 1-D strategy return series.
        benchmark: 1-D benchmark return series.
        window: Rolling window length in observations.
        ax: Existing matplotlib ``Axes`` to draw onto.

    Returns:
        The ``matplotlib.figure.Figure`` containing the plot.
    """
    strategy, bench = _paired_returns(returns, benchmark)
    corr = strategy.rolling(window).corr(bench)
    fig = _plot_metric(corr, f"Rolling {window}-period correlation", "correlation", ax)
    fig.axes[0].set_ylim(-1.0, 1.0)
    return fig


def plot_return_scatter(
    returns: Any,
    benchmark: Any,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot strategy returns against benchmark returns.

    Args:
        returns: 1-D strategy return series.
        benchmark: 1-D benchmark return series.
        ax: Existing matplotlib ``Axes`` to draw onto.

    Returns:
        The ``matplotlib.figure.Figure`` containing the scatter plot.
    """
    strategy, bench = _paired_returns(returns, benchmark)
    x = bench.to_numpy()
    y = strategy.to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)

    fig, ax = _new_axes(ax)
    ax.scatter(x[mask], y[mask], s=14, alpha=0.55, color="#1f77b4", edgecolors="none")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=0.5)
    if mask.sum() > 1 and np.nanvar(x[mask]) > 0.0:
        beta, alpha = np.polyfit(x[mask], y[mask], 1)
        line_x = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
        ax.plot(line_x, alpha + beta * line_x, color="#d62728", linewidth=1.2, label=f"beta {beta:.2f}")
        ax.legend(loc="best", frameon=False)
    ax.set_title("Strategy versus benchmark returns")
    ax.set_xlabel("benchmark return")
    ax.set_ylabel("strategy return")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_drawdown_underwater(
    returns: Any,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Underwater drawdown plot.

    Args:
        returns: 1-D series of periodic returns.
        ax: Existing matplotlib ``Axes`` to draw onto.

    Returns:
        The ``matplotlib.figure.Figure`` containing the filled-area
        drawdown plot.
    """
    fig, ax = _new_axes(ax)
    values, index = to_returns_and_index(returns)
    dd = drawdown(values)
    ax.fill_between(index, dd, 0.0, color="#d62728", alpha=0.4)
    ax.plot(index, dd, color="#d62728", linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_title("Drawdown (underwater)")
    ax.set_ylabel("drawdown")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


_HEATMAP_PERIOD_SPEC = {
    "month": {
        "col_attr": "month",
        "col_count": 12,
        "col_labels": [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
        "unit_label": "monthly",
    },
    "quarter": {
        "col_attr": "quarter",
        "col_count": 4,
        "col_labels": ["Q1", "Q2", "Q3", "Q4"],
        "unit_label": "quarterly",
    },
    "week": {
        "col_attr": "isocalendar_week",
        "col_count": 53,
        "col_labels": [str(i) for i in range(1, 54)],
        "unit_label": "weekly",
    },
}


def plot_returns_heatmap(
    returns: Any,
    *,
    period: Any = "month",
    ax: Axes | None = None,
) -> Figure:
    """Year-by-``period`` heatmap of compounded returns.

    Args:
        returns: 1-D series of periodic returns. Best results when the
            input has a ``DatetimeIndex``; otherwise the function
            assumes daily ('B') frequency starting at 2000-01-01.
        period: Calendar bucket per cell — ``"month"`` (default),
            ``"quarter"``, or ``"week"`` — or a
            :class:`finance_enums.Frequency` value.
        ax: Existing matplotlib ``Axes`` to draw onto.

    Returns:
        The ``matplotlib.figure.Figure`` containing the heatmap.
    """
    import pandas as pd

    freq = to_frequency(period) if Frequency is not None else str(period).lower()
    key = freq.value if Frequency is not None else freq
    if key not in _HEATMAP_PERIOD_SPEC:
        raise ValueError(f"plot_returns_heatmap: period={period!r} not supported (expected one of {sorted(_HEATMAP_PERIOD_SPEC)})")
    spec = _HEATMAP_PERIOD_SPEC[key]

    values, index = to_returns_and_index(returns)
    dt_index = _datetime_index(index, values.size)

    s = pd.Series(values, index=dt_index)
    bucketed = (1.0 + s).resample(_resample_rule(key)).prod() - 1.0
    if spec["col_attr"] == "isocalendar_week":
        col_vals = bucketed.index.isocalendar().week
    else:
        col_vals = getattr(bucketed.index, spec["col_attr"])
    table = bucketed.to_frame("r").assign(year=bucketed.index.year, col=col_vals).pivot_table(index="year", columns="col", values="r", aggfunc="sum")
    for m in range(1, spec["col_count"] + 1):
        if m not in table.columns:
            table[m] = np.nan
    table = table[sorted(table.columns)]

    fig, ax = _new_axes(ax)
    fig.set_size_inches(10, max(2.5, 0.4 * len(table)))
    arr = table.values * 100.0
    vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
    im = ax.imshow(
        arr,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(spec["col_labels"][: len(table.columns)])
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=f"{spec['unit_label']} return (%)")
    ax.set_title(f"{spec['unit_label'].capitalize()} returns")
    fig.tight_layout()
    return fig


def plot_returns_bar(
    returns: Any,
    *,
    period: Any = "year",
    ax: Axes | None = None,
) -> Figure:
    """Plot compounded returns by period as a bar chart.

    Args:
        returns: 1-D series of periodic returns.
        period: Calendar bucket: ``"day"``, ``"week"``, ``"month"``,
            ``"quarter"``, or ``"year"``.
        ax: Existing matplotlib ``Axes`` to draw onto.

    Returns:
        The ``matplotlib.figure.Figure`` containing the plot.
    """
    period_returns = _period_returns(returns, period)
    key = _period_key(period)
    label = _PERIOD_RETURN_SPEC[key]["label"]
    values = period_returns.to_numpy() * 100.0
    colors = np.where(values >= 0.0, "#2ca02c", "#d62728")

    fig, ax = _new_axes(ax)
    positions = np.arange(len(period_returns))
    ax.bar(positions, values, color=colors, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(x)[:10] for x in period_returns.index], rotation=45, ha="right")
    ax.set_title(f"{label.capitalize()} returns")
    ax.set_ylabel("return (%)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_returns_dist(
    returns: Any,
    *,
    period: Any = "month",
    bins: int = 20,
    ax: Axes | None = None,
) -> Figure:
    """Plot a histogram of compounded period returns.

    Args:
        returns: 1-D series of periodic returns.
        period: Calendar bucket: ``"day"``, ``"week"``, ``"month"``,
            ``"quarter"``, or ``"year"``.
        bins: Histogram bin count.
        ax: Existing matplotlib ``Axes`` to draw onto.

    Returns:
        The ``matplotlib.figure.Figure`` containing the plot.
    """
    period_returns = _period_returns(returns, period).dropna()
    key = _period_key(period)
    label = _PERIOD_RETURN_SPEC[key]["label"]
    values = period_returns.to_numpy() * 100.0

    fig, ax = _new_axes(ax)
    ax.hist(values, bins=bins, color="#1f77b4", alpha=0.75, edgecolor="white")
    if values.size:
        ax.axvline(np.nanmean(values), color="#d62728", linewidth=1.2, label="mean")
        ax.legend(loc="best", frameon=False)
    ax.set_title(f"Distribution of {label} returns")
    ax.set_xlabel("return (%)")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_returns_timeseries(
    returns: Any,
    *,
    period: Any = "month",
    ax: Axes | None = None,
) -> Figure:
    """Plot compounded period returns through time.

    Args:
        returns: 1-D series of periodic returns.
        period: Calendar bucket: ``"day"``, ``"week"``, ``"month"``,
            ``"quarter"``, or ``"year"``.
        ax: Existing matplotlib ``Axes`` to draw onto.

    Returns:
        The ``matplotlib.figure.Figure`` containing the plot.
    """
    period_returns = _period_returns(returns, period)
    key = _period_key(period)
    label = _PERIOD_RETURN_SPEC[key]["label"]

    fig, ax = _new_axes(ax)
    ax.plot(period_returns.index, period_returns.to_numpy() * 100.0, marker="o", linewidth=1.0, color="#1f77b4")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_title(f"{label.capitalize()} returns over time")
    ax.set_ylabel("return (%)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
