"""Alpha-analysis plots."""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._util import cumulative_returns, to_returns_and_index

__all__ = [
    "plot_ic_ts",
    "plot_ic_hist",
    "plot_ic_qq",
    "plot_ic_by_group",
    "plot_ic_heatmap",
    "plot_rolling_ic",
    "plot_quantile_returns_bar",
    "plot_top_bottom_quantile_turnover",
    "plot_cumulative_factor_returns",
]


def _new_axes(ax: Axes | None, *, figsize: tuple[float, float] = (8.0, 4.0)) -> tuple[Figure, Axes]:
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    return pd.DataFrame(data)


def _series(values: Any, *, name: str = "value") -> pd.Series:
    if isinstance(values, pd.Series):
        return values.copy()
    data, index = to_returns_and_index(values)
    return pd.Series(data, index=index, name=name)


def plot_ic_ts(ic: Any, *, rolling_window: int = 21, ax: Axes | None = None) -> Figure:
    """Plot an information-coefficient time series."""
    series = _series(ic, name="ic")
    fig, ax = _new_axes(ax)
    ax.plot(series.index, series.to_numpy(), color="#4c78a8", linewidth=1.1, label="IC")
    ax.plot(series.index, series.rolling(rolling_window).mean(), color="#f58518", linewidth=1.2, label="rolling mean")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_title("Information coefficient")
    ax.set_ylabel("IC")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_ic_hist(ic: Any, *, bins: int = 20, ax: Axes | None = None) -> Figure:
    """Plot an information-coefficient histogram."""
    values = _series(ic, name="ic").dropna().to_numpy()
    fig, ax = _new_axes(ax)
    ax.hist(values, bins=min(bins, max(1, values.size)), color="#4c78a8", alpha=0.75, edgecolor="white")
    if values.size:
        ax.axvline(np.nanmean(values), color="#d62728", linewidth=1.2, label="mean")
        ax.legend(loc="best", frameon=False)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_title("IC distribution")
    ax.set_xlabel("IC")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_ic_qq(ic: Any, *, ax: Axes | None = None) -> Figure:
    """Plot information coefficients against normal quantiles."""
    values = np.sort(_series(ic, name="ic").dropna().to_numpy())
    fig, ax = _new_axes(ax)
    if values.size:
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values, ddof=1)) if values.size > 1 else 1.0
        dist = NormalDist(mu=mean, sigma=std if std > 0 else 1.0)
        probs = (np.arange(1, values.size + 1) - 0.5) / values.size
        theoretical = np.array([dist.inv_cdf(float(p)) for p in probs])
        ax.scatter(theoretical, values, s=18, color="#4c78a8", alpha=0.75, edgecolors="none")
        low = min(theoretical.min(), values.min())
        high = max(theoretical.max(), values.max())
        ax.plot([low, high], [low, high], color="#d62728", linewidth=1.0)
    ax.set_title("IC Q-Q plot")
    ax.set_xlabel("normal quantile")
    ax.set_ylabel("observed IC")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_ic_by_group(data: Any, *, group_col: str = "group", ic_col: str = "ic", ax: Axes | None = None) -> Figure:
    """Plot mean information coefficient by group."""
    frame = _frame(data)
    grouped = frame.groupby(group_col, dropna=False)[ic_col].mean().sort_values(ascending=False)
    fig, ax = _new_axes(ax)
    ax.bar(grouped.index.astype(str), grouped.to_numpy(), color="#54a24b", alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_title("IC by group")
    ax.set_ylabel("mean IC")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_ic_heatmap(ic: Any, *, period: str = "month", ax: Axes | None = None) -> Figure:
    """Calendar heatmap of mean information coefficient."""
    series = _series(ic, name="ic")
    index = pd.to_datetime(series.index)
    if period == "quarter":
        labels = ["Q1", "Q2", "Q3", "Q4"]
        cols = index.quarter
    elif period == "week":
        labels = [str(i) for i in range(1, 54)]
        cols = index.isocalendar().week.astype(int)
    else:
        labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        cols = index.month
    table = pd.DataFrame({"ic": series.to_numpy(), "year": index.year, "col": cols}).pivot_table(
        index="year", columns="col", values="ic", aggfunc="mean"
    )
    for col in range(1, len(labels) + 1):
        if col not in table.columns:
            table[col] = np.nan
    table = table[sorted(table.columns)]

    fig, ax = _new_axes(ax, figsize=(10.0, max(2.5, 0.4 * max(1, len(table)))))
    arr = table.to_numpy()
    vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(labels[: len(table.columns)])
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    fig.colorbar(im, ax=ax, label="mean IC")
    ax.set_title("IC heatmap")
    fig.tight_layout()
    return fig


def plot_rolling_ic(ic: Any, *, window: int = 21, ax: Axes | None = None) -> Figure:
    """Plot rolling mean information coefficient."""
    series = _series(ic, name="ic")
    rolling = series.rolling(window).mean()
    fig, ax = _new_axes(ax)
    ax.plot(rolling.index, rolling.to_numpy(), color="#4c78a8", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_title(f"Rolling {window}-period IC")
    ax.set_ylabel("IC")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_quantile_returns_bar(data: Any, *, quantile_col: str = "quantile", return_col: str = "return", ax: Axes | None = None) -> Figure:
    """Plot mean return by signal quantile."""
    frame = _frame(data)
    grouped = frame.groupby(quantile_col, dropna=False)[return_col].mean().sort_index()
    values = grouped.to_numpy() * 100.0
    colors = np.where(values >= 0.0, "#54a24b", "#d62728")
    fig, ax = _new_axes(ax)
    ax.bar(grouped.index.astype(str), values, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_title("Returns by quantile")
    ax.set_ylabel("mean return (%)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_top_bottom_quantile_turnover(
    data: Any,
    *,
    quantile_col: str = "quantile",
    turnover_col: str = "turnover",
    ax: Axes | None = None,
) -> Figure:
    """Plot turnover for bottom and top quantiles."""
    frame = _frame(data)
    grouped = frame.groupby(quantile_col, dropna=False)[turnover_col].mean().sort_index()
    selected = grouped.loc[[grouped.index.min(), grouped.index.max()]]
    fig, ax = _new_axes(ax)
    ax.bar([str(x) for x in selected.index], selected.to_numpy() * 100.0, color=["#4c78a8", "#f58518"], alpha=0.9)
    ax.set_title("Top and bottom quantile turnover")
    ax.set_xlabel("quantile")
    ax.set_ylabel("turnover (%)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_cumulative_factor_returns(factor_returns: Any, *, ax: Axes | None = None) -> Figure:
    """Plot compounded factor returns."""
    values, index = to_returns_and_index(factor_returns)
    fig, ax = _new_axes(ax)
    ax.plot(index, cumulative_returns(values), color="#4c78a8", linewidth=1.4)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_title("Cumulative factor returns")
    ax.set_ylabel("cumulative return")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
