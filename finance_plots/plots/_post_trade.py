"""Post-trade diagnostic plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = ["plot_trading_cost_breakdown_bar", "plot_mfe_mae_scatter", "plot_execution_quality"]


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


def plot_trading_cost_breakdown_bar(
    costs: Any,
    *,
    component_col: str = "component",
    value_col: str = "total",
    ax: Axes | None = None,
) -> Figure:
    """Plot total trading cost by component."""
    frame = _frame(costs)
    grouped = frame.groupby(component_col, dropna=False)[value_col].sum().sort_values(ascending=False)

    fig, ax = _new_axes(ax)
    ax.bar(grouped.index.astype(str), grouped.to_numpy(), color="#4c78a8", alpha=0.9)
    ax.set_title("Trading cost breakdown")
    ax.set_ylabel("cost")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_mfe_mae_scatter(
    trades: Any,
    *,
    mae_col: str = "mae",
    mfe_col: str = "mfe",
    side_col: str = "side",
    ax: Axes | None = None,
) -> Figure:
    """Plot maximum adverse versus favorable excursion by trade."""
    frame = _frame(trades)
    fig, ax = _new_axes(ax)
    if side_col in frame.columns:
        for side, group in frame.groupby(side_col, dropna=False):
            ax.scatter(group[mae_col] * 100.0, group[mfe_col] * 100.0, s=34, alpha=0.7, label=str(side), edgecolors="none")
        ax.legend(loc="best", frameon=False)
    else:
        ax.scatter(frame[mae_col] * 100.0, frame[mfe_col] * 100.0, s=34, alpha=0.7, edgecolors="none")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_title("MAE versus MFE")
    ax.set_xlabel("maximum adverse excursion (%)")
    ax.set_ylabel("maximum favorable excursion (%)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_execution_quality(
    executions: Any,
    *,
    slippage_col: str = "implementation_shortfall_bps",
    bins: int = 20,
    ax: Axes | None = None,
) -> Figure:
    """Plot the distribution of execution-quality slippage in bps."""
    frame = _frame(executions)
    values = pd.to_numeric(frame[slippage_col], errors="coerce").dropna().to_numpy()

    fig, ax = _new_axes(ax)
    ax.hist(values, bins=min(bins, max(1, values.size)), color="#f58518", alpha=0.8, edgecolor="white")
    if values.size:
        ax.axvline(np.nanmean(values), color="#d62728", linewidth=1.2, label="mean")
        ax.legend(loc="best", frameon=False)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_title("Execution quality")
    ax.set_xlabel("implementation shortfall (bps)")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig
