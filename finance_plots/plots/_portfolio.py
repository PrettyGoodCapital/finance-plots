"""Portfolio construction and risk model plots."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "plot_efficient_frontier",
    "plot_market_impact_curve",
    "plot_execution_timeline",
    "plot_cost_breakdown_bar",
    "plot_return_attribution_stacked",
    "plot_portfolio_weight_evolution",
    "plot_weight_diff",
    "plot_risk_decomposition_stacked",
    "plot_factor_exposure_heatmap",
    "plot_correlation_matrix",
    "plot_covariance_eigenvalues",
]


def _new_axes(ax: Axes | None, *, figsize: tuple[float, float] = (8, 4)) -> tuple[Figure, Axes]:
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _matrix(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("covariance must be a square matrix")
    return arr


def _labels(labels: Sequence[str] | None, size: int) -> list[str]:
    if labels is None:
        return [f"asset {i + 1}" for i in range(size)]
    out = [str(label) for label in labels]
    if len(out) != size:
        raise ValueError("labels length must match matrix dimensions")
    return out


def _dataframe_like(data: Any):
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        return data
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    return pd.DataFrame(data)


def plot_efficient_frontier(expected_returns: Any, covariance: Any, *, points: int = 50, ax: Axes | None = None) -> Figure:
    """Plot a long-only unconstrained mean-variance efficient frontier."""
    mu = np.asarray(expected_returns, dtype=float).reshape(-1)
    cov = _matrix(covariance)
    if mu.size != cov.shape[0]:
        raise ValueError("expected_returns length must match covariance dimensions")
    fig, ax = _new_axes(ax)
    inv = np.linalg.pinv(cov)
    ones = np.ones(mu.size)
    a = float(ones @ inv @ ones)
    b = float(ones @ inv @ mu)
    c = float(mu @ inv @ mu)
    denom = a * c - b**2
    targets = np.linspace(mu.min(), mu.max(), max(points, 2))
    frontier_returns: list[float] = []
    frontier_vols: list[float] = []
    for target in targets:
        if abs(denom) < 1e-12:
            weights = ones / ones.size
        else:
            lam = (c - b * target) / denom
            gam = (a * target - b) / denom
            weights = inv @ (lam * ones + gam * mu)
        variance = float(weights @ cov @ weights)
        frontier_returns.append(float(weights @ mu))
        frontier_vols.append(float(np.sqrt(max(variance, 0.0))))
    asset_vols = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    ax.plot(frontier_vols, frontier_returns, color="#1f77b4", linewidth=1.6, label="frontier")
    ax.scatter(asset_vols, mu, color="#444", s=24, label="assets")
    ax.set_title("Efficient frontier")
    ax.set_xlabel("volatility")
    ax.set_ylabel("expected return")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    return fig


def plot_market_impact_curve(
    impact_frame: Any,
    *,
    participation_col: str = "participation_rate",
    impact_col: str = "impact_bps",
    ax: Axes | None = None,
) -> Figure:
    """Plot market impact against participation rate."""
    frame = _dataframe_like(impact_frame)
    x = np.asarray(frame[participation_col], dtype=float)
    y = np.asarray(frame[impact_col], dtype=float)
    order = np.argsort(x)

    fig, ax = _new_axes(ax)
    ax.plot(x[order], y[order], color="#b45309", linewidth=1.8)
    ax.scatter(x, y, color="#b45309", s=22, alpha=0.8)
    ax.set_title("Market impact curve")
    ax.set_xlabel("participation rate")
    ax.set_ylabel("impact (bps)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_execution_timeline(
    executions: Any,
    *,
    time_col: str = "timestamp",
    executed_col: str = "executed_qty",
    target_col: str | None = "target_qty",
    ax: Axes | None = None,
) -> Figure:
    """Plot cumulative executed quantity versus target trajectory."""
    frame = _dataframe_like(executions).sort_values(time_col)
    x = frame[time_col]
    executed = np.asarray(frame[executed_col], dtype=float)
    cumulative = np.cumsum(executed)

    fig, ax = _new_axes(ax)
    ax.plot(x, cumulative, color="#1f77b4", linewidth=1.8, label="executed")
    if target_col is not None and target_col in frame.columns:
        target = np.asarray(frame[target_col], dtype=float)
        ax.plot(x, target, color="#7f7f7f", linewidth=1.4, linestyle="--", label="target")
    ax.set_title("Execution timeline")
    ax.set_xlabel("time")
    ax.set_ylabel("quantity")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    return fig


def plot_cost_breakdown_bar(
    costs: Any,
    *,
    component_col: str = "component",
    value_col: str = "total",
    ax: Axes | None = None,
) -> Figure:
    """Plot signed cost contributions by component."""
    frame = _dataframe_like(costs)
    grouped = frame.groupby(component_col, dropna=False)[value_col].sum().sort_values(ascending=False)
    values = grouped.to_numpy(dtype=float)

    fig, ax = _new_axes(ax)
    colors = np.where(values >= 0.0, "#b91c1c", "#047857")
    ax.bar(grouped.index.astype(str), values, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title("Cost breakdown")
    ax.set_ylabel("cost")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_return_attribution_stacked(
    attribution: Any,
    *,
    time_col: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot stacked return attribution through time."""
    frame = _dataframe_like(attribution)
    if time_col is not None and time_col in frame.columns:
        x = frame[time_col].to_numpy()
        value_frame = frame.drop(columns=[time_col])
    else:
        x = frame.index.to_numpy() if hasattr(frame.index, "to_numpy") else np.arange(len(frame))
        value_frame = frame

    values = value_frame.to_numpy(dtype=float).T
    labels = [str(col) for col in value_frame.columns]

    fig, ax = _new_axes(ax)
    ax.stackplot(x, values, labels=labels, alpha=0.85)
    ax.set_title("Return attribution")
    ax.set_ylabel("return contribution")
    ax.grid(alpha=0.2, axis="y")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    return fig


def plot_portfolio_weight_evolution(weights: Any, *, ax: Axes | None = None) -> Figure:
    frame = _dataframe_like(weights)
    fig, ax = _new_axes(ax)
    x = frame.index.to_numpy() if hasattr(frame.index, "to_numpy") else np.arange(len(frame))
    labels = [str(column) for column in frame.columns]
    values = frame.to_numpy(dtype=float).T
    ax.stackplot(x, values, labels=labels, alpha=0.85)
    ax.set_title("Portfolio weight evolution")
    ax.set_ylabel("weight")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return fig


def plot_weight_diff(current: Mapping[str, float], target: Mapping[str, float], *, ax: Axes | None = None) -> Figure:
    assets = sorted(set(current) | set(target))
    diff = np.array([float(target.get(asset, 0.0)) - float(current.get(asset, 0.0)) for asset in assets])
    fig, ax = _new_axes(ax)
    colors = np.where(diff >= 0.0, "#2ca02c", "#d62728")
    ax.bar(assets, diff, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title("Target weight difference")
    ax.set_ylabel("target - current")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return fig


def plot_risk_decomposition_stacked(decomposition: Any, *, ax: Axes | None = None) -> Figure:
    frame = _dataframe_like(decomposition)
    fig, ax = _new_axes(ax)
    bottom = np.zeros(len(frame))
    x = np.arange(len(frame))
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=str(column))
        bottom += values
    ax.set_xticks(x, [str(idx) for idx in frame.index])
    ax.set_title("Risk decomposition")
    ax.set_ylabel("risk contribution")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return fig


def plot_factor_exposure_heatmap(exposures: Any, *, ax: Axes | None = None) -> Figure:
    frame = _dataframe_like(exposures)
    fig, ax = _new_axes(ax, figsize=(7, 5))
    values = frame.to_numpy(dtype=float)
    bound = max(float(np.nanmax(np.abs(values))), 1e-12)
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
    ax.set_xticks(np.arange(len(frame.columns)), [str(column) for column in frame.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(frame.index)), [str(idx) for idx in frame.index])
    ax.set_title("Factor exposures")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_correlation_matrix(covariance_or_correlation: Any, *, labels: Sequence[str] | None = None, ax: Axes | None = None) -> Figure:
    matrix = _matrix(covariance_or_correlation)
    diagonal = np.diag(matrix)
    if not np.allclose(diagonal, 1.0):
        std = np.sqrt(np.clip(diagonal, 0.0, None))
        denom = np.outer(std, std)
        matrix = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom > 0.0)
        np.fill_diagonal(matrix, 1.0)
    names = _labels(labels, matrix.shape[0])
    fig, ax = _new_axes(ax, figsize=(6, 5))
    image = ax.imshow(np.clip(matrix, -1.0, 1.0), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_title("Correlation matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_covariance_eigenvalues(covariance: Any, *, ax: Axes | None = None) -> Figure:
    cov = _matrix(covariance)
    eigvals = np.linalg.eigvalsh((cov + cov.T) / 2.0)[::-1]
    fig, ax = _new_axes(ax)
    ax.bar(np.arange(1, eigvals.size + 1), eigvals, color="#1f77b4")
    ax.set_title("Covariance eigenvalues")
    ax.set_xlabel("component")
    ax.set_ylabel("eigenvalue")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return fig
