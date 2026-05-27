"""Internal helpers for finance-plots.

Plots accept narwhals-compatible inputs (polars / pandas / numpy /
pyarrow) and convert internally to a numpy array of returns + a numpy
array of timestamps.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_returns_and_index(returns: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coerce ``returns`` to ``(values, index)`` numpy arrays.

    Args:
        returns: A narwhals-compatible 1-D Series (polars / pandas /
            pyarrow) or any 1-D numeric iterable.

    Returns:
        Tuple ``(values, index)``. ``values`` is a 1-D ``float64``
        array. ``index`` is taken from the input's ``.index`` attribute
        when available and shape-compatible; otherwise it is a 0-based
        integer position array.

    Raises:
        ValueError: If ``returns`` is not 1-D.
    """
    import narwhals.stable.v1 as nw

    try:
        s = nw.from_native(returns, series_only=True, allow_series=True)
        values = np.asarray(s.to_numpy(), dtype=float)
    except (TypeError, ValueError):
        values = np.asarray(returns, dtype=float)

    if values.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape={values.ndim}")

    idx = getattr(returns, "index", None)
    if idx is not None:
        try:
            index = np.asarray(idx)
        except Exception:
            index = np.arange(values.size)
    else:
        index = np.arange(values.size)
    if index.shape[0] != values.shape[0]:
        index = np.arange(values.size)
    return values, index


def cumulative_returns(values: np.ndarray) -> np.ndarray:
    """Cumulative compounded returns.

    Args:
        values: 1-D periodic returns. Non-finite entries are treated as
            zero.

    Returns:
        Array of ``(1 + r).cumprod() - 1``.
    """
    safe = np.where(np.isfinite(values), values, 0.0)
    return np.cumprod(1.0 + safe) - 1.0


def drawdown(values: np.ndarray) -> np.ndarray:
    """Per-period drawdown of an equity curve built from ``values``.

    Args:
        values: 1-D periodic returns.

    Returns:
        Array equal to ``equity / running_peak - 1``; always
        non-positive.
    """
    safe = np.where(np.isfinite(values), values, 0.0)
    equity = np.cumprod(1.0 + safe)
    peak = np.maximum.accumulate(equity)
    return equity / peak - 1.0
