"""Multi-panel technical-indicator plots."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .._util import to_returns_and_index


def _to_array(values: Any) -> np.ndarray:
    arr, _ = to_returns_and_index(values)
    return arr


def plot_indicator_panel(
    price: Any,
    panels: Sequence[dict] | None = None,
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Plot a price chart on top of N indicator sub-panels.

    Args:
        price: Narwhals-compatible 1-D price series.
        panels: Iterable of panel specifications. Each panel is a dict
            with keys ``title`` (str) and ``series`` (iterable of
            ``(label, values)`` pairs). Each ``values`` must be the
            same length as ``price``. If ``None`` or empty, only the
            price panel is drawn.
        figsize: Matplotlib figure size. Defaults to ``(10, 2 + 2*N)``
            where ``N`` is the number of indicator panels.
        title: Optional figure title.

    Returns:
        ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If a panel series length does not match ``price``.
    """
    import matplotlib.pyplot as plt

    panels = list(panels or [])
    n_panels = 1 + len(panels)
    if figsize is None:
        figsize = (10.0, 2.0 + 2.0 * n_panels)

    fig, axes = plt.subplots(
        n_panels,
        1,
        sharex=True,
        figsize=figsize,
        gridspec_kw={"height_ratios": [2.0] + [1.0] * len(panels)},
    )
    if n_panels == 1:
        axes = [axes]

    price_arr, index = to_returns_and_index(price)
    n = price_arr.size

    ax_price = axes[0]
    ax_price.plot(index, price_arr, color="#1f77b4", linewidth=1.2, label="price")
    ax_price.set_ylabel("price")
    ax_price.grid(True, alpha=0.3)
    if title:
        ax_price.set_title(title)

    for ax, panel in zip(axes[1:], panels):
        for label, values in panel.get("series", []):
            arr = _to_array(values)
            if arr.size != n:
                raise ValueError(f"panel series '{label}' length {arr.size} != price length {n}")
            ax.plot(index, arr, linewidth=1.0, label=label)
        ax.set_ylabel(panel.get("title", ""))
        ax.grid(True, alpha=0.3)
        if any(True for _ in panel.get("series", [])):
            ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("t")
    fig.tight_layout()
    return fig


def plot_price_with_overlays(
    price: Any,
    overlays: Iterable[tuple[str, Any]] | None = None,
    *,
    secondary_overlays: Iterable[tuple[str, Any]] | None = None,
    secondary_ylabel: str | None = None,
    figsize: tuple[float, float] = (10.0, 4.0),
    title: str | None = None,
):
    """Plot a price line with same-axis and optional secondary-axis overlays.

    Use ``overlays`` for moving averages, Bollinger / Donchian bands, or other
    indicator series measured in price units. Use ``secondary_overlays`` for
    bounded or differently-scaled indicators such as RSI.

    Args:
        price: Narwhals-compatible 1-D price series.
        overlays: Iterable of ``(label, values)`` pairs, each the same
            length as ``price``.
        secondary_overlays: Iterable of ``(label, values)`` pairs drawn
            on a right-hand y-axis.
        secondary_ylabel: Label for the right-hand y-axis.
        figsize: Matplotlib figure size.
        title: Optional figure title.

    Returns:
        ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If an overlay's length does not match ``price``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    price_arr, index = to_returns_and_index(price)
    n = price_arr.size
    ax.plot(index, price_arr, color="#1f77b4", linewidth=1.2, label="price")
    for label, values in overlays or ():
        arr = _to_array(values)
        if arr.size != n:
            raise ValueError(f"overlay '{label}' length {arr.size} != price length {n}")
        ax.plot(index, arr, linewidth=1.0, label=label)
    ax.set_ylabel("price")
    ax.grid(True, alpha=0.3)

    secondary_ax = None
    secondary_is_rsi = False
    secondary_colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for i, (label, values) in enumerate(secondary_overlays or ()):
        arr = _to_array(values)
        if arr.size != n:
            raise ValueError(f"secondary overlay '{label}' length {arr.size} != price length {n}")
        if secondary_ax is None:
            secondary_ax = ax.twinx()
        secondary_ax.plot(
            index,
            arr,
            color=secondary_colors[i % len(secondary_colors)],
            linewidth=1.0,
            linestyle="--",
            label=label,
        )
        secondary_is_rsi = secondary_is_rsi or "rsi" in label.lower()
    if secondary_ax is not None:
        ylabel = secondary_ylabel or "secondary"
        secondary_ax.set_ylabel(ylabel)
        if secondary_is_rsi or "rsi" in ylabel.lower():
            secondary_ax.set_ylim(0.0, 100.0)
            secondary_ax.axhline(30.0, color="#9467bd", linewidth=0.7, linestyle=":", alpha=0.35)
            secondary_ax.axhline(70.0, color="#9467bd", linewidth=0.7, linestyle=":", alpha=0.35)
            secondary_ax.tick_params(axis="y", colors="#9467bd")
            secondary_ax.yaxis.label.set_color("#9467bd")

    handles, labels = ax.get_legend_handles_labels()
    if secondary_ax is not None:
        secondary_handles, secondary_labels = secondary_ax.get_legend_handles_labels()
        handles.extend(secondary_handles)
        labels.extend(secondary_labels)
    ax.legend(handles, labels, loc="upper left", fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
