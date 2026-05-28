"""finance-plots — plot helpers."""

from ._alpha import (
    plot_cumulative_factor_returns,
    plot_ic_by_group,
    plot_ic_heatmap,
    plot_ic_hist,
    plot_ic_qq,
    plot_ic_ts,
    plot_quantile_returns_bar,
    plot_rolling_ic,
    plot_top_bottom_quantile_turnover,
)
from ._indicators import plot_indicator_panel, plot_price_with_overlays
from ._post_trade import plot_execution_quality, plot_mfe_mae_scatter, plot_trading_cost_breakdown_bar
from ._returns import (
    plot_drawdown_underwater,
    plot_return_scatter,
    plot_returns,
    plot_returns_bar,
    plot_returns_dist,
    plot_returns_heatmap,
    plot_returns_timeseries,
    plot_rolling_beta,
    plot_rolling_correlation,
    plot_rolling_returns,
    plot_rolling_sharpe,
    plot_rolling_volatility,
)

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
    "plot_indicator_panel",
    "plot_price_with_overlays",
    "plot_trading_cost_breakdown_bar",
    "plot_mfe_mae_scatter",
    "plot_execution_quality",
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
