# API

`finance-plots` exposes plotting and table helpers at the package root:

```python
from finance_plots import (
    plot_returns,
    plot_rolling_returns,
    plot_rolling_volatility,
    plot_rolling_sharpe,
    plot_rolling_beta,
    plot_rolling_correlation,
    plot_return_scatter,
    plot_drawdown_underwater,
    plot_returns_heatmap,
    plot_returns_bar,
    plot_returns_dist,
    plot_returns_timeseries,
    plot_indicator_panel,
    plot_price_with_overlays,
    plot_trading_cost_breakdown_bar,
    plot_mfe_mae_scatter,
    plot_execution_quality,
    plot_ic_ts,
    plot_ic_hist,
    plot_ic_qq,
    plot_ic_by_group,
    plot_ic_heatmap,
    plot_rolling_ic,
    plot_quantile_returns_bar,
    plot_top_bottom_quantile_turnover,
    plot_cumulative_factor_returns,
    perf_stats,
    table_perf_stats,
    table_period_returns,
    table_drawdowns,
    table_cost_breakdown,
    table_round_trip_stats,
    table_execution_quality,
    table_information,
    table_returns_by_quantile,
    table_turnover,
    table_quantile_statistics,
)
```

Plots accept Narwhals-compatible one-dimensional inputs such as pandas Series,
Polars Series, numpy arrays, and other supported backends. Plot functions return
`matplotlib.figure.Figure`. Table helpers either return a Python dictionary or a
`great_tables.GT` object.

______________________________________________________________________

## Return and Risk Plots

### `plot_returns(returns, live_start=None, log_scale=False, ax=None)`

Cumulative strategy returns without requiring a benchmark argument.

![plot_returns](../assets/gallery/plot_returns.png)

### `plot_rolling_returns(returns, benchmark=None, live_start=None, log_scale=False, ax=None)`

Cumulative strategy returns with optional benchmark and out-of-sample shading.

![plot_rolling_returns](../assets/gallery/plot_rolling_returns.png)

### `plot_rolling_volatility(returns, window=63, periods_per_year=252, ax=None)`

Rolling annualized volatility.

![plot_rolling_volatility](../assets/gallery/plot_rolling_volatility.png)

### `plot_rolling_sharpe(returns, window=63, periods_per_year=252, ax=None)`

Rolling annualized Sharpe ratio.

![plot_rolling_sharpe](../assets/gallery/plot_rolling_sharpe.png)

### `plot_rolling_beta(returns, benchmark, window=63, ax=None)`

Rolling beta versus a benchmark return series.

![plot_rolling_beta](../assets/gallery/plot_rolling_beta.png)

### `plot_rolling_correlation(returns, benchmark, window=63, ax=None)`

Rolling correlation versus a benchmark return series.

![plot_rolling_correlation](../assets/gallery/plot_rolling_correlation.png)

### `plot_return_scatter(returns, benchmark, ax=None)`

Strategy returns plotted against benchmark returns with a fitted beta line.

![plot_return_scatter](../assets/gallery/plot_return_scatter.png)

### `plot_drawdown_underwater(returns, ax=None)`

Underwater drawdown chart built from compounded returns.

![plot_drawdown_underwater](../assets/gallery/plot_drawdown_underwater.png)

### `plot_returns_heatmap(returns, period="month", ax=None)`

Calendar return heatmap for month, quarter, or week buckets.

![plot_returns_heatmap](../assets/gallery/plot_returns_heatmap.png)

### `plot_returns_bar(returns, period="year", ax=None)`

Compounded period returns as a bar chart.

![plot_returns_bar](../assets/gallery/plot_returns_bar.png)

### `plot_returns_dist(returns, period="month", bins=20, ax=None)`

Distribution of compounded period returns.

![plot_returns_dist](../assets/gallery/plot_returns_dist.png)

### `plot_returns_timeseries(returns, period="month", ax=None)`

Compounded period returns through time.

![plot_returns_timeseries](../assets/gallery/plot_returns_timeseries.png)

## Technical Indicator Plots

### `plot_price_with_overlays(price, overlays=None, secondary_overlays=None, secondary_ylabel=None, figsize=(10.0, 4.0), title=None)`

Price line with same-axis overlays and optional right-axis indicators such as RSI.

![plot_price_with_overlays](../assets/gallery/plot_price_with_overlays.png)

### `plot_indicator_panel(price, panels=None, figsize=None, title=None)`

Price chart with configurable aligned indicator sub-panels.

![plot_indicator_panel](../assets/gallery/plot_indicator_panel.png)

## Post-Trade Plots

### `plot_trading_cost_breakdown_bar(costs, component_col="component", value_col="total", ax=None)`

Trading cost attribution by component.

![plot_trading_cost_breakdown_bar](../assets/gallery/plot_trading_cost_breakdown_bar.png)

### `plot_mfe_mae_scatter(trades, mae_col="mae", mfe_col="mfe", side_col="side", ax=None)`

Maximum adverse versus favorable excursion by trade.

![plot_mfe_mae_scatter](../assets/gallery/plot_mfe_mae_scatter.png)

### `plot_execution_quality(executions, slippage_col="implementation_shortfall_bps", bins=20, ax=None)`

Distribution of implementation-shortfall slippage in basis points.

![plot_execution_quality](../assets/gallery/plot_execution_quality.png)

## Alpha-Analysis Plots

### `plot_ic_ts(ic, rolling_window=21, ax=None)`

Information-coefficient time series with a rolling mean overlay.

![plot_ic_ts](../assets/gallery/plot_ic_ts.png)

### `plot_ic_hist(ic, bins=20, ax=None)`

Information-coefficient distribution.

![plot_ic_hist](../assets/gallery/plot_ic_hist.png)

### `plot_ic_qq(ic, ax=None)`

Information-coefficient Q-Q plot against a normal distribution.

![plot_ic_qq](../assets/gallery/plot_ic_qq.png)

### `plot_ic_by_group(data, group_col="group", ic_col="ic", ax=None)`

Mean information coefficient by group.

![plot_ic_by_group](../assets/gallery/plot_ic_by_group.png)

### `plot_ic_heatmap(ic, period="month", ax=None)`

Calendar heatmap of mean information coefficient.

![plot_ic_heatmap](../assets/gallery/plot_ic_heatmap.png)

### `plot_rolling_ic(ic, window=21, ax=None)`

Rolling mean information coefficient.

![plot_rolling_ic](../assets/gallery/plot_rolling_ic.png)

### `plot_quantile_returns_bar(data, quantile_col="quantile", return_col="return", ax=None)`

Mean forward return by signal quantile.

![plot_quantile_returns_bar](../assets/gallery/plot_quantile_returns_bar.png)

### `plot_top_bottom_quantile_turnover(data, quantile_col="quantile", turnover_col="turnover", ax=None)`

Turnover for the bottom and top signal quantiles.

![plot_top_bottom_quantile_turnover](../assets/gallery/plot_top_bottom_quantile_turnover.png)

### `plot_cumulative_factor_returns(factor_returns, ax=None)`

Compounded long-short factor return path.

![plot_cumulative_factor_returns](../assets/gallery/plot_cumulative_factor_returns.png)

## Performance Tables

### `perf_stats(returns, periods_per_year=252)`

Compute scalar performance statistics.

```{include} ../assets/gallery/perf_stats.md
```

### `table_perf_stats(returns, benchmark=None, periods_per_year=252)`

Build a Great Tables performance summary.

```{include} ../assets/gallery/table_perf_stats.md
```

[table_perf_stats.html](../assets/gallery/table_perf_stats.html)

### `table_period_returns(returns, period="year")`

Build a Great Tables table of compounded period returns.

```{include} ../assets/gallery/table_period_returns.md
```

### `table_drawdowns(returns, top=5)`

Build a Great Tables table of the largest drawdown periods.

```{include} ../assets/gallery/table_drawdowns.md
```

## Post-Trade Tables

### `table_cost_breakdown(costs, component_col="component", value_col="total")`

Build a Great Tables trading-cost attribution summary.

```{include} ../assets/gallery/table_cost_breakdown.md
```

### `table_round_trip_stats(trades, pnl_col="pnl")`

Build a Great Tables round-trip trade-quality summary.

```{include} ../assets/gallery/table_round_trip_stats.md
```

### `table_execution_quality(executions, slippage_col="implementation_shortfall_bps")`

Build a Great Tables implementation-shortfall summary.

```{include} ../assets/gallery/table_execution_quality.md
```

## Alpha-Analysis Tables

### `table_information(ic)`

Build a Great Tables information-coefficient summary.

```{include} ../assets/gallery/table_information.md
```

### `table_returns_by_quantile(data, quantile_col="quantile", return_col="return")`

Build a Great Tables mean-return-by-quantile table.

```{include} ../assets/gallery/table_returns_by_quantile.md
```

### `table_turnover(data, quantile_col="quantile", turnover_col="turnover")`

Build a Great Tables quantile-turnover summary.

```{include} ../assets/gallery/table_turnover.md
```

### `table_quantile_statistics(data, quantile_col="quantile", signal_col="signal_mean", count_col="count")`

Build a Great Tables quantile count and signal-statistics summary.

```{include} ../assets/gallery/table_quantile_statistics.md
```

## Example Artifact Helper

| Function                                                                   | Description                                              |
| -------------------------------------------------------------------------- | -------------------------------------------------------- |
| `finance_plots.gallery.generate_gallery(output_dir="docs/assets/gallery")` | Write one example artifact per current public plot/table |

______________________________________________________________________

## Reference

```{eval-rst}
.. currentmodule:: finance_plots

.. autofunction:: plot_returns

.. autofunction:: plot_rolling_returns

.. autofunction:: plot_rolling_volatility

.. autofunction:: plot_rolling_sharpe

.. autofunction:: plot_rolling_beta

.. autofunction:: plot_rolling_correlation

.. autofunction:: plot_return_scatter

.. autofunction:: plot_drawdown_underwater

.. autofunction:: plot_returns_heatmap

.. autofunction:: plot_returns_bar

.. autofunction:: plot_returns_dist

.. autofunction:: plot_returns_timeseries

.. autofunction:: plot_price_with_overlays

.. autofunction:: plot_indicator_panel

.. autofunction:: plot_trading_cost_breakdown_bar

.. autofunction:: plot_mfe_mae_scatter

.. autofunction:: plot_execution_quality

.. autofunction:: plot_ic_ts

.. autofunction:: plot_ic_hist

.. autofunction:: plot_ic_qq

.. autofunction:: plot_ic_by_group

.. autofunction:: plot_ic_heatmap

.. autofunction:: plot_rolling_ic

.. autofunction:: plot_quantile_returns_bar

.. autofunction:: plot_top_bottom_quantile_turnover

.. autofunction:: plot_cumulative_factor_returns

.. autofunction:: perf_stats

.. autofunction:: table_perf_stats

.. autofunction:: table_period_returns

.. autofunction:: table_drawdowns

.. autofunction:: table_cost_breakdown

.. autofunction:: table_round_trip_stats

.. autofunction:: table_execution_quality

.. autofunction:: table_information

.. autofunction:: table_returns_by_quantile

.. autofunction:: table_turnover

.. autofunction:: table_quantile_statistics


.. currentmodule:: finance_plots.gallery

.. autofunction:: generate_gallery
```
