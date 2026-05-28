# Examples

These examples use `finance-datagen` for deterministic market-like inputs and
`finance-calcs` for returns and technical indicators. The plotting functions
then consume ordinary pandas Series, Polars Series, or numpy arrays.

______________________________________________________________________

## Shared Setup

```python
from datetime import date, datetime, timezone

import pandas as pd
import polars as pl
from finance_datagen import generate_prices, generate_signal

import finance_calcs as fc
import finance_plots as fp


def as_series(frame: pl.DataFrame, column: str, *, drop_nulls: bool = False) -> pd.Series:
    data = frame.select("timestamp", column)
    if drop_nulls:
        data = data.drop_nulls()
    return pd.Series(
        data[column].to_numpy(),
        index=data["timestamp"].to_pandas(),
        name=column,
    )


start_ms = int(datetime(2021, 1, 4, tzinfo=timezone.utc).timestamp() * 1000)
prices = generate_prices(n_steps=756, symbol="ACME", seed=7, start_ms=start_ms)
benchmark_prices = generate_prices(
    n_steps=756,
    symbol="BENCH",
    seed=11,
    start_ms=start_ms,
    mu=0.04,
    sigma=0.16,
)

price_frame = prices.with_columns(
    fc.simple_returns(pl.col("price")).alias("ret"),
    fc.sma(pl.col("price"), period=20).alias("sma20"),
    fc.ema(pl.col("price"), period=60).alias("ema60"),
    fc.rsi(pl.col("price"), period=14).alias("rsi14"),
    fc.macd_line(pl.col("price")).alias("macd"),
    fc.macd_signal(pl.col("price")).alias("macd_signal"),
)
benchmark_frame = benchmark_prices.with_columns(
    fc.simple_returns(pl.col("price")).alias("ret"),
)

returns = as_series(price_frame, "ret", drop_nulls=True)
benchmark = as_series(benchmark_frame, "ret", drop_nulls=True)
price = as_series(price_frame, "price")
sma20 = as_series(price_frame, "sma20")
ema60 = as_series(price_frame, "ema60")
rsi14 = as_series(price_frame, "rsi14")
macd = as_series(price_frame, "macd")
macd_signal = as_series(price_frame, "macd_signal")

trade_transactions = pl.DataFrame(
    {
        "timestamp": [date(2021, 1, 4), date(2021, 1, 6), date(2021, 1, 8), date(2021, 1, 11), date(2021, 1, 13)],
        "symbol": ["ACME", "ACME", "ACME", "BETA", "BETA"],
        "amount": [100.0, -40.0, -60.0, -80.0, 80.0],
        "price": [100.0, 106.0, 96.0, 50.0, 44.0],
        "commission": [1.0, 1.0, 1.0, 1.0, 1.0],
        "fees": [0.25, 0.25, 0.25, 0.25, 0.25],
        "bps": [4.0, 6.0, 5.0, 7.0, 4.0],
    }
)
cost_breakdown = fc.cost_attribution(trade_transactions)
round_trips = fc.extract_round_trips(trade_transactions)
excursion_prices = pl.DataFrame(
    {
        "timestamp": [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6), date(2021, 1, 7), date(2021, 1, 8)] * 2,
        "symbol": ["ACME"] * 5 + ["BETA"] * 5,
        "price": [100.0, 94.0, 106.0, 112.0, 96.0, 50.0, 53.0, 47.0, 43.0, 44.0],
    }
)
trades_with_excursions = fc.mae_mfe(round_trips, excursion_prices)
execution_quality = pd.DataFrame(
    {
        "timestamp": pd.date_range("2021-01-04", periods=12, freq="B"),
        "implementation_shortfall_bps": [9.0, 12.0, -3.0, 6.0, 15.0, 4.0, 8.0, -2.0, 11.0, 7.0, 5.0, 13.0],
    }
)

signals = generate_signal(n_dates=80, n_assets=40, ic=0.12, seed=23, start=date(2021, 1, 4)).with_columns(
    pl.when(pl.col("symbol").str.slice(-1).is_in(["0", "2", "4", "6", "8"]))
    .then(pl.lit("Tech"))
    .otherwise(pl.lit("Energy"))
    .alias("group")
)
signals = signals.with_columns(fc.assign_quantile(pl.col("signal"), 5).over("date").alias("quantile"))
ic_frame = signals.group_by("date").agg(fc.spearman_ic(pl.col("signal"), pl.col("fwd_returns")).alias("ic")).sort("date")
ic_by_group = signals.group_by("date", "group").agg(fc.spearman_ic(pl.col("signal"), pl.col("fwd_returns")).alias("ic")).sort("date")
changed = signals.sort("symbol", "date").with_columns(fc.quantile_changed(pl.col("quantile")).over("symbol").alias("changed"))
turnover = changed.group_by("date", "quantile").agg(fc.quantile_turnover(pl.col("changed")).alias("turnover"))
quantile_returns = signals.group_by("date", "quantile").agg(
    pl.col("fwd_returns").mean().alias("return"),
    pl.len().alias("count"),
    pl.col("signal").mean().alias("signal_mean"),
)
alpha_frame = quantile_returns.join(turnover, on=["date", "quantile"], how="left").sort("date", "quantile").to_pandas()
factor_returns_frame = signals.group_by("date").agg(
    fc.long_short_spread(pl.col("fwd_returns"), pl.col("quantile"), upper=4, lower=0).alias("factor_return")
).sort("date")
ic = pd.Series(ic_frame["ic"].to_numpy(), index=ic_frame["date"].to_pandas(), name="ic")
factor_returns = pd.Series(
    factor_returns_frame["factor_return"].to_numpy(),
    index=factor_returns_frame["date"].to_pandas(),
    name="factor_return",
)
```

______________________________________________________________________

## Return Path

```python
fig = fp.plot_returns(returns)
```

![plot_returns](../assets/gallery/plot_returns.png)

## Return Path With Benchmark

```python
fig = fp.plot_rolling_returns(
    returns,
    benchmark=benchmark,
    live_start=returns.index[int(len(returns) * 0.7)],
)
```

![plot_rolling_returns](../assets/gallery/plot_rolling_returns.png)

## Rolling Volatility, Sharpe, and Beta

```python
vol_fig = fp.plot_rolling_volatility(returns, window=63)
sharpe_fig = fp.plot_rolling_sharpe(returns, window=63)
beta_fig = fp.plot_rolling_beta(returns, benchmark, window=63)
```

![plot_rolling_volatility](../assets/gallery/plot_rolling_volatility.png)

![plot_rolling_sharpe](../assets/gallery/plot_rolling_sharpe.png)

![plot_rolling_beta](../assets/gallery/plot_rolling_beta.png)

## Benchmark Relationship

```python
corr_fig = fp.plot_rolling_correlation(returns, benchmark, window=63)
scatter_fig = fp.plot_return_scatter(returns, benchmark)
```

![plot_rolling_correlation](../assets/gallery/plot_rolling_correlation.png)

![plot_return_scatter](../assets/gallery/plot_return_scatter.png)

## Drawdown

```python
fig = fp.plot_drawdown_underwater(returns)
```

![plot_drawdown_underwater](../assets/gallery/plot_drawdown_underwater.png)

## Return Heatmap

```python
fig = fp.plot_returns_heatmap(returns, period="month")
```

![plot_returns_heatmap](../assets/gallery/plot_returns_heatmap.png)

Quarterly and weekly buckets use the same function:

```python
quarterly_fig = fp.plot_returns_heatmap(returns, period="quarter")
weekly_fig = fp.plot_returns_heatmap(returns, period="week")
```

## Period Return Bar, Distribution, and Timeseries

```python
annual_bar = fp.plot_returns_bar(returns, period="year")
monthly_dist = fp.plot_returns_dist(returns, period="month")
monthly_series = fp.plot_returns_timeseries(returns, period="month")
```

![plot_returns_bar](../assets/gallery/plot_returns_bar.png)

![plot_returns_dist](../assets/gallery/plot_returns_dist.png)

![plot_returns_timeseries](../assets/gallery/plot_returns_timeseries.png)

## Price Overlays

```python
fig = fp.plot_price_with_overlays(
    price,
    overlays=[("SMA 20", sma20), ("EMA 60", ema60)],
    secondary_overlays=[("RSI 14", rsi14)],
    secondary_ylabel="RSI",
    title="ACME price with moving averages and RSI",
)
```

![plot_price_with_overlays](../assets/gallery/plot_price_with_overlays.png)

## Indicator Panels

```python
fig = fp.plot_indicator_panel(
    price,
    panels=[{"title": "MACD", "series": [("MACD", macd), ("Signal", macd_signal)]}],
    title="ACME price and MACD",
)
```

![plot_indicator_panel](../assets/gallery/plot_indicator_panel.png)

## Performance Statistics

```python
stats = fp.perf_stats(returns)
```

```{include} ../assets/gallery/perf_stats.md
```

## Performance Table

```python
table = fp.table_perf_stats(returns, benchmark=benchmark)
html = table.as_raw_html()
```

```{include} ../assets/gallery/table_perf_stats.md
```

[table_perf_stats.html](../assets/gallery/table_perf_stats.html)

## Period Return Table

```python
period_table = fp.table_period_returns(returns, period="year")
```

```{include} ../assets/gallery/table_period_returns.md
```

## Drawdown Table

```python
drawdown_table = fp.table_drawdowns(returns, top=5)
```

```{include} ../assets/gallery/table_drawdowns.md
```

## Post-Trade Plots

```python
cost_fig = fp.plot_trading_cost_breakdown_bar(cost_breakdown)
mae_mfe_fig = fp.plot_mfe_mae_scatter(trades_with_excursions)
execution_fig = fp.plot_execution_quality(execution_quality)
```

![plot_trading_cost_breakdown_bar](../assets/gallery/plot_trading_cost_breakdown_bar.png)

![plot_mfe_mae_scatter](../assets/gallery/plot_mfe_mae_scatter.png)

![plot_execution_quality](../assets/gallery/plot_execution_quality.png)

## Post-Trade Tables

```python
cost_table = fp.table_cost_breakdown(cost_breakdown)
round_trip_table = fp.table_round_trip_stats(round_trips)
execution_table = fp.table_execution_quality(execution_quality)
```

```{include} ../assets/gallery/table_cost_breakdown.md
```

```{include} ../assets/gallery/table_round_trip_stats.md
```

```{include} ../assets/gallery/table_execution_quality.md
```

## Alpha IC Plots

```python
ic_ts_fig = fp.plot_ic_ts(ic)
ic_hist_fig = fp.plot_ic_hist(ic)
ic_qq_fig = fp.plot_ic_qq(ic)
ic_group_fig = fp.plot_ic_by_group(ic_by_group.to_pandas())
ic_heatmap_fig = fp.plot_ic_heatmap(ic, period="month")
rolling_ic_fig = fp.plot_rolling_ic(ic, window=21)
```

![plot_ic_ts](../assets/gallery/plot_ic_ts.png)

![plot_ic_hist](../assets/gallery/plot_ic_hist.png)

![plot_ic_qq](../assets/gallery/plot_ic_qq.png)

![plot_ic_by_group](../assets/gallery/plot_ic_by_group.png)

![plot_ic_heatmap](../assets/gallery/plot_ic_heatmap.png)

![plot_rolling_ic](../assets/gallery/plot_rolling_ic.png)

## Alpha Quantile Plots

```python
quantile_bar_fig = fp.plot_quantile_returns_bar(alpha_frame)
turnover_fig = fp.plot_top_bottom_quantile_turnover(alpha_frame)
factor_return_fig = fp.plot_cumulative_factor_returns(factor_returns)
```

![plot_quantile_returns_bar](../assets/gallery/plot_quantile_returns_bar.png)

![plot_top_bottom_quantile_turnover](../assets/gallery/plot_top_bottom_quantile_turnover.png)

![plot_cumulative_factor_returns](../assets/gallery/plot_cumulative_factor_returns.png)

## Alpha Analysis Tables

```python
information_table = fp.table_information(ic)
quantile_return_table = fp.table_returns_by_quantile(alpha_frame)
turnover_table = fp.table_turnover(alpha_frame)
quantile_stats_table = fp.table_quantile_statistics(alpha_frame)
```

```{include} ../assets/gallery/table_information.md
```

```{include} ../assets/gallery/table_returns_by_quantile.md
```

```{include} ../assets/gallery/table_turnover.md
```

```{include} ../assets/gallery/table_quantile_statistics.md
```

______________________________________________________________________

## Generate Every Example Artifact

Use the packaged helper when you want all examples written to disk:

```python
from finance_plots.gallery import generate_gallery

outputs = generate_gallery("docs/assets/gallery")
```
