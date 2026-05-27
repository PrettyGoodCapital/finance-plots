# Examples

These examples use `finance-datagen` for deterministic market-like inputs and
`finance-calcs` for returns and technical indicators. The plotting functions
then consume ordinary pandas Series, Polars Series, or numpy arrays.

______________________________________________________________________

## Shared Setup

```python
from datetime import datetime, timezone

import pandas as pd
import polars as pl
from finance_datagen import generate_prices

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

______________________________________________________________________

## Generate Every Example Artifact

Use the packaged helper when you want all examples written to disk:

```python
from finance_plots.gallery import generate_gallery

outputs = generate_gallery("docs/assets/gallery")
```
