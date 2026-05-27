# finance plots

Matplotlib plots and performance tables for financial return series, price
paths, and technical-indicator panels.

[![Build Status](https://github.com/prettygoodcapital/finance-plots/actions/workflows/build.yaml/badge.svg?branch=main&event=push)](https://github.com/prettygoodcapital/finance-plots/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/prettygoodcapital/finance-plots/branch/main/graph/badge.svg)](https://codecov.io/gh/prettygoodcapital/finance-plots)
[![License](https://img.shields.io/github/license/prettygoodcapital/finance-plots)](https://github.com/prettygoodcapital/finance-plots)
[![PyPI](https://img.shields.io/pypi/v/finance-plots.svg)](https://pypi.python.org/pypi/finance-plots)

## Overview

`finance-plots` is the presentation layer for the finance stack. It accepts
Narwhals-compatible inputs such as pandas, Polars, numpy, and other supported
series-like objects, then returns ordinary matplotlib figures or Great Tables
objects that can be saved, embedded in notebooks, or composed into tearsheets.

The initial release focuses on a compact, useful surface:

- Return/risk plots for cumulative returns, rolling volatility, rolling Sharpe,
  rolling beta/correlation, benchmark scatter, drawdowns, and period-return
  views.
- Technical-indicator plots for price overlays, secondary-axis indicators, and
  indicator sub-panels.
- Performance summary tables backed by `great-tables`.

## Install

```bash
pip install finance-plots
```

The gallery and documentation examples use the released data/calculation stack:

```bash
pip install "finance-plots[examples]"
```

## Quick Start

Generate deterministic prices with `finance-datagen`, compute returns with
`finance-calcs`, and plot them with `finance-plots`.

```python
from datetime import datetime, timezone

import polars as pl
from finance_datagen import generate_prices

import finance_calcs as fc
import finance_plots as fp

start_ms = int(datetime(2021, 1, 4, tzinfo=timezone.utc).timestamp() * 1000)
prices = generate_prices(symbol="ACME", seed=7, start_ms=start_ms)
returns = prices.with_columns(
    fc.simple_returns(pl.col("price")).alias("ret"),
).select("ret").drop_nulls()["ret"]

fig = fp.plot_rolling_returns(returns)
```

## Current Plot Catalog

| Function                                                         | Use it for                                                               |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `plot_returns(returns)`                                          | Simple cumulative return path                                            |
| `plot_rolling_returns(returns, benchmark=None, live_start=None)` | Cumulative return path with optional benchmark and out-of-sample shading |
| `plot_rolling_volatility(returns, window=63)`                    | Rolling annualized volatility                                            |
| `plot_rolling_sharpe(returns, window=63)`                        | Rolling annualized Sharpe ratio                                          |
| `plot_rolling_beta(returns, benchmark, window=63)`               | Rolling beta versus a benchmark                                          |
| `plot_rolling_correlation(returns, benchmark, window=63)`        | Rolling correlation versus a benchmark                                   |
| `plot_return_scatter(returns, benchmark)`                        | Strategy returns against benchmark returns with a fitted beta line       |
| `plot_drawdown_underwater(returns)`                              | Filled underwater drawdown chart                                         |
| `plot_returns_heatmap(returns, period="month")`                  | Year-by-month, year-by-quarter, or year-by-week return heatmap           |
| `plot_returns_bar(returns, period="year")`                       | Compounded period returns as a bar chart                                 |
| `plot_returns_dist(returns, period="month")`                     | Distribution of compounded period returns                                |
| `plot_returns_timeseries(returns, period="month")`               | Compounded period returns through time                                   |
| `plot_price_with_overlays(price, overlays, secondary_overlays)`  | Price line with moving averages and secondary-axis indicators            |
| `plot_indicator_panel(price, panels)`                            | Price chart with one or more aligned indicator sub-panels                |

## Current Table Catalog

| Function                                       | Use it for                                                                                               |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `perf_stats(returns)`                          | Dictionary of cumulative return, annualized return/volatility, Sharpe, Sortino, max drawdown, and Calmar |
| `table_perf_stats(returns, benchmark=None)`    | Great Tables performance summary with optional benchmark column                                          |
| `table_period_returns(returns, period="year")` | Great Tables period-return summary                                                                       |
| `table_drawdowns(returns, top=5)`              | Great Tables largest-drawdown-period summary                                                             |
