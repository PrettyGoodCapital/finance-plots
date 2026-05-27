"""Smoke tests for finance-plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import finance_plots as fp


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def returns_series() -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, size=500), index=idx, name="r")


@pytest.fixture
def benchmark_series(returns_series) -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(
        rng.normal(0.0003, 0.008, size=len(returns_series)),
        index=returns_series.index,
        name="bench",
    )


def test_plot_rolling_returns(returns_series, benchmark_series):
    fig = fp.plot_rolling_returns(returns_series, benchmark=benchmark_series)
    assert isinstance(fig, Figure)


def test_plot_rolling_returns_with_live_start(returns_series):
    cutoff = returns_series.index[300]
    fig = fp.plot_rolling_returns(returns_series, live_start=cutoff)
    assert isinstance(fig, Figure)


def test_plot_returns(returns_series):
    fig = fp.plot_returns(returns_series)
    assert isinstance(fig, Figure)


def test_plot_rolling_volatility(returns_series):
    fig = fp.plot_rolling_volatility(returns_series, window=63)
    assert isinstance(fig, Figure)


def test_plot_rolling_sharpe(returns_series):
    fig = fp.plot_rolling_sharpe(returns_series, window=63)
    assert isinstance(fig, Figure)


def test_plot_rolling_beta(returns_series, benchmark_series):
    fig = fp.plot_rolling_beta(returns_series, benchmark_series, window=63)
    assert isinstance(fig, Figure)


def test_plot_rolling_correlation(returns_series, benchmark_series):
    fig = fp.plot_rolling_correlation(returns_series, benchmark_series, window=63)
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_ylabel() == "correlation"


def test_plot_return_scatter(returns_series, benchmark_series):
    fig = fp.plot_return_scatter(returns_series, benchmark_series)
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_xlabel() == "benchmark return"


def test_plot_drawdown_underwater(returns_series):
    fig = fp.plot_drawdown_underwater(returns_series)
    assert isinstance(fig, Figure)


def test_plot_returns_heatmap(returns_series):
    fig = fp.plot_returns_heatmap(returns_series)
    assert isinstance(fig, Figure)


def test_period_plots_fallback_to_realistic_dates_for_numpy(returns_series):
    returns = returns_series.to_numpy()

    heatmap = fp.plot_returns_heatmap(returns)
    heatmap_years = [label.get_text() for label in heatmap.axes[0].get_yticklabels()]
    assert any(year.startswith("2000") for year in heatmap_years)
    assert all(not year.startswith("1970") for year in heatmap_years)

    bar = fp.plot_returns_bar(returns, period="year")
    bar_labels = [label.get_text() for label in bar.axes[0].get_xticklabels()]
    assert any(label.startswith("2000") for label in bar_labels)
    assert all(not label.startswith("1970") for label in bar_labels)

    timeseries = fp.plot_returns_timeseries(returns, period="month")
    first_x = timeseries.axes[0].lines[0].get_xdata()[0]
    assert pd.Timestamp(first_x).year == 2000


def test_plot_period_return_charts(returns_series):
    figures = [
        fp.plot_returns_bar(returns_series, period="year"),
        fp.plot_returns_dist(returns_series, period="month"),
        fp.plot_returns_timeseries(returns_series, period="month"),
    ]

    assert all(isinstance(fig, Figure) for fig in figures)


def test_perf_stats_keys(returns_series):
    stats = fp.perf_stats(returns_series)
    expected = {
        "cum_return",
        "ann_return",
        "ann_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
    }
    assert set(stats) == expected
    assert all(isinstance(v, float) for v in stats.values())


def test_perf_stats_max_drawdown_nonpositive(returns_series):
    stats = fp.perf_stats(returns_series)
    assert stats["max_drawdown"] <= 0.0


def test_table_perf_stats(returns_series, benchmark_series):
    gt = fp.table_perf_stats(returns_series, benchmark=benchmark_series)
    # Smoke: render to HTML successfully.
    html = gt.as_raw_html()
    assert "Performance summary" in html
    assert "Sharpe" in html


def test_additional_tables(returns_series):
    period_html = fp.table_period_returns(returns_series, period="year").as_raw_html()
    drawdown_html = fp.table_drawdowns(returns_series, top=3).as_raw_html()

    assert "Period returns" in period_html
    assert "Drawdown periods" in drawdown_html


def test_accepts_polars(returns_series):
    import polars as pl

    s = pl.Series("r", returns_series.values)
    fig = fp.plot_drawdown_underwater(s)
    assert isinstance(fig, Figure)
    stats = fp.perf_stats(s)
    assert np.isfinite(stats["ann_vol"])


def test_accepts_numpy(returns_series):
    arr = returns_series.values
    fig = fp.plot_rolling_returns(arr)
    assert isinstance(fig, Figure)


@pytest.fixture
def price_series(returns_series) -> pd.Series:
    return (1.0 + returns_series).cumprod() * 100.0


def test_plot_indicator_panel_smoke(price_series):
    rsi_like = pd.Series(
        50.0 + 10.0 * np.sin(np.linspace(0, 6, len(price_series))),
        index=price_series.index,
    )
    fig = fp.plot_indicator_panel(
        price_series,
        panels=[
            {"title": "RSI", "series": [("rsi", rsi_like)]},
        ],
        title="demo",
    )
    assert isinstance(fig, Figure)


def test_plot_indicator_panel_no_panels(price_series):
    fig = fp.plot_indicator_panel(price_series)
    assert isinstance(fig, Figure)


def test_plot_indicator_panel_length_mismatch(price_series):
    short = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        fp.plot_indicator_panel(
            price_series,
            panels=[{"title": "x", "series": [("x", short)]}],
        )


def test_plot_price_with_overlays(price_series):
    sma = price_series.rolling(20).mean()
    ema = price_series.ewm(span=20, adjust=False).mean()
    rsi_like = pd.Series(
        50.0 + 10.0 * np.sin(np.linspace(0, 6, len(price_series))),
        index=price_series.index,
    )
    fig = fp.plot_price_with_overlays(
        price_series,
        overlays=[("sma20", sma), ("ema20", ema)],
        secondary_overlays=[("rsi", rsi_like)],
        secondary_ylabel="RSI",
        title="overlays",
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylim() == (0.0, 100.0)
    guide_levels = [line.get_ydata()[0] for line in fig.axes[1].lines[1:]]
    assert any(np.isclose(level, 30.0) for level in guide_levels)
    assert any(np.isclose(level, 70.0) for level in guide_levels)


def test_gallery_sample_data_uses_realistic_dates():
    from finance_plots.gallery import _sample_data

    sample = _sample_data()

    assert sample["price"].index.min().year >= 2020


def test_generate_gallery_writes_every_public_artifact(tmp_path):
    from finance_plots.gallery import GALLERY_ARTIFACTS, generate_gallery

    expected = {
        "plot_returns": "plot_returns.png",
        "plot_rolling_returns": "plot_rolling_returns.png",
        "plot_rolling_volatility": "plot_rolling_volatility.png",
        "plot_rolling_sharpe": "plot_rolling_sharpe.png",
        "plot_rolling_beta": "plot_rolling_beta.png",
        "plot_rolling_correlation": "plot_rolling_correlation.png",
        "plot_return_scatter": "plot_return_scatter.png",
        "plot_drawdown_underwater": "plot_drawdown_underwater.png",
        "plot_returns_heatmap": "plot_returns_heatmap.png",
        "plot_returns_bar": "plot_returns_bar.png",
        "plot_returns_dist": "plot_returns_dist.png",
        "plot_returns_timeseries": "plot_returns_timeseries.png",
        "plot_indicator_panel": "plot_indicator_panel.png",
        "plot_price_with_overlays": "plot_price_with_overlays.png",
        "perf_stats": "perf_stats.md",
        "table_perf_stats": "table_perf_stats.html",
        "table_perf_stats_markdown": "table_perf_stats.md",
        "table_period_returns": "table_period_returns.html",
        "table_period_returns_markdown": "table_period_returns.md",
        "table_drawdowns": "table_drawdowns.html",
        "table_drawdowns_markdown": "table_drawdowns.md",
    }

    outputs = generate_gallery(tmp_path)

    assert GALLERY_ARTIFACTS == expected
    assert set(outputs) == set(expected)
    for name, path in outputs.items():
        assert path.name == expected[name]
        assert path.exists()
        assert path.stat().st_size > 0


def test_gallery_markdown_artifacts_render_inline_tables(tmp_path):
    from finance_plots.gallery import generate_gallery

    outputs = generate_gallery(tmp_path)

    perf_stats = outputs["perf_stats"].read_text(encoding="utf-8")
    table_perf = outputs["table_perf_stats_markdown"].read_text(encoding="utf-8")
    drawdowns = outputs["table_drawdowns_markdown"].read_text(encoding="utf-8")

    assert "| Metric | Value |" in perf_stats
    assert "Sharpe ratio" in table_perf
    assert "Drawdown" in drawdowns
