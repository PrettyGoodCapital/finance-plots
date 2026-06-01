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


@pytest.fixture
def post_trade_frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "component": ["commission", "fees", "slippage", "spread", "market_impact", "commission"],
            "total": [12.0, 4.0, 15.0, 6.0, 3.0, 8.0],
            "mae": [-0.04, -0.02, -0.08, -0.01, -0.03, -0.06],
            "mfe": [0.10, 0.04, 0.12, 0.03, 0.07, 0.05],
            "side": ["long", "short", "long", "short", "long", "short"],
            "implementation_shortfall_bps": [12.0, 8.0, -4.0, 15.0, 3.0, 10.0],
            "pnl": [100.0, -50.0, 75.0, -20.0, 30.0, 10.0],
        }
    )


@pytest.fixture
def alpha_frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=40, freq="B")
    rng = np.random.default_rng(123)
    ic = pd.Series(rng.normal(0.04, 0.08, len(idx)), index=idx, name="ic")
    return pd.DataFrame(
        {
            "date": idx,
            "ic": ic.to_numpy(),
            "group": np.where(np.arange(len(idx)) % 2 == 0, "Tech", "Energy"),
            "quantile": np.tile(np.arange(5), 8),
            "return": np.tile(np.linspace(-0.01, 0.02, 5), 8) + rng.normal(0.0, 0.002, len(idx)),
            "turnover": rng.uniform(0.1, 0.4, len(idx)),
            "factor_return": rng.normal(0.0004, 0.01, len(idx)),
            "count": rng.integers(20, 40, len(idx)),
            "signal_mean": rng.normal(0.0, 0.2, len(idx)),
        }
    )


def test_post_trade_plots(post_trade_frame):
    figures = [
        fp.plot_trading_cost_breakdown_bar(post_trade_frame),
        fp.plot_mfe_mae_scatter(post_trade_frame),
        fp.plot_execution_quality(post_trade_frame),
    ]

    assert all(isinstance(fig, Figure) for fig in figures)


def test_post_trade_tables(post_trade_frame):
    cost_html = fp.table_cost_breakdown(post_trade_frame).as_raw_html()
    round_trip_html = fp.table_round_trip_stats(post_trade_frame).as_raw_html()
    execution_html = fp.table_execution_quality(post_trade_frame).as_raw_html()

    assert "Cost breakdown" in cost_html
    assert "Round-trip statistics" in round_trip_html
    assert "Execution quality" in execution_html


def test_alpha_analysis_plots(alpha_frame):
    ic = alpha_frame.set_index("date")["ic"]
    figures = [
        fp.plot_ic_ts(ic),
        fp.plot_ic_hist(ic),
        fp.plot_ic_qq(ic),
        fp.plot_ic_by_group(alpha_frame),
        fp.plot_ic_heatmap(ic, period="month"),
        fp.plot_rolling_ic(ic, window=10),
        fp.plot_quantile_returns_bar(alpha_frame),
        fp.plot_top_bottom_quantile_turnover(alpha_frame),
        fp.plot_cumulative_factor_returns(alpha_frame.set_index("date")["factor_return"]),
    ]

    assert all(isinstance(fig, Figure) for fig in figures)


def test_alpha_analysis_tables(alpha_frame):
    info_html = fp.table_information(alpha_frame["ic"]).as_raw_html()
    quantile_html = fp.table_returns_by_quantile(alpha_frame).as_raw_html()
    turnover_html = fp.table_turnover(alpha_frame).as_raw_html()
    stats_html = fp.table_quantile_statistics(alpha_frame).as_raw_html()

    assert "Information coefficient" in info_html
    assert "Returns by quantile" in quantile_html
    assert "Quantile turnover" in turnover_html
    assert "Quantile statistics" in stats_html


def test_portfolio_construction_plots() -> None:
    cov = np.array(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16],
        ]
    )
    expected_returns = np.array([0.08, 0.10, 0.12])
    weights = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=3, freq="ME"),
            "A": [0.5, 0.4, 0.3],
            "B": [0.3, 0.4, 0.4],
            "C": [0.2, 0.2, 0.3],
        }
    ).set_index("date")
    exposures = pd.DataFrame(
        [[0.2, 0.5], [-0.1, 0.3], [0.4, -0.2]],
        index=["A", "B", "C"],
        columns=["value", "momentum"],
    )
    impact = pd.DataFrame(
        {
            "participation_rate": [0.05, 0.10, 0.20, 0.35],
            "impact_bps": [2.0, 3.6, 6.5, 11.2],
        }
    )
    execution = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=4, freq="15min"),
            "executed_qty": [1000, 2000, 1800, 1200],
            "target_qty": [1200, 2800, 4600, 6000],
        }
    )
    costs = pd.DataFrame(
        {
            "component": ["commission", "slippage", "market_impact", "rebate"],
            "total": [12.0, 20.0, 16.0, -3.0],
        }
    )
    attribution = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-31", periods=3, freq="ME"),
            "allocation": [0.001, -0.0005, 0.0008],
            "selection": [0.002, 0.001, -0.0002],
            "interaction": [0.0003, -0.0001, 0.0004],
        }
    )

    figures = [
        fp.plot_efficient_frontier(expected_returns, cov),
        fp.plot_market_impact_curve(impact),
        fp.plot_execution_timeline(execution),
        fp.plot_cost_breakdown_bar(costs),
        fp.plot_return_attribution_stacked(attribution, time_col="date"),
        fp.plot_portfolio_weight_evolution(weights),
        fp.plot_weight_diff({"A": 0.5, "B": 0.3, "C": 0.2}, {"A": 0.4, "B": 0.4, "C": 0.2}),
        fp.plot_risk_decomposition_stacked(pd.DataFrame({"systematic": [0.12, 0.08], "idiosyncratic": [0.04, 0.06]}, index=["A", "B"])),
        fp.plot_factor_exposure_heatmap(exposures),
        fp.plot_correlation_matrix(cov, labels=["A", "B", "C"]),
        fp.plot_covariance_eigenvalues(cov),
    ]

    assert all(isinstance(fig, Figure) for fig in figures)


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
        "plot_trading_cost_breakdown_bar": "plot_trading_cost_breakdown_bar.png",
        "plot_mfe_mae_scatter": "plot_mfe_mae_scatter.png",
        "plot_execution_quality": "plot_execution_quality.png",
        "plot_ic_ts": "plot_ic_ts.png",
        "plot_ic_hist": "plot_ic_hist.png",
        "plot_ic_qq": "plot_ic_qq.png",
        "plot_ic_by_group": "plot_ic_by_group.png",
        "plot_ic_heatmap": "plot_ic_heatmap.png",
        "plot_rolling_ic": "plot_rolling_ic.png",
        "plot_quantile_returns_bar": "plot_quantile_returns_bar.png",
        "plot_top_bottom_quantile_turnover": "plot_top_bottom_quantile_turnover.png",
        "plot_cumulative_factor_returns": "plot_cumulative_factor_returns.png",
        "perf_stats": "perf_stats.md",
        "table_perf_stats": "table_perf_stats.html",
        "table_perf_stats_markdown": "table_perf_stats.md",
        "table_period_returns": "table_period_returns.html",
        "table_period_returns_markdown": "table_period_returns.md",
        "table_drawdowns": "table_drawdowns.html",
        "table_drawdowns_markdown": "table_drawdowns.md",
        "table_cost_breakdown": "table_cost_breakdown.html",
        "table_cost_breakdown_markdown": "table_cost_breakdown.md",
        "table_round_trip_stats": "table_round_trip_stats.html",
        "table_round_trip_stats_markdown": "table_round_trip_stats.md",
        "table_execution_quality": "table_execution_quality.html",
        "table_execution_quality_markdown": "table_execution_quality.md",
        "table_information": "table_information.html",
        "table_information_markdown": "table_information.md",
        "table_returns_by_quantile": "table_returns_by_quantile.html",
        "table_returns_by_quantile_markdown": "table_returns_by_quantile.md",
        "table_turnover": "table_turnover.html",
        "table_turnover_markdown": "table_turnover.md",
        "table_quantile_statistics": "table_quantile_statistics.html",
        "table_quantile_statistics_markdown": "table_quantile_statistics.md",
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
    cost_breakdown = outputs["table_cost_breakdown_markdown"].read_text(encoding="utf-8")
    information = outputs["table_information_markdown"].read_text(encoding="utf-8")

    assert "| Metric | Value |" in perf_stats
    assert "Sharpe ratio" in table_perf
    assert "Drawdown" in drawdowns
    assert "Component" in cost_breakdown
    assert "Mean IC" in information
