"""Generate example artifacts for every public finance-plots chart."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from .plots import (
    plot_cumulative_factor_returns,
    plot_drawdown_underwater,
    plot_execution_quality,
    plot_ic_by_group,
    plot_ic_heatmap,
    plot_ic_hist,
    plot_ic_qq,
    plot_ic_ts,
    plot_indicator_panel,
    plot_mfe_mae_scatter,
    plot_price_with_overlays,
    plot_quantile_returns_bar,
    plot_return_scatter,
    plot_returns,
    plot_returns_bar,
    plot_returns_dist,
    plot_returns_heatmap,
    plot_returns_timeseries,
    plot_rolling_beta,
    plot_rolling_correlation,
    plot_rolling_ic,
    plot_rolling_returns,
    plot_rolling_sharpe,
    plot_rolling_volatility,
    plot_top_bottom_quantile_turnover,
    plot_trading_cost_breakdown_bar,
)
from .plots._returns import _period_returns
from .tables import (
    perf_stats,
    table_cost_breakdown,
    table_drawdowns,
    table_execution_quality,
    table_information,
    table_perf_stats,
    table_period_returns,
    table_quantile_statistics,
    table_returns_by_quantile,
    table_round_trip_stats,
    table_turnover,
)
from .tables._perf import _PERF_STAT_LABELS, _PERF_STAT_PERCENT_KEYS, _display_index_value, _drawdown_rows, _period_label

GALLERY_ARTIFACTS = {
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


def _epoch_ms(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000)


def _example_import_error(error: ImportError) -> ImportError:
    return ImportError(
        "Generating the finance-plots gallery requires the example stack. "
        "Install with `pip install finance-plots[examples]` or install "
        "finance-datagen and finance-calcs alongside finance-plots."
    ).with_traceback(error.__traceback__)


def _series(frame: Any, name: str, *, drop_nulls: bool = False):
    import pandas as pd

    selected = frame.select("timestamp", name)
    if drop_nulls:
        selected = selected.drop_nulls()
    return pd.Series(
        selected[name].to_numpy(),
        index=selected["timestamp"].to_pandas(),
        name=name,
    )


def _format_value(value: float, *, percent: bool = False) -> str:
    import numpy as np

    if not np.isfinite(value):
        return "n/a"
    if percent:
        return f"{value:.2%}"
    return f"{value:.2f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(["<!-- markdownlint-disable-file MD041 -->", header, separator, *body]) + "\n"


def _perf_stats_markdown(returns: Any) -> str:
    stats = perf_stats(returns)
    rows = [[_PERF_STAT_LABELS[key], _format_value(value, percent=key in _PERF_STAT_PERCENT_KEYS)] for key, value in stats.items()]
    return _markdown_table(["Metric", "Value"], rows)


def _table_perf_stats_markdown(returns: Any, benchmark: Any) -> str:
    strategy = perf_stats(returns)
    bench = perf_stats(benchmark)
    rows = [
        [
            _PERF_STAT_LABELS[key],
            _format_value(strategy[key], percent=key in _PERF_STAT_PERCENT_KEYS),
            _format_value(bench[key], percent=key in _PERF_STAT_PERCENT_KEYS),
        ]
        for key in strategy
    ]
    return _markdown_table(["Metric", "Strategy", "Benchmark"], rows)


def _period_returns_markdown(returns: Any, period: Any = "year") -> str:
    series = _period_returns(returns, period).dropna()
    rows = [[_period_label(index_value, period), _format_value(float(value), percent=True)] for index_value, value in series.items()]
    return _markdown_table(["Period", "Return"], rows)


def _drawdowns_markdown(returns: Any, top: int = 5) -> str:
    rows = [
        [
            str(row["rank"]),
            _display_index_value(row["start"]),
            _display_index_value(row["trough"]),
            _display_index_value(row["recovery"]),
            _format_value(row["drawdown"], percent=True),
            str(row["duration"]),
        ]
        for row in _drawdown_rows(returns, top=top)
    ]
    return _markdown_table(["Rank", "Start", "Trough", "Recovery", "Drawdown", "Duration"], rows)


def _cost_breakdown_markdown(costs: Any) -> str:
    frame = costs.to_pandas() if hasattr(costs, "to_pandas") else costs
    rows = [[str(row.component), _format_value(float(row.total)), _format_value(float(row.pct_total), percent=True)] for row in frame.itertuples()]
    return _markdown_table(["Component", "Total", "Pct total"], rows)


def _round_trip_stats_markdown(stats: dict[str, Any]) -> str:
    labels = {
        "n_trades": "Trades",
        "win_rate": "Win rate",
        "avg_pnl": "Average PnL",
        "total_pnl": "Total PnL",
        "profit_factor": "Profit factor",
        "payoff_ratio": "Payoff ratio",
    }
    rows = [[labels[key], _format_value(float(value), percent=key == "win_rate")] for key, value in stats.items()]
    return _markdown_table(["Metric", "Value"], rows)


def _execution_quality_markdown(frame: Any) -> str:
    import pandas as pd

    data = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    values = pd.to_numeric(data["implementation_shortfall_bps"], errors="coerce").dropna()
    rows = [
        ["Count", str(len(values))],
        ["Mean bps", _format_value(float(values.mean()))],
        ["Median bps", _format_value(float(values.median()))],
        ["Worst bps", _format_value(float(values.max()))],
        ["Best bps", _format_value(float(values.min()))],
    ]
    return _markdown_table(["Metric", "Value"], rows)


def _information_markdown(ic: Any) -> str:
    import numpy as np
    import pandas as pd

    values = pd.Series(ic).dropna().astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    ir = mean / std if std > 0 else float("nan")
    rows = [
        ["Mean IC", _format_value(mean)],
        ["IC volatility", _format_value(std)],
        ["ICIR", _format_value(ir)],
        ["t-stat", _format_value(ir * np.sqrt(len(values)))],
        ["Positive IC", _format_value(float((values > 0).mean()), percent=True)],
        ["Observations", str(len(values))],
    ]
    return _markdown_table(["Metric", "Value"], rows)


def _returns_by_quantile_markdown(frame: Any) -> str:
    data = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    grouped = data.groupby("quantile", dropna=False)["return"].agg(["count", "mean", "std"]).reset_index()
    rows = [
        [str(row.quantile), str(int(row.count)), _format_value(float(row.mean), percent=True), _format_value(float(row.std), percent=True)]
        for row in grouped.itertuples()
    ]
    return _markdown_table(["Quantile", "Count", "Mean return", "Volatility"], rows)


def _turnover_markdown(frame: Any) -> str:
    data = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    grouped = data.groupby("quantile", dropna=False)["turnover"].mean().reset_index()
    rows = [[str(row.quantile), _format_value(float(row.turnover), percent=True)] for row in grouped.itertuples()]
    return _markdown_table(["Quantile", "Turnover"], rows)


def _quantile_statistics_markdown(frame: Any) -> str:
    data = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    grouped = (
        data.groupby("quantile", dropna=False)
        .agg(count=("count", "sum"), signal_mean=("signal_mean", "mean"), signal_std=("signal_mean", "std"))
        .reset_index()
    )
    rows = [
        [str(row.quantile), str(int(row.count)), _format_value(float(row.signal_mean)), _format_value(float(row.signal_std))]
        for row in grouped.itertuples()
    ]
    return _markdown_table(["Quantile", "Count", "Signal mean", "Signal std"], rows)


def _sample_data() -> dict[str, Any]:
    try:
        import finance_calcs as fc
        import pandas as pd
        import polars as pl
        from finance_datagen import generate_prices, generate_signal
    except ImportError as error:  # pragma: no cover - exercised by users without examples extra
        raise _example_import_error(error) from error

    start_ms = _epoch_ms(2021, 1, 4)
    prices = generate_prices(n_steps=756, symbol="ACME", seed=7, start_ms=start_ms)
    benchmark_prices = generate_prices(
        n_steps=756,
        symbol="BENCH",
        seed=11,
        start_ms=start_ms,
        mu=0.04,
        sigma=0.16,
    )

    enriched = prices.with_columns(
        fc.simple_returns(pl.col("price")).alias("ret"),
        fc.sma(pl.col("price"), period=20).alias("sma20"),
        fc.ema(pl.col("price"), period=60).alias("ema60"),
        fc.rsi(pl.col("price"), period=14).alias("rsi14"),
        fc.macd_line(pl.col("price")).alias("macd"),
        fc.macd_signal(pl.col("price")).alias("macd_signal"),
    )
    benchmark = benchmark_prices.with_columns(
        fc.simple_returns(pl.col("price")).alias("ret"),
    )

    round_trip_transactions = pl.DataFrame(
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
    cost_breakdown = fc.cost_attribution(round_trip_transactions)
    round_trips = fc.extract_round_trips(round_trip_transactions)
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
        pl.when(pl.col("symbol").str.slice(-1).is_in(["0", "2", "4", "6", "8"])).then(pl.lit("Tech")).otherwise(pl.lit("Energy")).alias("group")
    )
    signals = signals.with_columns(fc.assign_quantile(pl.col("signal"), 5).over("date").alias("quantile"))
    ic = signals.group_by("date").agg(fc.spearman_ic(pl.col("signal"), pl.col("fwd_returns")).alias("ic")).sort("date")
    ic_by_group = signals.group_by("date", "group").agg(fc.spearman_ic(pl.col("signal"), pl.col("fwd_returns")).alias("ic")).sort("date")
    changed = signals.sort("symbol", "date").with_columns(fc.quantile_changed(pl.col("quantile")).over("symbol").alias("changed"))
    turnover = changed.group_by("date", "quantile").agg(fc.quantile_turnover(pl.col("changed")).alias("turnover"))
    quantile_returns = signals.group_by("date", "quantile").agg(
        pl.col("fwd_returns").mean().alias("return"),
        pl.len().alias("count"),
        pl.col("signal").mean().alias("signal_mean"),
    )
    alpha_frame = quantile_returns.join(turnover, on=["date", "quantile"], how="left").sort("date", "quantile").to_pandas()
    factor_returns = (
        signals.group_by("date")
        .agg(fc.long_short_spread(pl.col("fwd_returns"), pl.col("quantile"), upper=4, lower=0).alias("factor_return"))
        .sort("date")
    )
    ic_series = pd.Series(ic["ic"].to_numpy(), index=ic["date"].to_pandas(), name="ic")
    factor_return_series = pd.Series(factor_returns["factor_return"].to_numpy(), index=factor_returns["date"].to_pandas(), name="factor_return")

    return {
        "returns": _series(enriched, "ret", drop_nulls=True),
        "benchmark": _series(benchmark, "ret", drop_nulls=True),
        "price": _series(enriched, "price"),
        "sma20": _series(enriched, "sma20"),
        "ema60": _series(enriched, "ema60"),
        "rsi14": _series(enriched, "rsi14"),
        "macd": _series(enriched, "macd"),
        "macd_signal": _series(enriched, "macd_signal"),
        "cost_breakdown": cost_breakdown,
        "round_trips": round_trips,
        "round_trip_stats": fc.round_trip_stats(round_trips),
        "trades_with_excursions": trades_with_excursions,
        "execution_quality": execution_quality,
        "ic": ic_series,
        "ic_by_group": ic_by_group.to_pandas(),
        "alpha_frame": alpha_frame,
        "factor_returns": factor_return_series,
    }


def _gallery_figures(sample: dict[str, Any]) -> dict[str, Figure]:
    returns = sample["returns"]
    benchmark = sample["benchmark"]
    price = sample["price"]

    return {
        "plot_returns": plot_returns(returns),
        "plot_rolling_returns": plot_rolling_returns(
            returns,
            benchmark=benchmark,
            live_start=returns.index[int(len(returns) * 0.7)],
        ),
        "plot_rolling_volatility": plot_rolling_volatility(returns, window=63),
        "plot_rolling_sharpe": plot_rolling_sharpe(returns, window=63),
        "plot_rolling_beta": plot_rolling_beta(returns, benchmark, window=63),
        "plot_rolling_correlation": plot_rolling_correlation(returns, benchmark, window=63),
        "plot_return_scatter": plot_return_scatter(returns, benchmark),
        "plot_drawdown_underwater": plot_drawdown_underwater(returns),
        "plot_returns_heatmap": plot_returns_heatmap(returns, period="month"),
        "plot_returns_bar": plot_returns_bar(returns, period="year"),
        "plot_returns_dist": plot_returns_dist(returns, period="month"),
        "plot_returns_timeseries": plot_returns_timeseries(returns, period="month"),
        "plot_indicator_panel": plot_indicator_panel(
            price,
            panels=[{"title": "MACD", "series": [("MACD", sample["macd"]), ("Signal", sample["macd_signal"])]}],
            title="ACME price and MACD",
        ),
        "plot_price_with_overlays": plot_price_with_overlays(
            price,
            overlays=[("SMA 20", sample["sma20"]), ("EMA 60", sample["ema60"])],
            secondary_overlays=[("RSI 14", sample["rsi14"])],
            secondary_ylabel="RSI",
            title="ACME price with moving averages and RSI",
        ),
        "plot_trading_cost_breakdown_bar": plot_trading_cost_breakdown_bar(sample["cost_breakdown"]),
        "plot_mfe_mae_scatter": plot_mfe_mae_scatter(sample["trades_with_excursions"]),
        "plot_execution_quality": plot_execution_quality(sample["execution_quality"]),
        "plot_ic_ts": plot_ic_ts(sample["ic"]),
        "plot_ic_hist": plot_ic_hist(sample["ic"]),
        "plot_ic_qq": plot_ic_qq(sample["ic"]),
        "plot_ic_by_group": plot_ic_by_group(sample["ic_by_group"]),
        "plot_ic_heatmap": plot_ic_heatmap(sample["ic"], period="month"),
        "plot_rolling_ic": plot_rolling_ic(sample["ic"], window=21),
        "plot_quantile_returns_bar": plot_quantile_returns_bar(sample["alpha_frame"]),
        "plot_top_bottom_quantile_turnover": plot_top_bottom_quantile_turnover(sample["alpha_frame"]),
        "plot_cumulative_factor_returns": plot_cumulative_factor_returns(sample["factor_returns"]),
    }


def _write_table_pair(outputs: dict[str, Path], output_path: Path, html_key: str, markdown_key: str, html: str, markdown: str) -> None:
    html_path = output_path / GALLERY_ARTIFACTS[html_key]
    html_path.write_text(html, encoding="utf-8")
    outputs[html_key] = html_path

    markdown_path = output_path / GALLERY_ARTIFACTS[markdown_key]
    markdown_path.write_text(markdown, encoding="utf-8")
    outputs[markdown_key] = markdown_path


def generate_gallery(
    output_dir: str | Path = "docs/assets/gallery",
    *,
    dpi: int = 144,
    close_figures: bool = True,
) -> dict[str, Path]:
    """Generate image and table artifacts for the public example gallery.

    Args:
        output_dir: Directory where artifacts should be written.
        dpi: PNG resolution for matplotlib figures.
        close_figures: Close figures after saving to avoid leaking GUI state.

    Returns:
        Mapping from public plot/table name to the written artifact path.
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sample = _sample_data()
    outputs: dict[str, Path] = {}

    with plt.rc_context({"figure.max_open_warning": 0}):
        figures = _gallery_figures(sample)

    for name, figure in figures.items():
        path = output_path / GALLERY_ARTIFACTS[name]
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs[name] = path
        if close_figures:
            plt.close(figure)

    perf_stats_path = output_path / GALLERY_ARTIFACTS["perf_stats"]
    perf_stats_path.write_text(_perf_stats_markdown(sample["returns"]), encoding="utf-8")
    outputs["perf_stats"] = perf_stats_path

    table_perf_stats_path = output_path / GALLERY_ARTIFACTS["table_perf_stats"]
    table_perf_stats_path.write_text(
        table_perf_stats(sample["returns"], benchmark=sample["benchmark"]).as_raw_html(),
        encoding="utf-8",
    )
    outputs["table_perf_stats"] = table_perf_stats_path

    table_perf_stats_markdown_path = output_path / GALLERY_ARTIFACTS["table_perf_stats_markdown"]
    table_perf_stats_markdown_path.write_text(
        _table_perf_stats_markdown(sample["returns"], sample["benchmark"]),
        encoding="utf-8",
    )
    outputs["table_perf_stats_markdown"] = table_perf_stats_markdown_path

    table_period_returns_path = output_path / GALLERY_ARTIFACTS["table_period_returns"]
    table_period_returns_path.write_text(
        table_period_returns(sample["returns"], period="year").as_raw_html(),
        encoding="utf-8",
    )
    outputs["table_period_returns"] = table_period_returns_path

    table_period_returns_markdown_path = output_path / GALLERY_ARTIFACTS["table_period_returns_markdown"]
    table_period_returns_markdown_path.write_text(_period_returns_markdown(sample["returns"], period="year"), encoding="utf-8")
    outputs["table_period_returns_markdown"] = table_period_returns_markdown_path

    table_drawdowns_path = output_path / GALLERY_ARTIFACTS["table_drawdowns"]
    table_drawdowns_path.write_text(
        table_drawdowns(sample["returns"], top=5).as_raw_html(),
        encoding="utf-8",
    )
    outputs["table_drawdowns"] = table_drawdowns_path

    table_drawdowns_markdown_path = output_path / GALLERY_ARTIFACTS["table_drawdowns_markdown"]
    table_drawdowns_markdown_path.write_text(_drawdowns_markdown(sample["returns"], top=5), encoding="utf-8")
    outputs["table_drawdowns_markdown"] = table_drawdowns_markdown_path

    _write_table_pair(
        outputs,
        output_path,
        "table_cost_breakdown",
        "table_cost_breakdown_markdown",
        table_cost_breakdown(sample["cost_breakdown"]).as_raw_html(),
        _cost_breakdown_markdown(sample["cost_breakdown"]),
    )
    _write_table_pair(
        outputs,
        output_path,
        "table_round_trip_stats",
        "table_round_trip_stats_markdown",
        table_round_trip_stats(sample["round_trips"]).as_raw_html(),
        _round_trip_stats_markdown(sample["round_trip_stats"]),
    )
    _write_table_pair(
        outputs,
        output_path,
        "table_execution_quality",
        "table_execution_quality_markdown",
        table_execution_quality(sample["execution_quality"]).as_raw_html(),
        _execution_quality_markdown(sample["execution_quality"]),
    )
    _write_table_pair(
        outputs,
        output_path,
        "table_information",
        "table_information_markdown",
        table_information(sample["ic"]).as_raw_html(),
        _information_markdown(sample["ic"]),
    )
    _write_table_pair(
        outputs,
        output_path,
        "table_returns_by_quantile",
        "table_returns_by_quantile_markdown",
        table_returns_by_quantile(sample["alpha_frame"]).as_raw_html(),
        _returns_by_quantile_markdown(sample["alpha_frame"]),
    )
    _write_table_pair(
        outputs,
        output_path,
        "table_turnover",
        "table_turnover_markdown",
        table_turnover(sample["alpha_frame"]).as_raw_html(),
        _turnover_markdown(sample["alpha_frame"]),
    )
    _write_table_pair(
        outputs,
        output_path,
        "table_quantile_statistics",
        "table_quantile_statistics_markdown",
        table_quantile_statistics(sample["alpha_frame"]).as_raw_html(),
        _quantile_statistics_markdown(sample["alpha_frame"]),
    )

    return outputs


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for generating the example gallery."""
    parser = argparse.ArgumentParser(description="Generate finance-plots gallery artifacts.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="docs/assets/gallery",
        help="Directory where PNG/HTML artifacts should be written.",
    )
    parser.add_argument("--dpi", type=int, default=144, help="PNG resolution.")
    args = parser.parse_args(argv)

    outputs = generate_gallery(args.output_dir, dpi=args.dpi)
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
