"""Generate example artifacts for every public finance-plots chart."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from .plots import (
    plot_drawdown_underwater,
    plot_indicator_panel,
    plot_price_with_overlays,
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
from .plots._returns import _period_returns
from .tables import perf_stats, table_drawdowns, table_perf_stats, table_period_returns
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
    "perf_stats": "perf_stats.md",
    "table_perf_stats": "table_perf_stats.html",
    "table_perf_stats_markdown": "table_perf_stats.md",
    "table_period_returns": "table_period_returns.html",
    "table_period_returns_markdown": "table_period_returns.md",
    "table_drawdowns": "table_drawdowns.html",
    "table_drawdowns_markdown": "table_drawdowns.md",
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


def _sample_data() -> dict[str, Any]:
    try:
        import finance_calcs as fc
        import polars as pl
        from finance_datagen import generate_prices
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

    return {
        "returns": _series(enriched, "ret", drop_nulls=True),
        "benchmark": _series(benchmark, "ret", drop_nulls=True),
        "price": _series(enriched, "price"),
        "sma20": _series(enriched, "sma20"),
        "ema60": _series(enriched, "ema60"),
        "rsi14": _series(enriched, "rsi14"),
        "macd": _series(enriched, "macd"),
        "macd_signal": _series(enriched, "macd_signal"),
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
    }


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

    for name, figure in _gallery_figures(sample).items():
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
