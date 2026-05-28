"""finance-plots — table helpers."""

from ._alpha import table_information, table_quantile_statistics, table_returns_by_quantile, table_turnover
from ._perf import perf_stats, table_drawdowns, table_perf_stats, table_period_returns
from ._post_trade import table_cost_breakdown, table_execution_quality, table_round_trip_stats

__all__ = [
    "perf_stats",
    "table_perf_stats",
    "table_period_returns",
    "table_drawdowns",
    "table_cost_breakdown",
    "table_round_trip_stats",
    "table_execution_quality",
    "table_information",
    "table_returns_by_quantile",
    "table_turnover",
    "table_quantile_statistics",
]
