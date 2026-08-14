"""finance-plots — table helpers."""

from ._alpha import table_information, table_quantile_statistics, table_returns_by_quantile, table_turnover
from ._perf import performance_statistics, table_drawdowns, table_performance_statistics, table_period_returns
from ._post_trade import table_cost_breakdown, table_execution_quality, table_round_trip_stats

__all__ = [
    "performance_statistics",
    "table_cost_breakdown",
    "table_drawdowns",
    "table_execution_quality",
    "table_information",
    "table_performance_statistics",
    "table_period_returns",
    "table_quantile_statistics",
    "table_returns_by_quantile",
    "table_round_trip_stats",
    "table_turnover",
]
