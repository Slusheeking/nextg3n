"""
Initialize the visualization package for the NextG3N Trading System.

This module makes the visualization directory a Python package, enabling imports of classes
for generating charts, serving metrics, and providing a web-based dashboard for monitoring
trading activities and system performance.
"""

from .chart_generator import ChartGenerator
from .metrics_api import MetricsApi
from .trade_dashboard import TradeDashboard

__all__ = [
    "ChartGenerator",
    "MetricsApi",
    "TradeDashboard",
]