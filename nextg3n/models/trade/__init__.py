"""
Initialize the trade package for the NextG3N Trading System.

This module makes the models/trade directory a Python package, enabling imports of the
TradeExecutor class for executing and tracking trading orders.
"""

from .trade_executor import TradeExecutor

__all__ = [
    "TradeExecutor",
]