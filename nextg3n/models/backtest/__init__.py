"""
Initialize the backtest package for the NextG3N Trading System.

This module makes the models/backtest directory a Python package, enabling imports of the
BacktestEngine class for validating trading strategies.
"""

from .backtest_engine import BacktestEngine

__all__ = [
    "BacktestEngine",
]