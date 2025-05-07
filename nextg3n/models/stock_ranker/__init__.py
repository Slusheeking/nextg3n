"""
Initialize the stock_ranker package for the NextG3N Trading System.

This module makes the models/stock_ranker directory a Python package, enabling imports of the
StockRanker class for ranking stocks based on multiple factors.
"""

from .stock_ranker import StockRanker

__all__ = [
    "StockRanker",
]