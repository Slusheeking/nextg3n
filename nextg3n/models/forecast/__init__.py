"""
Initialize the forecast package for the NextG3N Trading System.

This module makes the models/forecast directory a Python package, enabling imports of the
ForecastModel class for predicting stock price movements.
"""

from .forecast_model import ForecastModel

__all__ = [
    "ForecastModel",
]