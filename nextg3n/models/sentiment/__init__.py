"""
Initialize the sentiment package for the NextG3N Trading System.

This module makes the models/sentiment directory a Python package, enabling imports of the
SentimentModel class for analyzing sentiment in financial texts.
"""

from .sentiment_model import SentimentModel

__all__ = [
    "SentimentModel",
]