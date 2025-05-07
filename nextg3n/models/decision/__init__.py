"""
Initialize the decision package for the NextG3N Trading System.

This module makes the models/decision directory a Python package, enabling imports of the
DecisionModel class for making trading decisions.
"""

from .decision_model import DecisionModel

__all__ = [
    "DecisionModel",
]