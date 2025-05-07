"""
Initialize the trainer package for the NextG3N Trading System.

This module makes the models/trainer directory a Python package, enabling imports of the
TrainerModel class for training and fine-tuning machine learning models.
"""

from .trainer_model import TrainerModel

__all__ = [
    "TrainerModel",
]