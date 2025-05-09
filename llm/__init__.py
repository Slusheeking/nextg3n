"""
Initialize the LLM (Large Language Model) package for the NextG3N Trading System.

This module makes the llm directory a Python package, enabling imports of LLM integration
classes and utilities for AI-powered trading decisions.
"""

from monitor.logging_utils import get_logger

# Initialize logger
logger = get_logger("llm")
logger.info("Initializing LLM package")

from .main_llm import LLMTradingIntegration

__all__ = [
    "LLMTradingIntegration",
]
