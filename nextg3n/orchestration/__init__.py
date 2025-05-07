"""
Initialize the orchestration package for the NextG3N Trading System.

This module makes the orchestration directory a Python package, enabling imports of the
TradeFlowOrchestrator class for coordinating trading workflows and system components.
"""

from .trade_flow_orchestrator import TradeFlowOrchestrator

__all__ = [
    "TradeFlowOrchestrator",
]