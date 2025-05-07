"""
Initialize the context package for the NextG3N Trading System.

This module makes the models/context directory a Python package, enabling imports of the
ContextRetriever class for managing the RAG system.
"""

from .context_retriever import ContextRetriever

__all__ = [
    "ContextRetriever",
]