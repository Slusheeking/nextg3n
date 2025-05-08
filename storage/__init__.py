"""
Initialize the storage package for the NextG3N Trading System.

This module makes the storage directory a Python package, enabling imports of storage classes
for Redis-based distributed caching and vector database operations for RAG.
"""

from monitor.logging_utils import get_logger
from .redis_server import RedisClusterManager
from. redis_message_broker import RedisMessageBroker

# Initialize logger
logger = get_logger("storage")
logger.info("Initializing storage package")

__all__ = [
    "RedisClusterManager",
    "RedisMessageBroker",
]
