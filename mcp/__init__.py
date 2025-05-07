"""
Initialize the MCP (Model Context Protocol) package for the NextG3N Trading System.

This module makes the mcp directory a Python package. All MCP servers are now standalone FastAPI services.
"""

from monitor.logging_utils import get_logger

# Initialize logger
logger = get_logger("mcp")
logger.info("Initializing MCP package")
