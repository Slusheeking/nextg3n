"""
Unit tests for BacktestEngine in NextG3N Trading System

Tests PPO strategy backtesting with Polygon data via MCP tools.
"""

import pytest
import asyncio
import pandas as pd
from unittest.mock import AsyncMock, patch
from models.backtest.backtest_engine import BacktestEngine

@pytest.mark.asyncio
class TestBacktestEngine:
    @pytest.fixture
    def config(self):
        return {
            "kafka": {"bootstrap_servers": "localhost:9092", "topic_prefix": "nextg3n-"},
            "storage": {"redis": {"host": "localhost", "port": 6379, "db": 0}}
        }

    @pytest.fixture
    def backtest_engine(self, config):
        return BacktestEngine(config)

    async def test_run_backtest_success(self, backtest_engine):
        with patch("services.mcp_client.MCPClient.call_tool", AsyncMock(return_value={
            "success": True,
            "bars": [
                {"timestamp": "2023-01-01T00:00:00Z", "open": 99.0, "high": 100.0, "low": 98.0, "close": 99.5, "volume": 1000},
                {"timestamp": "2023-01-02T00:00:00Z", "open": 99.5, "high": 101.0, "low": 98.5, "close": 100.0, "volume": 1200}
            ]
        })):
            result = await backtest_engine.run_backtest("AAPL", timeframe="1d")
            assert result["success"]
            assert "returns" in result
            assert result["symbol"] == "AAPL"

    async def test_run_backtest_no_data(self, backtest_engine):
        with patch("services.mcp_client.MCPClient.call_tool", AsyncMock(return_value={"success": False, "error": "No bars"})):
            result = await backtest_engine.run_backtest("AAPL", timeframe="1d")
            assert not result["success"]
            assert "error" in result