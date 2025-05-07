"""
Unit tests for StockRanker in NextG3N Trading System

Tests XGBoost and CNN-based stock screening with MCP tools.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from models.stock_ranker.stock_ranker import StockRanker

@pytest.mark.asyncio
class TestStockRanker:
    @pytest.fixture
    def config(self):
        return {
            "kafka": {"bootstrap_servers": "localhost:9092", "topic_prefix": "nextg3n-"},
            "storage": {"redis": {"host": "localhost", "port": 6379, "db": 0}}
        }

    @pytest.fixture
    def stock_ranker(self, config):
        return StockRanker(config)

    async def test_rank_stocks_success(self, stock_ranker):
        with patch("services.mcp_client.MCPClient.call_tool", side_effect=[
            AsyncMock(return_value={"success": True, "price": 100.0}),
            AsyncMock(return_value={"success": True, "bars": [{"high": 102, "low": 98, "close": 100, "volume": 3000000}]}),
            AsyncMock(return_value={"success": True, "fundamentals": {"pe_ratio": 15, "market_cap": 1000000000}})
        ]):
            result = await stock_ranker.rank_stocks(["AAPL", "GOOGL"])
            assert result["success"]
            assert len(result["stocks"]) <= 5
            assert all(s["symbol"] in ["AAPL", "GOOGL"] for s in result["stocks"])

    async def test_rank_stocks_no_data(self, stock_ranker):
        with patch("services.mcp_client.MCPClient.call_tool", AsyncMock(return_value={"success": False, "error": "No data"})):
            result = await stock_ranker.rank_stocks(["AAPL"])
            assert not result["success"]
            assert "error" in result