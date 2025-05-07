"""
Unit tests for TradeExecutor in NextG3N Trading System

Tests trade execution with LSTM/CNN peak detection and slippage controls.
"""

import pytest
import asyncio
import json
import datetime
from unittest.mock import AsyncMock, patch
from models.trade.trade_executor import TradeExecutor

@pytest.mark.asyncio
class TestTradeExecutor:
    @pytest.fixture
    def config(self):
        return {
            "kafka": {"bootstrap_servers": "localhost:9092", "topic_prefix": "nextg3n-"},
            "storage": {"redis": {"host": "localhost", "port": 6379, "db": 0}},
            "models": {"trade": {"alpaca_api_key": "key", "alpaca_api_secret": "secret"}}
        }

    @pytest.fixture
    def trade_executor(self, config):
        return TradeExecutor(config)

    async def test_execute_trade_success(self, trade_executor):
        with patch("services.mcp_client.MCPClient.call_tool", side_effect=[
            AsyncMock(return_value={"success": True, "price": 100.0}),
            AsyncMock(return_value={"success": True, "order_id": "123", "status": "filled"}),
            AsyncMock(return_value={"success": True, "filled_avg_price": 100.5})
        ]):
            result = await trade_executor.execute_trade("AAPL", "buy", 10)
            assert result["success"]
            assert result["order_id"] == "123"
            assert "slippage" in result

    async def test_monitor_trade_peak_detected(self, trade_executor):
        with patch("services.mcp_client.MCPClient.call_tool", side_effect=[
            AsyncMock(return_value={"success": True, "filled_avg_price": 100.0}),
            AsyncMock(return_value={"success": True, "price": 105.0})
        ]):
            self.redis.set(f"trade:AAPL", json.dumps({"price": 105.0, "timestamp": datetime.utcnow().isoformat()}))
            result = await trade_executor.monitor_trade("AAPL", "123")
            assert result["success"]
            assert result["should_exit"]  # Peak detected