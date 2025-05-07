"""
Unit tests for ForecastModel in NextG3N Trading System

Tests LSTM-based intraday price forecasting with MCP tools.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from models.forecast.forecast_model import ForecastModel

@pytest.mark.asyncio
class TestForecastModel:
    @pytest.fixture
    def config(self):
        return {
            "kafka": {"bootstrap_servers": "localhost:9092", "topic_prefix": "nextg3n-"},
            "storage": {"redis": {"host": "localhost", "port": 6379, "db": 0}}
        }

    @pytest.fixture
    def forecast_model(self, config):
        return ForecastModel(config)

    async def test_predict_price_success(self, forecast_model):
        with patch("services.mcp_client.MCPClient.call_tool", AsyncMock(return_value={
            "success": True,
            "bars": [{"open": 99.0, "high": 100.0, "low": 98.0, "close": 99.5, "volume": 1000}] * 60
        })):
            result = await forecast_model.predict_price("AAPL")
            assert result["success"]
            assert "prediction" in result
            assert result["symbol"] == "AAPL"

    async def test_predict_price_no_data(self, forecast_model):
        with patch("services.mcp_client.MCPClient.call_tool", AsyncMock(return_value={"success": False, "error": "No bars"})):
            result = await forecast_model.predict_price("AAPL")
            assert not result["success"]
            assert "error" in result