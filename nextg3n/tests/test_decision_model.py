"""
Unit tests for DecisionModel in NextG3N Trading System

Tests the decision-making process of the DecisionModel.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from models.decision.decision_model import DecisionModel

@pytest.mark.asyncio
class TestDecisionModel:
    @pytest.fixture
    def config(self):
        return {
            "kafka": {"bootstrap_servers": "localhost:9092", "topic_prefix": "nextg3n-"},
            "llm": {"model": "gpt-4", "base_url": "https://openrouter.ai/api/v1", "max_tokens": 512},
            "storage": {"redis": {"host": "localhost", "port": 6379, "db": 0}},
            "decision_logic": {"up_threshold": 0.05, "down_threshold": -0.05},
            "data": {"timeframe": "1m", "limit": 60}
        }

    @pytest.fixture
    def decision_model(self, config):
        return DecisionModel(config)

    async def test_make_decision_success(self, decision_model):
        # Mock the dependencies of the DecisionModel
        with patch("models.forecast.forecast_model.ForecastModel.predict") as mock_predict, \
             patch("services.market_data_service.MarketDataService.get_historical_bars") as mock_get_historical_bars:

            # Configure the mocks
            mock_get_historical_bars.return_value = [
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
                {"open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1200}
            ]
            mock_predict.return_value = {"prediction": 0.1, "direction": "up"}

            # Call the make_decision method
            symbol = "AAPL"
            result = await decision_model.make_decision(symbol)

            # Assertions
            assert result["success"]
            assert result["symbol"] == symbol
            assert result["action"] == "buy"  # Based on the up_threshold and prediction
            assert "forecast_result" in result

    async def test_make_decision_no_data(self, decision_model):
        # Mock the MarketDataService to return an empty list
        with patch("services.market_data_service.MarketDataService.get_historical_bars") as mock_get_historical_bars:
            # Configure the mocks
            mock_get_historical_bars.return_value = []

            # Call the make_decision method
            symbol = "AAPL"
            result = await decision_model.make_decision(symbol)

            # Assertions
            assert not result["success"]
            assert "error" in result