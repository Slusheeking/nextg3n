"""
Unit tests for BacktestEngine and TFTStrategy in the NextG3N Trading System.

Tests LLM-driven strategy generation, backtesting, parallel backtesting,
and Kafka event publishing.
"""

import pytest
import pandas as pd
import json
from unittest.mock import AsyncMock, patch
from datetime import datetime
from models.backtest.backtest_engine import BacktestEngine, TFTStrategy
import backtrader as bt

@pytest.mark.asyncio
class TestBacktestEngine:
    @pytest.fixture
    def config(self):
        return {
            "kafka": {
                "bootstrap_servers": "localhost:9092",
                "topic_prefix": "nextg3n-"
            },
            "llm": {
                "model": "openai/gpt-4",
                "base_url": "https://openrouter.ai/api/v1",
                "max_tokens": 200
            },
            "backtest": {
                "initial_cash": 100000,
                "commission": 0.001
            }
        }

    @pytest.fixture
    def backtest_engine(self, config):
        return BacktestEngine(config)

    @pytest.fixture
    def mock_data_service(self):
        class MockDataService:
            def get_historical_data(self, symbol, timeframe, limit):
                return {
                    "bars": [
                        {"timestamp": "2023-01-01", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000},
                        {"timestamp": "2023-01-02", "open": 101, "high": 103, "low": 100, "close": 102, "volume": 1100}
                    ]
                }
        return MockDataService()

    @pytest.fixture
    def mock_forecast_model(self):
        class MockForecastModel:
            def predict_price(self, symbol, timeframe, data):
                return {"action": "buy", "confidence": 0.7}
        return MockForecastModel()

    async def test_get_llm_strategy_success(self, backtest_engine):
        with patch("aiohttp.ClientSession.post", new=AsyncMock()) as mock_post:
            mock_post.return_value.status = 200
            mock_post.return_value.json = AsyncMock(return_value={
                "choices": [{"message": {"content": json.dumps({
                    "size": 200,
                    "stop_loss": 0.03,
                    "lookback": 20,
                    "confidence_threshold": 0.65
                })}}]
            })
            result = await backtest_engine.get_llm_strategy()
            assert result == {
                "size": 200,
                "stop_loss": 0.03,
                "lookback": 20,
                "confidence_threshold": 0.65
            }

    async def test_get_llm_strategy_invalid_response(self, backtest_engine):
        with patch("aiohttp.ClientSession.post", new=AsyncMock()) as mock_post:
            mock_post.return_value.status = 200
            mock_post.return_value.json = AsyncMock(return_value={
                "choices": [{"message": {"content": "Invalid JSON"}}]
            })
            result = await backtest_engine.get_llm_strategy()
            assert result == {
                "size": 100,
                "stop_loss": 0.05,
                "lookback": 30,
                "confidence_threshold": 0.6
            }

    async def test_run_backtest_success(self, backtest_engine, mock_data_service, mock_forecast_model):
        with patch("aiohttp.ClientSession.post", new=AsyncMock()) as mock_post, \
             patch("kafka.KafkaProducer.send") as mock_kafka, \
             patch("models.backtest.backtest_engine.MarketDataService", return_value=mock_data_service), \
             patch("models.backtest.backtest_engine.ForecastModel", return_value=mock_forecast_model):
            mock_post.return_value.status = 200
            mock_post.return_value.json = AsyncMock(return_value={
                "choices": [{"message": {"content": json.dumps({
                    "size": 150,
                    "stop_loss": 0.04,
                    "lookback": 25,
                    "confidence_threshold": 0.7
                })}}]
            })
            result = await backtest_engine.run_backtest("AAPL", timeframe="1d")
            assert "symbol" in result
            assert result["symbol"] == "AAPL"
            assert "llm_params" in result
            assert result["llm_params"] == {
                "size": 150,
                "stop_loss": 0.04,
                "lookback": 25,
                "confidence_threshold": 0.7
            }
            mock_kafka.assert_called_once()

    async def test_run_backtest_no_data(self, backtest_engine, mock_data_service):
        with patch("aiohttp.ClientSession.post", new=AsyncMock()) as mock_post, \
             patch("kafka.KafkaProducer.send") as mock_kafka, \
             patch("models.backtest.backtest_engine.MarketDataService", return_value=mock_data_service) as mock_service:
            mock_data_service.get_historical_data = lambda *args, **kwargs: {"bars": []}
            result = await backtest_engine.run_backtest("AAPL", timeframe="1d")
            assert result["symbol"] == "AAPL"
            assert result["error"] == "No data available"
            mock_kafka.assert_not_called()

    async def test_run_parallel_backtest(self, backtest_engine, mock_data_service, mock_forecast_model):
        with patch("aiohttp.ClientSession.post", new=AsyncMock()) as mock_post, \
             patch("kafka.KafkaProducer.send") as mock_kafka, \
             patch("models.backtest.backtest_engine.MarketDataService", return_value=mock_data_service), \
             patch("models.backtest.backtest_engine.ForecastModel", return_value=mock_forecast_model):
            mock_post.return_value.status = 200
            mock_post.return_value.json = AsyncMock(return_value={
                "choices": [{"message": {"content": json.dumps({
                    "size": 100,
                    "stop_loss": 0.05,
                    "lookback": 30,
                    "confidence_threshold": 0.6
                })}}]
            })
            results = await backtest_engine.run_parallel_backtest(["AAPL", "GOOGL"], timeframe="1d")
            assert len(results) == 2
            assert all(r["symbol"] in ["AAPL", "GOOGL"] for r in results)
            assert mock_kafka.call_count >= 2  # At least one per backtest + summary