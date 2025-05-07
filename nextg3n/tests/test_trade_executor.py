"""
Unit tests for TradingEngine in NextG3N Trading System

Tests the core trading cycle of the TradingEngine.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from models.trade.trading_engine import TradingEngine

@pytest.mark.asyncio
class TestTradingEngine:
    @pytest.fixture
    def config(self):
        return {
            "kafka": {"bootstrap_servers": "localhost:9092", "topic_prefix": "nextg3n-"},
            "storage": {"redis": {"host": "localhost", "port": 6379, "db": 0}},
            "trading_engine": {"top_n_stocks": 2},
            "trade": {"default_order_type": "market"},
            "models": {"trade": {"alpaca_api_key": "key", "alpaca_api_secret": "secret"}}
        }

    @pytest.fixture
    def trading_engine(self, config):
        return TradingEngine(config)

    async def test_run_trading_cycle_success(self, trading_engine):
        # Mock the dependencies of the TradingEngine
        with patch("models.stock_ranker.stock_ranker.StockRanker.rank_stocks") as mock_rank_stocks, \
             patch("models.decision.decision_model.DecisionModel.make_decision") as mock_make_decision, \
             patch("models.trade.trade_executor.TradeExecutor.place_order") as mock_place_order:

            # Configure the mocks
            mock_rank_stocks.return_value = [{"symbol": "AAPL", "rank": 1}, {"symbol": "GOOGL", "rank": 2}]
            mock_make_decision.return_value = {"action": "buy", "quantity": 1, "current_price": 150.0}
            mock_place_order.return_value = {"success": True, "order_id": "123"}

            # Call the run_trading_cycle method
            symbols = ["AAPL", "GOOGL"]
            executed_trades = await trading_engine.run_trading_cycle(symbols)

            # Assertions
            assert len(executed_trades) == 2
            assert executed_trades[0]["symbol"] == "AAPL"
            assert executed_trades[0]["action"] == "buy"
            assert executed_trades[0]["quantity"] == 1
            assert executed_trades[0]["status"] == "placed"
            assert executed_trades[1]["symbol"] == "GOOGL"
            assert executed_trades[1]["action"] == "buy"
            assert executed_trades[1]["quantity"] == 1
            assert executed_trades[1]["status"] == "placed"

    async def test_run_trading_cycle_no_stocks(self, trading_engine):
        # Mock the StockRanker to return an empty list
        with patch("models.stock_ranker.stock_ranker.StockRanker.rank_stocks") as mock_rank_stocks:
            mock_rank_stocks.return_value = []

            # Call the run_trading_cycle method
            symbols = ["AAPL", "GOOGL"]
            executed_trades = await trading_engine.run_trading_cycle(symbols)

            # Assertions
            assert len(executed_trades) == 0