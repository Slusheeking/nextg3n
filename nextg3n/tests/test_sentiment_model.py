"""
Unit tests for SentimentModel in NextG3N Trading System

Tests FinBERT-based sentiment analysis with MCP tools.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from models.sentiment.sentiment_model import SentimentModel

@pytest.mark.asyncio
class TestSentimentModel:
    @pytest.fixture
    def config(self):
        return {
            "kafka": {"bootstrap_servers": "localhost:9092", "topic_prefix": "nextg3n-"},
            "storage": {"redis": {"host": "localhost", "port": 6379, "db": 0}}
        }

    @pytest.fixture
    def sentiment_model(self, config):
        return SentimentModel(config)

    async def test_analyze_sentiment_success(self, sentiment_model):
        with patch("services.mcp_client.MCPClient.call_tool", side_effect=[
            AsyncMock(return_value={"success": True, "posts": [{"title": "Bullish on AAPL", "text": "Great news"}]}),
            AsyncMock(return_value={"success": True, "flow": [{"type": "call", "premium": 100000}]}),
            AsyncMock(return_value={"success": True, "news": [{"title": "AAPL earnings beat", "summary": "Strong results"}]}),
        ]):
            result = await sentiment_model.analyze_sentiment("AAPL")
            assert result["success"]
            assert "sentiment_score" in result
            assert result["symbol"] == "AAPL"

    async def test_analyze_sentiment_no_data(self, sentiment_model):
        with patch("services.mcp_client.MCPClient.call_tool", AsyncMock(return_value={"success": False, "error": "No data"})):
            result = await sentiment_model.analyze_sentiment("AAPL")
            assert not result["success"]
            assert "error" in result