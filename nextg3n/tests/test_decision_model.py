"""
Unit tests for DecisionModel in NextG3N Trading System

Tests chat room, PPO, and CNN-based decision-making with MCP tools.
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
            "storage": {"redis": {"host": "localhost", "port": 6379, "db": 0}}
        }

    @pytest.fixture
    def decision_model(self, config):
        return DecisionModel(config)

    async def test_make_decision_success(self, decision_model):
        with patch("services.mcp_client.MCPClient.call_tool", side_effect=[
            AsyncMock(return_value={"success": True, "sentiment_score": 0.6}),
            AsyncMock(return_value={"success": True, "prediction": 100.0}),
            AsyncMock(return_value={"success": True, "bars": [{"close": 99.0}, {"close": 100.0}]}),
        ]), patch("autogen.GroupChatManager.initiate_chat", return_value="Buy recommendation"):
            result = await decision_model.make_decision("AAPL")
            assert result["success"]
            assert result["symbol"] == "AAPL"
            assert result["action"] in ["buy", "sell", "hold"]
            assert "cnn_action" in result

    async def test_make_decision_data_failure(self, decision_model):
        with patch("services.mcp_client.MCPClient.call_tool", AsyncMock(return_value={"success": False, "error": "No data"})):
            result = await decision_model.make_decision("AAPL")
            assert not result["success"]
            assert "error" in result