"""
Unit tests for TradeFlowOrchestrator in NextG3N Trading System

Tests AutoGen 0.9.0 agent coordination and MCP tool interactions.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from orchestration.trade_flow_orchestrator import TradeFlowOrchestrator

@pytest.mark.asyncio
class TestTradeFlowOrchestrator:
    @pytest.fixture
    def config(self):
        return {
            "kafka": {"bootstrap_servers": "localhost:9092", "topic_prefix": "nextg3n-"},
            "llm": {"model": "gpt-4", "base_url": "https://openrouter.ai/api/v1"},
            "storage": {"redis": {"host": "localhost", "port": 6379, "db": 0}}
        }

    @pytest.fixture
    def orchestrator(self, config):
        return TradeFlowOrchestrator(config)

    async def test_run_trading_workflow_success(self, orchestrator):
        with patch("services.mcp_client.MCPClient.call_tool", side_effect=[
            AsyncMock(return_value={"success": True, "stocks": [{"symbol": "AAPL"}]}),
            AsyncMock(return_value={"success": True, "sentiment_score": 0.6}),
            AsyncMock(return_value={"success": True, "prediction": 100.0}),
            AsyncMock(return_value={"success": True, "context_summary": "Positive news"}),
            AsyncMock(return_value={"success": True, "order_id": "123"}),
            AsyncMock(return_value={"success": True, "should_exit": False})
        ]), patch("autogen.GroupChatManager.initiate_chat", return_value="Buy recommendation"):
            result = await orchestrator.run_trading_workflow()
            assert result["success"]
            assert "decisions" in result
            assert len(result["decisions"]) > 0

    async def test_run_trading_workflow_no_stocks(self, orchestrator):
        with patch("services.mcp_client.MCPClient.call_tool", AsyncMock(return_value={"success": False, "error": "No stocks"})):
            result = await orchestrator.run_trading_workflow()
            assert not result["success"]
            assert "error" in result