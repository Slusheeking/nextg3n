"""
Unit tests for the TradeFlowOrchestrator class in the NextG3N Trading System.
"""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from nextg3n.orchestration.trade_flow_orchestrator import TradeFlowOrchestrator
import asyncio

class TestTradeFlowOrchestrator(unittest.TestCase):
    def setUp(self):
        self.config = {
            "models": {
                "stock_ranker": {"enabled": True},
                "sentiment": {"enabled": True},
                "forecast": {"enabled": True},
                "context": {"enabled": True},
                "decision": {"enabled": True},
                "trade": {"enabled": True},
                "backtest": {"enabled": True},
                "trainer": {"enabled": True}
            },
            "visualization": {
                "chart_generator": {"enabled": True},
                "metrics_api": {"enabled": True},
                "trade_dashboard": {"enabled": True, "port": 3050}
            },
            "kafka": {"bootstrap_servers": "localhost:9092"},
            "llm": {
                "enabled": True,
                "provider": "openrouter",
                "model": "openai/gpt-4",
                "max_tokens": 512,
                "temperature": 0.7,
                "retry_attempts": 3,
                "retry_delay": 1000
            }
        }
        self.orchestrator = TradeFlowOrchestrator(self.config)
        self.orchestrator.logger = MagicMock()

    @patch('autogen.ConversableAgent')
    @patch('autogen.GroupChat')
    @patch('autogen.GroupChatManager')
    async def test_start_success(self, mock_group_chat_manager, mock_group_chat, mock_agent):
        # Mock AutoGen components
        mock_group_chat_instance = MagicMock()
        mock_group_chat.return_value = mock_group_chat_instance
        mock_group_chat_manager_instance = AsyncMock()
        mock_group_chat_manager.return_value = mock_group_chat_manager_instance
        mock_group_chat_manager_instance.initiate_chat = AsyncMock(return_value=None)
        
        # Mock agents
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        
        # Run start
        await self.orchestrator.start()
        
        self.orchestrator.logger.info.assert_any_call("Starting TradeFlowOrchestrator")
        self.orchestrator.logger.info.assert_any_call("TradeFlowOrchestrator started successfully")
        mock_group_chat_manager_instance.initiate_chat.assert_called()

    async def test_shutdown_success(self):
        # Mock shutdown methods
        for component in list(self.orchestrator.models.values()) + \
                         list(self.orchestrator.visualization.values()):
            if hasattr(component, "shutdown"):
                component.shutdown = MagicMock()
        
        # Run shutdown
        self.orchestrator.shutdown()
        
        self.orchestrator.logger.info.assert_any_call("Shutting down TradeFlowOrchestrator")
        self.orchestrator.logger.info.assert_any_call("TradeFlowOrchestrator shutdown complete")
        for component in list(self.orchestrator.models.values()) + \
                         list(self.orchestrator.visualization.values()):
            if hasattr(component, "shutdown"):
                component.shutdown.assert_called()

if __name__ == '__main__':
    unittest.main()