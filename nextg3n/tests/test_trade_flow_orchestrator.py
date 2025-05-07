"""
Unit tests for TradeFlowOrchestrator in the NextG3N Trading System.

Tests AutoGen agent interactions, Kafka event processing for trainer and backtest events,
and workflow orchestration with LLM-driven agents.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch
from datetime import datetime
from orchestration.trade_flow_orchestrator import TradeFlowOrchestrator

@pytest.mark.asyncio
class TestTradeFlowOrchestrator:
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
                "max_tokens": 512
            }
        }

    @pytest.fixture
    def orchestrator(self, config):
        return TradeFlowOrchestrator(config)

    async def test_init_agents(self, orchestrator):
        assert set(orchestrator.agents.keys()) == {
            "stock_picker", "sentiment_analyzer", "predictor", "context_analyzer",
            "trade_decider", "trainer", "hyperparameter_optimizer", "strategy_generator",
            "user_proxy"
        }
        assert orchestrator.group_chat_manager is not None

    @pytest.mark.asyncio
    async def test_start_workflow_success(self, orchestrator):
        with patch("autogen.GroupChatManager.initiate_chat", new=AsyncMock()) as mock_chat:
            await orchestrator.start_workflow("Run daily trading cycle")
            mock_chat.assert_called_once()
            assert mock_chat.call_args[1]["message"] == "Run daily trading cycle"

    @pytest.mark.asyncio
    async def test_consume_trainer_events(self, orchestrator):
        mock_message = {
            "value": {
                "event": "model_trained",
                "data": {
                    "model_name": "sentiment",
                    "metrics": {"accuracy": 0.85, "loss": 0.2},
                    "operation_id": "test_op",
                    "timestamp": datetime.utcnow().isoformat()
                }
            },
            "topic": "nextg3n-trainer-events"
        }
        with patch("kafka.KafkaConsumer.__iter__", return_value=[mock_message]), \
             patch("kafka.KafkaConsumer.__next__", return_value=mock_message), \
             patch("autogen.ConversableAgent.initiate_chat", new=AsyncMock()) as mock_chat, \
             patch("kafka.KafkaProducer.send") as mock_kafka:
            await orchestrator.consume_kafka_events()
            mock_chat.assert_called_once()
            assert "Optimize hyperparameters" in mock_chat.call_args[1]["message"]
            mock_kafka.assert_called_once()

    @pytest.mark.asyncio
    async def test_consume_backtest_events(self, orchestrator):
        mock_message = {
            "value": {
                "event": "backtest_completed",
                "data": {
                    "symbol": "AAPL",
                    "returns": 5.2,
                    "win_rate": 60.0,
                    "sharpe_ratio": 1.2,
                    "operation_id": "test_op",
                    "llm_params": {"size": 100, "stop_loss": 0.05}
                }
            },
            "topic": "nextg3n-backtest-events"
        }
        with patch("kafka.KafkaConsumer.__iter__", return_value=[mock_message]), \
             patch("kafka.KafkaConsumer.__next__", return_value=mock_message), \
             patch("autogen.ConversableAgent.initiate_chat", new=AsyncMock()) as mock_chat, \
             patch("kafka.KafkaProducer.send") as mock_kafka:
            await orchestrator.consume_kafka_events()
            mock_chat.assert_called_once()
            assert "Refine trading strategy" in mock_chat.call_args[1]["message"]
            mock_kafka.assert_called_once()

    async def test_consume_invalid_event(self, orchestrator):
        mock_message = {
            "value": {
                "event": "unknown_event",
                "data": {}
            },
            "topic": "nextg3n-trainer-events"
        }
        with patch("kafka.KafkaConsumer.__iter__", return_value=[mock_message]), \
             patch("kafka.KafkaConsumer.__next__", return_value=mock_message), \
             patch("autogen.ConversableAgent.initiate_chat", new=AsyncMock()) as mock_chat, \
             patch("kafka.KafkaProducer.send") as mock_kafka:
            await orchestrator.consume_kafka_events()
            mock_chat.assert_not_called()
            mock_kafka.assert_not_called()