"""
Trade Flow Orchestrator for NextG3N Trading System

This module implements the TradeFlowOrchestrator class, coordinating AutoGen agents
to manage trading workflows, including stock selection, sentiment analysis, prediction,
context retrieval, trading decisions, training, and backtesting with LLM-driven optimization.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv
from kafka import KafkaConsumer, KafkaProducer
from autogen import ConversableAgent, GroupChat, GroupChatManager

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

# Model imports (assumed to exist)
from models.sentiment.sentiment_model import SentimentModel
from models.forecast.forecast_model import ForecastModel
from models.context.context_retriever import ContextRetriever
from models.decision.decision_model import DecisionModel
from models.stock_ranker.stock_ranker import StockRanker
from models.trade.trade_executor import TradeExecutor
from models.trainer.trainer_model import TrainerModel
from models.backtest.backtest_engine import BacktestEngine

class TradeFlowOrchestrator:
    """
    Orchestrates trading workflows using AutoGen agents, integrating LLM-driven training
    and backtesting for the NextG3N system.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TradeFlowOrchestrator with configuration.

        Args:
            config: Configuration dictionary
        """
        load_dotenv()
        self.logger = MetricsLogger(component_name="trade_flow_orchestrator")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.config = config
        self.kafka_config = config.get("kafka", {})
        self.llm_config = config.get("llm", {})

        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize models (placeholder; replace with actual implementations)
        self.models = {
            "sentiment": SentimentModel(config),
            "forecast": ForecastModel(config),
            "context": ContextRetriever(config),
            "decision": DecisionModel(config),
            "stock_ranker": StockRanker(config),
            "trade": TradeExecutor(config),
            "trainer": TrainerModel(config),
            "backtest": BacktestEngine(config)
        }

        # Initialize AutoGen agents
        self.agents = {
            "stock_picker": ConversableAgent(
                name="StockPicker",
                llm_config={
                    "config_list": [{"model": self.llm_config.get("model", "openai/gpt-4"), "api_key": os.getenv("OPENROUTER_API_KEY")}]
                },
                system_message="Select promising stocks based on rankings and market data."
            ),
            "sentiment_analyzer": ConversableAgent(
                name="SentimentAnalyzer",
                llm_config={
                    "config_list": [{"model": self.llm_config.get("model", "openai/gpt-4"), "api_key": os.getenv("OPENROUTER_API_KEY")}]
                },
                system_message="Analyze sentiment from news and social data."
            ),
            "predictor": ConversableAgent(
                name="Predictor",
                llm_config={
                    "config_list": [{"model": self.llm_config.get("model", "openai/gpt-4"), "api_key": os.getenv("OPENROUTER_API_KEY")}]
                },
                system_message="Generate price predictions using ForecastModel."
            ),
            "context_analyzer": ConversableAgent(
                name="ContextAnalyzer",
                llm_config={
                    "config_list": [{"model": self.llm_config.get("model", "openai/gpt-4"), "api_key": os.getenv("OPENROUTER_API_KEY")}]
                },
                system_message="Retrieve and summarize market context."
            ),
            "trade_decider": ConversableAgent(
                name="TradeDecider",
                llm_config={
                    "config_list": [{"model": self.llm_config.get("model", "openai/gpt-4"), "api_key": os.getenv("OPENROUTER_API_KEY")}]
                },
                system_message="Make trading decisions based on predictions and context."
            ),
            "trainer": ConversableAgent(
                name="Trainer",
                llm_config={
                    "config_list": [{"model": self.llm_config.get("model", "openai/gpt-4"), "api_key": os.getenv("OPENROUTER_API_KEY")}]
                },
                system_message="Train and fine-tune models."
            ),
            "hyperparameter_optimizer": ConversableAgent(
                name="HyperparameterOptimizer",
                llm_config={
                    "config_list": [{"model": self.llm_config.get("model", "openai/gpt-4"), "api_key": os.getenv("OPENROUTER_API_KEY")}]
                },
                system_message="Optimize model hyperparameters based on training metrics."
            ),
            "strategy_generator": ConversableAgent(
                name="StrategyGenerator",
                llm_config={
                    "config_list": [{"model": self.llm_config.get("model", "openai/gpt-4"), "api_key": os.getenv("OPENROUTER_API_KEY")}]
                },
                system_message="Generate and refine trading strategies for backtesting."
            ),
            "user_proxy": ConversableAgent(
                name="UserProxy",
                is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
                human_input_mode="NEVER",
                system_message="Proxy for user interactions."
            )
        }

        # Initialize group chat
        self.group_chat = GroupChat(
            agents=list(self.agents.values()),
            messages=[],
            max_round=10
        )
        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config={
                "config_list": [{"model": self.llm_config.get("model", "openai/gpt-4"), "api_key": os.getenv("OPENROUTER_API_KEY")}]
            }
        )

        self.logger.info("TradeFlowOrchestrator initialized")

    async def start_workflow(self, task: str):
        """
        Start a trading workflow using AutoGen agents.

        Args:
            task: Task description (e.g., "Run daily trading cycle")
        """
        operation_id = f"workflow_{int(time.time())}"
        self.logger.info(f"Starting workflow: {task}, Operation: {operation_id}")

        try:
            await self.group_chat_manager.initiate_chat(
                self.agents["user_proxy"],
                message=task
            )
            self.logger.info(f"Workflow completed: {task}")
        except Exception as e:
            self.logger.error(f"Workflow error: {e}")

    async def consume_kafka_events(self):
        """
        Consume Kafka events and trigger agent actions.
        """
        operation_id = f"kafka_{int(time.time())}"
        consumer = KafkaConsumer(
            f"{self.kafka_config.get('topic_prefix', 'nextg3n-')}trainer-events",
            f"{self.kafka_config.get('topic_prefix', 'nextg3n-')}backtest-events",
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            group_id="orchestrator",
            auto_offset_reset="latest",
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )

        self.logger.info("Starting Kafka consumer for trainer and backtest events")

        for message in consumer:
            try:
                event = message.value.get("event")
                data = message.value.get("data")
                topic = message.topic

                if topic.endswith("trainer-events"):
                    if event == "model_trained":
                        self.logger.info(f"Received model_trained event: {data['model_name']}")
                        # Trigger hyperparameter optimization
                        await self.agents["hyperparameter_optimizer"].initiate_chat(
                            self.agents["user_proxy"],
                            message=f"Optimize hyperparameters for {data['model_name']} based on metrics: {data.get('metrics', {})}"
                        )
                    elif event == "model_evaluated":
                        self.logger.info(f"Received model_evaluated event: {data['model_name']}")
                        # Trigger retraining if metrics are poor
                        if data.get("metrics", {}).get("accuracy", 0) < 0.7:
                            await self.agents["trainer"].initiate_chat(
                                self.agents["user_proxy"],
                                message=f"Retrain {data['model_name']} with suggestions: {data.get('llm_suggestions', {})}"
                            )

                elif topic.endswith("backtest-events"):
                    if event == "backtest_completed":
                        self.logger.info(f"Received backtest_completed event: {data['symbol']}")
                        # Trigger strategy refinement
                        await self.agents["strategy_generator"].initiate_chat(
                            self.agents["user_proxy"],
                            message=f"Refine trading strategy for {data['symbol']} based on backtest: {data}"
                        )
                    elif event == "parallel_backtest_completed":
                        self.logger.info(f"Received parallel_backtest_completed event: {data['total_results']} results")
                        # Trigger analysis of best strategies
                        await self.agents["strategy_generator"].initiate_chat(
                            self.agents["user_proxy"],
                            message=f"Analyze parallel backtest results: {data}"
                        )

                # Publish orchestration event
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n-')}orchestration-events",
                    {"event": "agent_action", "data": {"operation_id": operation_id, "topic": topic, "event": event}}
                )

            except Exception as e:
                self.logger.error(f"Error processing Kafka event: {e}")

    def shutdown(self):
        """
        Shutdown the orchestrator and close resources.
        """
        self.logger.info("Shutting down TradeFlowOrchestrator")
        self.producer.close()
        self.logger.info("Kafka producer closed")
