"""
Trade Flow Orchestrator for NextG3N Trading System

This module implements the TradeFlowOrchestrator class, coordinating agents and workflows
for trading operations, enhanced with AutoGen for LLM-powered agent interactions.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import yaml
import os
from autogen import ConversableAgent, GroupChat, GroupChatManager

# System imports
from nextg3n import (
    StockRanker,
    SentimentModel,
    ForecastModel,
    ContextRetriever,
    DecisionModel,
    TradeExecutor,
    BacktestEngine,
    TrainerModel,
    ChartGenerator,
    MetricsApi,
    TradeDashboard,
    MetricsLogger
)

class TradeFlowOrchestrator:
    """
    Orchestrates trading workflows and agents in the NextG3N system, using AutoGen for
    LLM-powered agent interactions.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TradeFlowOrchestrator with configuration and system components.

        Args:
            config: Configuration dictionary with system settings
        """
        init_start_time = time.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="trade_flow_orchestrator")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        
        # Initialize components
        self.models = self._initialize_models()
        self.services = self._initialize_services()
        self.storage = self._initialize_storage()
        self.monitoring = self._initialize_monitoring()
        self.visualization = self._initialize_visualization()
        
        # Initialize AutoGen agents
        self.agents = self._initialize_agents()
        self.group_chat = None
        self.group_chat_manager = None
        self._initialize_group_chat()
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("trade_flow_orchestrator.initialization_time_ms", init_duration)
        self.logger.info("TradeFlowOrchestrator initialized")

    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initialize model components.

        Returns:
            Dictionary of initialized models
        """
        models = {}
        model_configs = self.config.get("models", {})
        if model_configs.get("stock_ranker", {}).get("enabled", True):
            models["stock_ranker"] = StockRanker(self.config)
            self.logger.info("Initialized StockRanker")
        if model_configs.get("sentiment", {}).get("enabled", True):
            models["sentiment"] = SentimentModel(self.config)
            self.logger.info("Initialized SentimentModel")
        if model_configs.get("forecast", {}).get("enabled", True):
            models["forecast"] = ForecastModel(self.config)
            self.logger.info("Initialized ForecastModel")
        if model_configs.get("context", {}).get("enabled", True):
            models["context"] = ContextRetriever(self.config)
            self.logger.info("Initialized ContextRetriever")
        if model_configs.get("decision", {}).get("enabled", True):
            models["decision"] = DecisionModel(self.config)
            self.logger.info("Initialized DecisionModel")
        if model_configs.get("trade", {}).get("enabled", True):
            models["trade"] = TradeExecutor(self.config)
            self.logger.info("Initialized TradeExecutor")
        if model_configs.get("backtest", {}).get("enabled", True):
            models["backtest"] = BacktestEngine(self.config)
            self.logger.info("Initialized BacktestEngine")
        if model_configs.get("trainer", {}).get("enabled", True):
            models["trainer"] = TrainerModel(self.config)
            self.logger.info("Initialized TrainerModel")
        return models

    def _initialize_services(self) -> Dict[str, Any]:
        """
        Initialize service components (placeholder).

        Returns:
            Dictionary of initialized services
        """
        return {}

    def _initialize_storage(self) -> Dict[str, Any]:
        """
        Initialize storage components (placeholder).

        Returns:
            Dictionary of initialized storage components
        """
        return {}

    def _initialize_monitoring(self) -> Dict[str, Any]:
        """
        Initialize monitoring components (placeholder).

        Returns:
            Dictionary of initialized monitoring components
        """
        return {}

    def _initialize_visualization(self) -> Dict[str, Any]:
        """
        Initialize visualization components.

        Returns:
            Dictionary of initialized visualization components
        """
        visualization = {}
        viz_configs = self.config.get("visualization", {})
        if viz_configs.get("chart_generator", {}).get("enabled", True):
            visualization["chart_generator"] = ChartGenerator(self.config)
            self.logger.info("Initialized ChartGenerator")
        if viz_configs.get("metrics_api", {}).get("enabled", True):
            visualization["metrics_api"] = MetricsApi(self.config)
            asyncio.create_task(visualization["metrics_api"].start_server())
            self.logger.info("Initialized MetricsApi")
        if viz_configs.get("trade_dashboard", {}).get("enabled", True):
            visualization["trade_dashboard"] = TradeDashboard(self.config)
            asyncio.create_task(visualization["trade_dashboard"].start_dashboard())
            self.logger.info("Initialized TradeDashboard")
        return visualization

    def _initialize_agents(self) -> Dict[str, ConversableAgent]:
        """
        Initialize AutoGen agents with LLM configuration.

        Returns:
            Dictionary of initialized agents
        """
        llm_config = {
            "model": self.config["llm"]["model"],
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": self.config["llm"]["base_url"],
            "max_tokens": self.config["llm"]["max_tokens"],
            "temperature": self.config["llm"]["temperature"]
        }
        
        agents = {
            "stock_picker": ConversableAgent(
                name="StockPickerAgent",
                llm_config=llm_config,
                system_message="Rank stocks based on provided data."
            ),
            "sentiment": ConversableAgent(
                name="SentimentAgent",
                llm_config=llm_config,
                system_message="Analyze sentiment of financial texts."
            ),
            "predictor": ConversableAgent(
                name="PredictorAgent",
                llm_config=llm_config,
                system_message="Forecast stock price movements."
            ),
            "context": ConversableAgent(
                name="ContextAgent",
                llm_config=llm_config,
                system_message="Retrieve and summarize relevant context."
            ),
            "trade": ConversableAgent(
                name="TradeAgent",
                llm_config=llm_config,
                system_message="Execute trading decisions."
            ),
            "monitor": ConversableAgent(
                name="MonitorAgent",
                llm_config=llm_config,
                system_message="Monitor system performance and generate insights."
            ),
            "user_proxy": ConversableAgent(
                name="UserProxyAgent",
                llm_config=False,  # No LLM for user proxy
                human_input_mode="NEVER",
                system_message="Proxy for user interactions."
            )
        }
        
        return agents

    def _initialize_group_chat(self):
        """
        Initialize AutoGen group chat for agent collaboration.
        """
        self.group_chat = GroupChat(
            agents=list(self.agents.values()),
            messages=[],
            max_round=10
        )
        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config={
                "model": self.config["llm"]["model"],
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "base_url": self.config["llm"]["base_url"],
                "max_tokens": self.config["llm"]["max_tokens"],
                "temperature": self.config["llm"]["temperature"]
            }
        )

    async def start(self):
        """
        Start the orchestration process, coordinating agents and workflows.
        """
        self.logger.info("Starting TradeFlowOrchestrator")
        
        try:
            # Example workflow: Coordinate agents to analyze and trade
            task = """
            Analyze the market for AAPL:
            1. Rank stocks using StockRanker.
            2. Analyze sentiment using SentimentModel.
            3. Forecast price using ForecastModel.
            4. Retrieve context using ContextRetriever.
            5. Make a trading decision using DecisionModel.
            6. Execute the trade using TradeExecutor.
            7. Generate visualizations using ChartGenerator and TradeDashboard.
            """
            await self.group_chat_manager.initiate_chat(
                self.agents["user_proxy"],
                message=task
            )
            
            self.logger.info("TradeFlowOrchestrator started successfully")
        
        except Exception as e:
            self.logger.error(f"Error starting TradeFlowOrchestrator: {e}")
            raise

    def shutdown(self):
        """
        Shutdown the orchestrator and close resources.
        """
        self.logger.info("Shutting down TradeFlowOrchestrator")
        self.executor.shutdown(wait=True)
        for component in list(self.models.values()) + list(self.services.values()) + \
                        list(self.storage.values()) + list(self.monitoring.values()) + \
                        list(self.visualization.values()):
            if hasattr(component, "shutdown"):
                component.shutdown()
        self.logger.info("TradeFlowOrchestrator shutdown complete")