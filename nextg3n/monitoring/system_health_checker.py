"""
System Health Checker for NextG3N Trading System

This module implements the SystemHealthChecker class, performing health checks on system
components, including models, services, storage, messaging, orchestration, visualization,
and external APIs.
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
import aiohttp
import redis
import chromadb
from kafka import KafkaProducer, KafkaConsumer
import os
from dotenv import load_dotenv
import json
import time

# Placeholder imports (replace with actual paths when available)
# from nextg3n.models.sentiment.sentiment_model import SentimentModel
# from nextg3n.models.context.context_retriever import ContextRetriever
# from nextg3n.models.decision.decision_model import DecisionModel
# from nextg3n.orchestration.trade_flow_orchestrator import TradeFlowOrchestrator
from .metrics_logger import MetricsLogger

class SystemHealthChecker:
    """
    Class for performing health checks on NextG3N system components.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SystemHealthChecker with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = MetricsLogger(component_name="system_health_checker")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        # Placeholder for orchestrator initialization
        # self.orchestrator = TradeFlowOrchestrator(config)
        self.producer = KafkaProducer(
            bootstrap_servers=config["kafka"].get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    async def check_health(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check on all system components.

        Returns:
            Dictionary with health check results
        """
        start_time = time.time()
        operation_id = f"health_check_{int(start_time)}"
        self.logger.info(f"Starting system health check - Operation: {operation_id}")
        errors = []

        # Run all health checks concurrently
        checks = await asyncio.gather(
            self.check_models(),
            self.check_services(),
            self.check_storage(),
            self.check_messaging(),
            self.check_orchestration(),
            self.check_visualization(),
            return_exceptions=True
        )

        # Aggregate results
        for check_result in checks:
            if isinstance(check_result, dict) and not check_result["success"]:
                errors.extend(check_result["errors"])

        result = {
            "success": len(errors) == 0,
            "errors": errors,
            "operation_id": operation_id,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Publish to Kafka
        self.producer.send(
            f"{self.config['kafka'].get('topic_prefix', 'nextg3n-')}health-events",
            {"event": "health_check", "data": result}
        )

        duration = (time.time() - start_time) * 1000
        self.logger.timing("system_health_checker.check_health_time_ms", duration)
        if result["success"]:
            self.logger.info("System health check passed")
            self.logger.counter("system_health_checker.checks_passed", 1)
        else:
            self.logger.error(f"System health check failed with {len(errors)} errors")
            self.logger.counter("system_health_checker.checks_failed", 1)
            # Send alert for failures
            await self.logger.send_alert(f"Health check failed: {errors}")

        return result

    async def check_models(self) -> Dict[str, Any]:
        """Check health of all models."""
        errors = []
        try:
            # Placeholder for model checks (uncomment when imports are available)
            """
            # SentimentModel
            sentiment_model = SentimentModel(self.config)
            result = await sentiment_model.analyze_sentiment(["Test news"], use_llm=True)
            if not result["success"]:
                errors.append("SentimentModel LLM analysis failed")
            result = await sentiment_model.analyze_sentiment(["Test news"], use_llm=False)
            if not result["success"]:
                errors.append("SentimentModel RoBERTa analysis failed")

            # ContextRetriever
            context_retriever = ContextRetriever(self.config)
            result = await context_retriever.store_context(["Test article"], [{"source": "test"}])
            if not result["success"]:
                errors.append("ContextRetriever store_context failed")
            result = await context_retriever.retrieve_context("Test query", generate_summary=True)
            if not result["success"]:
                errors.append("ContextRetriever retrieve_context with LLM summary failed")

            # DecisionModel
            decision_model = DecisionModel(self.config)
            state = {
                "price_prediction": {"predicted_price": 150},
                "sentiment": {"sentiment_score": 0.5},
                "technical_indicators": {"rsi": 70, "macd": 0.1},
                "context": {"context_score": 0.2}
            }
            result = await decision_model.make_decision("AAPL", state, explain=True)
            if not result["success"]:
                errors.append("DecisionModel make_decision with LLM explanation failed")

            # Other models (placeholder for initialization check)
            for model_name in ["forecast", "stock_ranker", "trade", "backtest", "trainer"]:
                model = self.orchestrator.models.get(model_name)
                if not model:
                    errors.append(f"{model_name.capitalize()}Model not initialized")
            """
            errors.append("Model checks skipped: Placeholder implementation")
        except Exception as e:
            errors.append(f"Model health check failed: {str(e)}")

        return {"success": len(errors) == 0, "errors": errors}

    async def check_services(self) -> Dict[str, Any]:
        """Check health of external services."""
        errors = []
        async with aiohttp.ClientSession() as session:
            try:
                # Alpaca (MarketDataService, TradeService)
                headers = {
                    "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
                    "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY")
                }
                async with session.get(
                    self.config["services"]["market_data"]["base_url"] + "/v2/stocks/AAPL/quotes/latest",
                    headers=headers
                ) as response:
                    if response.status != 200:
                        errors.append("Alpaca API connectivity failed")

                # NewsAPI
                async with session.get(
                    "https://newsapi.org/v2/top-headlines?country=us&apiKey=" + os.getenv("NEWS_API_KEY")
                ) as response:
                    if response.status != 200:
                        errors.append("NewsAPI connectivity failed")

                # Reddit (placeholder for API check)
                errors.append("Reddit API check not implemented")

                # OpenRouter
                headers = {"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"}
                payload = {
                    "model": self.config["llm"]["model"],
                    "messages": [{"role": "user", "content": "Health check"}],
                    "max_tokens": 10
                }
                async with session.post(
                    self.config["llm"]["base_url"] + "/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        errors.append("OpenRouter API connectivity failed")

            except Exception as e:
                errors.append(f"Service health check failed: {str(e)}")

        return {"success": len(errors) == 0, "errors": errors}

    async def check_storage(self) -> Dict[str, Any]:
        """Check health of storage systems."""
        errors = []
        try:
            # Redis
            redis_client = redis.Redis(
                host=self.config["storage"]["redis"]["host"],
                port=self.config["storage"]["redis"]["port"],
                db=self.config["storage"]["redis"]["db"]
            )
            redis_client.set("health_check", "test")
            if redis_client.get("health_check") != b"test":
                errors.append("Redis read/write failed")
            redis_client.delete("health_check")

            # ChromaDB
            chroma_client = chromadb.HttpClient(
                host=self.config["storage"]["vector_db"]["host"],
                port=self.config["storage"]["vector_db"]["port"]
            )
            collection = chroma_client.get_or_create_collection("health_check")
            collection.add(ids=["test"], documents=["Test document"], embeddings=[[0.1] * 384])
            result = collection.query(query_embeddings=[[0.1] * 384], n_results=1)
            if not result["ids"]:
                errors.append("ChromaDB query failed")
            chroma_client.delete_collection("health_check")

        except Exception as e:
            errors.append(f"Storage health check failed: {str(e)}")

        return {"success": len(errors) == 0, "errors": errors}

    async def check_messaging(self) -> Dict[str, Any]:
        """Check health of Kafka messaging."""
        errors = []
        try:
            # Producer
            self.producer.send(
                f"{self.config['kafka']['topic_prefix']}health-events",
                {"event": "test", "data": {"test": "health_check"}}
            )
            self.producer.flush()

            # Consumer
            consumer = KafkaConsumer(
                f"{self.config['kafka']['topic_prefix']}health-events",
                bootstrap_servers=self.config["kafka"]["bootstrap_servers"],
                auto_offset_reset="latest",
                group_id="health_check"
            )
            consumer.poll(timeout_ms=1000)
            consumer.close()

        except Exception as e:
            errors.append(f"Kafka health check failed: {str(e)}")

        return {"success": len(errors) == 0, "errors": errors}

    async def check_orchestration(self) -> Dict[str, Any]:
        """Check health of TradeFlowOrchestrator."""
        errors = []
        try:
            # Placeholder for orchestrator check
            errors.append("Orchestration check skipped: Placeholder implementation")
            """
            if not self.orchestrator.agents:
                errors.append("TradeFlowOrchestrator agents not initialized")
            task = "Health check task"
            await self.orchestrator.group_chat_manager.initiate_chat(
                self.orchestrator.agents["user_proxy"],
                message=task
            )
            """
        except Exception as e:
            errors.append(f"Orchestration health check failed: {str(e)}")

        return {"success": len(errors) == 0, "errors": errors}

    async def check_visualization(self) -> Dict[str, Any]:
        """Check health of visualization components."""
        errors = []
        async with aiohttp.ClientSession() as session:
            try:
                # TradeDashboard
                async with session.get(
                    f"http://{self.config['visualization']['trade_dashboard']['host']}:{self.config['visualization']['trade_dashboard']['port']}"
                ) as response:
                    if response.status != 200:
                        errors.append("TradeDashboard not accessible")

                # MetricsApi
                async with session.get(
                    f"http://{self.config['visualization']['metrics_api']['host']}:{self.config['visualization']['metrics_api']['port']}/system_metrics"
                ) as response:
                    if response.status != 200:
                        errors.append("MetricsApi not accessible")

            except Exception as e:
                errors.append(f"Visualization health check failed: {str(e)}")

        return {"success": len(errors) == 0, "errors": errors}

    def shutdown(self):
        """Shutdown the health checker."""
        self.logger.info("Shutting down SystemHealthChecker")
        self.producer.close()
        self.logger.info("Kafka producer closed")