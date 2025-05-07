"""
Metrics API for NextG3N Trading System

This module implements the MetricsApi class, providing a programmatic interface to retrieve
and aggregate system, model, and trading metrics. It supports the TradeDashboard and MonitorAgent
in TradeFlowOrchestrator for real-time monitoring.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import redis
import aiohttp

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

# Kafka imports
from kafka import KafkaProducer

class MetricsApiHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for MetricsApi endpoints.
    """
    def __init__(self, metrics_api, *args, **kwargs):
        self.metrics_api = metrics_api
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """
        Handle GET requests for metrics endpoints.
        """
        try:
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            query_params = parse_qs(parsed_url.query)
            
            response = {}
            status_code = 200
            
            if path == "/system_metrics":
                response = asyncio.run(self.metrics_api.get_system_metrics())
            elif path == "/model_metrics":
                model_name = query_params.get("model_name", [None])[0]
                response = asyncio.run(self.metrics_api.get_model_metrics(model_name))
            elif path == "/trading_metrics":
                response = asyncio.run(self.metrics_api.get_trading_metrics())
            else:
                response = {"success": False, "error": "Invalid endpoint"}
                status_code = 404
            
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
            # Log request
            self.metrics_api.logger.info(f"Handled GET request: {path}")
            self.metrics_api.logger.counter("metrics_api.requests_handled", 1)
            
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))
            self.metrics_api.logger.error(f"Error handling GET request: {e}")
            self.metrics_api.logger.counter("metrics_api.request_errors", 1)

class MetricsApi:
    """
    Class for serving metrics data via a lightweight API in the NextG3N system.
    Supports the TradeDashboard and MonitorAgent for real-time monitoring.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MetricsApi with configuration and API settings.

        Args:
            config: Configuration dictionary with API and Kafka settings
        """
        init_start_time = time.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="metrics_api")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.api_config = config.get("visualization", {}).get("metrics_api", {})
        self.kafka_config = config.get("kafka", {})
        self.redis_config = config.get("storage", {}).get("redis", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize Redis client
        self.redis = None
        self._initialize_redis()
        
        # Initialize API server
        self.server = None
        self.host = self.api_config.get("host", "localhost")
        self.port = self.api_config.get("port", 8000)
        self.running = False
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("metrics_api.initialization_time_ms", init_duration)
        self.logger.info("MetricsApi initialized")

    def _initialize_redis(self):
        """
        Initialize the Redis client for metrics storage.
        """
        try:
            self.redis = redis.Redis(
                host=self.redis_config.get("host", "localhost"),
                port=self.redis_config.get("port", 6379),
                db=self.redis_config.get("db", 0),
                decode_responses=True
            )
            self.redis.ping()
            self.logger.info("Connected to Redis for metrics storage")
        except redis.RedisError as e:
            self.logger.error(f"Failed to initialize Redis client: {e}")
            self.redis = None
            raise

    async def start_server(self):
        """
        Start the HTTP server to serve metrics endpoints.
        """
        self.logger.info(f"Starting MetricsApi server on {self.host}:{self.port}")
        self.running = True
        
        def create_handler(*args, **kwargs):
            return MetricsApiHandler(self, *args, **kwargs)
        
        self.server = HTTPServer((self.host, self.port), create_handler)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self.server.serve_forever
        )

    async def get_system_metrics(self) -> Dict[str, Any]:
        """
        Retrieve system health metrics (e.g., CPU, GPU, memory usage).

        Returns:
            Dictionary containing system metrics
        """
        start_time = time.time()
        operation_id = f"get_system_metrics_{int(start_time)}"
        self.logger.info(f"Retrieving system metrics - Operation: {operation_id}")

        try:
            async with aiohttp.ClientSession() as session:
                if not self.redis:
                    raise redis.RedisError("Redis client not initialized")
                
                # Retrieve metrics from Redis
                metrics = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: json.loads(self.redis.get("system_metrics") or "{}")
                )

                result = {
                    "success": True,
                    "metrics": metrics or {
                        "cpu_percent": 0.0,
                        "memory_percent": 0.0,
                        "gpu_metrics": [],
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}metrics_api_events",
                    {"event": "system_metrics_retrieved", "data": result}
                )

                duration = (time.time() - start_time) * 1000
                self.logger.timing("metrics_api.get_system_metrics_time_ms", duration)
                self.logger.info("System metrics retrieved")
                self.logger.counter("metrics_api.metrics_retrieved", 1)
                return result

        except redis.RedisError as e:
            self.logger.error(f"Redis error retrieving system metrics: {e}")
            self.logger.counter("metrics_api.redis_errors", 1)
            return {
                "success": False,
                "error": f"Redis error: {e}",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error retrieving system metrics: {e}")
            self.logger.counter("metrics_api.metrics_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def get_model_metrics(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve model performance metrics (e.g., accuracy, loss).

        Args:
            model_name: Optional model name to filter metrics (e.g., 'sentiment', 'forecast')

        Returns:
            Dictionary containing model metrics
        """
        start_time = time.time()
        operation_id = f"get_model_metrics_{int(start_time)}"
        self.logger.info(f"Retrieving model metrics for {model_name or 'all'} - Operation: {operation_id}")

        try:
            async with aiohttp.ClientSession() as session:
                if not self.redis:
                    raise redis.RedisError("Redis client not initialized")
                
                # Retrieve metrics from Redis
                metrics = {}
                if model_name:
                    key = f"model_metrics:{model_name}"
                    metrics[model_name] = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: json.loads(self.redis.get(key) or "{}")
                    )
                else:
                    # Retrieve all model metrics
                    model_names = ["sentiment", "forecast", "decision"]
                    for name in model_names:
                        key = f"model_metrics:{name}"
                        metrics[name] = await asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            lambda: json.loads(self.redis.get(key) or "{}")
                        )

                result = {
                    "success": True,
                    "metrics": metrics,
                    "model_name": model_name,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}metrics_api_events",
                    {"event": "model_metrics_retrieved", "data": result}
                )

                duration = (time.time() - start_time) * 1000
                self.logger.timing("metrics_api.get_model_metrics_time_ms", duration)
                self.logger.info(f"Model metrics retrieved for {model_name or 'all'}")
                self.logger.counter("metrics_api.metrics_retrieved", 1)
                return result

        except redis.RedisError as e:
            self.logger.error(f"Redis error retrieving model metrics: {e}")
            self.logger.counter("metrics_api.redis_errors", 1)
            return {
                "success": False,
                "error": f"Redis error: {e}",
                "model_name": model_name,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error retrieving model metrics: {e}")
            self.logger.counter("metrics_api.metrics_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def get_trading_metrics(self) -> Dict[str, Any]:
        """
        Retrieve trading performance metrics (e.g., P&L, win rate).

        Returns:
            Dictionary containing trading metrics
        """
        start_time = time.time()
        operation_id = f"get_trading_metrics_{int(start_time)}"
        self.logger.info(f"Retrieving trading metrics - Operation: {operation_id}")

        try:
            async with aiohttp.ClientSession() as session:
                if not self.redis:
                    raise redis.RedisError("Redis client not initialized")
                
                # Retrieve metrics from Redis
                metrics = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: json.loads(self.redis.get("trading_metrics") or "{}")
                )

                result = {
                    "success": True,
                    "metrics": metrics or {
                        "cumulative_pnl": 0.0,
                        "win_rate": 0.0,
                        "trade_count": 0,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}metrics_api_events",
                    {"event": "trading_metrics_retrieved", "data": result}
                )

                duration = (time.time() - start_time) * 1000
                self.logger.timing("metrics_api.get_trading_metrics_time_ms", duration)
                self.logger.info("Trading metrics retrieved")
                self.logger.counter("metrics_api.metrics_retrieved", 1)
                return result

        except redis.RedisError as e:
            self.logger.error(f"Redis error retrieving trading metrics: {e}")
            self.logger.counter("metrics_api.redis_errors", 1)
            return {
                "success": False,
                "error": f"Redis error: {e}",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error retrieving trading metrics: {e}")
            self.logger.counter("metrics_api.metrics_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def shutdown(self):
        """
        Shutdown the MetricsApi and close resources.
        """
        self.logger.info("Shutting down MetricsApi")
        self.running = False
        if self.server:
            self.server.server_close()
        self.executor.shutdown(wait=True)
        self.producer.close()
        self.logger.info("Kafka producer and HTTP server closed")