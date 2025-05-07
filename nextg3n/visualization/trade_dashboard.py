"""
Trade Dashboard for NextG3N Trading System

This module implements the TradeDashboard class, providing a web-based interface for
visualizing and monitoring trading activities, system performance, model metrics, and
LLM-generated summaries. It integrates with ChartGenerator and MetricsApi to support
the TradeFlowOrchestrator.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import requests
import aiohttp
from flask import Flask, render_template, jsonify

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

# Kafka imports
from kafka import KafkaProducer

app = Flask(__name__, template_folder='templates')

class TradeDashboard:
    """
    Class for serving a web-based dashboard to visualize and monitor the NextG3N system.
    Integrates with ChartGenerator and MetricsApi for real-time insights.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TradeDashboard with configuration and dashboard settings.

        Args:
            config: Configuration dictionary with dashboard and Kafka settings
        """
        init_start_time = time.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="trade_dashboard")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.dashboard_config = config.get("visualization", {}).get("trade_dashboard", {})
        self.kafka_config = config.get("kafka", {})
        self.llm_config = config.get("llm", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize dashboard parameters
        self.host = self.dashboard_config.get("host", "localhost")
        self.port = self.dashboard_config.get("port", 3050)  # Default to port 3050
        self.metrics_api_url = self.dashboard_config.get("metrics_api_url", "http://localhost:8000")
        self.chart_dir = self.dashboard_config.get("chart_dir", "./charts")
        self.refresh_interval = self.dashboard_config.get("refresh_interval", 60)  # Seconds
        self.default_symbol = self.dashboard_config.get("default_symbol", "AAPL")
        
        # Initialize Flask app routes
        self._setup_routes()
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialize running state
        self.running = False
        
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("trade_dashboard.initialization_time_ms", init_duration)
        self.logger.info("TradeDashboard initialized")

    def _setup_routes(self):
        """
        Setup Flask routes for the dashboard.
        """
        @app.route('/')
        def index():
            operation_id = f"dashboard_index_{int(time.time())}"
            self.logger.info(f"Rendering dashboard index - Operation: {operation_id}")
            
            try:
                # Fetch metrics from MetricsApi
                system_metrics = requests.get(f"{self.metrics_api_url}/system_metrics").json()
                model_metrics = requests.get(f"{self.metrics_api_url}/model_metrics").json()
                trading_metrics = requests.get(f"{self.metrics_api_url}/trading_metrics").json()
                
                # Fetch chart paths (example for default symbol)
                chart_generator = self.config.get("chart_generator")  # Assume injected via config
                price_data = {
                    "bars": [
                        {"timestamp": datetime.utcnow().isoformat(), "open": 150, "high": 152, "low": 149, "close": 151, "volume": 1000000}
                    ]
                }  # Placeholder
                sentiment_data = {
                    "results": [{"sentiment_score": 0.5, "timestamp": datetime.utcnow().isoformat()}]
                }  # Placeholder
                trade_history = [
                    {"symbol": self.default_symbol, "action": "buy", "quantity": 10, "price": 150, "timestamp": datetime.utcnow().isoformat()}
                ]  # Placeholder
                
                price_chart = chart_generator.generate_price_chart(
                    self.default_symbol, price_data, indicators=["rsi", "macd"]
                ).get("chart_path", "")
                sentiment_chart = chart_generator.generate_sentiment_chart(
                    self.default_symbol, sentiment_data
                ).get("chart_path", "")
                performance_chart = chart_generator.generate_performance_chart(
                    trade_history
                ).get("chart_path", "")
                
                chart_paths = {
                    "price_chart": price_chart,
                    "sentiment_chart": sentiment_chart,
                    "performance_chart": performance_chart
                }
                
                # Generate LLM summary
                summary = asyncio.run(self._generate_market_summary(self.default_symbol, trading_metrics))
                
                result = {
                    "success": True,
                    "system_metrics": system_metrics.get("metrics", {}),
                    "model_metrics": model_metrics.get("metrics", {}),
                    "trading_metrics": trading_metrics.get("metrics", {}),
                    "chart_paths": chart_paths,
                    "summary": summary,
                    "refresh_interval": self.refresh_interval,
                    "symbol": self.default_symbol,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}dashboard_events",
                    {"event": "dashboard_rendered", "data": result}
                )
                
                self.logger.counter("trade_dashboard.pages_rendered", 1)
                self.logger.track_llm_usage(tokens=200, model=self.llm_config.get("model"))  # Estimate tokens
                return render_template('index.html', **result)
            
            except Exception as e:
                self.logger.error(f"Error rendering dashboard index: {e}")
                self.logger.counter("trade_dashboard.render_errors", 1)
                return jsonify({"success": False, "error": str(e)}), 500

        @app.route('/api/update_metrics')
        async def update_metrics():
            operation_id = f"update_metrics_{int(time.time())}"
            self.logger.info(f"Updating dashboard metrics - Operation: {operation_id}")
            
            try:
                async with aiohttp.ClientSession() as session:
                    system_metrics = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: requests.get(f"{self.metrics_api_url}/system_metrics").json()
                    )
                    model_metrics = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: requests.get(f"{self.metrics_api_url}/model_metrics").json()
                    )
                    trading_metrics = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: requests.get(f"{self.metrics_api_url}/trading_metrics").json()
                    )
                
                # Generate LLM summary
                summary = await self._generate_market_summary(self.default_symbol, trading_metrics)
                
                result = {
                    "success": True,
                    "system_metrics": system_metrics.get("metrics", {}),
                    "model_metrics": model_metrics.get("metrics", {}),
                    "trading_metrics": trading_metrics.get("metrics", {}),
                    "summary": summary,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}dashboard_events",
                    {"event": "metrics_updated", "data": result}
                )
                
                self.logger.counter("trade_dashboard.metrics_updated", 1)
                self.logger.track_llm_usage(tokens=200, model=self.llm_config.get("model"))  # Estimate tokens
                return jsonify(result)
            
            except Exception as e:
                self.logger.error(f"Error updating dashboard metrics: {e}")
                self.logger.counter("trade_dashboard.update_errors", 1)
                return jsonify({"success": False, "error": str(e)}), 500

    async def _generate_market_summary(self, symbol: str, trading_metrics: Dict[str, Any]) -> str:
        """
        Generate a market summary using OpenRouter LLM.

        Args:
            symbol: Stock symbol
            trading_metrics: Trading metrics dictionary

        Returns:
            Summary text
        """
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }
        metrics_summary = (
            f"Cumulative P&L: {trading_metrics.get('metrics', {}).get('cumulative_pnl', 0.0):.2f}, "
            f"Win Rate: {trading_metrics.get('metrics', {}).get('win_rate', 0.0):.2f}, "
            f"Trade Count: {trading_metrics.get('metrics', {}).get('trade_count', 0)}"
        )
        payload = {
            "model": self.llm_config.get("model", "openai/gpt-4"),
            "messages": [
                {"role": "user", "content": f"Generate a 100-word summary of the market for {symbol} based on: {metrics_summary}"}
            ],
            "max_tokens": self.llm_config.get("max_tokens", 512),
            "temperature": self.llm_config.get("temperature", 0.7)
        }
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.llm_config.get("retry_attempts", 3)):
                try:
                    async with session.post(
                        self.llm_config.get("base_url", "https://openrouter.ai/api/v1") + "/chat/completions",
                        json=payload,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result["choices"][0]["message"]["content"]
                        else:
                            await asyncio.sleep(self.llm_config.get("retry_delay", 1000) / 1000)
                except Exception as e:
                    if attempt == self.llm_config.get("retry_attempts", 3) - 1:
                        self.logger.error(f"Failed to generate market summary: {e}")
                        return "Failed to generate market summary"
                    await asyncio.sleep(self.llm_config.get("retry_delay", 1000) / 1000)
        return "Failed to generate market summary"

    async def start_dashboard(self):
        """
        Start the Flask web server to serve the dashboard.
        """
        self.logger.info(f"Starting TradeDashboard on {self.host}:{self.port}")
        self.running = True
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            lambda: app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
        )

    async def fetch_metrics(self) -> Dict[str, Any]:
        """
        Fetch metrics data from MetricsApi.

        Returns:
            Dictionary containing aggregated metrics
        """
        start_time = time.time()
        operation_id = f"fetch_metrics_{int(start_time)}"
        self.logger.info(f"Fetching metrics - Operation: {operation_id}")

        try:
            async with aiohttp.ClientSession() as session:
                system_metrics = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: requests.get(f"{self.metrics_api_url}/system_metrics").json()
                )
                model_metrics = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: requests.get(f"{self.metrics_api_url}/model_metrics").json()
                )
                trading_metrics = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: requests.get(f"{self.metrics_api_url}/trading_metrics").json()
                )

                result = {
                    "success": True,
                    "system_metrics": system_metrics.get("metrics", {}),
                    "model_metrics": model_metrics.get("metrics", {}),
                    "trading_metrics": trading_metrics.get("metrics", {}),
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}dashboard_events",
                    {"event": "metrics_fetched", "data": result}
                )

                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_dashboard.fetch_metrics_time_ms", duration)
                self.logger.info("Metrics fetched")
                self.logger.counter("trade_dashboard.metrics_fetched", 1)
                return result

        except Exception as e:
            self.logger.error(f"Error fetching metrics: {e}")
            self.logger.counter("trade_dashboard.fetch_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def fetch_charts(
        self,
        symbol: str,
        price_data: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        trade_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fetch charts from ChartGenerator.

        Args:
            symbol: Stock symbol
            price_data: Price data for chart generation
            sentiment_data: Sentiment data for chart generation
            trade_history: Trade history for chart generation

        Returns:
            Dictionary containing chart file paths
        """
        start_time = time.time()
        operation_id = f"fetch_charts_{int(start_time)}"
        self.logger.info(f"Fetching charts for {symbol} - Operation: {operation_id}")

        try:
            async with aiohttp.ClientSession() as session:
                chart_generator = self.config.get("chart_generator")  # Assume injected via config
                price_chart = await chart_generator.generate_price_chart(symbol, price_data, indicators=["rsi", "macd"])
                sentiment_chart = await chart_generator.generate_sentiment_chart(symbol, sentiment_data)
                performance_chart = await chart_generator.generate_performance_chart(trade_history)

                result = {
                    "success": True,
                    "symbol": symbol,
                    "price_chart": price_chart.get("chart_path", ""),
                    "sentiment_chart": sentiment_chart.get("chart_path", ""),
                    "performance_chart": performance_chart.get("chart_path", ""),
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}dashboard_events",
                    {"event": "charts_fetched", "data": result}
                )

                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_dashboard.fetch_charts_time_ms", duration)
                self.logger.info(f"Charts fetched for {symbol}")
                self.logger.counter("trade_dashboard.charts_fetched", 1)
                return result

        except Exception as e:
            self.logger.error(f"Error fetching charts for {symbol}: {e}")
            self.logger.counter("trade_dashboard.fetch_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def shutdown(self):
        """
        Shutdown the TradeDashboard and close resources.
        """
        self.logger.info("Shutting down TradeDashboard")
        self.running = False
        self.executor.shutdown(wait=True)
        self.producer.close()
        self.logger.info("Kafka producer closed")