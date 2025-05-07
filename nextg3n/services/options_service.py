"""
Options Service for NextG3N Trading System

This module implements the OptionsService, fetching and analyzing options flow data from the
Unusual Whales API to gauge market sentiment. It provides tools for retrieving options flow
and performing preliminary sentiment analysis, integrated with the StockPickerAgent and
PredictorAgent in TradeFlowOrchestrator.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
from kafka import KafkaProducer

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

class OptionsService:
    """
    Service for fetching and analyzing options flow data in the NextG3N system.
    Provides tools for retrieving options flow and preliminary sentiment analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OptionsService with configuration and API settings.

        Args:
            config: Configuration dictionary with Unusual Whales and Kafka settings
        """
        init_start_time = time.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="options_service")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.uw_config = config.get("services", {}).get("options", {}).get("unusual_whales", {})
        self.kafka_config = config.get("kafka", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize API settings
        self.api_key = self.uw_config.get("api_key") or os.environ.get("UNUSUAL_WHALES_API_KEY")
        self.base_url = self.uw_config.get("base_url", "https://api.unusualwhales.com/v1")
        self.rate_limit_delay = self.uw_config.get("rate_limit_delay", 1.0)  # Seconds between requests
        
        if not self.api_key:
            self.logger.error("Unusual Whales API key missing")
            raise ValueError("Unusual Whales API key not provided")

        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("options_service.initialization_time_ms", init_duration)
        self.logger.info("OptionsService initialized")

    async def get_options_flow(
        self,
        symbol: str,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Fetch recent options flow data for a stock from Unusual Whales.

        Args:
            symbol: Stock symbol
            limit: Maximum number of flow entries to fetch

        Returns:
            Dictionary containing options flow data
        """
        start_time = time.time()
        operation_id = f"flow_{int(start_time)}"
        self.logger.info(f"Fetching options flow for {symbol} - Operation: {operation_id}")

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/options/flow"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                params = {"symbol": symbol, "limit": limit}
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        error = f"API error: {response.status} - {await response.text()}"
                        self.logger.error(error)
                        return {
                            "success": False,
                            "error": error,
                            "symbol": symbol,
                            "operation_id": operation_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    
                    data = await response.json()
                
                # Process flow data (assumed format: list of trades)
                flow = [
                    {
                        "symbol": item.get("symbol", symbol),
                        "type": item.get("type", "").lower(),  # call/put
                        "premium": float(item.get("premium", 0)),
                        "strike_price": float(item.get("strike_price", 0)),
                        "expiration_date": item.get("expiration_date", ""),
                        "volume": int(item.get("volume", 0)),
                        "open_interest": int(item.get("open_interest", 0)),
                        "timestamp": item.get("timestamp", datetime.utcnow().isoformat())
                    }
                    for item in data[:limit]
                ]

                result = {
                    "success": True,
                    "symbol": symbol,
                    "flow": flow,
                    "count": len(flow),
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}options_events",
                    {"event": "flow_fetched", "data": result}
                )

                duration = (time.time() - start_time) * 1000
                self.logger.timing("options_service.get_options_flow_time_ms", duration)
                self.logger.info(f"Fetched {result['count']} options flow entries for {symbol}")
                self.logger.counter("options_service.flow_fetched", result['count'])
                
                # Respect rate limits
                await asyncio.sleep(self.rate_limit_delay)
                
                return result

        except Exception as e:
            self.logger.error(f"Error fetching options flow for {symbol}: {e}")
            self.logger.counter("options_service.flow_errors", 1)
            result = {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}options_events",
                {"event": "flow_fetch_failed", "data": result}
            )
            return result

    async def get_options_sentiment(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Analyze options flow for bullish/bearish sentiment.

        Args:
            symbol: Stock symbol

        Returns:
            Sentiment analysis dictionary
        """
        start_time = time.time()
        operation_id = f"sentiment_{int(start_time)}"
        self.logger.info(f"Analyzing options sentiment for {symbol} - Operation: {operation_id}")

        try:
            # Fetch options flow data
            flow_result = await self.get_options_flow(symbol, limit=50)
            
            if not flow_result["success"] or not flow_result["flow"]:
                self.logger.warning(f"No options flow data for {symbol}")
                return {
                    "success": False,
                    "error": "No options flow data available",
                    "symbol": symbol,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

            flow = flow_result["flow"]
            
            # Calculate sentiment based on call/put ratios and premiums
            call_volume = sum(item["volume"] for item in flow if item["type"] == "call")
            put_volume = sum(item["volume"] for item in flow if item["type"] == "put")
            call_premium = sum(item["premium"] for item in flow if item["type"] == "call")
            put_premium = sum(item["premium"] for item in flow if item["type"] == "put")
            
            total_volume = call_volume + put_volume
            total_premium = call_premium + put_premium
            
            call_volume_ratio = call_volume / total_volume if total_volume > 0 else 0
            call_premium_ratio = call_premium / total_premium if total_premium > 0 else 0
            
            # Weighted sentiment score (0-1, higher is more bullish)
            sentiment_score = (call_volume_ratio * 0.5 + call_premium_ratio * 0.5)
            sentiment_label = (
                "bullish" if sentiment_score > 0.6 else
                "bearish" if sentiment_score < 0.4 else
                "neutral"
            )

            result = {
                "success": True,
                "symbol": symbol,
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "call_volume": call_volume,
                "put_volume": put_volume,
                "call_premium": call_premium,
                "put_premium": put_premium,
                "flow_count": len(flow),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}options_events",
                {"event": "sentiment_analyzed", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("options_service.get_options_sentiment_time_ms", duration)
            self.logger.info(f"Options sentiment analyzed for {symbol}: Score={sentiment_score:.2f}, Label={sentiment_label}")
            self.logger.counter("options_service.sentiment_analyses", 1)
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing options sentiment for {symbol}: {e}")
            self.logger.counter("options_service.sentiment_errors", 1)
            result = {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}options_events",
                {"event": "sentiment_analysis_failed", "data": result}
            )
            return result

    def shutdown(self):
        """
        Shutdown the service and close resources.
        """
        self.logger.info("Shutting down OptionsService")
        self.executor.shutdown(wait=True)
        self.producer.close()
        self.logger.info("Kafka producer closed")