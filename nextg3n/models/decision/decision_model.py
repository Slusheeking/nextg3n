"""
Decision Model for NextG3N Trading System

Implements trading decisions using directly integrated AI/ML models for low latency.
Fetches data directly, runs models, applies deterministic logic, and publishes to Kafka.
"""

import logging
import json
import time
import datetime
import torch
from typing import Dict, Any, List, Optional

# Import necessary models and services directly
from models.forecast.forecast_model import ForecastModel
from services.market_data_service import MarketDataService

from kafka import KafkaProducer
from redis import Redis
from monitoring.metrics_logger import MetricsLogger

class DecisionModel:
    def __init__(self, config: Dict[str, Any]):
        self.logger = MetricsLogger(component_name="decision_model")
        self.logger.setLevel(logging.WARNING)  # Reduce logging level in production
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.config = config
        self.kafka_config = config.get("kafka", {})
        self.redis_config = config.get("storage", {}).get("redis", {})
        self.decision_logic_config = config.get("decision_logic", {})

        # Initialize directly integrated components
        try:
            self.forecast_model = ForecastModel(config.get("forecast_model", {}))
            self.logger.info("ForecastModel initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize ForecastModel: {e}")
            self.forecast_model = None

        try:
            self.market_data_service = MarketDataService(config.get("market_data_service", {}))
            self.logger.info("MarketDataService initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize MarketDataService: {e}")
            self.market_data_service = None

        # Initialize Kafka Producer and Redis
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            self.logger.info("KafkaProducer initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize KafkaProducer: {e}")
            self.producer = None

        try:
            self.redis = Redis(
                host=self.redis_config.get("host", "localhost"),
                port=self.redis_config.get("port", 6379),
                db=self.redis_config.get("db", 0)
            )
            self.logger.info("Redis client initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis client: {e}")
            self.redis = None

        self.logger.info("DecisionModel initialized with directly integrated components.")

    async def make_decision(self, symbol: str) -> Dict[str, Any]:
        operation_id = f"decision_{int(time.time())}"
        self.logger.info(f"Making decision for {symbol} - Operation: {operation_id}")

        # Check for essential components
        if not all([self.forecast_model, self.market_data_service, self.producer, self.redis]):
            self.logger.error("DecisionModel essential components not fully initialized. Cannot make decision.")
            return {"success": False, "error": "Decision system not ready", "symbol": symbol, "operation_id": operation_id}

        try:
            # 1. Fetch Data Directly (Historical Bars Only)
            self.logger.info(f"Fetching historical market data for {symbol}")
            historical_bars = await self.market_data_service.get_historical_bars(
                symbol=symbol,
                timeframe=self.config.get("data", {}).get("timeframe", "1m"),
                limit=self.config.get("data", {}).get("limit", 60)
            )

            if not historical_bars:
                 self.logger.warning(f"No historical data fetched for {symbol}. Cannot proceed with decision.")
                 return {"success": False, "error": "No historical data available", "symbol": symbol, "operation_id": operation_id}

            # 2. Run Integrated Forecast Model
            self.logger.info(f"Running integrated forecast model for {symbol}")

            # Prepare data for ForecastModel (assuming it expects a torch.Tensor)
            try:
                # Assuming the LSTM model expects a tensor of shape (batch_size, sequence_length, input_size)
                # where input_size is 5 (OHLCV)
                forecast_input_data = torch.tensor(
                    [[b["open"], b["high"], b["low"], b["close"], b["volume"]] for b in historical_bars],
                    dtype=torch.float32
                ).unsqueeze(0) # Add batch dimension
                self.logger.debug(f"Prepared forecast input data with shape: {forecast_input_data.shape}")
            except Exception as data_prep_e:
                 self.logger.error(f"Error preparing forecast input data: {data_prep_e}")
                 return {"success": False, "error": "Failed to prepare forecast data", "symbol": symbol, "operation_id": operation_id}

            forecast_result = await self.forecast_model.predict(forecast_input_data)
            self.logger.info(f"Forecast result for {symbol}: {forecast_result}")

            # 3. Apply Deterministic Decision Logic (Based on Forecast Only)
            self.logger.info(f"Applying deterministic decision logic for {symbol} based on forecast.")

            final_action = "hold"

            forecast_direction = forecast_result.get("direction", "neutral")

            # Simple Strategy: Buy if strong upward forecast, Sell if strong downward forecast
            # Thresholds should be configurable
            # Cache decision logic config values
            up_threshold = self.decision_logic_config.get("up_threshold")
            down_threshold = self.decision_logic_config.get("down_threshold")

            # Validate required configuration parameters for decision logic
            if up_threshold is None or down_threshold is None:
                 error_msg = "DecisionModel requires 'up_threshold' and 'down_threshold' in 'decision_logic' configuration."
                 self.logger.error(error_msg)
                 return {"success": False, "error": error_msg, "symbol": symbol, "operation_id": operation_id}


            # Decision logic based on forecast prediction value relative to thresholds
            forecast_prediction_value = forecast_result.get("prediction")

            if forecast_prediction_value is None:
                 self.logger.warning(f"Forecast prediction value is missing for {symbol}. Cannot make decision.")
                 return {"success": False, "error": "Forecast prediction value missing", "symbol": symbol, "operation_id": operation_id}


            if forecast_prediction_value > up_threshold:
                 final_action = "buy"
            elif forecast_prediction_value < down_threshold:
                 final_action = "sell"
            else:
                 final_action = "hold"


            self.logger.info(f"Final decision for {symbol}: Action={final_action}")

            # 4. Prepare and Publish Result
            result = {
                "success": True,
                "symbol": symbol,
                "action": final_action,
                "forecast_result": forecast_result,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Redis Cache (with expiry)
            try:
                redis_expiration_time = self.config.get("decision_model", {}).get("redis_expiration_time", 300)
                self.redis.setex(f"decision:{symbol}", redis_expiration_time, json.dumps(result))
                self.logger.info(f"Decision for {symbol} cached in Redis.")
            except Exception as redis_e:
                self.logger.error(f"Failed to cache decision in Redis for {symbol}: {redis_e}")

            # Publish to Kafka Topic
            try:
                topic = f"{self.kafka_config.get('topic_prefix', 'nextg3n-')}decision-events"
                self.producer.send(
                    topic,
                    {"event": "decision_made", "data": result}
                )
                self.producer.flush()
                self.logger.info(f"Decision for {symbol} published to Kafka topic {topic}.")
            except Exception as kafka_e:
                self.logger.error(f"Failed to publish decision to Kafka for {symbol}: {kafka_e}")

            return result

        except Exception as e:
            self.logger.error(f"Error making decision for {symbol}: {e}")
            return {"success": False, "error": str(e), "symbol": symbol, "operation_id": operation_id}

    async def shutdown(self):
        self.logger.info("DecisionModel shutting down...")
        if self.producer:
            self.producer.close()
            self.logger.info("KafkaProducer closed.")
        if self.redis:
            self.redis.close()
            self.logger.info("Redis client closed.")
        if hasattr(self.market_data_service, 'shutdown'):
            await self.market_data_service.shutdown()
            self.logger.info("MarketDataService shutdown.")
        if hasattr(self.forecast_model, 'shutdown'):
            await self.forecast_model.shutdown()
            self.logger.info("ForecastModel shutdown.")

        self.logger.info("DecisionModel shutdown complete.")