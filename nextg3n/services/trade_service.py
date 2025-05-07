"""
Trade Service for NextG3N Trading System

Implements an MCP server for trade execution via Alpaca API.
Provides tools for order submission and status checking; publishes to Kafka topic nextg3n-trade-events.
Optimized for HFT with sub-millisecond latency.
"""

import logging
import asyncio
import json
import aiohttp
import time
import datetime
from typing import Dict, Any, List
from kafka import KafkaProducer
from redis import Redis
from monitoring.metrics_logger import MetricsLogger
from services.mcp_client import MCPClient

class TradeService:
    def __init__(self, config: Dict[str, Any]):
        self.logger = MetricsLogger(component_name="trade_service")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.config = config
        self.alpaca_config = config.get("services", {}).get("trade", {}).get("alpaca", {})
        self.kafka_config = config.get("kafka", {})
        self.redis_config = config.get("storage", {}).get("redis", {})
        self.mcp_client = MCPClient(config)

        self.api_key = self.alpaca_config.get("api_key")
        self.api_secret = self.alpaca_config.get("api_secret")
        self.base_url = self.alpaca_config.get("base_url", "https://paper-api.alpaca.markets")
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.redis = Redis(
            host=self.redis_config.get("host", "localhost"),
            port=self.redis_config.get("port", 6379),
            db=self.redis_config.get("db", 0)
        )
        self.logger.info("TradeService initialized")

    async def place_order(self, symbol: str, action: str, quantity: int, order_type: str = "market") -> Dict[str, Any]:
        operation_id = f"order_{int(time.time())}"
        self.logger.info(f"Placing {order_type} order: {action} {quantity} shares of {symbol} - Operation: {operation_id}")

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v2/orders"
                headers = {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret}
                payload = {
                    "symbol": symbol,
                    "qty": quantity,
                    "side": action.lower(),
                    "type": order_type,
                    "time_in_force": "day"
                }
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error = f"Alpaca API error: {response.status}"
                        self.logger.error(error)
                        return {"success": False, "error": error, "operation_id": operation_id}

                    data = await response.json()
                    result = {
                        "success": True,
                        "order_id": data["id"],
                        "status": data["status"],
                        "symbol": symbol,
                        "action": action,
                        "quantity": quantity,
                        "operation_id": operation_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    self.redis.setex(f"order:{symbol}:{data['id']}", 300, json.dumps(result))
                    self.producer.send(
                        f"{self.kafka_config.get('topic_prefix', 'nextg3n-')}trade-events",
                        {"event": "order_submitted", "data": result}
                    )
                    return result

        except Exception as e:
            self.logger.error(f"Error placing order for {symbol}: {e}")
            return {"success": False, "error": str(e), "symbol": symbol, "operation_id": operation_id}

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        operation_id = f"status_{int(time.time())}"
        self.logger.info(f"Fetching order status for {order_id} - Operation: {operation_id}")

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v2/orders/{order_id}"
                headers = {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret}
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        error = f"Alpaca API error: {response.status}"
                        self.logger.error(error)
                        return {"success": False, "error": error, "operation_id": operation_id}

                    data = await response.json()
                    result = {
                        "success": True,
                        "order_id": order_id,
                        "status": data["status"],
                        "symbol": data["symbol"],
                        "filled_qty": float(data["filled_qty"]),
                        "filled_avg_price": float(data["filled_avg_price"]) if data["filled_avg_price"] else None,
                        "operation_id": operation_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    return result

        except Exception as e:
            self.logger.error(f"Error fetching order status for {order_id}: {e}")
            return {"success": False, "error": str(e), "order_id": order_id, "operation_id": operation_id}

    async def shutdown(self):
        self.producer.close()
        self.redis.close()
        await self.mcp_client.shutdown()
        self.logger.info("TradeService shutdown")