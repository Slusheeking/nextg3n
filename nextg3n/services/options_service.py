"""
Options Service for NextG3N Trading System

Implements an MCP server for options flow data from Unusual Whales.
Provides tools for retrieving flow and congressional trades; publishes to Kafka topic nextg3n-options-flow-events.
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

class OptionsService:
    def __init__(self, config: Dict[str, Any]):
        self.logger = MetricsLogger(component_name="options_service")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.config = config
        self.uw_config = config.get("services", {}).get("options", {}).get("unusual_whales", {})
        self.kafka_config = config.get("kafka", {})
        self.redis_config = config.get("storage", {}).get("redis", {})
        self.mcp_client = MCPClient(config)

        self.api_key = self.uw_config.get("api_key")
        self.base_url = self.uw_config.get("base_url", "https://api.unusualwhales.com/v1")
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.redis = Redis(
            host=self.redis_config.get("host", "localhost"),
            port=self.redis_config.get("port", 6379),
            db=self.redis_config.get("db", 0)
        )
        self.logger.info("OptionsService initialized")

    async def get_options_flow(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        operation_id = f"flow_{int(time.time())}"
        self.logger.info(f"Fetching options flow for {symbol} - Operation: {operation_id}")

        try:
            cached = self.redis.get(f"options_flow:{symbol}")
            if cached:
                return json.loads(cached)

            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/options/flow"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                params = {"symbol": symbol, "limit": limit}
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        error = f"API error: {response.status}"
                        self.logger.error(error)
                        return {"success": False, "error": error, "symbol": symbol, "operation_id": operation_id}

                    data = await response.json()
                    flow = [
                        {
                            "symbol": item.get("symbol", symbol),
                            "type": item.get("type", "").lower(),
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
                    self.redis.setex(f"options_flow:{symbol}", 300, json.dumps(result))
                    self.producer.send(
                        f"{self.kafka_config.get('topic_prefix', 'nextg3n-')}options-flow-events",
                        {"event": "flow_fetched", "data": result}
                    )
                    return result

        except Exception as e:
            self.logger.error(f"Error fetching options flow for {symbol}: {e}")
            return {"success": False, "error": str(e), "symbol": symbol, "operation_id": operation_id}

    async def shutdown(self):
        self.producer.close()
        self.redis.close()
        await self.mcp_client.shutdown()
        self.logger.info("OptionsService shutdown")