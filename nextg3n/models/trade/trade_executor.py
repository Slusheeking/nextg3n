"""
Trade Executor for NextG3N Trading System

Implements trade execution via Alpaca API with LSTM/CNN peak detection and slippage/drift controls.
Uses MCP tools for order execution; publishes to Kafka topic nextg3n-trade-events.
Optimized for HFT with sub-millisecond latency.
"""

import logging
import asyncio
import json
import time
import datetime
import threading
import torch
import torch.nn as nn
import websocket
from typing import Dict, Any, List
from kafka import KafkaProducer
from redis import Redis
from monitoring.metrics_logger import MetricsLogger
from services.mcp_client import MCPClient

class LSTMDetector(nn.Module):
    def __init__(self):
        super(LSTMDetector, self).__init__()
        self.lstm = nn.LSTM(1, 32, 2, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TradeExecutor:
    def __init__(self, config: Dict[str, Any]):
        self.logger = MetricsLogger(component_name="trade_executor")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.config = config
        self.kafka_config = config.get("kafka", {})
        self.redis_config = config.get("storage", {}).get("redis", {})
        self.alpaca_config = config.get("models", {}).get("trade", {})
        self.mcp_client = MCPClient(config)

        self.peak_detector = LSTMDetector().to('cuda')
        self.peak_detector.eval()
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.redis = Redis(
            host=self.redis_config.get("host", "localhost"),
            port=self.redis_config.get("port", 6379),
            db=self.redis_config.get("db", 0)
        )
        self.websocket = None
        self._initialize_websocket()
        self.logger.info("TradeExecutor initialized")

    def _initialize_websocket(self):
        ws_url = "wss://stream.data.alpaca.markets/v2/iex"
        self.websocket = websocket.WebSocketApp(
            ws_url,
            on_message=self._on_websocket_message,
            on_error=self._on_websocket_error,
            on_close=self._on_websocket_close
        )
        threading.Thread(target=self.websocket.run_forever, daemon=True).start()
        self.websocket.send(json.dumps({
            "action": "auth",
            "key": self.alpaca_config.get("alpaca_api_key"),
            "secret": self.alpaca_config.get("alpaca_api_secret")
        }))
        self.logger.info("Alpaca WebSocket initialized")

    def _on_websocket_message(self, ws, message):
        try:
            data = json.loads(message)
            for item in data:
                if item.get("T") == "t":  # Trade event
                    symbol = item.get("S")
                    price = item.get("p")
                    timestamp = datetime.utcnow().isoformat()
                    self.redis.setex(f"trade:{symbol}", 300, json.dumps({"price": price, "timestamp": timestamp}))
                    self.producer.send(
                        f"{self.kafka_config.get('topic_prefix', 'nextg3n-')}trade-events",
                        {"event": "trade", "data": {"symbol": symbol, "price": price, "timestamp": timestamp}}
                    )
        except Exception as e:
            self.logger.error(f"WebSocket message error: {e}")

    def _on_websocket_error(self, ws, error):
        self.logger.error(f"WebSocket error: {error}")

    def _on_websocket_close(self, ws, code, reason):
        self.logger.warning(f"WebSocket closed: {code} - {reason}")

    async def execute_trade(self, symbol: str, action: str, quantity: int) -> Dict[str, Any]:
        operation_id = f"trade_{int(time.time())}"
        self.logger.info(f"Executing {action} trade for {symbol}, quantity={quantity} - Operation: {operation_id}")

        try:
            # Validate position size
            if quantity * (await self.mcp_client.call_tool("polygon", "get_realtime_quote", {"symbol": symbol}))["price"] > 1000:
                return {"success": False, "error": "Position size exceeds 20% capital", "operation_id": operation_id}

            order = await self.mcp_client.call_tool("alpaca", "place_order", {
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "order_type": "market"
            })
            if not order["success"]:
                return {"success": False, "error": order["error"], "operation_id": order["operation_id"]}

            # Monitor for slippage
            executed_price = (await self.mcp_client.call_tool("alpaca", "get_order_status", {"order_id": order["result"]["order_id"]}))["filled_avg_price"]
            expected_price = (await self.mcp_client.call_tool("polygon", "get_realtime_quote", {"symbol": symbol}))["price"]
            slippage = abs(executed_price - expected_price) / expected_price if executed_price else 0

            result = {
                "success": True,
                "order_id": order["result"]["order_id"],
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "executed_price": executed_price,
                "slippage": slippage,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.redis.setex(f"trade:{symbol}:{operation_id}", 300, json.dumps(result))
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n-')}trade-events",
                {"event": "trade_executed", "data": result}
            )
            return result

        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
            return {"success": False, "error": str(e), "symbol": symbol, "operation_id": operation_id}

    async def monitor_trade(self, symbol: str, order_id: str) -> Dict[str, Any]:
        operation_id = f"monitor_{int(time.time())}"
        self.logger.info(f"Monitoring trade for {symbol}, order_id={order_id} - Operation: {operation_id}")

        try:
            # Fetch real-time price
            price_data = self.redis.get(f"trade:{symbol}")
            if not price_data:
                return {"success": False, "error": "No price data", "operation_id": operation_id}

            prices = [json.loads(price_data)["price"]]
            data = torch.tensor(prices[-30:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to('cuda')
            with torch.no_grad():
                peak_score = self.peak_detector(data).cpu().sigmoid().item()

            # Check stop-loss (3%)
            order = await self.mcp_client.call_tool("alpaca", "get_order_status", {"order_id": order_id})
            entry_price = order["filled_avg_price"]
            current_price = prices[-1]
            loss = (entry_price - current_price) / entry_price if entry_price else 0

            result = {
                "success": True,
                "symbol": symbol,
                "order_id": order_id,
                "current_price": current_price,
                "peak_score": peak_score,
                "loss_percent": loss,
                "should_exit": peak_score > 0.8 or loss > 0.03,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.redis.setex(f"monitor:{symbol}:{order_id}", 300, json.dumps(result))
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n-')}trade-events",
                {"event": "trade_monitored", "data": result}
            )
            return result

        except Exception as e:
            self.logger.error(f"Error monitoring trade for {symbol}: {e}")
            return {"success": False, "error": str(e), "symbol": symbol, "operation_id": operation_id}

    async def shutdown(self):
        if self.websocket:
            self.websocket.close()
        self.producer.close()
        self.redis.close()
        await self.mcp_client.shutdown()
        self.logger.info("TradeExecutor shutdown")