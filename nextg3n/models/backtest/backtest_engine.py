"""
Backtest Engine for NextG3N Trading System

Implements strategy backtesting using Backtrader and PPO optimization.
Uses Polygon data via MCP tools; publishes to Kafka topic nextg3n-backtest-events.
"""

import logging
import asyncio
import json
import time
import datetime
import pandas as pd
import backtrader as bt
from typing import Dict, Any, List
from kafka import KafkaProducer
from redis import Redis
from monitoring.metrics_logger import MetricsLogger
from services.mcp_client import MCPClient


class PPOStrategy(bt.Strategy):
    params = (('size', 100), ('stop_loss', 0.03),
              ('lookback', 20), ('confidence_threshold', 0.7))

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None

    def next(self):
        if self.order:
            return
        if len(self.dataclose) < self.params.lookback:
            return
        forecast = self.dataclose[-1] / \
            self.dataclose[-self.params.lookback] - 1
        if forecast > self.params.confidence_threshold:
            self.order = self.buy(size=self.params.size)
        elif forecast < -self.params.confidence_threshold:
            self.order = self.sell(size=self.params.size)


class BacktestEngine:
    def __init__(self, config: Dict[str, Any]):
        self.logger = MetricsLogger(component_name="backtest_engine")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.config = config
        self.kafka_config = config.get("kafka", {})
        self.redis_config = config.get("storage", {}).get("redis", {})
        self.mcp_client = MCPClient(config)

        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get(
                "bootstrap_servers",
                "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        self.redis = Redis(
            host=self.redis_config.get("host", "localhost"),
            port=self.redis_config.get("port", 6379),
            db=self.redis_config.get("db", 0)
        )
        self.logger.info("BacktestEngine initialized")

    async def run_backtest(
            self, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        operation_id = f"backtest_{int(time.time())}"
        self.logger.info(
            f"Running backtest for {symbol}, timeframe={timeframe} - Operation: {operation_id}")

        try:
            limit = self.config.get("backtest", {}).get("historical_data_limit", 1000)
            bars = await self.mcp_client.call_tool("polygon", "get_historical_bars", {"symbol": symbol, "timeframe": timeframe, "limit": limit})
            if not bars["success"]:
                return {
                    "success": False,
                    "error": bars.get("error", "No bars available"),
                    "symbol": symbol,
                    "operation_id": operation_id}

            cerebro = bt.Cerebro()
            data = bt.feeds.PandasData(dataname=pd.DataFrame(bars["bars"]))
            cerebro.adddata(data)
            cerebro.addstrategy(PPOStrategy)
            initial_cash = self.config.get("backtest", {}).get("initial_cash", 10000)
            cerebro.broker.setcash(initial_cash)
            cerebro.run()

            result = {
                "success": True,
                "symbol": symbol,
                "returns": cerebro.broker.getvalue() / initial_cash - 1,
                "win_rate": None,
                "sharpe_ratio": None,
                "operation_id": operation_id,
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
            self.redis.setex(
                f"backtest:{symbol}:{operation_id}",
                86400,
                json.dumps(result))
            topic = f"{self.kafka_config.get('topic_prefix', 'nextg3n-')}backtest-events"
            try:
                self.producer.send(
                    topic,
                    {"event": "backtest_completed", "data": result}
                )
                self.logger.info(f"Published to Kafka topic: {topic}")
                return result
            except Exception as e:
                self.logger.error(f"Failed to send to Kafka: {e}")
                return {
                    "success": False,
                    "error": f"Failed to send to Kafka: {e}",
                    "symbol": symbol,
                    "operation_id": operation_id}

        except KeyError as e:
            self.logger.error(f"Missing configuration key: {e}")
            return {
                "success": False,
                "error": f"Missing configuration key: {e}",
                "symbol": symbol,
            }
        except Exception as e:
            self.logger.exception(f"Error running backtest for {symbol}: {e}")
            return {
                "success": False,
                "error": "Backtest failed",
                "symbol": symbol,
                "operation_id": operation_id}

    async def shutdown(self):
        self.producer.close()
        self.redis.close()
        await self.mcp_client.shutdown()
        self.logger.info("BacktestEngine shutdown")
