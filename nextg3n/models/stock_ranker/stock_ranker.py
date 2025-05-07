"""
Stock Ranker for NextG3N Trading System

Implements stock screening using XGBoost and CNN for volatility, liquidity, and fundamentals.
Uses MCP tools for Polygon and Yahoo Finance data; publishes to Kafka topic nextg3n-context-events.
"""

import logging
import asyncio
import json
import torch
import time
import datetime
import torch.nn as nn
import xgboost as xgb
from typing import Dict, Any, List
from kafka import KafkaProducer
from redis import Redis
from monitoring.metrics_logger import MetricsLogger
from services.mcp_client import MCPClient

class CNNRanker(nn.Module):
    def __init__(self):
        super(CNNRanker, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 13 * 13, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 13 * 13)
        return self.fc(x)

class StockRanker:
    def __init__(self, config: Dict[str, Any]):
        self.logger = MetricsLogger(component_name="stock_ranker")
        self.logger.setLevel(logging.WARNING)  # Reduce logging level in production
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.config = config
        self.kafka_config = config.get("kafka", {})
        self.redis_config = config.get("storage", {}).get("redis", {})
        self.mcp_client = MCPClient(config)

        self.xgb_model = xgb.XGBRanker(tree_method='gpu_hist')
        self.cnn_model = CNNRanker().to('cuda')
        self.cnn_model.eval()
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.redis = Redis(
            host=self.redis_config.get("host", "localhost"),
            port=self.redis_config.get("port", 6379),
            db=self.redis_config.get("db", 0)
        )
        self.logger.info("StockRanker initialized")

    async def rank_stocks(self, symbols: List[str]) -> Dict[str, Any]:
        operation_id = f"rank_{int(time.time())}"
        self.logger.info(f"Ranking stocks: {symbols} - Operation: {operation_id}")

        try:
            features = []
            for symbol in symbols:
                quote = await self.mcp_client.call_tool("polygon", "get_realtime_quote", {"symbol": symbol})
                bars = await self.mcp_client.call_tool("polygon", "get_historical_bars", {"symbol": symbol, "timeframe": "1d", "limit": 30})
                fundamentals = await self.mcp_client.call_tool("yahoo_finance", "get_fundamentals", {"symbol": symbol})

                if not (quote["success"] and bars["success"] and fundamentals["success"]):
                    continue

                volatility = (max(b["high"] for b in bars["bars"]) - min(b["low"] for b in bars["bars"])) / bars["bars"][-1]["close"]
                liquidity = sum(b["volume"] for b in bars["bars"]) / len(bars["bars"])
                pe_ratio = fundamentals["fundamentals"].get("pe_ratio", 0)
                market_cap = fundamentals["fundamentals"].get("market_cap", 0)

                features.append({
                    "symbol": symbol,
                    "volatility": volatility,
                    "liquidity": liquidity,
                    "pe_ratio": pe_ratio,
                    "market_cap": market_cap
                })

            # XGBoost ranking
            X = [[f["volatility"], f["liquidity"], f["pe_ratio"], f["market_cap"]] for f in features]
            if not X:
                return {"success": False, "error": "No valid features", "operation_id": operation_id}

            dmatrix = xgb.DMatrix(X)
            scores = self.xgb_model.predict(dmatrix)

            # CNN ranking
            cnn_scores = []
            for f in features:
                bars = await self.mcp_client.call_tool("polygon", "get_historical_bars", {"symbol": f["symbol"], "timeframe": "1m", "limit": 30})
                if bars["success"]:
                    data = torch.tensor([[b["close"] for b in bars["bars"]]], dtype=torch.float32).unsqueeze(0).to('cuda')
                    with torch.no_grad():
                        score = self.cnn_model(data).cpu().item()
                    cnn_scores.append(score)
                else:
                    cnn_scores.append(0)

            ranked_stocks = [
                {
                    "symbol": f["symbol"],
                    "score": 0.6 * s + 0.4 * c,  # Weighted combination
                    "volatility": f["volatility"],
                    "liquidity": f["liquidity"],
                    "market_cap": f["market_cap"]
                }
                for f, s, c in zip(features, scores, cnn_scores)
                if f["volatility"] > 0.03 and f["liquidity"] > 2_000_000 and f["market_cap"] > 500_000_000
            ]
            ranked_stocks.sort(key=lambda x: x["score"], reverse=True)
            ranked_stocks = ranked_stocks[:5]  # Top 5 for $5,000 capital

            result = {
                "success": True,
                "stocks": ranked_stocks,
                "count": len(ranked_stocks),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.redis.setex(f"ranking:{operation_id}", 300, json.dumps(result))
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n-')}context-events",
                {"event": "stocks_ranked", "data": result}
            )
            return result

        except Exception as e:
            self.logger.error(f"Error ranking stocks: {e}")
            return {"success": False, "error": str(e), "operation_id": operation_id}

    async def shutdown(self):
        self.producer.close()
        self.redis.close()
        await self.mcp_client.shutdown()
        self.logger.info("StockRanker shutdown")