"""
News Service for NextG3N Trading System

Implements an MCP server for news data, integrating Polygon.io and Yahoo Finance.
Uses FinBERT for article filtering; publishes to Kafka topic nextg3n-news-events.
"""

import logging
import asyncio
import json
import aiohttp
import time
import datetime
import yfinance as yf
from typing import Dict, Any, List
from transformers import pipeline
from kafka import KafkaProducer
from redis import Redis
from monitoring.metrics_logger import MetricsLogger
from services.mcp_client import MCPClient

class NewsService:
    def __init__(self, config: Dict[str, Any]):
        self.logger = MetricsLogger(component_name="news_service")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.config = config
        self.polygon_config = config.get("services", {}).get("news", {}).get("polygon", {})
        self.yahoo_config = config.get("services", {}).get("news", {}).get("yahoo", {})
        self.kafka_config = config.get("kafka", {})
        self.redis_config = config.get("storage", {}).get("redis", {})
        self.mcp_client = MCPClient(config)

        self.sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        self.llm_model_name = self.config.get("services", {}).get("news", {}).get("llm", {}).get("model_name", "google/flan-t5-base")
        self.llm_pipeline = pipeline("text2text-generation", model=self.llm_model_name)
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.redis = Redis(
            host=self.redis_config.get("host", "localhost"),
            port=self.redis_config.get("port", 6379),
            db=self.redis_config.get("db", 0)
        )
        self.logger.info("NewsService initialized")

    async def get_news(self, symbols: List[str], limit: int = 30) -> Dict[str, Any]:
        operation_id = f"news_{int(time.time())}"
        self.logger.info(f"Fetching news for {symbols} - Operation: {operation_id}")

        try:
            cached = self.redis.get(f"news:{','.join(symbols)}")
            if cached:
                return json.loads(cached)

            articles = []
            async with aiohttp.ClientSession() as session:
                for symbol in symbols:
                    url = f"{self.polygon_config.get('base_url', 'https://api.polygon.io/v2')}/reference/news"
                    headers = {"Authorization": f"Bearer {self.polygon_config.get('api_key')}"}
                    params = {"ticker": symbol, "limit": limit // len(symbols)}
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            for item in data.get("results", []):
                                article = {
                                    "symbol": symbol,
                                    "title": item.get("title", ""),
                                    "summary": item.get("description", ""),
                                    "published_utc": item.get("published_utc", ""),
                                    "source": "Polygon",
                                    "url": item.get("article_url", "")
                                }
                                result = self.sentiment_analyzer(article["title"] + " " + article["summary"])[0]
                                if result["score"] > 0.7:  # Filter relevant articles
                                    llm_analysis = await self.analyze_news_with_llm(article["title"], article["summary"])
                                    article["llm_analysis"] = llm_analysis
                                    articles.append(article)
                                    await asyncio.sleep(self.polygon_config.get("rate_limit_delay", 12.0))

            if self.yahoo_config.get("enabled", True):
                for symbol in symbols:
                    ticker = yf.Ticker(symbol)
                    news = ticker.news[:limit // len(symbols)]
                    for item in news:
                        article = {
                            "symbol": symbol,
                            "title": item.get("title", ""),
                            "summary": item.get("summary", ""),
                            "published_utc": datetime.utcfromtimestamp(item.get("provider_publish_time", 0)).isoformat(),
                            "source": "Yahoo",
                            "url": item.get("link", "")
                        }
                        result = self.sentiment_analyzer(article["title"] + " " + article["summary"])[0]
                        if result["score"] > 0.7:
                            llm_analysis = await self.analyze_news_with_llm(article["title"], article["summary"])
                            article["llm_analysis"] = llm_analysis
                            articles.append(article)
                            await asyncio.sleep(1.0)

            result = {
                "success": True,
                "symbols": symbols,
                "news": articles[:limit],
                "count": len(articles),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.redis.setex(f"news:{','.join(symbols)}", 300, json.dumps(result))
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n-')}news-events",
                {"event": "news_fetched", "data": result}
            )
            return result

        except Exception as e:
            self.logger.error(f"Error fetching news for {symbols}: {e}")
            return {"success": False, "error": str(e), "symbols": symbols, "operation_id": operation_id}

    async def analyze_news_with_llm(self, title: str, summary: str) -> str:
        """
        Analyzes news content using a large language model.
        """
        try:
            prompt = f"Analyze the following news headline and summary: Title: {title}, Summary: {summary}. Provide a detailed analysis of the potential impact on the stock market."
            llm_response = self.llm_pipeline(prompt, max_length=150, num_return_sequences=1)[0]["generated_text"]
            return llm_response
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return "LLM analysis failed."

    async def shutdown(self):
        self.producer.close()
        self.redis.close()
        await self.mcp_client.shutdown()
        self.logger.info("NewsService shutdown")
