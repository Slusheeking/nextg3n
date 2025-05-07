"""
Context Retriever for NextG3N Trading System

Implements RAG-based context retrieval using Sentence Transformers and FinBERT.
Uses MCP tools for news and Reddit data; publishes to Kafka topic nextg3n-context-events.
Optimized for day trading with GPU acceleration.
"""

import logging
import asyncio
import json
import time
import datetime
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from kafka import KafkaProducer
from redis import Redis
from chromadb import Client
from monitoring.metrics_logger import MetricsLogger
from services.mcp_client import MCPClient

class ContextRetriever:
    def __init__(self, config: Dict[str, Any]):
        self.logger = MetricsLogger(component_name="context_retriever")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.config = config
        self.kafka_config = config.get("kafka", {})
        self.redis_config = config.get("storage", {}).get("redis", {})
        self.vector_db_config = config.get("storage", {}).get("vector_db", {})
        self.mcp_client = MCPClient(config)

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')
        self.summarizer = pipeline("summarization", model="ProsusAI/finbert")
        self.vector_db = Client().get_or_create_collection(
            name=self.vector_db_config.get("collection_name", "nextg3n_vectors"),
            embedding_function=self.embedder.encode
        )
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.redis = Redis(
            host=self.redis_config.get("host", "localhost"),
            port=self.redis_config.get("port", 6379),
            db=self.redis_config.get("db", 0)
        )
        self.logger.info("ContextRetriever initialized")

    async def retrieve_context(self, symbol: str) -> Dict[str, Any]:
        operation_id = f"context_{int(time.time())}"
        self.logger.info(f"Retrieving context for {symbol} - Operation: {operation_id}")

        try:
            cached = self.redis.get(f"context:{symbol}")
            if cached:
                return json.loads(cached)

            # Fetch data via MCP tools
            news = await self.mcp_client.call_tool("yahoo_finance", "get_news_articles", {"symbol": symbol, "limit": 10})
            reddit = await self.mcp_client.call_tool("reddit", "get_reddit_posts", {"symbol": symbol, "limit": 10})

            texts = []
            if news["success"]:
                texts.extend([a["title"] + " " + a["summary"] for a in news["result"]["news"]])
            if reddit["success"]:
                texts.extend([p["title"] + " " + p["text"] for p in reddit["result"]["posts"]])

            # Embed and store in vector DB
            embeddings = self.embedder.encode(texts, convert_to_tensor=True).to('cuda')
            ids = [f"{symbol}_{i}" for i in range(len(texts))]
            self.vector_db.add(ids=ids, embeddings=embeddings, metadatas=[{"text": t} for t in texts])

            # Retrieve top contexts
            query_embedding = self.embedder.encode(symbol, convert_to_tensor=True).to('cuda')
            results = self.vector_db.query(query_embeddings=[query_embedding], n_results=5)
            contexts = [meta["text"] for meta in results["metadatas"][0]]

            # Summarize with FinBERT
            summary = self.summarizer(" ".join(contexts), max_length=100, min_length=30)[0]["summary_text"]

            result = {
                "success": True,
                "symbol": symbol,
                "context_summary": summary,
                "context_count": len(contexts),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.redis.setex(f"context:{symbol}", 300, json.dumps(result))
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n-')}context-events",
                {"event": "context_retrieved", "data": result}
            )
            return result

        except Exception as e:
            self.logger.error(f"Error retrieving context for {symbol}: {e}")
            return {"success": False, "error": str(e), "symbol": symbol, "operation_id": operation_id}

    async def shutdown(self):
        self.producer.close()
        self.redis.close()
        await self.mcp_client.shutdown()
        self.logger.info("ContextRetriever shutdown")