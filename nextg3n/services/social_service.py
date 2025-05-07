"""
Social Service for NextG3N Trading System

Implements an MCP server for social media data from Reddit.
Integrates with Unusual Whales for sentiment; publishes to Kafka topic nextg3n-social-events.
"""

import logging
import asyncio
import json
import aiohttp
import praw
import time
import datetime
from typing import Dict, Any, List
from kafka import KafkaProducer
from redis import Redis
from monitoring.metrics_logger import MetricsLogger
from services.mcp_client import MCPClient

class SocialService:
    def __init__(self, config: Dict[str, Any]):
        self.logger = MetricsLogger(component_name="social_service")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.config = config
        self.reddit_config = config.get("services", {}).get("social", {}).get("reddit", {})
        self.kafka_config = config.get("kafka", {})
        self.redis_config = config.get("storage", {}).get("redis", {})
        self.mcp_client = MCPClient(config)

        self.reddit = praw.Reddit(
            client_id=self.reddit_config.get("client_id"),
            client_secret=self.reddit_config.get("client_secret"),
            user_agent=self.reddit_config.get("user_agent")
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
        self.subreddits = self.reddit_config.get("subreddits", ["wallstreetbets", "stocks", "investing"])
        self.logger.info("SocialService initialized")

    async def get_reddit_posts(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        operation_id = f"posts_{int(time.time())}"
        self.logger.info(f"Fetching Reddit posts for {symbol} - Operation: {operation_id}")

        try:
            cached = self.redis.get(f"reddit_posts:{symbol}")
            if cached:
                return json.loads(cached)

            posts = []
            for subreddit_name in self.subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                for submission in subreddit.search(f"{symbol}", limit=limit // len(self.subreddits)):
                    post_data = {
                        "id": submission.id,
                        "title": submission.title,
                        "text": submission.selftext,
                        "created_utc": datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                        "score": submission.score,
                        "subreddit": subreddit_name,
                        "type": "post"
                    }
                    if submission.score > 100:  # Filter high-impact posts
                        posts.append(post_data)
                await asyncio.sleep(1)  # Respect rate limits

            result = {
                "success": True,
                "symbol": symbol,
                "platform": "reddit",
                "posts": posts[:limit],
                "count": len(posts),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.redis.setex(f"reddit_posts:{symbol}", 300, json.dumps(result))
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n-')}social-events",
                {"event": "posts_fetched", "data": result}
            )
            return result

        except Exception as e:
            self.logger.error(f"Error fetching Reddit posts for {symbol}: {e}")
            return {"success": False, "error": str(e), "symbol": symbol, "operation_id": operation_id}

    async def shutdown(self):
        self.producer.close()
        self.redis.close()
        await self.mcp_client.shutdown()
        self.logger.info("SocialService shutdown")