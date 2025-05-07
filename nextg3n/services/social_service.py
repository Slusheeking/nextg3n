"""
Social Service for NextG3N Trading System

This module implements the SocialService, fetching and analyzing social media data from Reddit
to gauge market sentiment. It provides tools for retrieving posts/comments and performing
preliminary sentiment analysis, integrated with the SentimentAgent in TradeFlowOrchestrator.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
from kafka import KafkaProducer

# Reddit API imports
try:
    import praw
    HAVE_PRAW = True
except ImportError:
    HAVE_PRAW = False
    logging.warning("praw not installed. Social media features will be unavailable.")

# Sentiment analysis imports
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAVE_VADER = True
except ImportError:
    HAVE_VADER = False
    logging.warning("vaderSentiment not installed. Sentiment analysis will be unavailable.")

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

class SocialService:
    """
    Service for fetching and analyzing social media data in the NextG3N system.
    Provides tools for retrieving posts/comments and preliminary sentiment analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SocialService with configuration and Reddit client.

        Args:
            config: Configuration dictionary with Reddit and Kafka settings
        """
        init_start_time = time.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="social_service")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.reddit_config = config.get("services", {}).get("social", {}).get("reddit", {})
        self.kafka_config = config.get("kafka", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize Reddit client
        self.reddit = None
        if HAVE_PRAW:
            self._initialize_reddit_client()
        else:
            self.logger.error("PRAW library not found. Social media features disabled.")
        
        # Initialize VADER sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer() if HAVE_VADER else None
        if not HAVE_VADER:
            self.logger.error("VADER library not found. Sentiment analysis disabled.")
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Default subreddits
        self.subreddits = self.reddit_config.get("subreddits", ["wallstreetbets", "stocks", "investing"])
        
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("social_service.initialization_time_ms", init_duration)
        self.logger.info("SocialService initialized")

    def _initialize_reddit_client(self):
        """
        Initialize the Reddit client using PRAW.
        """
        try:
            client_id = self.reddit_config.get("client_id") or os.environ.get("REDDIT_CLIENT_ID")
            client_secret = self.reddit_config.get("client_secret") or os.environ.get("REDDIT_CLIENT_SECRET")
            user_agent = self.reddit_config.get("user_agent") or os.environ.get("REDDIT_USER_AGENT")

            if not client_id or not client_secret or not user_agent:
                self.logger.error("Reddit API credentials missing")
                raise ValueError("Reddit API credentials not provided")

            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            
            # Test connection
            self.reddit.user.me()
            self.logger.info("Connected to Reddit API")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Reddit client: {e}")
            self.reddit = None
            raise

    async def get_social_posts(
        self,
        symbol: str,
        platform: str = "reddit",
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Fetch social media posts mentioning a stock.

        Args:
            symbol: Stock symbol
            platform: Social media platform (default: "reddit")
            limit: Maximum number of posts/comments to fetch

        Returns:
            Dictionary containing posts/comments
        """
        start_time = time.time()
        operation_id = f"posts_{int(start_time)}"
        self.logger.info(f"Fetching social posts for {symbol} on {platform} - Operation: {operation_id}")

        if platform.lower() != "reddit":
            self.logger.error(f"Unsupported platform: {platform}")
            return {
                "success": False,
                "error": f"Unsupported platform: {platform}",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        if not self.reddit:
            self.logger.error("Reddit client not initialized")
            return {
                "success": False,
                "error": "Reddit client not initialized",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                posts = await loop.run_in_executor(
                    self.executor,
                    lambda: self._fetch_reddit_posts(symbol, limit)
                )

            result = {
                "success": True,
                "symbol": symbol,
                "platform": platform,
                "posts": posts,
                "count": len(posts),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}social_events",
                {"event": "posts_fetched", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("social_service.get_social_posts_time_ms", duration)
            self.logger.info(f"Fetched {result['count']} posts for {symbol}")
            self.logger.counter("social_service.posts_fetched", result['count'])
            return result

        except Exception as e:
            self.logger.error(f"Error fetching posts for {symbol}: {e}")
            self.logger.counter("social_service.posts_errors", 1)
            result = {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "platform": platform,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}social_events",
                {"event": "posts_fetch_failed", "data": result}
            )
            return result

    def _fetch_reddit_posts(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """
        Fetch Reddit posts/comments mentioning a stock (synchronous helper).

        Args:
            symbol: Stock symbol
            limit: Maximum number of posts/comments

        Returns:
            List of post/comment dictionaries
        """
        posts = []
        try:
            for subreddit_name in self.subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                # Search for posts mentioning the symbol
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
                    posts.append(post_data)
                
                # Fetch comments from recent posts
                for submission in subreddit.hot(limit=10):
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list()[:limit // len(self.subreddits)]:
                        if symbol.lower() in comment.body.lower():
                            comment_data = {
                                "id": comment.id,
                                "text": comment.body,
                                "created_utc": datetime.utcfromtimestamp(comment.created_utc).isoformat(),
                                "score": comment.score,
                                "subreddit": subreddit_name,
                                "type": "comment"
                            }
                            posts.append(comment_data)
                
                # Respect Reddit API rate limits (60 requests/minute)
                time.sleep(1)
        
        except Exception as e:
            self.logger.error(f"Error in _fetch_reddit_posts: {e}")
            raise
        
        return posts[:limit]

    async def get_social_sentiment(
        self,
        symbol: str,
        platform: str = "reddit"
    ) -> Dict[str, Any]:
        """
        Perform preliminary sentiment analysis on social media posts.

        Args:
            symbol: Stock symbol
            platform: Social media platform (default: "reddit")

        Returns:
            Sentiment analysis dictionary
        """
        start_time = time.time()
        operation_id = f"sentiment_{int(start_time)}"
        self.logger.info(f"Analyzing social sentiment for {symbol} on {platform} - Operation: {operation_id}")

        if platform.lower() != "reddit":
            self.logger.error(f"Unsupported platform: {platform}")
            return {
                "success": False,
                "error": f"Unsupported platform: {platform}",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        if not self.reddit or not self.sentiment_analyzer:
            self.logger.error("Reddit client or VADER not initialized")
            return {
                "success": False,
                "error": "Reddit client or VADER not initialized",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                posts = await loop.run_in_executor(
                    self.executor,
                    lambda: self._fetch_reddit_posts(symbol, limit=50)
                )

            if not posts:
                self.logger.warning(f"No posts found for {symbol}")
                return {
                    "success": False,
                    "error": "No posts found",
                    "symbol": symbol,
                    "platform": platform,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

            # Analyze sentiment
            sentiment_scores = []
            for post in posts:
                text = post.get("title", "") + " " + post.get("text", "")
                if text.strip():
                    score = self.sentiment_analyzer.polarity_scores(text)
                    sentiment_scores.append(score["compound"])
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            sentiment_label = "positive" if avg_sentiment > 0.05 else "negative" if avg_sentiment < -0.05 else "neutral"

            result = {
                "success": True,
                "symbol": symbol,
                "platform": platform,
                "sentiment_score": avg_sentiment,
                "sentiment_label": sentiment_label,
                "post_count": len(posts),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}social_events",
                {"event": "sentiment_analyzed", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("social_service.get_social_sentiment_time_ms", duration)
            self.logger.info(f"Sentiment analyzed for {symbol}: Score={avg_sentiment:.2f}, Label={sentiment_label}")
            self.logger.counter("social_service.sentiment_analyses", 1)
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            self.logger.counter("social_service.sentiment_errors", 1)
            result = {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "platform": platform,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}social_events",
                {"event": "sentiment_analysis_failed", "data": result}
            )
            return result

    def shutdown(self):
        """
        Shutdown the service and close resources.
        """
        self.logger.info("Shutting down SocialService")
        self.executor.shutdown(wait=True)
        self.producer.close()
        self.logger.info("Kafka producer closed")