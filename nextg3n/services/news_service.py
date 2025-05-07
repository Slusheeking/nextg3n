"""
News Service for NextG3N Trading System

This module implements the NewsService, fetching and analyzing financial news data from
Polygon.io and Yahoo Finance to gauge market sentiment. It provides tools for retrieving
news articles and performing preliminary sentiment analysis, integrated with the SentimentAgent
in TradeFlowOrchestrator.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from dotenv import load_dotenv
import aiohttp
from kafka import KafkaProducer

# Yahoo Finance imports
try:
    import yfinance as yf
    HAVE_YFINANCE = True
except ImportError:
    HAVE_YFINANCE = False
    logging.warning("yfinance not installed. Yahoo Finance news will be unavailable.")

# Sentiment analysis imports
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAVE_VADER = True
except ImportError:
    HAVE_VADER = False
    logging.warning("vaderSentiment not installed. Sentiment analysis will be unavailable.")

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

class NewsService:
    """
    Service for fetching and analyzing financial news data in the NextG3N system.
    Provides tools for retrieving news articles and preliminary sentiment analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the NewsService with configuration and API settings.

        Args:
            config: Configuration dictionary with Polygon.io, Yahoo Finance, and Kafka settings
        """
        init_start_time = time.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="news_service")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.polygon_config = config.get("services", {}).get("news", {}).get("polygon", {})
        self.yahoo_config = config.get("services", {}).get("news", {}).get("yahoo", {})
        self.kafka_config = config.get("kafka", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize API settings
        self.polygon_api_key = self.polygon_config.get("api_key") or os.environ.get("POLYGON_API_KEY")
        self.polygon_base_url = self.polygon_config.get("base_url", "https://api.polygon.io/v2")
        self.polygon_rate_limit_delay = self.polygon_config.get("rate_limit_delay", 12.0)  # 5 requests/minute = 12s delay
        self.yahoo_enabled = self.yahoo_config.get("enabled", True) and HAVE_YFINANCE
        
        if not self.polygon_api_key:
            self.logger.error("Polygon API key missing")
            raise ValueError("Polygon API key not provided")
        
        if not self.yahoo_enabled and not HAVE_YFINANCE:
            self.logger.warning("Yahoo Finance disabled due to missing yfinance library")

        # Initialize VADER sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer() if HAVE_VADER else None
        if not HAVE_VADER:
            self.logger.error("VADER library not found. Sentiment analysis disabled.")
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        start_time = time.time()
        init_duration = (time.time() - start_time) * 1000
        self.logger.timing("news_service.initialization_time_ms", init_duration)
        self.logger.info("NewsService initialized")

    async def get_news(
        self,
        symbols: List[str],
        limit: int = 30
    ) -> Dict[str, Any]:
        """
        Fetch recent news articles for specified stocks from Polygon.io and Yahoo Finance.

        Args:
            symbols: List of stock symbols
            limit: Maximum number of articles to fetch

        Returns:
            Dictionary containing news articles
        """
        start_time = time.time()
        operation_id = f"news_{int(start_time)}"
        self.logger.info(f"Fetching news for symbols: {symbols} - Operation: {operation_id}")

        try:
            articles = []
            
            # Fetch from Polygon.io
            polygon_articles = await self._fetch_polygon_news(symbols, limit // 2)
            articles.extend(polygon_articles)
            
            # Fetch from Yahoo Finance (if enabled)
            if self.yahoo_enabled:
                yahoo_articles = await self._fetch_yahoo_news(symbols, limit - len(polygon_articles))
                articles.extend(yahoo_articles)
            
            # Limit total articles
            articles = articles[:limit]
            
            result = {
                "success": True,
                "symbols": symbols,
                "news": articles,
                "count": len(articles),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}news_events",
                {"event": "news_fetched", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("news_service.get_news_time_ms", duration)
            self.logger.info(f"Fetched {result['count']} news articles for {symbols}")
            self.logger.counter("news_service.news_fetched", result['count'])
            return result

        except Exception as e:
            self.logger.error(f"Error fetching news for {symbols}: {e}")
            self.logger.counter("news_service.news_errors", 1)
            result = {
                "success": False,
                "error": str(e),
                "symbols": symbols,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}news_events",
                {"event": "news_fetch_failed", "data": result}
            )
            return result

    async def _fetch_polygon_news(self, symbols: List[str], limit: int) -> List[Dict[str, Any]]:
        """
        Fetch news from Polygon.io (asynchronous).

        Args:
            symbols: List of stock symbols
            limit: Maximum number of articles

        Returns:
            List of article dictionaries
        """
        articles = []
        try:
            async with aiohttp.ClientSession() as session:
                for symbol in symbols:
                    url = f"{self.polygon_base_url}/reference/news"
                    headers = {"Authorization": f"Bearer {self.polygon_api_key}"}
                    params = {"ticker": symbol, "limit": limit // len(symbols)}
                    
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status != 200:
                            self.logger.warning(f"Polygon API error for {symbol}: {response.status}")
                            continue
                        
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
                            articles.append(article)
                    
                    # Respect Polygon rate limits (5 requests/minute for free tier)
                    await asyncio.sleep(self.polygon_rate_limit_delay)
        
        except Exception as e:
            self.logger.error(f"Error in _fetch_polygon_news: {e}")
        
        return articles

    async def _fetch_yahoo_news(self, symbols: List[str], limit: int) -> List[Dict[str, Any]]:
        """
        Fetch news from Yahoo Finance (asynchronous).

        Args:
            symbols: List of stock symbols
            limit: Maximum number of articles

        Returns:
            List of article dictionaries
        """
        articles = []
        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                for symbol in symbols:
                    ticker = await loop.run_in_executor(self.executor, lambda: yf.Ticker(symbol))
                    news = await loop.run_in_executor(self.executor, lambda: ticker.news)
                    
                    for item in news[:limit // len(symbols)]:
                        article = {
                            "symbol": symbol,
                            "title": item.get("title", ""),
                            "summary": item.get("summary", ""),
                            "published_utc": datetime.utcfromtimestamp(item.get("provider_publish_time", 0)).isoformat(),
                            "source": "Yahoo",
                            "url": item.get("link", "")
                        }
                        articles.append(article)
                    
                    # Avoid overwhelming Yahoo API
                    await asyncio.sleep(1.0)
        
        except Exception as e:
            self.logger.error(f"Error in _fetch_yahoo_news: {e}")
        
        return articles

    async def get_news_sentiment(
        self,
        symbol: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Perform preliminary sentiment analysis on news articles.

        Args:
            symbol: Stock symbol
            days: Number of days to consider for news

        Returns:
            Sentiment analysis dictionary
        """
        start_time = time.time()
        operation_id = f"sentiment_{int(start_time)}"
        self.logger.info(f"Analyzing news sentiment for {symbol} - Operation: {operation_id}")

        if not self.sentiment_analyzer:
            self.logger.error("VADER not initialized")
            return {
                "success": False,
                "error": "VADER not initialized",
                "symbol": symbol,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            # Fetch news articles
            news_result = await self.get_news([symbol], limit=30)
            
            if not news_result["success"] or not news_result["news"]:
                self.logger.warning(f"No news found for {symbol}")
                return {
                    "success": False,
                    "error": "No news found",
                    "symbol": symbol,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

            articles = news_result["news"]
            
            # Filter articles by date (within specified days)
            from_date = datetime.utcnow() - timedelta(days=days)
            articles = [
                a for a in articles
                if datetime.fromisoformat(a["published_utc"].replace("Z", "+00:00")) >= from_date
            ]
            
            # Analyze sentiment
            sentiment_scores = []
            for article in articles:
                text = article.get("title", "") + " " + article.get("summary", "")
                if text.strip():
                    score = self.sentiment_analyzer.polarity_scores(text)
                    sentiment_scores.append(score["compound"])
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            sentiment_label = (
                "positive" if avg_sentiment > 0.05 else
                "negative" if avg_sentiment < -0.05 else
                "neutral"
            )

            result = {
                "success": True,
                "symbol": symbol,
                "sentiment_score": avg_sentiment,
                "sentiment_label": sentiment_label,
                "article_count": len(articles),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}news_events",
                {"event": "sentiment_analyzed", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("news_service.get_news_sentiment_time_ms", duration)
            self.logger.info(f"News sentiment analyzed for {symbol}: Score={avg_sentiment:.2f}, Label={sentiment_label}")
            self.logger.counter("news_service.sentiment_analyses", 1)
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment for {symbol}: {e}")
            self.logger.counter("news_service.sentiment_errors", 1)
            result = {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}news_events",
                {"event": "sentiment_analysis_failed", "data": result}
            )
            return result

    def shutdown(self):
        """
        Shutdown the service and close resources.
        """
        self.logger.info("Shutting down NewsService")
        self.executor.shutdown(wait=True)
        self.producer.close()
        self.logger.info("Kafka producer closed")