"""
Stock Ranker for NextG3N Trading System

This module implements the StockRanker class, ranking stocks based on multiple factors
(price predictions, sentiment, options flow, technical indicators). It supports the
StockPickerAgent in TradeFlowOrchestrator by identifying high-potential stocks.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

# Kafka imports
from kafka import KafkaProducer

class StockRanker:
    """
    Class for ranking stocks based on multiple factors in the NextG3N system.
    Supports the StockPickerAgent in TradeFlowOrchestrator.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the StockRanker with configuration and ranking settings.

        Args:
            config: Configuration dictionary with ranking and Kafka settings
        """
        init_start_time = time.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="stock_ranker")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.ranking_config = config.get("models", {}).get("stock_ranker", {})
        self.kafka_config = config.get("kafka", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize ranking weights
        self.weights = self.ranking_config.get("weights", {
            "price_prediction": 0.4,
            "sentiment_score": 0.3,
            "options_sentiment": 0.2,
            "technical_score": 0.1
        })
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("stock_ranker.initialization_time_ms", init_duration)
        self.logger.info("StockRanker initialized")

    async def rank_stocks(
        self,
        stock_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Rank stocks based on aggregated data.

        Args:
            stock_data: List of dictionaries containing stock data (symbol, predictions, sentiment, etc.)

        Returns:
            Dictionary containing ranked stocks
        """
        start_time = time.time()
        operation_id = f"rank_stocks_{int(start_time)}"
        self.logger.info(f"Ranking {len(stock_data)} stocks - Operation: {operation_id}")

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                
                # Process stock data
                ranked_stocks = await loop.run_in_executor(
                    self.executor,
                    lambda: self._process_and_rank(stock_data)
                )

            result = {
                "success": True,
                "ranked_stocks": ranked_stocks,
                "stock_count": len(ranked_stocks),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}ranking_events",
                {"event": "stocks_ranked", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("stock_ranker.rank_stocks_time_ms", duration)
            self.logger.info(f"Ranked {len(ranked_stocks)} stocks")
            self.logger.counter("stock_ranker.rankings_completed", len(ranked_stocks))
            return result

        except Exception as e:
            self.logger.error(f"Error ranking stocks: {e}")
            self.logger.counter("stock_ranker.ranking_errors", len(stock_data))
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def _process_and_rank(self, stock_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process stock data and compute ranking scores.

        Args:
            stock_data: List of dictionaries containing stock data

        Returns:
            Sorted list of ranked stocks with scores
        """
        try:
            # Convert to DataFrame for efficient processing
            df = pd.DataFrame(stock_data)
            
            # Normalize features (scale to [0, 1])
            def normalize(series):
                min_val, max_val = series.min(), series.max()
                if max_val == min_val:
                    return np.zeros_like(series)
                return (series - min_val) / (max_val - min_val)
            
            # Extract and normalize features
            df["price_score"] = normalize(df.get("price_prediction", {}).apply(lambda x: x.get("predicted_price_change", 0.0)))
            df["sentiment_score"] = normalize(df.get("sentiment", {}).apply(lambda x: x.get("sentiment_score", 0.0)))
            df["options_score"] = normalize(df.get("options_sentiment", {}).apply(lambda x: x.get("sentiment_score", 0.0)))
            
            # Technical score (e.g., combine RSI and MACD)
            df["technical_score"] = normalize(
                df.get("technical_indicators", {}).apply(lambda x: (
                    (x.get("rsi", 50.0) - 50) / 50 +  # Normalize RSI around 50
                    x.get("macd", 0.0) / abs(x.get("macd", 0.0) or 1.0)  # Normalize MACD
                ) / 2)
            )
            
            # Compute composite score
            df["composite_score"] = (
                self.weights["price_prediction"] * df["price_score"] +
                self.weights["sentiment_score"] * df["sentiment_score"] +
                self.weights["options_sentiment"] * df["options_score"] +
                self.weights["technical_score"] * df["technical_score"]
            )
            
            # Sort by composite score (descending)
            df = df.sort_values(by="composite_score", ascending=False)
            
            # Prepare ranked stocks
            ranked_stocks = [
                {
                    "symbol": row.get("symbol", ""),
                    "composite_score": row["composite_score"],
                    "price_score": row["price_score"],
                    "sentiment_score": row["sentiment_score"],
                    "options_score": row["options_score"],
                    "technical_score": row["technical_score"]
                }
                for _, row in df.iterrows()
            ]
            
            return ranked_stocks
        
        except Exception as e:
            self.logger.error(f"Error processing and ranking stocks: {e}")
            return []

    def shutdown(self):
        """
        Shutdown the StockRanker and close resources.
        """
        self.logger.info("Shutting down StockRanker")
        self.executor.shutdown(wait=True)
        self.producer.close()
        self.logger.info("Kafka producer closed")