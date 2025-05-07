"""
Sentiment Model for NextG3N Trading System

Implements sentiment analysis using FinBERT for provided text data.
Designed for low-latency inference when integrated directly.
"""

import logging
import time
import json
import datetime
from typing import Dict, Any, List
from transformers import pipeline

from monitoring.metrics_logger import MetricsLogger


class SentimentModel:
    def __init__(self, config: Dict[str, Any]):
        self.logger = MetricsLogger(component_name="sentiment_model")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.config = config

        # Initialize the sentiment analysis pipeline with a pre-trained FinBERT model
        # This model is loaded once at initialization for faster inference
        # later
        try:
            model_name = self.config.get("model_name", "ProsusAI/finbert")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", model=model_name)
            self.logger.info(
                "FinBERT sentiment analysis pipeline initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize FinBERT pipeline: {e}")
            self.sentiment_analyzer = None  # Handle failure gracefully

        self.logger.info("SentimentModel initialized.")

    async def analyze(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of a list of text inputs.

        Args:
            texts: A list of strings to analyze.

        Returns:
            A dictionary containing the sentiment analysis results.
        """
        operation_id = f"sentiment_analysis_{int(time.time())}"
        self.logger.info(
            f"Analyzing sentiment for {len(texts)} texts - Operation: {operation_id}")

        if not self.sentiment_analyzer:
            self.logger.error(
                "Sentiment analyzer not initialized. Cannot perform analysis.")
            return {
                "success": False,
                "error": "Sentiment analyzer not ready",
                "operation_id": operation_id}

        if not texts:
            self.logger.warning("No texts provided for sentiment analysis.")
            return {
                "success": True,
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "text_count": 0,
                "operation_id": operation_id}

        try:
            # Perform sentiment analysis
            # The pipeline handles batching internally for efficiency
            results = self.sentiment_analyzer(texts)

            sentiment_scores = []
            for result in results:
                score = 1.0 if result["label"] == "positive" else - \
                    1.0 if result["label"] == "negative" else 0.0
                sentiment_scores.append(score * result["score"])

            # Calculate average sentiment score
            avg_sentiment = sum(sentiment_scores) / \
                len(sentiment_scores) if sentiment_scores else 0

            # Determine overall sentiment label
            positive_threshold = self.config.get("positive_threshold", 0.05)
            negative_threshold = self.config.get("negative_threshold", -0.05)
            sentiment_label = "positive" if avg_sentiment > positive_threshold else "negative" if avg_sentiment < negative_threshold else "neutral"

            result = {
                "success": True,
                # Format for consistency
                "sentiment_score": float(f"{avg_sentiment:.4f}"),
                "sentiment_label": sentiment_label,
                "text_count": len(texts),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()  # Use datetime from import
            }

            self.logger.info(f"Sentiment analysis complete. Result: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error during sentiment analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id}

    async def shutdown(self):
        # No specific shutdown needed for the pipeline in this context, but
        # include for consistency
        self.logger.info("SentimentModel shutdown.")

# Note: This model is designed to be integrated directly into a trading engine
# and receive text data as input. It does not handle data fetching, Kafka,
# or Redis internally.
