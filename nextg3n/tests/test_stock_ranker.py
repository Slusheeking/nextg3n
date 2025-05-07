"""
Unit tests for the StockRanker class in the NextG3N Trading System.
"""

import unittest
from unittest.mock import patch, MagicMock
from nextg3n.models.stock_ranker.stock_ranker import StockRanker
import pandas as pd
import asyncio

class TestStockRanker(unittest.TestCase):
    def setUp(self):
        self.config = {
            "models": {
                "stock_ranker": {
                    "weights": {
                        "price_prediction": 0.4,
                        "sentiment_score": 0.3,
                        "options_sentiment": 0.2,
                        "technical_score": 0.1
                    }
                }
            },
            "kafka": {"bootstrap_servers": "localhost:9092"}
        }
        self.stock_ranker = StockRanker(self.config)
        self.stock_ranker.logger = MagicMock()

    async def test_rank_stocks_success(self):
        # Test data
        stock_data = [
            {
                "symbol": "AAPL",
                "price_prediction": {"predicted_price_change": 0.05},
                "sentiment": {"sentiment_score": 0.7},
                "options_sentiment": {"sentiment_score": 0.6},
                "technical_indicators": {"rsi": 70, "macd": 0.1}
            },
            {
                "symbol": "GOOGL",
                "price_prediction": {"predicted_price_change": 0.03},
                "sentiment": {"sentiment_score": 0.5},
                "options_sentiment": {"sentiment_score": 0.4},
                "technical_indicators": {"rsi": 50, "macd": 0.0}
            }
        ]
        
        # Run rank_stocks
        result = await self.stock_ranker.rank_stocks(stock_data)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["stock_count"], 2)
        self.assertEqual(result["ranked_stocks"][0]["symbol"], "AAPL")  # Higher score
        self.stock_ranker.logger.info.assert_called()

    async def test_rank_stocks_empty_data(self):
        # Test with empty data
        result = await self.stock_ranker.rank_stocks([])
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.stock_ranker.logger.error.assert_called()

if __name__ == '__main__':
    unittest.main()