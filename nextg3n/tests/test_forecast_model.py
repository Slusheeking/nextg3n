"""
Unit tests for the ForecastModel class in the NextG3N Trading System.
"""

import unittest
from unittest.mock import patch, MagicMock
import torch
import pandas as pd
from nextg3n.models.forecast.forecast_model import ForecastModel
import asyncio

class TestForecastModel(unittest.TestCase):
    def setUp(self):
        self.config = {
            "models": {"forecast": {"max_encoder_length": 30, "max_prediction_length": 1}},
            "kafka": {"bootstrap_servers": "localhost:9092"}
        }
        self.forecast_model = ForecastModel(self.config)
        self.forecast_model.logger = MagicMock()

    @patch('pytorch_forecasting.TemporalFusionTransformer')
    async def test_predict_price_success(self, mock_tft):
        # Mock model inference
        self.forecast_model.model = MagicMock()
        mock_output = torch.tensor([0.02])  # 2% price increase
        self.forecast_model.model.forward.return_value = mock_output
        
        # Test data
        data = {
            "bars": [
                {"timestamp": "2023-01-01", "close": 100, "volume": 1000},
                {"timestamp": "2023-01-02", "close": 101, "volume": 1100}
            ],
            "technical_indicators": {"rsi": 70, "macd": 0.1},
            "sentiment": {"news_sentiment": 0.5, "social_sentiment": 0.3}
        }
        
        # Run predict_price
        result = await self.forecast_model.predict_price("AAPL", "1d", data)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["predicted_price_change"], 0.02)
        self.assertEqual(result["direction"], "up")
        self.forecast_model.logger.info.assert_called()

    async def test_predict_price_empty_data(self):
        # Test with empty data
        result = await self.forecast_model.predict_price("AAPL", "1d", {})
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.forecast_model.logger.error.assert_called()

if __name__ == '__main__':
    unittest.main()