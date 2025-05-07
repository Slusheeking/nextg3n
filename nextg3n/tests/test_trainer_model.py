"""
Unit tests for the TrainerModel class in the NextG3N Trading System.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from nextg3n.models.trainer.trainer_model import TrainerModel
import asyncio

class TestTrainerModel(unittest.TestCase):
    def setUp(self):
        self.config = {
            "models": {"trainer": {"max_epochs": 10, "batch_size": 32}},
            "kafka": {"bootstrap_servers": "localhost:9092"}
        }
        self.trainer_model = TrainerModel(self.config)
        self.trainer_model.logger = MagicMock()

    @patch('pytorch_lightning.Trainer')
    async def test_train_model_success(self, mock_trainer):
        # Mock model and trainer
        mock_model = MagicMock()
        mock_model.train_model = asyncio.coroutine(lambda *args, **kwargs: {"success": True})
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.fit.return_value = None
        
        # Test data
        training_data = pd.DataFrame({
            "text": ["Test article"],
            "label": [2]
        })
        
        # Run train_model
        result = await self.trainer_model.train_model(mock_model, "sentiment", training_data)
        
        self.assertTrue(result["success"])
        self.assertIn("checkpoint_path", result)
        self.trainer_model.logger.info.assert_called()

    async def test_evaluate_model_success(self):
        # Mock model
        mock_model = MagicMock()
        mock_model.analyze_sentiment.return_value = {
            "success": True,
            "results": [{"positive": 0.7, "neutral": 0.2, "negative": 0.1}]
        }
        
        # Test data
        validation_data = pd.DataFrame({
            "text": ["Test article"],
            "label": [2]
        })
        
        # Run evaluate_model
        result = await self.trainer_model.evaluate_model(mock_model, "sentiment", validation_data)
        
        self.assertTrue(result["success"])
        self.assertIn("metrics", result)
        self.assertEqual(result["metrics"]["accuracy"], 1.0)
        self.trainer_model.logger.info.assert_called()

if __name__ == '__main__':
    unittest.main()