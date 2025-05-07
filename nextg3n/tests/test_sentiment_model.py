"""
Unit tests for the SentimentModel class in the NextG3N Trading System.
"""

import unittest
from unittest.mock import patch, MagicMock
import torch
from nextg3n.models.sentiment.sentiment_model import SentimentModel
import asyncio

class TestSentimentModel(unittest.TestCase):
    def setUp(self):
        self.config = {
            "models": {"sentiment": {"model_name": "roberta-base", "max_length": 512}},
            "kafka": {"bootstrap_servers": "localhost:9092"}
        }
        self.sentiment_model = SentimentModel(self.config)
        self.sentiment_model.logger = MagicMock()

    @patch('transformers.RobertaForSequenceClassification')
    @patch('transformers.RobertaTokenizer')
    async def test_analyze_sentiment_success(self, mock_tokenizer, mock_model):
        # Mock model and tokenizer
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.1, 0.2, 0.7]])  # Positive sentiment
        mock_model_instance.return_value = mock_output
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.encode.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        
        # Test data
        texts = ["Positive news article"]
        
        # Run analyze_sentiment
        result = await self.sentiment_model.analyze_sentiment(texts)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["text_count"], 1)
        self.assertAlmostEqual(result["results"][0]["positive"], 0.7, places=2)
        self.sentiment_model.logger.info.assert_called()

    async def test_analyze_sentiment_empty_texts(self):
        # Test with empty texts
        result = await self.sentiment_model.analyze_sentiment([])
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.sentiment_model.logger.error.assert_called()

if __name__ == '__main__':
    unittest.main()