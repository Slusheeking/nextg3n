"""
Unit tests for the DecisionModel class in the NextG3N Trading System.
"""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import torch
import numpy as np
from nextg3n.models.decision.decision_model import DecisionModel
import asyncio

class TestDecisionModel(unittest.TestCase):
    def setUp(self):
        self.config = {
            "models": {
                "decision": {
                    "state_dim": 64,
                    "action_dim": 3,
                    "hidden_size": 256,
                    "n_layer": 6,
                    "n_head": 8
                }
            },
            "kafka": {"bootstrap_servers": "localhost:9092"},
            "llm": {
                "enabled": True,
                "provider": "openrouter",
                "model": "openai/gpt-4",
                "max_tokens": 512,
                "temperature": 0.7,
                "retry_attempts": 3,
                "retry_delay": 1000
            }
        }
        self.decision_model = DecisionModel(self.config)
        self.decision_model.logger = MagicMock()

    @patch('torch.nn.Module')
    async def test_make_decision_success(self, mock_module):
        # Mock model inference
        self.decision_model.model = MagicMock()
        mock_output = torch.tensor([[[0.1, 0.8, 0.1]]])  # High probability for sell
        self.decision_model.model.forward.return_value = mock_output
        
        # Test state
        state = {
            "price_prediction": {"predicted_price": 150, "confidence": 0.9},
            "sentiment": {"sentiment_score": 0.5},
            "technical_indicators": {"rsi": 70, "macd": 0.1},
            "context": {"context_score": 0.2}
        }
        
        # Run make_decision
        result = await self.decision_model.make_decision("AAPL", state, explain=False)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "sell")
        self.assertAlmostEqual(result["confidence"], 0.8, places=2)
        self.decision_model.logger.info.assert_called()

    @patch('torch.nn.Module')
    @patch('aiohttp.ClientSession.post')
    async def test_make_decision_with_explanation_success(self, mock_post, mock_module):
        # Mock model inference
        self.decision_model.model = MagicMock()
        mock_output = torch.tensor([[[0.1, 0.8, 0.1]]])  # High probability for sell
        self.decision_model.model.forward.return_value = mock_output
        
        # Mock OpenRouter response for explanation
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "Explanation for selling AAPL"}}]
        })
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Test state
        state = {
            "price_prediction": {"predicted_price": 150, "confidence": 0.9},
            "sentiment": {"sentiment_score": 0.5},
            "technical_indicators": {"rsi": 70, "macd": 0.1},
            "context": {"context_score": 0.2}
        }
        
        # Run make_decision with explanation
        result = await self.decision_model.make_decision("AAPL", state, explain=True)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "sell")
        self.assertAlmostEqual(result["confidence"], 0.8, places=2)
        self.assertEqual(result["explanation"], "Explanation for selling AAPL")
        self.decision_model.logger.info.assert_called()
        self.decision_model.logger.track_llm_usage.assert_called_with(tokens=200, model="openai/gpt-4")

if __name__ == '__main__':
    unittest.main()