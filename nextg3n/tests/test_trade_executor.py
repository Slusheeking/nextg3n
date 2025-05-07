"""
Unit tests for the TradeExecutor class in the NextG3N Trading System.
"""

import unittest
from unittest.mock import patch, MagicMock
from nextg3n.models.trade.trade_executor import TradeExecutor
import asyncio

class TestTradeExecutor(unittest.TestCase):
    def setUp(self):
        self.config = {
            "models": {
                "trade": {
                    "capital": 100000,
                    "max_position_size": 0.05,
                    "alpaca_api_key": "test_key",
                    "alpaca_api_secret": "test_secret"
                }
            },
            "kafka": {"bootstrap_servers": "localhost:9092"}
        }
        self.trade_executor = TradeExecutor(self.config)
        self.trade_executor.logger = MagicMock()

    @patch('alpaca_trade_api.rest.REST')
    async def test_execute_trade_success(self, mock_alpaca):
        # Mock Alpaca API
        mock_alpaca_instance = MagicMock()
        mock_alpaca.return_value = mock_alpaca_instance
        mock_account = MagicMock(cash="100000")
        mock_alpaca_instance.get_account.return_value = mock_account
        mock_order = MagicMock(id="order_123")
        mock_alpaca_instance.submit_order.return_value = mock_order
        
        # Test decision
        decision = {"symbol": "AAPL", "action": "buy", "confidence": 0.9}
        current_price = 150.0
        
        # Run execute_trade
        result = await self.trade_executor.execute_trade(decision, current_price)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["action"], "buy")
        self.assertEqual(result["order_id"], "order_123")
        self.trade_executor.logger.info.assert_called()

    @patch('alpaca_trade_api.rest.REST')
    async def test_execute_trade_invalid_action(self, mock_alpaca):
        # Test with invalid action
        decision = {"symbol": "AAPL", "action": "invalid", "confidence": 0.9}
        current_price = 150.0
        
        # Run execute_trade
        result = await self.trade_executor.execute_trade(decision, current_price)
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.trade_executor.logger.error.assert_called()

if __name__ == '__main__':
    unittest.main()