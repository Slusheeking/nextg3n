"""
Unit tests for the BacktestEngine class in the NextG3N Trading System.
"""

import unittest
from unittest.mock import patch, MagicMock
from nextg3n.models.backtest.backtest_engine import BacktestEngine
import pandas as pd

class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        self.config = {
            "models": {"backtest": {"enabled": True}},
            "kafka": {"bootstrap_servers": "localhost:9092"}
        }
        self.backtest_engine = BacktestEngine(self.config)
        self.backtest_engine.logger = MagicMock()

    @patch('backtrader.Backtrader')
    def test_run_backtest_success(self, mock_backtrader):
        # Mock Backtrader strategy and data
        mock_strategy = MagicMock()
        mock_backtrader.Strategy.return_value = mock_strategy
        mock_data = pd.DataFrame({
            "timestamp": ["2023-01-01", "2023-01-02"],
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 1100]
        })
        
        # Mock backtest result
        mock_cerebro = MagicMock()
        mock_backtrader.Cerebro.return_value = mock_cerebro
        mock_cerebro.run.return_value = [{"pnl": 500, "trades": 2}]
        
        # Run backtest
        result = self.backtest_engine.run_backtest("AAPL", mock_data, "1d")
        
        self.assertTrue(result["success"])
        self.assertEqual(result["pnl"], 500)
        self.assertEqual(result["trade_count"], 2)
        self.backtest_engine.logger.info.assert_called()

    @patch('backtrader.Backtrader')
    def test_run_backtest_empty_data(self, mock_backtrader):
        # Test with empty data
        mock_data = pd.DataFrame()
        
        result = self.backtest_engine.run_backtest("AAPL", mock_data, "1d")
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.backtest_engine.logger.error.assert_called()

if __name__ == '__main__':
    unittest.main()