"""
Backtest Engine for NextG3N Trading System

This module implements the backtesting model using Backtrader, validating trading strategies
based on predictions from the ForecastModel. It supports the PredictorAgent in evaluating
strategy performance before live trading.
"""

import os
import logging
import pandas as pd
import backtrader as bt
from typing import Dict, Any, List
from multiprocessing import Pool
from datetime import datetime
import numpy as np

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

# Service imports (assumed to exist)
from services.market_data_service import MarketDataService

# Model imports (assumed to exist)
from models.forecast.forecast_model import ForecastModel

class TFTStrategy(bt.Strategy):
    """
    Backtrader strategy driven by TFT predictions from ForecastModel.
    """
    params = (
        ("size", 100),  # Number of shares per trade
        ("stop_loss", 0.05),  # 5% stop-loss
        ("lookback", 30),  # Lookback period for predictions
    )

    def __init__(self):
        self.logger = MetricsLogger(component_name="tft_strategy")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        self.forecast_model = ForecastModel(config={})  # Initialize with config
        self.order = None
        self.trades = []
        self.position_value = 0

    def notify_order(self, order):
        """
        Handle order notifications.
        """
        if order.status in [order.Completed]:
            if order.isbuy():
                self.logger.info(f"BUY EXECUTED: {order.executed.price}, Size: {order.executed.size}")
            elif order.issell():
                self.logger.info(f"SELL EXECUTED: {order.executed.price}, Size: {order.executed.size}")
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.logger.warning(f"Order failed: {order.status}")
            self.order = None

    def notify_trade(self, trade):
        """
        Track trade outcomes.
        """
        if trade.isclosed:
            self.trades.append({
                "profit": trade.pnl,
                "entry_price": trade.price,
                "exit_price": trade.price + trade.pnl / trade.size,
                "entry_time": trade.dtopen,
                "exit_time": trade.dtclose,
                "win": trade.pnl > 0
            })
            self.logger.info(f"TRADE CLOSED: Profit={trade.pnl:.2f}, Win={trade.pnl > 0}")

    def next(self):
        """
        Execute strategy logic for each data point.
        """
        if self.order:  # Skip if an order is pending
            return
        
        # Fetch recent data
        data = {
            "close": list(self.data.close.get(size=self.params.lookback)),
            "timestamp": [self.data.datetime.datetime(i) for i in range(-self.params.lookback + 1, 1)]
        }
        
        try:
            prediction = self.forecast_model.predict_price(
                symbol=self.data._name,
                timeframe="1d",
                data=data
            )
            
            if prediction["confidence"] < 0.6:  # Skip low-confidence predictions
                return
            
            if prediction["action"] == "buy" and not self.position:
                self.order = self.buy(size=self.params.size)
                self.position_value = self.params.size * self.data.close[0]
                self.logger.info(f"BUY ORDER: {self.data._name}, Size={self.params.size}")
            
            elif prediction["action"] == "sell" and self.position:
                current_price = self.data.close[0]
                if current_price <= self.position_value * (1 - self.params.stop_loss):
                    self.order = self.sell(size=self.position.size)
                    self.logger.info(f"STOP-LOSS SELL: {self.data._name}, Size={self.position.size}")
                elif prediction["confidence"] > 0.7:
                    self.order = self.sell(size=self.position.size)
                    self.logger.info(f"SELL ORDER: {self.data._name}, Size={self.position.size}")
        
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")

class BacktestEngine:
    """
    Backtesting model for NextG3N, using Backtrader to validate trading strategies.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BacktestEngine.

        Args:
            config: Configuration dictionary
        """
        self.logger = MetricsLogger(component_name="backtest_engine")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        self.config = config
        self.data_service = MarketDataService(config)
        self.initial_cash = config.get("backtest", {}).get("initial_cash", 100000)
        self.commission = config.get("backtest", {}).get("commission", 0.001)
        self.logger.info("BacktestEngine initialized")

    def run_backtest(self, symbol: str, timeframe: str = "1d", start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a backtest for a single stock.

        Args:
            symbol: Stock symbol
            timeframe: Data timeframe (e.g., '1d')
            start_date: Start date (YYYY-MM-DD, optional)
            end_date: End date (YYYY-MM-DD, optional)

        Returns:
            Backtest results (returns, win rate, Sharpe ratio, etc.)
        """
        start_time = datetime.time()
        self.logger.info(f"Running backtest for {symbol}, timeframe={timeframe}")
        
        try:
            # Fetch historical data
            data = self.data_service.get_historical_data(symbol, timeframe, limit=1000)
            if not data.get("bars"):
                self.logger.error(f"No data for {symbol}")
                return {"symbol": symbol, "error": "No data available"}
            
            # Prepare data for Backtrader
            df = pd.DataFrame(data["bars"])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Initialize Backtrader
            cerebro = bt.Cerebro()
            cerebro.addstrategy(TFTStrategy, size=100, stop_loss=0.05, lookback=30)
            cerebro.adddata(bt.feeds.PandasData(dataname=df))
            cerebro.broker.set_cash(self.initial_cash)
            cerebro.broker.setcommission(commission=self.commission)
            
            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            
            # Run backtest
            results = cerebro.run()
            strat = results[0]
            
            # Calculate metrics
            final_value = cerebro.broker.getvalue()
            returns = (final_value - self.initial_cash) / self.initial_cash * 100
            sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            max_drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
            
            trades = strat.trades
            total_trades = len(trades)
            win_trades = sum(1 for trade in trades if trade["win"])
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
            
            result = {
                "symbol": symbol,
                "returns": returns,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "total_trades": total_trades,
                "win_trades": win_trades,
                "execution_time_ms": (time.time() - start_time) * 1000
            }
            
            self.logger.info(f"Backtest completed for {symbol}: Returns={returns:.2f}%, Win Rate={win_rate:.2f}%")
            self.logger.timing("backtest.execution_time_ms", result["execution_time_ms"])
            return result
        
        except Exception as e:
            self.logger.error(f"Backtest error for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e), "execution_time_ms": (time.time() - start_time) * 1000}

    def run_parallel_backtest(self, symbols: List[str], timeframe: str = "1d", start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run backtests for multiple stocks in parallel.

        Args:
            symbols: List of stock symbols
            timeframe: Data timeframe
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            List of backtest results
        """
        start_time = datetime.time()
        self.logger.info(f"Running parallel backtest for {len(symbols)} symbols")
        
        try:
            with Pool(processes=os.cpu_count()) as pool:
                results = pool.starmap(
                    self.run_backtest,
                    [(symbol, timeframe, start_date, end_date) for symbol in symbols]
                )
            
            total_duration = (datetime.time() - start_time) * 1000
            self.logger.timing("backtest.parallel_execution_time_ms", total_duration)
            self.logger.info(f"Parallel backtest completed for {len(symbols)} symbols")
            return results
        
        except Exception as e:
            self.logger.error(f"Parallel backtest error: {e}")
            return [{"symbol": symbol, "error": str(e)} for symbol in symbols]
