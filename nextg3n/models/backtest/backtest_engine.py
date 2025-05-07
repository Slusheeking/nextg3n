"""
Backtest Engine for NextG3N Trading System

This module implements the backtesting model using Backtrader, validating trading strategies
based on predictions from the ForecastModel and LLM-generated strategies via OpenRouter.
It supports the PredictorAgent in evaluating strategy performance before live trading.
"""

import os
import logging
import time
import pandas as pd
import backtrader as bt
from typing import Dict, Any, List, Optional
from multiprocessing import Pool
from datetime import datetime
import numpy as np
import aiohttp
import json
import asyncio
from kafka import KafkaProducer
from dotenv import load_dotenv

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

# Service imports (assumed to exist)
from services.market_data_service import MarketDataService

# Model imports (assumed to exist)
from models.forecast.forecast_model import ForecastModel

class TFTStrategy(bt.Strategy):
    """
    Backtrader strategy driven by TFT predictions and LLM-generated parameters.
    """
    params = (
        ("size", 100),  # Number of shares per trade
        ("stop_loss", 0.05),  # 5% stop-loss
        ("lookback", 30),  # Lookback period for predictions
        ("confidence_threshold", 0.6),  # Minimum prediction confidence
        ("llm_params", None),  # LLM-generated parameters (e.g., custom indicators)
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

        # Apply LLM-generated parameters if provided
        if self.params.llm_params:
            self.logger.info(f"Applying LLM-generated parameters: {self.params.llm_params}")
            for key, value in self.params.llm_params.items():
                if key in self.params._getkeys():
                    self.params._set(key, value)

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
            
            if prediction["confidence"] < self.params.confidence_threshold:
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
                elif prediction["confidence"] > self.params.confidence_threshold + 0.1:
                    self.order = self.sell(size=self.position.size)
                    self.logger.info(f"SELL ORDER: {self.data._name}, Size={self.position.size}")
        
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")

class BacktestEngine:
    """
    Backtesting model for NextG3N, using Backtrader to validate trading strategies
    with LLM-generated enhancements via OpenRouter.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BacktestEngine.

        Args:
            config: Configuration dictionary
        """
        load_dotenv()
        self.logger = MetricsLogger(component_name="backtest_engine")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        self.config = config
        self.data_service = MarketDataService(config)
        self.initial_cash = config.get("backtest", {}).get("initial_cash", 100000)
        self.commission = config.get("backtest", {}).get("commission", 0.001)
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=config.get("kafka", {}).get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.logger.info("BacktestEngine initialized")

    async def get_llm_strategy(self, backtest_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate or refine strategy parameters using an LLM via OpenRouter.

        Args:
            backtest_results: Previous backtest results (optional, for refinement)

        Returns:
            Dictionary of LLM-generated strategy parameters
        """
        try:
            prompt = "Generate trading strategy parameters for a Backtrader strategy. Parameters include 'size' (int, shares per trade), 'stop_loss' (float, 0.01-0.1), 'lookback' (int, 10-50), and 'confidence_threshold' (float, 0.5-0.9). "
            if backtest_results:
                prompt += f"Refine based on previous backtest: Returns={backtest_results.get('returns', 0):.2f}%, Win Rate={backtest_results.get('win_rate', 0):.2f}%, Sharpe Ratio={backtest_results.get('sharpe_ratio', 0):.2f}, Max Drawdown={backtest_results.get('max_drawdown', 0):.2f}%."
            else:
                prompt += "Provide a balanced strategy for a stock trading system."

            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"}
                payload = {
                    "model": self.config.get("llm", {}).get("model", "openai/gpt-4"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200
                }
                async with session.post(
                    self.config.get("llm", {}).get("base_url", "https://openrouter.ai/api/v1") + "/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        self.logger.error("Failed to get LLM strategy")
                        return {}
                    result = await response.json()
                    llm_response = result["choices"][0]["message"]["content"]
                    # Parse LLM response (assume JSON-like output)
                    try:
                        params = json.loads(llm_response)
                        return {
                            "size": params.get("size", 100),
                            "stop_loss": params.get("stop_loss", 0.05),
                            "lookback": params.get("lookback", 30),
                            "confidence_threshold": params.get("confidence_threshold", 0.6)
                        }
                    except json.JSONDecodeError:
                        self.logger.warning("LLM response not JSON; using default parameters")
                        return {
                            "size": 100,
                            "stop_loss": 0.05,
                            "lookback": 30,
                            "confidence_threshold": 0.6
                        }
        except Exception as e:
            self.logger.error(f"LLM strategy generation error: {e}")
            return {}

    async def run_backtest(self, symbol: str, timeframe: str = "1d", start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a backtest for a single stock with LLM-generated strategy parameters.

        Args:
            symbol: Stock symbol
            timeframe: Data timeframe (e.g., '1d')
            start_date: Start date (YYYY-MM-DD, optional)
            end_date: End date (YYYY-MM-DD, optional)

        Returns:
            Backtest results (returns, win rate, Sharpe ratio, etc.)
        """
        start_time = time.time()
        operation_id = f"backtest_{symbol}_{int(start_time)}"
        self.logger.info(f"Running backtest for {symbol}, timeframe={timeframe}, Operation: {operation_id}")
        
        try:
            # Get LLM-generated strategy parameters
            llm_params = await self.get_llm_strategy()
            
            # Fetch historical data
            data = self.data_service.get_historical_data(symbol, timeframe, limit=1000)
            if not data.get("bars"):
                self.logger.error(f"No data for {symbol}")
                return {"symbol": symbol, "error": "No data available", "operation_id": operation_id}
            
            # Prepare data for Backtrader
            df = pd.DataFrame(data["bars"])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Initialize Backtrader
            cerebro = bt.Cerebro()
            cerebro.addstrategy(TFTStrategy, llm_params=llm_params)
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
                "execution_time_ms": (time.time() - start_time) * 1000,
                "operation_id": operation_id,
                "llm_params": llm_params
            }
            
            # Publish to Kafka
            self.kafka_producer.send(
                f"{self.config.get('kafka', {}).get('topic_prefix', 'nextg3n-')}backtest-events",
                {"event": "backtest_completed", "data": result}
            )
            
            self.logger.info(f"Backtest completed for {symbol}: Returns={returns:.2f}%, Win Rate={win_rate:.2f}%")
            self.logger.timing("backtest.execution_time_ms", result["execution_time_ms"])
            return result
        
        except Exception as e:
            self.logger.error(f"Backtest error for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "execution_time_ms": (time.time() - start_time) * 1000,
                "operation_id": operation_id
            }

    async def run_parallel_backtest(self, symbols: List[str], timeframe: str = "1d", start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run backtests for multiple stocks in parallel with LLM-generated strategies.

        Args:
            symbols: List of stock symbols
            timeframe: Data timeframe
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            List of backtest results
        """
        start_time = time.time()
        operation_id = f"parallel_backtest_{int(start_time)}"
        self.logger.info(f"Running parallel backtest for {len(symbols)} symbols, Operation: {operation_id}")
        
        try:
            # Run backtests asynchronously
            tasks = [self.run_backtest(symbol, timeframe, start_date, end_date) for symbol in symbols]
            results = await asyncio.gather(*tasks)
            
            total_duration = (time.time() - start_time) * 1000
            self.logger.timing("backtest.parallel_execution_time_ms", total_duration)
            self.logger.info(f"Parallel backtest completed for {len(symbols)} symbols")
            
            # Publish summary to Kafka
            summary = {
                "operation_id": operation_id,
                "symbols": symbols,
                "total_results": len(results),
                "successful": sum(1 for r in results if "error" not in r),
                "execution_time_ms": total_duration
            }
            self.kafka_producer.send(
                f"{self.config.get('kafka', {}).get('topic_prefix', 'nextg3n-')}backtest-events",
                {"event": "parallel_backtest_completed", "data": summary}
            )
            
            return results
        
        except Exception as e:
            self.logger.error(f"Parallel backtest error: {e}")
            return [{"symbol": symbol, "error": str(e), "operation_id": operation_id} for symbol in symbols]