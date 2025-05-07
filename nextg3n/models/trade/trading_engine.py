"""
High-Performance Trading Engine for NextG3N Trading System

Implements the core trading path as a single application component.
Integrates low-latency AI/ML models, direct data access, and deterministic
decision logic for efficient trade execution.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List

# Monitoring imports
from nextg3n.monitoring.metrics_logger import MetricsLogger

# Assuming direct imports for models and services based on architecture plan
from nextg3n.models.decision.decision_model import DecisionModel
from nextg3n.models.stock_ranker.stock_ranker import StockRanker
from nextg3n.services.market_data_service import MarketDataService
from nextg3n.models.trade.trade_executor import TradeExecutor


class TradingEngine:
    """
    Core trading engine handling the critical trading path.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the TradingEngine with necessary components.

        Args:
            config: System configuration dictionary.
        """
        self.metrics_logger = MetricsLogger(component_name="trading_engine")
        self.metrics_logger.setLevel(logging.WARNING)  # Reduce logging level in production
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.metrics_logger.addHandler(handler)


        self.config = config
        self.market_data_service = MarketDataService(config)
        self.decision_model = DecisionModel(config)
        self.stock_ranker = StockRanker(config)
        self.trade_executor = TradeExecutor(config)

        self.metrics_logger.info("TradingEngine initialized")

    async def run_trading_cycle(
            self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Executes a single trading cycle for the given symbols.

        Args:
            symbols: A list of stock symbols to process.

        Returns:
            A list of executed trades or recommended actions.
        """
        operation_id = f"trading_cycle_{int(asyncio.get_event_loop().time())}"
        self.metrics_logger.info(
            f"Running trading cycle - Operation: {operation_id} for symbols: {symbols}")

        executed_trades = []
        cycle_start_time = time.time()

        try:
            # Step 1: Stock Ranking (using the integrated StockRanker)
            ranked_stocks = await self.stock_ranker.rank_stocks(symbols)
            if not ranked_stocks:
                self.metrics_logger.warning(
                    "Stock ranking returned no results for symbols: %s", symbols)
                return []

            self.metrics_logger.timing(
                "trading_engine.stock_ranking_latency",
                time.time() - cycle_start_time)

            # Process top ranked stocks
            top_n = self.config.get("trading_engine", {}).get("top_n_stocks", 5)
            top_symbols = [stock["symbol"] for stock in ranked_stocks[:top_n]]
            self.metrics_logger.info(
                f"Processing top {len(top_symbols)} stocks: {top_symbols}")

            for symbol in top_symbols:
                self.metrics_logger.info(f"Analyzing symbol: {symbol}")
                symbol_start_time = time.time()

                # Step 2: Deterministic Decision Logic (integrated DecisionModel)
                trade_decision = await self.decision_model.make_decision(symbol)
                self.metrics_logger.info(
                    "Decision for %s: %s", symbol, trade_decision)

                # Step 3: Efficient Trade Execution Integration (integrated TradeExecutor)
                if trade_decision and trade_decision.get("action") in ["buy", "sell"]:
                    quantity = self._calculate_trade_quantity(
                        current_price=trade_decision.get("current_price", 0))
                    if quantity > 0:
                        order_params = {
                            "symbol": symbol,
                            "action": trade_decision["action"],
                            "quantity": quantity,
                            "order_type": trade_decision.get("order_type", self.config.get("trade", {}).get("default_order_type", "market")),
                            "price": trade_decision.get("price"),
                        }
                        self.metrics_logger.info(
                            "Placing order for %s: %s", symbol, order_params)
                        trade_start_time = time.time()
                        trade_result = await self.trade_executor.place_order(order_params)
                        trade_latency = time.time() - trade_start_time

                        if trade_result and trade_result.get("success"):
                            self.metrics_logger.info(
                                "Order placed successfully for %s: %s",
                                symbol,
                                trade_result.get("order_id"),
                            )
                            self.metrics_logger.timing(
                                "trading_engine.trade_execution_latency", trade_latency)
                            executed_trades.append(
                                {
                                    "symbol": symbol,
                                    "action": trade_decision["action"],
                                    "quantity": quantity,
                                    "order_id": trade_result.get("order_id"),
                                    "status": "placed",
                                }
                            )
                        else:
                            self.metrics_logger.error(
                                "Failed to place order for %s: %s",
                                symbol,
                                trade_result.get("error", "Unknown error"),
                            )
                    else:
                        self.metrics_logger.info(
                            "Calculated quantity is zero for %s. Skipping trade.", symbol)
                else:
                    self.metrics_logger.info(
                        "No trade decision to execute for %s.", symbol)

                self.metrics_logger.timing(
                    "trading_engine.symbol_analysis_latency",
                    time.time() - symbol_start_time)

            self.metrics_logger.info(
                "Trading cycle completed. Executed trades: %s",
                executed_trades)
            self.metrics_logger.timing(
                "trading_engine.trading_cycle_latency",
                time.time() - cycle_start_time)
            return executed_trades

        except Exception as e:
            self.metrics_logger.error(f"Error during trading cycle operation {operation_id}: {e}")
            return []

    def _calculate_trade_quantity(self, current_price: float) -> int:
        """
        Calculates the trade quantity based on risk management rules and capital.
        Args:
            current_price: The current market price of the stock.
        Returns:
            The calculated quantity to trade.
        """
        # Cache configuration values for efficiency
        notional_capital = self.config.get("trading_engine", {}).get("notional_capital", 10000)
        risk_percentage = self.config.get("trading_engine", {}).get("risk_percentage_per_trade", 0.01)

        if current_price <= 0:
            return 0

        # Calculate maximum capital to allocate for this trade based on risk
        capital_allocation = notional_capital * risk_percentage

        # Calculate potential quantity
        quantity = int(capital_allocation / current_price)

        # Ensure minimum quantity is 1 if a trade is recommended and feasible
        return max(1, quantity) if quantity > 0 else 0

    async def shutdown(self):
        """
        Shuts down the trading engine and its components.
        """
        self.metrics_logger.info("Shutting down TradingEngine")
        await self.market_data_service.shutdown()
        await self.decision_model.shutdown()
        await self.stock_ranker.shutdown()
        await self.trade_executor.shutdown()
        self.metrics_logger.info("TradingEngine shutdown complete")
