"""
Trade Executor for NextG3N Trading System

This module implements the TradeExecutor class, executing trading orders based on decisions
from the DecisionModel and interacting with the TradeService. It supports the TradeAgent in
TradeFlowOrchestrator.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from alpaca_trade_api.rest import REST, APIError

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

# Kafka imports
from kafka import KafkaProducer

class TradeExecutor:
    """
    Class for executing and tracking trading orders in the NextG3N system.
    Supports the TradeAgent in TradeFlowOrchestrator.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TradeExecutor with configuration and trading settings.

        Args:
            config: Configuration dictionary with trading and Kafka settings
        """
        init_start_time = datetime.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="trade_executor")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.trading_config = config.get("models", {}).get("trade", {})
        self.kafka_config = config.get("kafka", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize Alpaca API client
        self.alpaca = None
        self._initialize_alpaca_client()
        
        # Initialize trading parameters
        self.capital = self.trading_config.get("capital", 100000.0)  # Default $100,000
        self.max_position_size = self.trading_config.get("max_position_size", 0.05)  # 5% of capital
        self.stop_loss_percent = self.trading_config.get("stop_loss_percent", 0.02)  # 2% stop-loss
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        init_duration = (datetime.time() - init_start_time) * 1000
        self.logger.timing("trade_executor.initialization_time_ms", init_duration)
        self.logger.info("TradeExecutor initialized")

    def _initialize_alpaca_client(self):
        """
        Initialize the Alpaca API client.
        """
        try:
            api_key = self.trading_config.get("alpaca_api_key") or os.environ.get("ALPACA_API_KEY")
            api_secret = self.trading_config.get("alpaca_api_secret") or os.environ.get("ALPACA_SECRET_KEY")
            base_url = self.trading_config.get("alpaca_base_url", "https://paper-api.alpaca.markets")
            
            if not api_key or not api_secret:
                self.logger.error("Alpaca API credentials missing")
                raise ValueError("Alpaca API credentials not provided")
            
            self.alpaca = REST(api_key, api_secret, base_url)
            
            # Test connection
            self.alpaca.get_account()
            self.logger.info("Connected to Alpaca API")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca client: {e}")
            self.alpaca = None
            raise

    async def execute_trade(
        self,
        decision: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """
        Execute a trading order based on a decision.

        Args:
            decision: Decision dictionary (symbol, action, confidence)
            current_price: Current stock price

        Returns:
            Dictionary containing trade execution result
        """
        start_time = datetime.time()
        operation_id = f"execute_trade_{int(start_time)}"
        self.logger.info(f"Executing trade for {decision.get('symbol', '')} - Operation: {operation_id}")

        if not self.alpaca:
            self.logger.error("Alpaca client not initialized")
            return {
                "success": False,
                "error": "Alpaca client not initialized",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            symbol = decision.get("symbol")
            action = decision.get("action")
            confidence = decision.get("confidence", 0.5)
            
            if not symbol or action not in ["buy", "sell", "hold"]:
                self.logger.error(f"Invalid decision: {decision}")
                return {
                    "success": False,
                    "error": "Invalid decision",
                    "symbol": symbol,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                
                if action == "hold":
                    result = {
                        "success": True,
                        "symbol": symbol,
                        "action": action,
                        "order_id": None,
                        "quantity": 0,
                        "price": current_price,
                        "operation_id": operation_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    # Calculate position size
                    quantity = await loop.run_in_executor(
                        self.executor,
                        lambda: self._calculate_position_size(current_price)
                    )
                    
                    if quantity <= 0:
                        self.logger.warning(f"Insufficient capital or invalid quantity for {symbol}")
                        return {
                            "success": False,
                            "error": "Insufficient capital or invalid quantity",
                            "symbol": symbol,
                            "operation_id": operation_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    
                    # Place order
                    order = await loop.run_in_executor(
                        self.executor,
                        lambda: self.alpaca.submit_order(
                            symbol=symbol,
                            qty=quantity,
                            side=action,
                            type="market",
                            time_in_force="gtc",
                            stop_loss={"stop_price": current_price * (1 - self.stop_loss_percent) if action == "buy" else None}
                        )
                    )
                    
                    result = {
                        "success": True,
                        "symbol": symbol,
                        "action": action,
                        "order_id": order.id,
                        "quantity": quantity,
                        "price": current_price,
                        "operation_id": operation_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }

                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}trade_events",
                    {"event": "trade_executed", "data": result}
                )

                duration = (datetime.time() - start_time) * 1000
                self.logger.timing("trade_executor.execute_trade_time_ms", duration)
                self.logger.info(f"Trade executed: {action} {result['quantity']} shares of {symbol} at ${current_price:.2f}")
                self.logger.counter("trade_executor.trades_executed", 1)
                return result

        except APIError as e:
            self.logger.error(f"Alpaca API error executing trade for {symbol}: {e}")
            self.logger.counter("trade_executor.trade_errors", 1)
            return {
                "success": False,
                "error": f"Alpaca API error: {e}",
                "symbol": symbol,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
            self.logger.counter("trade_executor.trade_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def _calculate_position_size(self, current_price: float) -> int:
        """
        Calculate the number of shares to trade based on position sizing rules.

        Args:
            current_price: Current stock price

        Returns:
            Number of shares (integer)
        """
        try:
            # Fixed percentage of capital
            position_value = self.capital * self.max_position_size
            quantity = int(position_value / current_price)
            
            # Ensure quantity is positive and within risk limits
            if quantity <= 0:
                return 0
            
            # Check available capital
            account = self.alpaca.get_account()
            available_cash = float(account.cash)
            if quantity * current_price > available_cash:
                self.logger.warning("Insufficient cash for position")
                return 0
            
            return quantity
        
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0

    async def track_order(self, order_id: str) -> Dict[str, Any]:
        """
        Track the status of an executed order.

        Args:
            order_id: Order ID to track

        Returns:
            Dictionary containing order status
        """
        start_time = datetime.time()
        operation_id = f"track_order_{int(start_time)}"
        self.logger.info(f"Tracking order {order_id} - Operation: {operation_id}")

        if not self.alpaca:
            self.logger.error("Alpaca client not initialized")
            return {
                "success": False,
                "error": "Alpaca client not initialized",
                "order_id": order_id,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                
                # Retrieve order status
                order = await loop.run_in_executor(
                    self.executor,
                    lambda: self.alpaca.get_order(order_id)
                )
                
                result = {
                    "success": True,
                    "order_id": order_id,
                    "symbol": order.symbol,
                    "status": order.status,
                    "quantity": float(order.qty),
                    "filled_qty": float(order.filled_qty),
                    "side": order.side,
                    "avg_fill_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}trade_events",
                    {"event": "order_tracked", "data": result}
                )

                duration = (datetime.time() - start_time) * 1000
                self.logger.timing("trade_executor.track_order_time_ms", duration)
                self.logger.info(f"Order {order_id} status: {order.status}")
                self.logger.counter("trade_executor.orders_tracked", 1)
                return result

        except APIError as e:
            self.logger.error(f"Alpaca API error tracking order {order_id}: {e}")
            self.logger.counter("trade_executor.track_errors", 1)
            return {
                "success": False,
                "error": f"Alpaca API error: {e}",
                "order_id": order_id,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error tracking order {order_id}: {e}")
            self.logger.counter("trade_executor.track_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "order_id": order_id,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def shutdown(self):
        """
        Shutdown the TradeExecutor and close resources.
        """
        self.logger.info("Shutting down TradeExecutor")
        self.executor.shutdown(wait=True)
        self.producer.close()
        self.logger.info("Kafka producer closed")