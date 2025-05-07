"""
Trade Service for NextG3N Trading System

This module implements the TradeService, handling trading operations via the Alpaca API.
It provides tools for submitting orders, retrieving order status, fetching account information,
and managing portfolio positions, integrated with the TradeAgent in TradeFlowOrchestrator.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import aiohttp
from kafka import KafkaProducer

# Alpaca API imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    HAVE_ALPACA = True
except ImportError:
    HAVE_ALPACA = False
    logging.warning("alpaca-trade-api not installed. Trading features will be unavailable.")

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

class TradeService:
    """
    Service for handling trading operations via Alpaca API in the NextG3N system.
    Provides tools for order submission, status checking, account info, and position management.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TradeService with configuration and Alpaca client.

        Args:
            config: Configuration dictionary with Alpaca and Kafka settings
        """
        init_start_time = time.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="trade_service")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.alpaca_config = config.get("services", {}).get("trade", {}).get("alpaca", {})
        self.kafka_config = config.get("kafka", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize Alpaca client
        self.trading_client = None
        if HAVE_ALPACA:
            self._initialize_alpaca_client()
        else:
            self.logger.error("Alpaca library not found. Trading features disabled.")
        
        # Initialize thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("trade_service.initialization_time_ms", init_duration)
        self.logger.info("TradeService initialized")

    def _initialize_alpaca_client(self):
        """
        Initialize the Alpaca trading client.
        """
        try:
            api_key = self.alpaca_config.get("api_key") or os.environ.get("ALPACA_API_KEY")
            api_secret = self.alpaca_config.get("api_secret") or os.environ.get("ALPACA_SECRET_KEY")
            paper_trading = self.alpaca_config.get("paper_trading", True)

            if not api_key or not api_secret:
                self.logger.error("Alpaca API credentials missing")
                raise ValueError("Alpaca API credentials not provided")

            self.trading_client = TradingClient(api_key, api_secret, paper=paper_trading)
            
            # Test connection
            account = self.trading_client.get_account()
            self.logger.info(f"Connected to Alpaca account: {account.id} (Paper: {paper_trading})")
            self.logger.gauge("trade_service.account_equity", float(account.equity))
            self.logger.gauge("trade_service.account_buying_power", float(account.buying_power))
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca client: {e}")
            self.trading_client = None
            raise

    async def submit_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "market",
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Submit a trade order to Alpaca.

        Args:
            symbol: Stock symbol
            action: Trade action ("buy" or "sell")
            quantity: Number of shares
            order_type: Order type ("market" or "limit")
            price: Limit price (for limit orders)

        Returns:
            Order result dictionary
        """
        start_time = time.time()
        operation_id = f"order_{int(start_time)}"
        self.logger.info(f"Submitting {order_type} order: {action} {quantity} shares of {symbol} - ID: {operation_id}")

        if not self.trading_client:
            self.logger.error("Trading client not initialized")
            return {"success": False, "error": "Trading client not initialized", "operation_id": operation_id}

        try:
            order_side = OrderSide.BUY if action.lower() == "buy" else OrderSide.SELL
            time_in_force = TimeInForce.DAY

            if order_type.lower() == "market":
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=time_in_force
                )
            elif order_type.lower() == "limit" and price is not None:
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    limit_price=price,
                    time_in_force=time_in_force
                )
            else:
                raise ValueError(f"Invalid order type or missing price for limit order: {order_type}")

            async with aiohttp.ClientSession() as session:
                # Execute order in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                order = await loop.run_in_executor(
                    self.executor,
                    lambda: self.trading_client.submit_order(order_request)
                )

            result = {
                "success": True,
                "order_id": order.id,
                "status": order.status,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "order_type": order_type,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}trade_events",
                {"event": "order_submitted", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("trade_service.submit_order_time_ms", duration)
            self.logger.info(f"Order submitted: {result['order_id']} for {symbol}")
            self.logger.counter("trade_service.orders_submitted", 1)
            return result

        except Exception as e:
            self.logger.error(f"Error submitting order for {symbol}: {e}")
            self.logger.counter("trade_service.order_errors", 1)
            result = {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}trade_events",
                {"event": "order_failed", "data": result}
            )
            return result

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Retrieve the status of a specific order.

        Args:
            order_id: Order ID

        Returns:
            Order status dictionary
        """
        start_time = time.time()
        operation_id = f"status_{int(start_time)}"
        self.logger.info(f"Fetching order status for ID: {order_id} - Operation: {operation_id}")

        if not self.trading_client:
            self.logger.error("Trading client not initialized")
            return {"success": False, "error": "Trading client not initialized", "operation_id": operation_id}

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                order = await loop.run_in_executor(
                    self.executor,
                    lambda: self.trading_client.get_order_by_id(order_id)
                )

            result = {
                "success": True,
                "order_id": order.id,
                "status": order.status,
                "symbol": order.symbol,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            duration = (time.time() - start_time) * 1000
            self.logger.timing("trade_service.get_order_status_time_ms", duration)
            self.logger.info(f"Order status retrieved: {order_id}, Status={order.status}")
            self.logger.counter("trade_service.order_status_checks", 1)
            return result

        except Exception as e:
            self.logger.error(f"Error fetching order status for {order_id}: {e}")
            self.logger.counter("trade_service.order_status_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "order_id": order_id,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Fetch account information (e.g., buying power, equity).

        Returns:
            Account information dictionary
        """
        start_time = time.time()
        operation_id = f"account_{int(start_time)}"
        self.logger.info(f"Fetching account info - Operation: {operation_id}")

        if not self.trading_client:
            self.logger.error("Trading client not initialized")
            return {"success": False, "error": "Trading client not initialized", "operation_id": operation_id}

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                account = await loop.run_in_executor(
                    self.executor,
                    lambda: self.trading_client.get_account()
                )

            result = {
                "success": True,
                "account_id": account.id,
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            self.logger.gauge("trade_service.account_equity", result["equity"])
            self.logger.gauge("trade_service.account_buying_power", result["buying_power"])
            
            duration = (time.time() - start_time) * 1000
            self.logger.timing("trade_service.get_account_info_time_ms", duration)
            self.logger.info(f"Account info retrieved: ID={account.id}")
            self.logger.counter("trade_service.account_info_requests", 1)
            return result

        except Exception as e:
            self.logger.error(f"Error fetching account info: {e}")
            self.logger.counter("trade_service.account_info_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Retrieve current portfolio positions.

        Returns:
            List of position dictionaries
        """
        start_time = time.time()
        operation_id = f"positions_{int(start_time)}"
        self.logger.info(f"Fetching portfolio positions - Operation: {operation_id}")

        if not self.trading_client:
            self.logger.error("Trading client not initialized")
            return [{"success": False, "error": "Trading client not initialized", "operation_id": operation_id}]

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                positions = await loop.run_in_executor(
                    self.executor,
                    lambda: self.trading_client.get_all_positions()
                )

            result = [
                {
                    "success": True,
                    "symbol": pos.symbol,
                    "quantity": float(pos.qty),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                for pos in positions
            ]

            duration = (time.time() - start_time) * 1000
            self.logger.timing("trade_service.get_positions_time_ms", duration)
            self.logger.info(f"Retrieved {len(result)} positions")
            self.logger.counter("trade_service.positions_requests", 1)
            return result

        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            self.logger.counter("trade_service.positions_errors", 1)
            return [
                {
                    "success": False,
                    "error": str(e),
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            ]

    def shutdown(self):
        """
        Shutdown the service and close resources.
        """
        self.logger.info("Shutting down TradeService")
        self.executor.shutdown(wait=True)
        self.producer.close()
        self.logger.info("Kafka producer closed")
