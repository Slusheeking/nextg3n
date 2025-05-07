"""
Market Data Service for NextG3N Trading System

This module implements the MarketDataService, fetching historical and real-time market data
from Polygon.io and Alpaca, and calculating technical indicators. It provides tools for data
retrieval and analysis, integrated with the StockPickerAgent, PredictorAgent, and TradeAgent
in TradeFlowOrchestrator.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
from kafka import KafkaProducer
import pandas as pd
import websocket
import threading
import ta

# Alpaca API imports
try:
    from alpaca.data import StockDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.enums import DataFeed
    HAVE_ALPACA = True
except ImportError:
    HAVE_ALPACA = False
    logging.warning("alpaca-trade-api not installed. Alpaca data features will be unavailable.")

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

class MarketDataService:
    """
    Service for fetching and analyzing market data in the NextG3N system.
    Provides tools for retrieving historical/real-time data and calculating technical indicators.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MarketDataService with configuration and API settings.

        Args:
            config: Configuration dictionary with Polygon.io, Alpaca, and Kafka settings
        """
        init_start_time = time.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="market_data_service")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.polygon_config = config.get("services", {}).get("market_data", {}).get("polygon", {})
        self.alpaca_config = config.get("services", {}).get("market_data", {}).get("alpaca", {})
        self.kafka_config = config.get("kafka", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize API settings
        self.polygon_api_key = self.polygon_config.get("api_key") or os.environ.get("POLYGON_API_KEY")
        self.polygon_base_url = self.polygon_config.get("base_url", "https://api.polygon.io/v2")
        self.polygon_rate_limit_delay = self.polygon_config.get("rate_limit_delay", 12.0)  # 5 requests/minute
        self.alpaca_api_key = self.alpaca_config.get("api_key") or os.environ.get("ALPACA_API_KEY")
        self.alpaca_api_secret = self.alpaca_config.get("api_secret") or os.environ.get("ALPACA_SECRET_KEY")
        self.alpaca_enabled = self.alpaca_config.get("enabled", True) and HAVE_ALPACA
        
        if not self.polygon_api_key:
            self.logger.error("Polygon API key missing")
            raise ValueError("Polygon API key not provided")
        
        if not self.alpaca_enabled and not HAVE_ALPACA:
            self.logger.warning("Alpaca disabled due to missing alpaca-trade-api library")
        elif not (self.alpaca_api_key and self.alpaca_api_secret):
            self.logger.warning("Alpaca API credentials missing; disabling Alpaca features")
            self.alpaca_enabled = False

        # Initialize Alpaca client
        self.alpaca_client = None
        if self.alpaca_enabled:
            self.alpaca_client = StockDataClient(self.alpaca_api_key, self.alpaca_api_secret)
        
        # Initialize WebSocket for real-time data
        self.websocket = None
        self.websocket_thread = None
        self.realtime_data = {}
        self._initialize_websocket()

        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("market_data_service.initialization_time_ms", init_duration)
        self.logger.info("MarketDataService initialized")

    def _initialize_websocket(self):
        """
        Initialize WebSocket for real-time data from Alpaca.
        """
        if not self.alpaca_enabled:
            self.logger.warning("WebSocket disabled due to Alpaca configuration")
            return
        
        try:
            ws_url = "wss://stream.data.alpaca.markets/v2/iex"
            self.websocket = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_websocket_message,
                on_error=self._on_websocket_error,
                on_close=self._on_websocket_close
            )
            self.websocket_thread = threading.Thread(target=self.websocket.run_forever)
            self.websocket_thread.daemon = True
            self.websocket_thread.start()
            self.logger.info("WebSocket initialized for Alpaca real-time data")
            
            # Authenticate WebSocket
            self.websocket.send(json.dumps({
                "action": "auth",
                "key": self.alpaca_api_key,
                "secret": self.alpaca_api_secret
            }))
        
        except Exception as e:
            self.logger.error(f"Failed to initialize WebSocket: {e}")
            self.websocket = None

    def _on_websocket_message(self, ws, message):
        """
        Handle incoming WebSocket messages.
        """
        try:
            data = json.loads(message)
            if isinstance(data, list):
                for item in data:
                    if item.get("T") == "q":  # Quote message
                        symbol = item.get("S")
                        price = item.get("bp")  # Bid price
                        self.realtime_data[symbol] = {
                            "price": price,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        self.logger.debug(f"Received real-time data for {symbol}: Price={price}")
        except Exception as e:
            self.logger.error(f"WebSocket message error: {e}")

    def _on_websocket_error(self, ws, error):
        """
        Handle WebSocket errors.
        """
        self.logger.error(f"WebSocket error: {error}")

    def _on_websocket_close(self, ws, close_status_code, close_msg):
        """
        Handle WebSocket closure.
        """
        self.logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Fetch historical OHLCV data for a stock from Polygon.io or Alpaca.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            limit: Maximum number of data points

        Returns:
            Dictionary containing historical data
        """
        start_time = time.time()
        operation_id = f"historical_{int(start_time)}"
        self.logger.info(f"Fetching historical data for {symbol}, timeframe={timeframe} - Operation: {operation_id}")

        try:
            bars = []
            
            # Try Polygon.io first
            async with aiohttp.ClientSession() as session:
                url = f"{self.polygon_base_url}/aggs/ticker/{symbol}/range/1/{timeframe}/from/2020-01-01/to/{datetime.utcnow().strftime('%Y-%m-%d')}"
                headers = {"Authorization": f"Bearer {self.polygon_api_key}"}
                params = {"limit": limit}
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        bars = [
                            {
                                "timestamp": datetime.utcfromtimestamp(item["t"] / 1000).isoformat(),
                                "open": float(item["o"]),
                                "high": float(item["h"]),
                                "low": float(item["l"]),
                                "close": float(item["c"]),
                                "volume": int(item["v"])
                            }
                            for item in data.get("results", [])[:limit]
                        ]
                    else:
                        self.logger.warning(f"Polygon API error for {symbol}: {response.status}")
                
                # Respect Polygon rate limits
                await asyncio.sleep(self.polygon_rate_limit_delay)
            
            # Fallback to Alpaca if needed
            if not bars and self.alpaca_enabled:
                async with aiohttp.ClientSession() as session:
                    loop = asyncio.get_event_loop()
                    request = StockBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=timeframe,
                        limit=limit
                    )
                    bars_data = await loop.run_in_executor(
                        self.executor,
                        lambda: self.alpaca_client.get_stock_bars(request)
                    )
                    bars = [
                        {
                            "timestamp": bar.timestamp.isoformat(),
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": int(bar.volume)
                        }
                        for bar in bars_data[symbol]
                    ]

            if not bars:
                self.logger.warning(f"No data found for {symbol}")
                return {
                    "success": False,
                    "error": "No data available",
                    "symbol": symbol,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

            result = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "bars": bars,
                "count": len(bars),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}market_data",
                {"event": "historical_data_fetched", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("market_data_service.get_historical_data_time_ms", duration)
            self.logger.info(f"Fetched {result['count']} historical data points for {symbol}")
            self.logger.counter("market_data_service.historical_data_fetched", result['count'])
            return result

        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            self.logger.counter("market_data_service.historical_data_errors", 1)
            result = {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}market_data",
                {"event": "historical_data_fetch_failed", "data": result}
            )
            return result

    async def get_realtime_data(
        self,
        symbol: str,
        timeframe: str = "1m"
    ) -> Dict[str, Any]:
        """
        Fetch real-time price data for a stock from Alpaca WebSocket or Polygon.io REST.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe (e.g., '1m')

        Returns:
            Dictionary containing real-time data
        """
        start_time = time.time()
        operation_id = f"realtime_{int(start_time)}"
        self.logger.info(f"Fetching real-time data for {symbol}, timeframe={timeframe} - Operation: {operation_id}")

        try:
            # Try WebSocket first (Alpaca)
            if self.websocket and symbol in self.realtime_data:
                data = self.realtime_data[symbol]
                result = {
                    "success": True,
                    "symbol": symbol,
                    "price": data["price"],
                    "timestamp": data["timestamp"],
                    "source": "alpaca_websocket",
                    "operation_id": operation_id,
                    "timestamp_fetch": datetime.utcnow().isoformat()
                }
                self.logger.info(f"Retrieved real-time data for {symbol} from WebSocket")
                return result
            
            # Fallback to Polygon.io REST
            async with aiohttp.ClientSession() as session:
                url = f"{self.polygon_base_url}/last/trade/{symbol}"
                headers = {"Authorization": f"Bearer {self.polygon_api_key}"}
                
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        error = f"Polygon API error: {response.status} - {await response.text()}"
                        self.logger.warning(error)
                        return {
                            "success": False,
                            "error": error,
                            "symbol": symbol,
                            "operation_id": operation_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    
                    data = await response.json()
                    result = {
                        "success": True,
                        "symbol": symbol,
                        "price": float(data.get("results", {}).get("p", 0)),
                        "timestamp": datetime.utcfromtimestamp(data.get("results", {}).get("t", 0) / 1000).isoformat(),
                        "source": "polygon_rest",
                        "operation_id": operation_id,
                        "timestamp_fetch": datetime.utcnow().isoformat()
                    }
                
                # Respect Polygon rate limits
                await asyncio.sleep(self.polygon_rate_limit_delay)

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}market_data",
                {"event": "realtime_data_fetched", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("market_data_service.get_realtime_data_time_ms", duration)
            self.logger.info(f"Fetched real-time data for {symbol}: Price={result['price']}")
            self.logger.counter("market_data_service.realtime_data_fetched", 1)
            return result

        except Exception as e:
            self.logger.error(f"Error fetching real-time data for {symbol}: {e}")
            self.logger.counter("market_data_service.realtime_data_errors", 1)
            result = {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}market_data",
                {"event": "realtime_data_fetch_failed", "data": result}
            )
            return result

    async def get_technical_indicators(
        self,
        symbol: str,
        indicators: List[str] = ["rsi", "macd"]
    ) -> Dict[str, Any]:
        """
        Calculate technical indicators for a stock.

        Args:
            symbol: Stock symbol
            indicators: List of indicators to calculate (e.g., ["rsi", "macd"])

        Returns:
            Dictionary containing indicator values
        """
        start_time = time.time()
        operation_id = f"indicators_{int(start_time)}"
        self.logger.info(f"Calculating technical indicators for {symbol}: {indicators} - Operation: {operation_id}")

        try:
            # Fetch historical data for indicators
            data_result = await self.get_historical_data(symbol, timeframe="1d", limit=100)
            
            if not data_result["success"] or not data_result["bars"]:
                self.logger.warning(f"No data for {symbol}")
                return {
                    "success": False,
                    "error": "No data available",
                    "symbol": symbol,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

            # Prepare data for indicators
            df = pd.DataFrame(data_result["bars"])
            df['close'] = df['close'].astype(float)
            
            result = {
                "success": True,
                "symbol": symbol,
                "indicators": {},
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Calculate indicators
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                for indicator in indicators:
                    if indicator.lower() == "rsi":
                        rsi = await loop.run_in_executor(
                            self.executor,
                            lambda: ta.momentum.RSIIndicator(df['close']).rsi().iloc[-1]
                        )
                        result["indicators"]["rsi"] = float(rsi) if not pd.isna(rsi) else None
                    elif indicator.lower() == "macd":
                        macd = await loop.run_in_executor(
                            self.executor,
                            lambda: ta.trend.MACD(df['close']).macd_diff().iloc[-1]
                        )
                        result["indicators"]["macd"] = float(macd) if not pd.isna(macd) else None
                    else:
                        self.logger.warning(f"Unsupported indicator: {indicator}")

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}market_data",
                {"event": "indicators_calculated", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("market_data_service.get_technical_indicators_time_ms", duration)
            self.logger.info(f"Calculated indicators for {symbol}: {result['indicators']}")
            self.logger.counter("market_data_service.indicators_calculated", len(indicators))
            return result

        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}")
            self.logger.counter("market_data_service.indicators_errors", 1)
            result = {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}market_data",
                {"event": "indicators_calculation_failed", "data": result}
            )
            return result

    def shutdown(self):
        """
        Shutdown the service and close resources.
        """
        self.logger.info("Shutting down MarketDataService")
        self.executor.shutdown(wait=True)
        if self.websocket:
            self.websocket.close()
        self.producer.close()
        self.logger.info("Kafka producer closed")