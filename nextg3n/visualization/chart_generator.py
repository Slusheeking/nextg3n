"""
Chart Generator for NextG3N Trading System

This module implements the ChartGenerator class, creating visualizations for stock prices,
technical indicators, sentiment trends, and trade performance. It supports the TradeDashboard
in TradeFlowOrchestrator for monitoring and analysis.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

# Kafka imports
from kafka import KafkaProducer

class ChartGenerator:
    """
    Class for generating visualizations in the NextG3N system.
    Supports the TradeDashboard for monitoring and analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ChartGenerator with configuration and visualization settings.

        Args:
            config: Configuration dictionary with visualization and Kafka settings
        """
        init_start_time = datetime.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="chart_generator")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.viz_config = config.get("visualization", {}).get("chart_generator", {})
        self.kafka_config = config.get("kafka", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize visualization parameters
        self.output_dir = self.viz_config.get("output_dir", "./charts")
        os.makedirs(self.output_dir, exist_ok=True)
        self.figure_size = tuple(self.viz_config.get("figure_size", [12, 8]))
        self.style = self.viz_config.get("style", "seaborn")
        
        # Set Matplotlib style
        plt.style.use(self.style)
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        init_duration = (datetime.time() - init_start_time) * 1000
        self.logger.timing("chart_generator.initialization_time_ms", init_duration)
        self.logger.info("ChartGenerator initialized")

    async def generate_price_chart(
        self,
        symbol: str,
        data: Dict[str, Any],
        indicators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a stock price chart with OHLCV and optional technical indicators.

        Args:
            symbol: Stock symbol
            data: Dictionary containing OHLCV data (e.g., from MarketDataService)
            indicators: Optional list of technical indicators to plot (e.g., ['rsi', 'macd'])

        Returns:
            Dictionary containing chart file path and metadata
        """
        start_time = datetime.time()
        operation_id = f"price_chart_{int(start_time)}"
        self.logger.info(f"Generating price chart for {symbol} - Operation: {operation_id}")

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                
                # Generate chart
                chart_path = await loop.run_in_executor(
                    self.executor,
                    lambda: self._generate_price_chart(symbol, data, indicators)
                )

                result = {
                    "success": True,
                    "symbol": symbol,
                    "chart_type": "price",
                    "chart_path": chart_path,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}visualization_events",
                    {"event": "price_chart_generated", "data": result}
                )

                duration = (datetime.time() - start_time) * 1000
                self.logger.timing("chart_generator.generate_price_chart_time_ms", duration)
                self.logger.info(f"Price chart generated for {symbol}: {chart_path}")
                self.logger.counter("chart_generator.charts_generated", 1)
                return result

        except Exception as e:
            self.logger.error(f"Error generating price chart for {symbol}: {e}")
            self.logger.counter("chart_generator.chart_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "chart_type": "price",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def _generate_price_chart(self, symbol: str, data: Dict[str, Any], indicators: Optional[List[str]]) -> str:
        """
        Internal method to generate a stock price chart.

        Args:
            symbol: Stock symbol
            data: OHLCV data dictionary
            indicators: Optional technical indicators to plot

        Returns:
            File path of the generated chart
        """
        try:
            # Process data
            bars = data.get("bars", [])
            df = pd.DataFrame(bars)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            
            # Plot OHLC
            ax1.plot(df.index, df["close"], label="Close Price", color="blue")
            ax1.set_title(f"{symbol} Price Chart")
            ax1.set_ylabel("Price ($)")
            ax1.legend()
            ax1.grid(True)
            
            # Plot volume
            ax2.bar(df.index, df["volume"], color="gray", alpha=0.5)
            ax2.set_ylabel("Volume")
            ax2.set_xlabel("Date")
            ax2.grid(True)
            
            # Format x-axis dates
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            
            # Plot technical indicators if specified
            if indicators:
                technical_data = data.get("technical_indicators", {})
                if "rsi" in indicators and "rsi" in technical_data:
                    ax_rsi = ax1.twinx()
                    ax_rsi.plot(df.index, [technical_data.get("rsi", 50.0)] * len(df), label="RSI", color="orange")
                    ax_rsi.set_ylabel("RSI")
                    ax_rsi.legend(loc="upper right")
                if "macd" in indicators and "macd" in technical_data:
                    ax_macd = ax1.twinx()
                    ax_macd.plot(df.index, [technical_data.get("macd", 0.0)] * len(df), label="MACD", color="green")
                    ax_macd.set_ylabel("MACD")
                    ax_macd.legend(loc="lower right")
            
            # Save chart
            chart_path = os.path.join(self.output_dir, f"{symbol}_price_{int(start_time)}.png")
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close(fig)
            
            return chart_path
        
        except Exception as e:
            self.logger.error(f"Error in _generate_price_chart: {e}")
            raise

    async def generate_sentiment_chart(
        self,
        symbol: str,
        sentiment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a sentiment score trend chart.

        Args:
            symbol: Stock symbol
            sentiment_data: Dictionary containing sentiment scores (e.g., from SentimentModel)

        Returns:
            Dictionary containing chart file path and metadata
        """
        start_time = datetime.time()
        operation_id = f"sentiment_chart_{int(start_time)}"
        self.logger.info(f"Generating sentiment chart for {symbol} - Operation: {operation_id}")

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                
                # Generate chart
                chart_path = await loop.run_in_executor(
                    self.executor,
                    lambda: self._generate_sentiment_chart(symbol, sentiment_data)
                )

                result = {
                    "success": True,
                    "symbol": symbol,
                    "chart_type": "sentiment",
                    "chart_path": chart_path,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}visualization_events",
                    {"event": "sentiment_chart_generated", "data": result}
                )

                duration = (datetime.time() - start_time) * 1000
                self.logger.timing("chart_generator.generate_sentiment_chart_time_ms", duration)
                self.logger.info(f"Sentiment chart generated for {symbol}: {chart_path}")
                self.logger.counter("chart_generator.charts_generated", 1)
                return result

        except Exception as e:
            self.logger.error(f"Error generating sentiment chart for {symbol}: {e}")
            self.logger.counter("chart_generator.chart_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "chart_type": "sentiment",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def _generate_sentiment_chart(self, symbol: str, sentiment_data: Dict[str, Any]) -> str:
        """
        Internal method to generate a sentiment score trend chart.

        Args:
            symbol: Stock symbol
            sentiment_data: Sentiment data dictionary

        Returns:
            File path of the generated chart
        """
        try:
            # Process data
            results = sentiment_data.get("results", [])
            df = pd.DataFrame(results)
            if "timestamp" not in df:
                df["timestamp"] = [datetime.utcnow()] * len(df)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Plot sentiment scores
            ax.plot(df.index, df["sentiment_score"], label="Sentiment Score", color="purple")
            ax.set_title(f"{symbol} Sentiment Trend")
            ax.set_ylabel("Sentiment Score")
            ax.set_xlabel("Date")
            ax.legend()
            ax.grid(True)
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            
            # Save chart
            chart_path = os.path.join(self.output_dir, f"{symbol}_sentiment_{int(start_time)}.png")
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close(fig)
            
            return chart_path
        
        except Exception as e:
            self.logger.error(f"Error in _generate_sentiment_chart: {e}")
            raise

    async def generate_performance_chart(
        self,
        trade_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a trade performance chart (e.g., cumulative P&L, win rate).

        Args:
            trade_history: List of trade records (e.g., from TradeExecutor)

        Returns:
            Dictionary containing chart file path and metadata
        """
        start_time = datetime.time()
        operation_id = f"performance_chart_{int(start_time)}"
        self.logger.info(f"Generating performance chart - Operation: {operation_id}")

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                
                # Generate chart
                chart_path = await loop.run_in_executor(
                    self.executor,
                    lambda: self._generate_performance_chart(trade_history)
                )

                result = {
                    "success": True,
                    "chart_type": "performance",
                    "chart_path": chart_path,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}visualization_events",
                    {"event": "performance_chart_generated", "data": result}
                )

                duration = (datetime.time() - start_time) * 1000
                self.logger.timing("chart_generator.generate_performance_chart_time_ms", duration)
                self.logger.info(f"Performance chart generated: {chart_path}")
                self.logger.counter("chart_generator.charts_generated", 1)
                return result

        except Exception as e:
            self.logger.error(f"Error generating performance chart: {e}")
            self.logger.counter("chart_generator.chart_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "chart_type": "performance",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def _generate_performance_chart(self, trade_history: List[Dict[str, Any]]) -> str:
        """
        Internal method to generate a trade performance chart.

        Args:
            trade_history: List of trade records

        Returns:
            File path of the generated chart
        """
        try:
            # Process data
            df = pd.DataFrame(trade_history)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            
            # Calculate cumulative P&L (placeholder; assumes price and quantity)
            df["pnl"] = df.apply(
                lambda row: (row["price"] * row["quantity"] * (-1 if row["action"] == "buy" else 1)), axis=1
            )
            df["cumulative_pnl"] = df["pnl"].cumsum()
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Plot cumulative P&L
            ax.plot(df.index, df["cumulative_pnl"], label="Cumulative P&L", color="blue")
            ax.set_title("Trade Performance")
            ax.set_ylabel("Cumulative P&L ($)")
            ax.set_xlabel("Date")
            ax.legend()
            ax.grid(True)
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            
            # Save chart
            chart_path = os.path.join(self.output_dir, f"performance_{int(start_time)}.png")
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close(fig)
            
            return chart_path
        
        except Exception as e:
            self.logger.error(f"Error in _generate_performance_chart: {e}")
            raise

    def shutdown(self):
        """
        Shutdown the ChartGenerator and close resources.
        """
        self.logger.info("Shutting down ChartGenerator")
        self.executor.shutdown(wait=True)
        self.producer.close()
        plt.close('all')
        self.logger.info("Kafka producer and Matplotlib closed")