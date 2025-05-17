#!/usr/bin/env python3
"""
IB Gateway Async Test Script

This script demonstrates how to connect to the Interactive Brokers Gateway
asynchronously using ib_insync library and retrieve real-time market data.

Usage:
    python ib_async_test.py [--port PORT] [--symbols SYMBOLS]

Options:
    --port PORT       Port to connect to IB Gateway (default: 4002)
    --symbols SYMBOLS Comma-separated list of symbols to subscribe to (default: AAPL,MSFT,AMZN)
"""

import asyncio
import argparse
import logging
import signal
import sys
from typing import List, Dict, Any
from datetime import datetime

from ib_insync import IB, Stock, Contract, util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
ib = IB()
tickers = {}
running = True


async def connect_to_ib(host: str = '127.0.0.1', port: int = 4002, client_id: int = 1) -> bool:
    """
    Connect to Interactive Brokers Gateway asynchronously.
    
    Args:
        host: IB Gateway host address
        port: IB Gateway port
        client_id: Client ID for the connection
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        logger.info(f"Connecting to IB Gateway at {host}:{port} with client ID {client_id}")
        await ib.connectAsync(host, port, clientId=client_id)
        logger.info(f"Connected to IB Gateway: {ib.isConnected()}")
        
        # Set up disconnect handler
        ib.disconnectedEvent += on_disconnect
        
        return ib.isConnected()
    except Exception as e:
        logger.error(f"Failed to connect to IB Gateway: {e}")
        return False


def on_disconnect() -> None:
    """Handler for disconnect events."""
    global running
    logger.warning("Disconnected from IB Gateway")
    running = False


def on_ticker_update(ticker) -> None:
    """
    Handler for ticker updates.
    
    Args:
        ticker: The ticker object with updated market data
    """
    symbol = ticker.contract.symbol
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    # Extract relevant market data
    data = {
        "timestamp": timestamp,
        "symbol": symbol,
        "bid": ticker.bid,
        "ask": ticker.ask,
        "last": ticker.last,
        "volume": ticker.volume,
        "high": ticker.high,
        "low": ticker.low,
    }
    
    # Print market data update
    logger.info(f"Market Data: {data}")
    
    # Here you would typically:
    # 1. Process the data (e.g., feed into a model)
    # 2. Store in a database or queue (e.g., Redis)
    # 3. Trigger any trading signals


async def subscribe_to_market_data(symbols: List[str]) -> Dict[str, Any]:
    """
    Subscribe to market data for the specified symbols.
    
    Args:
        symbols: List of stock symbols to subscribe to
        
    Returns:
        Dict mapping symbols to their ticker objects
    """
    tickers = {}
    
    for symbol in symbols:
        try:
            # Create a stock contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Request market data
            # For delayed data, we need to use the correct parameters:
            # - genericTickList: '' (empty string for default ticks)
            # - snapshot: False (for streaming data)
            # - regulatorySnapshot: False
            logger.info(f"Subscribing to market data for {symbol}")
            
            # First try with delayed data option
            ticker = ib.reqMktData(contract, '', False, False)
            
            # Add a note that we're using whatever data is available (real-time or delayed)
            logger.info(f"Successfully subscribed to {symbol}")
            tickers[symbol] = ticker
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
    
    return tickers


async def monitor_market_data() -> None:
    """Monitor market data updates in a loop."""
    global running
    
    try:
        while running and ib.isConnected():
            # Sleep for a short interval to allow for updates
            # This is more efficient than constantly polling
            await asyncio.sleep(0.05)
            
            # Process any pending IB messages
            ib.sleep(0)
            
            # You can add additional monitoring logic here
    except asyncio.CancelledError:
        logger.info("Market data monitoring task cancelled")
    except Exception as e:
        logger.error(f"Error in market data monitoring: {e}")
        running = False


async def shutdown() -> None:
    """Gracefully shutdown the connection to IB Gateway."""
    global running
    
    logger.info("Shutting down...")
    running = False
    
    # Cancel any pending market data requests
    for ticker in ib.tickers():
        ib.cancelMktData(ticker.contract)
    
    # Disconnect from IB Gateway
    if ib.isConnected():
        ib.disconnect()
    
    logger.info("Shutdown complete")


async def main() -> None:
    """Main async function."""
    global tickers, running
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='IB Gateway Async Test')
    parser.add_argument('--port', type=int, default=4002, help='Port to connect to IB Gateway')
    parser.add_argument('--symbols', type=str, default='AAPL,MSFT,AMZN', 
                        help='Comma-separated list of symbols to subscribe to')
    args = parser.parse_args()
    
    # Extract symbols from command line arguments
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    try:
        # Connect to IB Gateway
        if not await connect_to_ib(port=args.port):
            logger.error("Failed to connect to IB Gateway. Exiting.")
            return
        
        # Register event handler for ticker updates
        ib.pendingTickersEvent += on_ticker_update
        
        # Subscribe to market data for the specified symbols
        tickers = await subscribe_to_market_data(symbols)
        
        # Set up signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
        
        # Monitor market data
        await monitor_market_data()
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Ensure we disconnect properly
        await shutdown()


if __name__ == "__main__":
    # Enable asyncio debugging (remove in production)
    # asyncio.get_event_loop().set_debug(True)
    
    # Run the main async function
    util.patchAsyncio()
    util.run(main())
