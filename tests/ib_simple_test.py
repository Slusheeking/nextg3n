#!/usr/bin/env python3
"""
Simple IB Gateway Test Script

This script demonstrates a simple way to connect to the Interactive Brokers Gateway
using ib_insync library and retrieve market data.

Usage:
    python ib_simple_test.py [--port PORT] [--symbols SYMBOLS]

Options:
    --port PORT       Port to connect to IB Gateway (default: 4002)
    --symbols SYMBOLS Comma-separated list of symbols to subscribe to (default: AAPL,MSFT,AMZN)
"""

import argparse
import asyncio
import logging
from typing import List
from datetime import datetime

from ib_insync import IB, Stock, util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def on_ticker_update(ticker):
    """
    Handler for ticker updates.
    
    Args:
        ticker: The ticker object with updated market data
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    symbol = ticker.contract.symbol
    
    # Log the market data update
    logger.info(f"[{timestamp}] {symbol}: bid={ticker.bid}, ask={ticker.ask}")


async def main():
    """Main async function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simple IB Gateway Test')
    parser.add_argument('--port', type=int, default=4002, help='Port to connect to IB Gateway')
    parser.add_argument('--symbols', type=str, default='AAPL,MSFT,AMZN', 
                        help='Comma-separated list of symbols to subscribe to')
    args = parser.parse_args()
    
    # Extract symbols from command line arguments
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Create IB instance
    ib = IB()
    
    try:
        # Connect to IB Gateway
        logger.info(f"Connecting to IB Gateway at 127.0.0.1:{args.port}")
        await ib.connectAsync('127.0.0.1', args.port, clientId=1)
        logger.info(f"Connected: {ib.isConnected()}")
        
        # Register event handler for ticker updates
        ib.pendingTickersEvent += on_ticker_update
        
        # Subscribe to market data for each symbol
        for symbol in symbols:
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                logger.info(f"Subscribing to market data for {symbol}")
                ib.reqMktData(contract, '', False, False)
                logger.info(f"Successfully subscribed to {symbol}")
            except Exception as e:
                logger.error(f"Failed to subscribe to {symbol}: {e}")
        
        # Keep the script running
        logger.info("Monitoring market data (press Ctrl+C to exit)...")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Disconnect from IB Gateway
        if ib.isConnected():
            ib.disconnect()
            logger.info("Disconnected from IB Gateway")


if __name__ == "__main__":
    # Enable asyncio debugging (remove in production)
    # asyncio.get_event_loop().set_debug(True)
    
    # Run the main async function
    util.patchAsyncio()
    util.run(main())
