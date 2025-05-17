#!/usr/bin/env python3
"""
Synchronous IB Gateway Test Script

This script demonstrates a simple way to connect to the Interactive Brokers Gateway
using ib_insync library in synchronous mode.

Usage:
    python ib_sync_test.py [--port PORT] [--symbol SYMBOL]

Options:
    --port PORT     Port to connect to IB Gateway (default: 4002)
    --symbol SYMBOL Stock symbol to subscribe to (default: AAPL)
"""

import argparse
import logging
from ib_insync import *

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
    symbol = ticker.contract.symbol
    logger.info(f"[PRICE] {symbol}: bid={ticker.bid}, ask={ticker.ask}")


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Synchronous IB Gateway Test')
    parser.add_argument('--port', type=int, default=4002, help='Port to connect to IB Gateway')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol to subscribe to')
    args = parser.parse_args()
    
    # Create IB instance
    ib = IB()
    
    try:
        # Connect to IB Gateway
        logger.info(f"Connecting to IB Gateway at 127.0.0.1:{args.port}")
        ib.connect('127.0.0.1', args.port, clientId=1)
        logger.info(f"Connected: {ib.isConnected()}")
        
        # Register event handler for ticker updates
        ib.pendingTickersEvent += on_ticker_update
        
        # Create a stock contract
        contract = Stock(args.symbol, 'SMART', 'USD')
        
        # Request market data
        logger.info(f"Subscribing to market data for {args.symbol}")
        ib.reqMktData(contract, '', False, False)
        logger.info(f"Successfully subscribed to {args.symbol}")
        
        # Keep the script running
        logger.info("Monitoring market data (press Ctrl+C to exit)...")
        ib.run()
        
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
    main()
