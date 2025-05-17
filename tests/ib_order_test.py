#!/usr/bin/env python3
"""
IB Gateway Order Placement Test Script

This script demonstrates how to place orders with the Interactive Brokers Gateway
using ib_insync library. It includes both synchronous and asynchronous implementations.

Usage:
    python ib_order_test.py [--port PORT] [--symbol SYMBOL] [--async]

Options:
    --port PORT       Port to connect to IB Gateway (default: 4002)
    --symbol SYMBOL   Stock symbol to trade (default: AAPL)
    --price PRICE     Limit price for the order (default: 150.00)
    --quantity QTY    Number of shares to trade (default: 10)
    --action ACTION   Order action: BUY or SELL (default: BUY)
    --async           Use asynchronous mode (default: False)
"""

import argparse
import asyncio
import logging
import sys
from ib_insync import IB, Stock, LimitOrder, util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def place_order_sync(port, symbol, action, quantity, price):
    """
    Place an order using the synchronous API.
    
    Args:
        port: Port to connect to IB Gateway
        symbol: Stock symbol to trade
        action: Order action (BUY or SELL)
        quantity: Number of shares to trade
        price: Limit price for the order
    """
    ib = IB()
    
    try:
        # Connect to IB Gateway with a unique client ID
        logger.info(f"Connecting to IB Gateway at 127.0.0.1:{port}")
        ib.connect('127.0.0.1', port, clientId=2)  # Using clientId=2 for order placement
        logger.info(f"Connected: {ib.isConnected()}")
        
        # Create a stock contract
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Create a limit order
        order = LimitOrder(action, quantity, price)
        
        # Place the order
        logger.info(f"Placing {action} order for {quantity} shares of {symbol} at ${price:.2f}")
        trade = ib.placeOrder(contract, order)
        
        # Log order details
        logger.info(f"Order placed: {trade.order}")
        logger.info(f"Order status: {trade.orderStatus.status}")
        
        # Wait for order status updates
        logger.info("Waiting for order status updates (press Ctrl+C to exit)...")
        ib.sleep(2)  # Give some time for initial status updates
        
        # Monitor order status for a short period
        for _ in range(5):
            logger.info(f"Order status: {trade.orderStatus.status}")
            ib.sleep(1)
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Disconnect from IB Gateway
        if ib.isConnected():
            ib.disconnect()
            logger.info("Disconnected from IB Gateway")


async def place_order_async(port, symbol, action, quantity, price):
    """
    Place an order using the asynchronous API.
    
    Args:
        port: Port to connect to IB Gateway
        symbol: Stock symbol to trade
        action: Order action (BUY or SELL)
        quantity: Number of shares to trade
        price: Limit price for the order
    """
    ib = IB()
    
    try:
        # Connect to IB Gateway with a unique client ID
        logger.info(f"Connecting to IB Gateway at 127.0.0.1:{port}")
        await ib.connectAsync('127.0.0.1', port, clientId=2)  # Using clientId=2 for order placement
        logger.info(f"Connected: {ib.isConnected()}")
        
        # Create a stock contract
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Create a limit order
        order = LimitOrder(action, quantity, price)
        
        # Place the order
        logger.info(f"Placing {action} order for {quantity} shares of {symbol} at ${price:.2f}")
        trade = ib.placeOrder(contract, order)
        
        # Log order details
        logger.info(f"Order placed: {trade.order}")
        logger.info(f"Order status: {trade.orderStatus.status}")
        
        # Wait for order status updates
        logger.info("Waiting for order status updates (press Ctrl+C to exit)...")
        await asyncio.sleep(2)  # Give some time for initial status updates
        
        # Monitor order status for a short period
        for _ in range(5):
            logger.info(f"Order status: {trade.orderStatus.status}")
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


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='IB Gateway Order Placement Test')
    parser.add_argument('--port', type=int, default=4002, help='Port to connect to IB Gateway')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol to trade')
    parser.add_argument('--price', type=float, default=150.00, help='Limit price for the order')
    parser.add_argument('--quantity', type=int, default=10, help='Number of shares to trade')
    parser.add_argument('--action', type=str, default='BUY', choices=['BUY', 'SELL'], help='Order action')
    parser.add_argument('--async', dest='async_mode', action='store_true', help='Use asynchronous mode')
    args = parser.parse_args()
    
    # Run in either synchronous or asynchronous mode
    if args.async_mode:
        logger.info("Running in asynchronous mode")
        util.patchAsyncio()
        util.run(place_order_async(args.port, args.symbol, args.action, args.quantity, args.price))
    else:
        logger.info("Running in synchronous mode")
        place_order_sync(args.port, args.symbol, args.action, args.quantity, args.price)


if __name__ == "__main__":
    main()
