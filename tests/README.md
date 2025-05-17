# IB Gateway Integration

This directory contains tools for interacting with the Interactive Brokers Gateway container, including both synchronous and asynchronous implementations.

## Overview

The IB Gateway container provides a way to connect to Interactive Brokers' trading platform programmatically. This integration uses the `ib_insync` library to establish connections to the IB Gateway, allowing for efficient retrieval of market data and order placement.

## Container Configuration

The IB Gateway container is configured with the following ports:

- **4001:4003**: Primary API port (mapped from container port 4003 to host port 4001)
- **4002:4004**: Secondary API port (mapped from container port 4004 to host port 4002)
- **5900:5900**: VNC port for GUI access

## Test Scripts

### 1. Comprehensive Async Test (`ib_async_test.py`)

A comprehensive asynchronous implementation that demonstrates:

- Connecting to the IB Gateway asynchronously
- Subscribing to real-time market data for multiple symbols
- Processing market data updates through event handlers
- Handling disconnections and gracefully shutting down

```bash
# Basic usage (connects to port 4002 and subscribes to AAPL, MSFT, AMZN)
./ib_async_test.py

# Connect to a different port
./ib_async_test.py --port 4001

# Subscribe to specific symbols
./ib_async_test.py --symbols "AAPL,GOOGL,TSLA,SPY"
```

### 2. Simple Async Test (`ib_simple_test.py`)

A simplified asynchronous implementation focused on market data retrieval:

```bash
# Basic usage
./ib_simple_test.py

# Connect to a different port
./ib_simple_test.py --port 4001

# Subscribe to specific symbols
./ib_simple_test.py --symbols "AAPL,GOOGL,TSLA,SPY"
```

### 3. Synchronous Test (`ib_sync_test.py`)

A synchronous implementation for market data retrieval:

```bash
# Basic usage
./ib_sync_test.py

# Connect to a different port
./ib_sync_test.py --port 4001

# Subscribe to a specific symbol
./ib_sync_test.py --symbol AAPL
```

### 4. Order Placement Test (`ib_order_test.py`)

Demonstrates order placement functionality with both synchronous and asynchronous implementations:

```bash
# Basic usage (synchronous mode)
./ib_order_test.py

# Asynchronous mode
./ib_order_test.py --async

# Customize order parameters
./ib_order_test.py --symbol MSFT --price 250.00 --quantity 5 --action SELL
```

## Integration with Trading Systems

To integrate these scripts with a trading system:

1. **Market Data Integration**:
   - Modify the `on_ticker_update` function to process market data according to your needs
   - Consider implementing a message queue (e.g., Redis) for distributing market data to other components

2. **Order Execution**:
   - Use the order placement functionality from `ib_order_test.py` as a template
   - Implement your trading strategy to generate order signals
   - Monitor order status and handle fills appropriately

## Performance Considerations

For low-latency applications:

- Minimize processing in the event handlers
- Use asynchronous implementations for better performance
- Consider using dedicated client IDs for different functions (market data vs. order placement)
- Adjust the sleep intervals based on your latency requirements
