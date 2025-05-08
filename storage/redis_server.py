"""
Redis Server Configuration for Trading System

This module configures and manages a Redis server instance for the automated trading system,
handling data flow between Market Analysis ML, Main LLM, and Trade LLM components.
It sets up appropriate data structures, handles persistence, and provides connection utilities.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import redis.asyncio as aioredis
import redis
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("redis_server")

# Load environment variables
load_dotenv()

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Key prefixes for different data types
MARKET_DATA_PREFIX = "market_data:"
POSITION_PREFIX = "position:"
TRADE_ORDER_PREFIX = "trade_order:"
ORDER_STATUS_PREFIX = "order_status:"
ML_ANALYSIS_PREFIX = "ml_analysis:"
SIGNAL_PREFIX = "signal:"

# TTL for different data types (in seconds)
MARKET_DATA_TTL = 60 * 60 * 24  # 24 hours
POSITION_TTL = 60 * 60 * 24 * 7  # 7 days
TRADE_ORDER_TTL = 60 * 60 * 24 * 3  # 3 days
ORDER_STATUS_TTL = 60 * 60 * 24 * 3  # 3 days
ML_ANALYSIS_TTL = 60 * 60 * 12  # 12 hours
SIGNAL_TTL = 60 * 60 * 24  # 24 hours

class RedisServer:
    """Manages Redis server connections and operations for the trading system."""
    
    def __init__(self):
        self.redis_client = None
        self.pubsub_client = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize Redis connections."""
        if self.initialized:
            return
        
        try:
            # Main Redis client for operations
            self.redis_client = aioredis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Create separate client for PubSub to avoid blocking
            self.pubsub_client = self.redis_client.duplicate()
            
            self.initialized = True
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            
            # Set up Redis persistence
            await self._configure_persistence()
            
            # Set up key expiration policies
            await self._set_expiration_policies()
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            if self.redis_client:
                await self.redis_client.close()
                self.redis_client = None
            if self.pubsub_client:
                await self.pubsub_client.close()
                self.pubsub_client = None
    
    async def _configure_persistence(self):
        """Configure Redis persistence settings."""
        try:
            # Set up RDB persistence (periodic snapshots)
            await self.redis_client.config_set('save', '900 1 300 10 60 10000')
            
            # Set up AOF persistence (append-only file for durability)
            await self.redis_client.config_set('appendonly', 'yes')
            await self.redis_client.config_set('appendfsync', 'everysec')
            
            logger.info("Redis persistence configured")
        except Exception as e:
            logger.error(f"Error configuring Redis persistence: {e}")
    
    async def _set_expiration_policies(self):
        """Set up key expiration policies."""
        # This is primarily for documentation - actual TTLs are set when keys are created
        logger.info("Redis expiration policies set up")
    
    async def store_market_data(self, symbol: str, data: Dict[str, Any]) -> bool:
        """Store market data for a symbol."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Create key with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            key = f"{MARKET_DATA_PREFIX}{symbol}:{timestamp}"
            
            # Store data as JSON
            await self.redis_client.set(key, json.dumps(data))
            await self.redis_client.expire(key, MARKET_DATA_TTL)
            
            # Update latest pointer
            latest_key = f"{MARKET_DATA_PREFIX}{symbol}:latest"
            await self.redis_client.set(latest_key, json.dumps(data))
            
            # Add to sorted set for time-series querying
            time_key = f"{MARKET_DATA_PREFIX}{symbol}:timeseries"
            score = datetime.now().timestamp()
            await self.redis_client.zadd(time_key, {timestamp: score})
            
            # Publish update for subscribers
            channel = f"{MARKET_DATA_PREFIX}{symbol}"
            await self.redis_client.publish(channel, json.dumps({
                "timestamp": timestamp,
                "type": "market_data_update",
                "data": data
            }))
            
            return True
        except Exception as e:
            logger.error(f"Error storing market data for {symbol}: {e}")
            return False
    
    async def get_latest_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest market data for a symbol."""
        if not self.initialized:
            await self.initialize()
        
        try:
            key = f"{MARKET_DATA_PREFIX}{symbol}:latest"
            data = await self.redis_client.get(key)
            
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def store_position(self, position_id: str, position_data: Dict[str, Any]) -> bool:
        """Store position information."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Create key
            key = f"{POSITION_PREFIX}{position_id}"
            
            # Store position data
            await self.redis_client.set(key, json.dumps(position_data))
            await self.redis_client.expire(key, POSITION_TTL)
            
            # Add to positions list by symbol
            symbol = position_data.get("symbol")
            if symbol:
                symbol_positions_key = f"{POSITION_PREFIX}symbol:{symbol}"
                await self.redis_client.sadd(symbol_positions_key, position_id)
            
            # Add to active positions set if position is open
            if position_data.get("status") == "open":
                await self.redis_client.sadd(f"{POSITION_PREFIX}active", position_id)
            else:
                await self.redis_client.srem(f"{POSITION_PREFIX}active", position_id)
            
            # Publish update
            await self.redis_client.publish(f"{POSITION_PREFIX}{position_id}", json.dumps({
                "type": "position_update",
                "position_id": position_id,
                "data": position_data
            }))
            
            return True
        except Exception as e:
            logger.error(f"Error storing position {position_id}: {e}")
            return False
    
    async def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get position information by ID."""
        if not self.initialized:
            await self.initialize()
        
        try:
            key = f"{POSITION_PREFIX}{position_id}"
            data = await self.redis_client.get(key)
            
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting position {position_id}: {e}")
            return None
    
    async def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get active position IDs
            active_positions = await self.redis_client.smembers(f"{POSITION_PREFIX}active")
            
            # Get position data for each ID
            positions = []
            for position_id in active_positions:
                position_data = await self.get_position(position_id)
                if position_data:
                    positions.append(position_data)
            
            return positions
        except Exception as e:
            logger.error(f"Error getting active positions: {e}")
            return []
    
    async def store_trade_order(self, order_id: str, order_data: Dict[str, Any]) -> bool:
        """Store trade order information."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Create key
            key = f"{TRADE_ORDER_PREFIX}{order_id}"
            
            # Store order data
            await self.redis_client.set(key, json.dumps(order_data))
            await self.redis_client.expire(key, TRADE_ORDER_TTL)
            
            # Add to orders list by symbol
            symbol = order_data.get("symbol")
            if symbol:
                symbol_orders_key = f"{TRADE_ORDER_PREFIX}symbol:{symbol}"
                await self.redis_client.sadd(symbol_orders_key, order_id)
            
            # Add to pending orders set
            if order_data.get("status") == "pending":
                await self.redis_client.sadd(f"{TRADE_ORDER_PREFIX}pending", order_id)
            
            # Publish update
            await self.redis_client.publish(f"{TRADE_ORDER_PREFIX}{order_id}", json.dumps({
                "type": "order_update",
                "order_id": order_id,
                "data": order_data
            }))
            
            return True
        except Exception as e:
            logger.error(f"Error storing trade order {order_id}: {e}")
            return False
    
    async def update_order_status(self, order_id: str, status: str, 
                                  status_data: Dict[str, Any]) -> bool:
        """Update the status of an order."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get current order data
            order_key = f"{TRADE_ORDER_PREFIX}{order_id}"
            order_data_str = await self.redis_client.get(order_key)
            
            if not order_data_str:
                logger.warning(f"Order {order_id} not found for status update")
                return False
                
            order_data = json.loads(order_data_str)
            
            # Update status
            order_data["status"] = status
            order_data["last_update"] = datetime.now().isoformat()
            order_data["status_history"] = order_data.get("status_history", []) + [{
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "data": status_data
            }]
            
            # Store updated order
            await self.redis_client.set(order_key, json.dumps(order_data))
            
            # Update pending orders set
            if status != "pending":
                await self.redis_client.srem(f"{TRADE_ORDER_PREFIX}pending", order_id)
            
            # Store status separately for quick lookup
            status_key = f"{ORDER_STATUS_PREFIX}{order_id}"
            await self.redis_client.set(status_key, json.dumps({
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "data": status_data
            }))
            await self.redis_client.expire(status_key, ORDER_STATUS_TTL)
            
            # Publish status update
            await self.redis_client.publish(f"{ORDER_STATUS_PREFIX}{order_id}", json.dumps({
                "type": "status_update",
                "order_id": order_id,
                "status": status,
                "data": status_data
            }))
            
            return True
        except Exception as e:
            logger.error(f"Error updating order status for {order_id}: {e}")
            return False
    
    async def store_ml_analysis(self, analysis_id: str, analysis_data: Dict[str, Any]) -> bool:
        """Store ML analysis results."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Create key
            key = f"{ML_ANALYSIS_PREFIX}{analysis_id}"
            
            # Store analysis data
            await self.redis_client.set(key, json.dumps(analysis_data))
            await self.redis_client.expire(key, ML_ANALYSIS_TTL)
            
            # Add to ML analysis list by type
            analysis_type = analysis_data.get("type", "general")
            type_key = f"{ML_ANALYSIS_PREFIX}type:{analysis_type}"
            timestamp = datetime.now().timestamp()
            await self.redis_client.zadd(type_key, {analysis_id: timestamp})
            
            # Publish update
            await self.redis_client.publish(f"{ML_ANALYSIS_PREFIX}{analysis_type}", json.dumps({
                "type": "ml_analysis_update",
                "analysis_id": analysis_id,
                "analysis_type": analysis_type,
                "data": analysis_data
            }))
            
            return True
        except Exception as e:
            logger.error(f"Error storing ML analysis {analysis_id}: {e}")
            return False
    
    async def get_latest_ml_analysis(self, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get the latest ML analysis of a specific type."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get latest analysis ID for this type
            type_key = f"{ML_ANALYSIS_PREFIX}type:{analysis_type}"
            latest = await self.redis_client.zrevrange(type_key, 0, 0, withscores=True)
            
            if not latest:
                return None
                
            analysis_id = latest[0][0]
            
            # Get analysis data
            key = f"{ML_ANALYSIS_PREFIX}{analysis_id}"
            data = await self.redis_client.get(key)
            
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting latest ML analysis for {analysis_type}: {e}")
            return None
    
    async def store_signal(self, symbol: str, signal_type: str, 
                           signal_data: Dict[str, Any]) -> bool:
        """Store a trading signal."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Create unique signal ID
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            signal_id = f"{symbol}_{signal_type}_{timestamp}"
            
            # Add timestamp to data
            signal_data["timestamp"] = datetime.now().isoformat()
            signal_data["symbol"] = symbol
            signal_data["signal_type"] = signal_type
            signal_data["signal_id"] = signal_id
            
            # Store signal
            key = f"{SIGNAL_PREFIX}{signal_id}"
            await self.redis_client.set(key, json.dumps(signal_data))
            await self.redis_client.expire(key, SIGNAL_TTL)
            
            # Add to signals by symbol
            symbol_key = f"{SIGNAL_PREFIX}symbol:{symbol}"
            score = datetime.now().timestamp()
            await self.redis_client.zadd(symbol_key, {signal_id: score})
            
            # Add to signals by type
            type_key = f"{SIGNAL_PREFIX}type:{signal_type}"
            await self.redis_client.zadd(type_key, {signal_id: score})
            
            # Publish signal
            await self.redis_client.publish(f"{SIGNAL_PREFIX}{symbol}", json.dumps({
                "type": "signal",
                "signal_id": signal_id,
                "symbol": symbol,
                "signal_type": signal_type,
                "data": signal_data
            }))
            
            # Also publish to signal type channel
            await self.redis_client.publish(f"{SIGNAL_PREFIX}type:{signal_type}", json.dumps({
                "type": "signal",
                "signal_id": signal_id,
                "symbol": symbol,
                "signal_type": signal_type,
                "data": signal_data
            }))
            
            return True
        except Exception as e:
            logger.error(f"Error storing signal for {symbol}: {e}")
            return False
    
    async def get_recent_signals(self, symbol: str = None, signal_type: str = None,
                                hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent signals, optionally filtered by symbol or type."""
        if not self.initialized:
            await self.initialize()
        
        try:
            signals = []
            
            # Determine which key to use
            if symbol and signal_type:
                # We need to do an intersection of two sorted sets
                symbol_key = f"{SIGNAL_PREFIX}symbol:{symbol}"
                type_key = f"{SIGNAL_PREFIX}type:{signal_type}"
                
                # Get IDs from both sets
                min_score = (datetime.now() - timedelta(hours=hours)).timestamp()
                max_score = float('inf')
                
                symbol_ids = await self.redis_client.zrangebyscore(
                    symbol_key, min_score, max_score)
                type_ids = await self.redis_client.zrangebyscore(
                    type_key, min_score, max_score)
                
                # Find intersection
                signal_ids = set(symbol_ids).intersection(set(type_ids))
                
            elif symbol:
                # Get by symbol
                symbol_key = f"{SIGNAL_PREFIX}symbol:{symbol}"
                min_score = (datetime.now() - timedelta(hours=hours)).timestamp()
                max_score = float('inf')
                
                signal_ids = await self.redis_client.zrangebyscore(
                    symbol_key, min_score, max_score)
                
            elif signal_type:
                # Get by type
                type_key = f"{SIGNAL_PREFIX}type:{signal_type}"
                min_score = (datetime.now() - timedelta(hours=hours)).timestamp()
                max_score = float('inf')
                
                signal_ids = await self.redis_client.zrangebyscore(
                    type_key, min_score, max_score)
                
            else:
                # No filter - need to scan keys
                pattern = f"{SIGNAL_PREFIX}*"
                signal_ids = []
                
                async for key in self.redis_client.scan_iter(match=pattern):
                    if key.count(':') == 0:  # Skip index keys
                        signal_id = key.replace(SIGNAL_PREFIX, '')
                        signal_ids.append(signal_id)
            
            # Get data for each signal
            for signal_id in signal_ids:
                key = f"{SIGNAL_PREFIX}{signal_id}"
                data = await self.redis_client.get(key)
                
                if data:
                    signal_data = json.loads(data)
                    
                    # Check timestamp if no filter was used
                    if not symbol and not signal_type:
                        timestamp = datetime.fromisoformat(signal_data.get("timestamp", ""))
                        if (datetime.now() - timestamp).total_seconds() > hours * 3600:
                            continue
                    
                    signals.append(signal_data)
            
            return signals
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return []
    
    async def subscribe_to_updates(self, channel_pattern: str, callback) -> None:
        """Subscribe to Redis PubSub updates."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Create pubsub object
            pubsub = self.pubsub_client.pubsub()
            
            # Subscribe to pattern
            await pubsub.psubscribe(**{channel_pattern: callback})
            
            logger.info(f"Subscribed to channel pattern: {channel_pattern}")
            
            # Return pubsub object for later management
            return pubsub
        except Exception as e:
            logger.error(f"Error subscribing to {channel_pattern}: {e}")
            return None
    
    async def close(self):
        """Close Redis connections."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
        
        if self.pubsub_client:
            await self.pubsub_client.close()
            self.pubsub_client = None
        
        self.initialized = False
        logger.info("Redis connections closed")


# Example usage
async def example_usage():
    """Example usage of RedisServer."""
    redis_server = RedisServer()
    await redis_server.initialize()
    
    # Store market data
    await redis_server.store_market_data("AAPL", {
        "price": 150.25,
        "volume": 1000000,
        "timestamp": datetime.now().isoformat()
    })
    
    # Get market data
    aapl_data = await redis_server.get_latest_market_data("AAPL")
    print(f"AAPL data: {aapl_data}")
    
    # Store position
    position_id = "pos_123456"
    await redis_server.store_position(position_id, {
        "symbol": "AAPL",
        "quantity": 100,
        "entry_price": 150.25,
        "current_price": 150.25,
        "status": "open",
        "timestamp": datetime.now().isoformat()
    })
    
    # Store ML analysis
    analysis_id = "analysis_market_20230501"
    await redis_server.store_ml_analysis(analysis_id, {
        "type": "market_analysis",
        "predictions": [
            {"symbol": "AAPL", "score": 0.85, "recommendation": "buy"},
            {"symbol": "MSFT", "score": 0.75, "recommendation": "hold"},
            {"symbol": "GOOGL", "score": 0.65, "recommendation": "sell"}
        ],
        "timestamp": datetime.now().isoformat()
    })
    
    # Store signal
    await redis_server.store_signal("AAPL", "breakout", {
        "direction": "up",
        "strength": 0.85,
        "price": 150.25,
        "volume": 1000000
    })
    
    # Close connection
    await redis_server.close()

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())