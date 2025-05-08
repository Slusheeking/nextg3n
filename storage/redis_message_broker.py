"""
Redis Message Broker for Trading System

This module implements a Redis message broker that connects all components
of the trading system as shown in the architecture diagram:
- Data Processing (MCP Tools, ML Analysis, ML Monitor)
- Main LLM (Trade Analysis, Position Sizing)
- Trade LLM
- Alpaca Broker

It handles all data flows between components using Redis PubSub channels and data structures.
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
logger = logging.getLogger("redis_broker")

# Load environment variables
load_dotenv()

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Channel definitions
MARKET_DATA_CHANNEL = "channel:market_data"
POSITION_DATA_CHANNEL = "channel:position_data"
MARKET_ANALYSIS_CHANNEL = "channel:market_analysis"
POSITION_MONITOR_CHANNEL = "channel:position_monitor"
MAIN_LLM_CHANNEL = "channel:main_llm"
TRADE_ANALYSIS_CHANNEL = "channel:trade_analysis"
POSITION_SIZING_CHANNEL = "channel:position_sizing"
TRADE_LLM_CHANNEL = "channel:trade_llm"
TRADE_ORDER_CHANNEL = "channel:trade_order"
ORDER_STATUS_CHANNEL = "channel:order_status"
POSITION_CREATED_CHANNEL = "channel:position_created"

class RedisBroker:
    """Message broker that connects all trading system components via Redis."""
    
    def __init__(self):
        self.redis_client = None
        self.pubsub_client = None
        self.initialized = False
        self.tasks = []
    
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
            
            # Set up listeners for all channels
            await self._setup_listeners()
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            if self.redis_client:
                await self.redis_client.close()
                self.redis_client = None
            if self.pubsub_client:
                await self.pubsub_client.close()
                self.pubsub_client = None
    
    async def _setup_listeners(self):
        """Set up listeners for all channels."""
        try:
            # Create pubsub handler
            pubsub = self.pubsub_client.pubsub()
            
            # Subscribe to all channels
            await pubsub.subscribe(
                MARKET_DATA_CHANNEL,
                POSITION_DATA_CHANNEL,
                MARKET_ANALYSIS_CHANNEL,
                POSITION_MONITOR_CHANNEL,
                MAIN_LLM_CHANNEL,
                TRADE_ANALYSIS_CHANNEL,
                POSITION_SIZING_CHANNEL,
                TRADE_LLM_CHANNEL,
                TRADE_ORDER_CHANNEL,
                ORDER_STATUS_CHANNEL,
                POSITION_CREATED_CHANNEL
            )
            
            # Start listener task
            listener_task = asyncio.create_task(self._listen_for_messages(pubsub))
            self.tasks.append(listener_task)
            
            logger.info("Redis message listeners set up")
        except Exception as e:
            logger.error(f"Error setting up Redis listeners: {e}")
    
    async def _listen_for_messages(self, pubsub):
        """Listen for messages on all channels and route to appropriate handlers."""
        try:
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message:
                    channel = message["channel"]
                    data = message["data"]
                    
                    # Try to parse JSON data
                    try:
                        if isinstance(data, str):
                            data = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning(f"Received non-JSON data on channel {channel}")
                    
                    # Route message to appropriate handler
                    await self._route_message(channel, data)
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("Message listener task cancelled")
        except Exception as e:
            logger.error(f"Error in message listener: {e}")
    
    async def _route_message(self, channel, data):
        """Route messages to appropriate handlers based on channel."""
        try:
            logger.debug(f"Received message on channel {channel}")
            
            if channel == MARKET_DATA_CHANNEL:
                await self._handle_market_data(data)
                
            elif channel == POSITION_DATA_CHANNEL:
                await self._handle_position_data(data)
                
            elif channel == MARKET_ANALYSIS_CHANNEL:
                await self._handle_market_analysis(data)
                
            elif channel == POSITION_MONITOR_CHANNEL:
                await self._handle_position_monitor(data)
                
            elif channel == MAIN_LLM_CHANNEL:
                await self._handle_main_llm(data)
                
            elif channel == TRADE_ANALYSIS_CHANNEL:
                await self._handle_trade_analysis(data)
                
            elif channel == POSITION_SIZING_CHANNEL:
                await self._handle_position_sizing(data)
                
            elif channel == TRADE_LLM_CHANNEL:
                await self._handle_trade_llm(data)
                
            elif channel == TRADE_ORDER_CHANNEL:
                await self._handle_trade_order(data)
                
            elif channel == ORDER_STATUS_CHANNEL:
                await self._handle_order_status(data)
                
            elif channel == POSITION_CREATED_CHANNEL:
                await self._handle_position_created(data)
                
            else:
                logger.warning(f"Received message on unknown channel: {channel}")
        except Exception as e:
            logger.error(f"Error routing message from channel {channel}: {e}")
    
    # Handler methods for each channel
    async def _handle_market_data(self, data):
        """Handle market data from MCP Tools."""
        logger.debug(f"Handling market data: {data}")
        
        # Store market data in Redis
        symbol = data.get("symbol")
        if symbol:
            await self.redis_client.set(
                f"market_data:{symbol}:latest",
                json.dumps(data)
            )
            
            # Forward to Market Analysis ML
            await self.redis_client.publish(
                MARKET_ANALYSIS_CHANNEL,
                json.dumps({
                    "type": "market_data_update",
                    "data": data
                })
            )
    
    async def _handle_position_data(self, data):
        """Handle position data from MCP Tools."""
        logger.debug(f"Handling position data: {data}")
        
        # Store position data in Redis
        position_id = data.get("position_id")
        if position_id:
            await self.redis_client.set(
                f"position:{position_id}",
                json.dumps(data)
            )
            
            # Forward to Position Monitor ML
            await self.redis_client.publish(
                POSITION_MONITOR_CHANNEL,
                json.dumps({
                    "type": "position_data_update",
                    "data": data
                })
            )
    
    async def _handle_market_analysis(self, data):
        """Handle market analysis results from ML Analysis."""
        logger.debug(f"Handling market analysis: {data}")
        
        # Store analysis in Redis
        analysis_id = data.get("analysis_id", datetime.now().strftime("%Y%m%d%H%M%S"))
        await self.redis_client.set(
            f"market_analysis:{analysis_id}",
            json.dumps(data)
        )
        
        # Also store as latest
        await self.redis_client.set(
            "market_analysis:latest",
            json.dumps(data)
        )
        
        # Forward to Main LLM
        await self.redis_client.publish(
            MAIN_LLM_CHANNEL,
            json.dumps({
                "type": "market_analysis_update",
                "data": data
            })
        )
    
    async def _handle_position_monitor(self, data):
        """Handle position monitor signals from ML Monitor."""
        logger.debug(f"Handling position monitor signal: {data}")
        
        # Store signal in Redis
        signal_id = data.get("signal_id", datetime.now().strftime("%Y%m%d%H%M%S"))
        await self.redis_client.set(
            f"position_signal:{signal_id}",
            json.dumps(data)
        )
        
        # Also store by position
        position_id = data.get("position_id")
        if position_id:
            await self.redis_client.set(
                f"position:{position_id}:latest_signal",
                json.dumps(data)
            )
        
        # Forward to Trade LLM
        await self.redis_client.publish(
            TRADE_LLM_CHANNEL,
            json.dumps({
                "type": "position_monitor_signal",
                "data": data
            })
        )
    
    async def _handle_main_llm(self, data):
        """Handle requests or responses from Main LLM."""
        logger.debug(f"Handling Main LLM message: {data}")
        
        message_type = data.get("type")
        
        if message_type == "analysis_request":
            # Forward to Trade Analysis
            await self.redis_client.publish(
                TRADE_ANALYSIS_CHANNEL,
                json.dumps(data)
            )
        elif message_type == "sizing_request":
            # Forward to Position Sizing
            await self.redis_client.publish(
                POSITION_SIZING_CHANNEL,
                json.dumps(data)
            )
    
    async def _handle_trade_analysis(self, data):
        """Handle trade analysis results."""
        logger.debug(f"Handling trade analysis: {data}")
        
        # Store analysis in Redis
        analysis_id = data.get("analysis_id", datetime.now().strftime("%Y%m%d%H%M%S"))
        await self.redis_client.set(
            f"trade_analysis:{analysis_id}",
            json.dumps(data)
        )
        
        # Forward back to Main LLM
        await self.redis_client.publish(
            MAIN_LLM_CHANNEL,
            json.dumps({
                "type": "trade_analysis_response",
                "data": data
            })
        )
    
    async def _handle_position_sizing(self, data):
        """Handle position sizing results."""
        logger.debug(f"Handling position sizing: {data}")
        
        # Store sizing in Redis
        order_id = data.get("order_id", datetime.now().strftime("%Y%m%d%H%M%S"))
        await self.redis_client.set(
            f"trade_order:{order_id}",
            json.dumps(data)
        )
        
        # Forward to Alpaca Broker
        await self.redis_client.publish(
            TRADE_ORDER_CHANNEL,
            json.dumps({
                "type": "trade_order",
                "data": data
            })
        )
    
    async def _handle_trade_llm(self, data):
        """Handle exit orders from Trade LLM."""
        logger.debug(f"Handling Trade LLM message: {data}")
        
        message_type = data.get("type")
        
        if message_type == "exit_order":
            # Store exit order in Redis
            order_id = data.get("order_id", datetime.now().strftime("%Y%m%d%H%M%S"))
            await self.redis_client.set(
                f"trade_order:{order_id}",
                json.dumps(data)
            )
            
            # Forward to Alpaca Broker
            await self.redis_client.publish(
                TRADE_ORDER_CHANNEL,
                json.dumps({
                    "type": "exit_order",
                    "data": data
                })
            )
    
    async def _handle_trade_order(self, data):
        """Handle trade orders going to Alpaca."""
        logger.debug(f"Handling trade order: {data}")
        
        # Nothing to do here as this is just passing through to Alpaca
        pass
    
    async def _handle_order_status(self, data):
        """Handle order status updates from Alpaca."""
        logger.debug(f"Handling order status: {data}")
        
        # Store order status in Redis
        order_id = data.get("order_id")
        if order_id:
            await self.redis_client.set(
                f"order_status:{order_id}",
                json.dumps(data)
            )
            
            # Forward to Redis for storage
            await self.redis_client.publish(
                "internal:order_status_update",
                json.dumps(data)
            )
    
    async def _handle_position_created(self, data):
        """Handle position created notifications from Alpaca."""
        logger.debug(f"Handling position created: {data}")
        
        # Store position in Redis
        position_id = data.get("position_id")
        if position_id:
            await self.redis_client.set(
                f"position:{position_id}",
                json.dumps(data)
            )
            
            # Forward to Redis for storage
            await self.redis_client.publish(
                "internal:position_created",
                json.dumps(data)
            )
    
    async def publish_market_data(self, data: Dict[str, Any]) -> bool:
        """Publish market data to the system."""
        if not self.initialized:
            await self.initialize()
        
        try:
            await self.redis_client.publish(
                MARKET_DATA_CHANNEL,
                json.dumps(data)
            )
            return True
        except Exception as e:
            logger.error(f"Error publishing market data: {e}")
            return False
    
    async def publish_position_data(self, data: Dict[str, Any]) -> bool:
        """Publish position data to the system."""
        if not self.initialized:
            await self.initialize()
        
        try:
            await self.redis_client.publish(
                POSITION_DATA_CHANNEL,
                json.dumps(data)
            )
            return True
        except Exception as e:
            logger.error(f"Error publishing position data: {e}")
            return False
    
    async def publish_main_llm_request(self, data: Dict[str, Any]) -> bool:
        """Publish request to Main LLM."""
        if not self.initialized:
            await self.initialize()
        
        try:
            await self.redis_client.publish(
                MAIN_LLM_CHANNEL,
                json.dumps(data)
            )
            return True
        except Exception as e:
            logger.error(f"Error publishing Main LLM request: {e}")
            return False
    
    async def publish_trade_llm_signal(self, data: Dict[str, Any]) -> bool:
        """Publish signal to Trade LLM."""
        if not self.initialized:
            await self.initialize()
        
        try:
            await self.redis_client.publish(
                TRADE_LLM_CHANNEL,
                json.dumps(data)
            )
            return True
        except Exception as e:
            logger.error(f"Error publishing Trade LLM signal: {e}")
            return False
    
    async def close(self):
        """Close Redis connections and cancel tasks."""
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
            
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Close Redis connections
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
        
        if self.pubsub_client:
            await self.pubsub_client.close()
            self.pubsub_client = None
        
        self.initialized = False
        logger.info("Redis broker shut down")


# Example usage
async def example_usage():
    """Example usage of RedisBroker."""
    broker = RedisBroker()
    await broker.initialize()
    
    # Simulate market data
    await broker.publish_market_data({
        "symbol": "AAPL",
        "price": 150.25,
        "volume": 1000000,
        "timestamp": datetime.now().isoformat()
    })
    
    # Simulate position data
    await broker.publish_position_data({
        "position_id": "pos_123456",
        "symbol": "AAPL",
        "quantity": 100,
        "entry_price": 150.25,
        "current_price": 150.25,
        "status": "open",
        "timestamp": datetime.now().isoformat()
    })
    
    # Run for a bit to let messages process
    await asyncio.sleep(5)
    
    # Clean shutdown
    await broker.close()

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())