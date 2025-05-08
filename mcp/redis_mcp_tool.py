#!/usr/bin/env python
"""
Redis MCP Tool for NextG3N Trading System

This module implements a standardized MCP server that provides:
1. A consistent interface for AI models to access Redis data
2. Functions to query and manipulate Redis data
3. Standardized data formats for LLM consumption
4. WebSocket support for real-time data updates

It serves as a bridge between the trading system components and Redis.
"""

import os
import json
import time
import asyncio
import logging
import argparse
import yaml
import uvicorn
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta, timezone
import aiohttp
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("redis_mcp_tool")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Models for API requests and responses
class MarketDataRequest(BaseModel):
    symbol: str

class PositionRequest(BaseModel):
    position_id: Optional[str] = None
    symbol: Optional[str] = None

class OrderRequest(BaseModel):
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    order_type: Optional[str] = None

class ContextRequest(BaseModel):
    symbols: Optional[List[str]] = None
    include_positions: bool = True
    include_market_data: bool = True
    include_analysis: bool = True

# Create FastAPI app
app = FastAPI(
    title="Redis MCP Tool",
    description="Model Context Protocol Tool for Redis data access",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Global variables
config = {}
redis_client = None
websocket_connections = []

# Tool functions
async def get_market_data(symbol: str) -> Dict[str, Any]:
    """Get market data for a specific symbol."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection not initialized")
        
    try:
        data = await redis_client.get(f"market_data:{symbol}:latest")
        if data:
            return json.loads(data)
        return {"error": f"No market data found for {symbol}"}
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_position(position_id: str = None, symbol: str = None) -> Dict[str, Any]:
    """Get position data by position ID or symbol."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection not initialized")
        
    try:
        if position_id:
            data = await redis_client.get(f"position:{position_id}:active")
            if data:
                return json.loads(data)
        elif symbol:
            # Look for positions with this symbol
            position_keys = await redis_client.keys(f"position:*:active")
            for key in position_keys:
                pos_data = await redis_client.get(key)
                if pos_data:
                    pos = json.loads(pos_data)
                    if pos.get("symbol") == symbol:
                        return pos
        
        return {"error": "Position not found"}
    except Exception as e:
        logger.error(f"Error getting position: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_all_positions() -> List[Dict[str, Any]]:
    """Get all active positions."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection not initialized")
        
    try:
        position_keys = await redis_client.keys("position:*:active")
        positions = []
        
        for key in position_keys:
            data = await redis_client.get(key)
            if data:
                positions.append(json.loads(data))
        
        return positions
    except Exception as e:
        logger.error(f"Error getting all positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_order(order_id: str) -> Dict[str, Any]:
    """Get order data by order ID."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection not initialized")
        
    try:
        data = await redis_client.get(f"order:{order_id}")
        if data:
            return json.loads(data)
        return {"error": f"Order not found: {order_id}"}
    except Exception as e:
        logger.error(f"Error getting order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_orders_by_symbol(symbol: str) -> List[Dict[str, Any]]:
    """Get all orders for a specific symbol."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection not initialized")
        
    try:
        # Get all order keys
        order_keys = await redis_client.keys("order:*")
        orders = []
        
        for key in order_keys:
            data = await redis_client.get(key)
            if data:
                order = json.loads(data)
                if order.get("symbol") == symbol:
                    orders.append(order)
        
        return orders
    except Exception as e:
        logger.error(f"Error getting orders for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_account_info() -> Dict[str, Any]:
    """Get account information."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection not initialized")
        
    try:
        data = await redis_client.get("alpaca:account_info")
        if data:
            return json.loads(data)
        return {"error": "Account information not found"}
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_watchlist() -> List[str]:
    """Get trading watchlist."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection not initialized")
        
    try:
        data = await redis_client.get("trading:watchlist")
        if data:
            return json.loads(data)
        return []
    except Exception as e:
        logger.error(f"Error getting watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_latest_analysis() -> Dict[str, Any]:
    """Get latest market analysis."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection not initialized")
        
    try:
        data = await redis_client.get("market_analysis:latest")
        if data:
            return json.loads(data)
        return {"error": "No market analysis found"}
    except Exception as e:
        logger.error(f"Error getting latest analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def create_context(symbols: List[str] = None, include_positions: bool = True,
                        include_market_data: bool = True, include_analysis: bool = True) -> Dict[str, Any]:
    """Create a context object with relevant trading data."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection not initialized")
        
    try:
        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Add account info
        account_info = await get_account_info()
        if "error" not in account_info:
            context["account"] = account_info
        
        # Add positions if requested
        if include_positions:
            positions = await get_all_positions()
            context["positions"] = positions
        
        # Get symbols from positions if not provided
        if not symbols and include_positions:
            position_symbols = [p.get("symbol") for p in context.get("positions", []) if p.get("symbol")]
            watchlist = await get_watchlist()
            symbols = list(set(position_symbols + watchlist))
        
        # Add market data if requested
        if include_market_data and symbols:
            market_data = {}
            for symbol in symbols[:50]:  # Limit to 50 symbols to avoid excessive processing
                symbol_data = await get_market_data(symbol)
                if "error" not in symbol_data:
                    market_data[symbol] = symbol_data
            context["market_data"] = market_data
        
        # Add market analysis if requested
        if include_analysis:
            analysis = await get_latest_analysis()
            if "error" not in analysis:
                context["market_analysis"] = analysis
        
        return context
    except Exception as e:
        logger.error(f"Error creating context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def subscribe_to_updates(channel_pattern: str, callback: Callable) -> Any:
    """Subscribe to Redis PubSub updates."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection not initialized")
        
    try:
        # Create pubsub object
        pubsub = redis_client.pubsub()
        
        # Subscribe to pattern
        await pubsub.psubscribe(**{channel_pattern: callback})
        
        logger.info(f"Subscribed to channel pattern: {channel_pattern}")
        
        # Return pubsub object for later management
        return pubsub
    except Exception as e:
        logger.error(f"Error subscribing to {channel_pattern}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """Verify API key."""
    if not config.get("security", {}).get("enable_auth", True):
        return None
    
    if not credentials:
        raise HTTPException(status_code=401, detail="API key required")
    
    valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")
    if credentials.credentials not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return credentials.credentials

# API routes
@app.get("/server_info")
async def get_server_info():
    """Get server information."""
    return {
        "name": "Redis MCP Tool",
        "version": "1.0.0",
        "status": "running",
        "redis_connected": redis_client is not None,
        "tools": [
            "get_market_data",
            "get_position",
            "get_all_positions", 
            "get_order",
            "get_orders_by_symbol",
            "get_account_info",
            "get_watchlist",
            "get_latest_analysis",
            "create_context"
        ],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/execute_tool/{tool_name}", dependencies=[Depends(verify_api_key)])
async def execute_tool(tool_name: str, request: Request):
    """Execute a tool with the provided arguments."""
    try:
        arguments = await request.json()
        
        if tool_name == "get_market_data":
            symbol = arguments.get("symbol")
            if not symbol:
                raise HTTPException(status_code=400, detail="Symbol is required")
            return await get_market_data(symbol)
        
        elif tool_name == "get_position":
            position_id = arguments.get("position_id")
            symbol = arguments.get("symbol")
            if not position_id and not symbol:
                raise HTTPException(status_code=400, detail="Position ID or symbol is required")
            return await get_position(position_id, symbol)
        
        elif tool_name == "get_all_positions":
            return await get_all_positions()
        
        elif tool_name == "get_order":
            order_id = arguments.get("order_id")
            if not order_id:
                raise HTTPException(status_code=400, detail="Order ID is required")
            return await get_order(order_id)
        
        elif tool_name == "get_orders_by_symbol":
            symbol = arguments.get("symbol")
            if not symbol:
                raise HTTPException(status_code=400, detail="Symbol is required")
            return await get_orders_by_symbol(symbol)
        
        elif tool_name == "get_account_info":
            return await get_account_info()
        
        elif tool_name == "get_watchlist":
            return await get_watchlist()
        
        elif tool_name == "get_latest_analysis":
            return await get_latest_analysis()
        
        elif tool_name == "create_context":
            symbols = arguments.get("symbols")
            include_positions = arguments.get("include_positions", True)
            include_market_data = arguments.get("include_market_data", True)
            include_analysis = arguments.get("include_analysis", True)
            return await create_context(symbols, include_positions, include_market_data, include_analysis)
        
        else:
            raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing tool: {str(e)}")

@app.get("/market_data/{symbol}", dependencies=[Depends(verify_api_key)])
async def market_data_endpoint(symbol: str):
    """Get market data for a symbol."""
    return await get_market_data(symbol)

@app.get("/position/{position_id}", dependencies=[Depends(verify_api_key)])
async def position_endpoint(position_id: str):
    """Get position by ID."""
    return await get_position(position_id=position_id)

@app.get("/positions", dependencies=[Depends(verify_api_key)])
async def positions_endpoint():
    """Get all positions."""
    return await get_all_positions()

@app.get("/account", dependencies=[Depends(verify_api_key)])
async def account_endpoint():
    """Get account information."""
    return await get_account_info()

@app.get("/watchlist", dependencies=[Depends(verify_api_key)])
async def watchlist_endpoint():
    """Get trading watchlist."""
    return await get_watchlist()

@app.get("/analysis", dependencies=[Depends(verify_api_key)])
async def analysis_endpoint():
    """Get latest market analysis."""
    return await get_latest_analysis()

@app.post("/context", dependencies=[Depends(verify_api_key)])
async def context_endpoint(request: ContextRequest):
    """Create a context with relevant trading data."""
    return await create_context(
        request.symbols,
        request.include_positions,
        request.include_market_data,
        request.include_analysis
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    
    # Add to connections list
    websocket_connections.append(websocket)
    
    try:
        # Create a Redis PubSub client
        pubsub = redis_client.pubsub()
        
        # Handle initial subscription message
        data = await websocket.receive_text()
        subscription_data = json.loads(data)
        channels = subscription_data.get("channels", [])
        
        if not channels:
            channels = ["market_data:*", "position:*", "order:*"]
        
        # Subscribe to channels
        for channel in channels:
            await pubsub.psubscribe(channel)
        
        # Message handling task
        async def message_handler():
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message:
                    await websocket.send_json({
                        "channel": message["channel"],
                        "data": json.loads(message["data"]) if isinstance(message["data"], str) else message["data"],
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                await asyncio.sleep(0.01)
        
        # Start message handler
        handler_task = asyncio.create_task(message_handler())
        
        # Keep the connection alive
        while True:
            data = await websocket.receive_text()
            
            # Process commands
            try:
                cmd = json.loads(data)
                
                if cmd.get("type") == "subscribe":
                    for channel in cmd.get("channels", []):
                        await pubsub.psubscribe(channel)
                        
                elif cmd.get("type") == "unsubscribe":
                    for channel in cmd.get("channels", []):
                        await pubsub.punsubscribe(channel)
                
                elif cmd.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()})
            
            except json.JSONDecodeError:
                continue
                
    except WebSocketDisconnect:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)
        
        # Clean up PubSub
        await pubsub.close()
        
        if handler_task:
            handler_task.cancel()
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)
        
        # Clean up PubSub
        if pubsub:
            await pubsub.close()
        
        if handler_task:
            handler_task.cancel()

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    global config, redis_client
    
    parser = argparse.ArgumentParser(description='Redis MCP Tool')
    parser.add_argument('--config', type=str, default='config/llm_config.yaml', help='Path to configuration file')
    args, _ = parser.parse_known_args()
    
    try:
        # Load configuration
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Configuration file {args.config} not found, using default configuration")
            config = {}
        
        # Redis configuration
        redis_host = os.getenv("REDIS_HOST", config.get("redis", {}).get("host", "localhost"))
        redis_port = int(os.getenv("REDIS_PORT", config.get("redis", {}).get("port", "6379")))
        redis_db = int(os.getenv("REDIS_DB", config.get("redis", {}).get("db", "0")))
        redis_password = os.getenv("REDIS_PASSWORD", config.get("redis", {}).get("password"))
        
        # Connect to Redis
        redis_client = aioredis.Redis(
            host=redis_host,
            port=redis_port,
            
db=redis_db,
            password=redis_password,
            decode_responses=True
        )
        
        # Test connection
        await redis_client.ping()
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    global redis_client
    
    # Close all WebSocket connections
    for ws in websocket_connections:
        try:
            await ws.close()
        except:
            pass
    
    # Close Redis connection
    if redis_client:
        await redis_client.close()
        redis_client = None
    
    logger.info("Redis MCP Tool shutdown complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Redis MCP Tool')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8011, help='Port to bind the server to')
    parser.add_argument('--config', type=str, default='config/llm_config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)