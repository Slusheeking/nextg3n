#!/usr/bin/env python
"""
Alpaca MCP Tool for NextG3N Trading System

This module implements a standardized MCP server that provides:
1. Connection to Alpaca API for market data and trading
2. Execution of orders from the AI trading system
3. Retrieving portfolio and account information
4. Real-time market data streaming

It serves as a bridge between the trading system components and Alpaca.
"""

import os
import json
import time
import asyncio
import logging
import argparse
import yaml
import uvicorn
from typing import Dict, List, Any, Optional, Union
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
logger = logging.getLogger("alpaca_mcp_tool")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Models for API requests and responses
class OrderRequest(BaseModel):
    symbol: str
    quantity: float
    side: str = "buy"
    order_type: str = "market"
    time_in_force: str = "day"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None
    extended_hours: bool = False

class PositionRequest(BaseModel):
    symbol: str

class MarketDataRequest(BaseModel):
    symbols: List[str]
    timeframe: str = "1Min"
    start: Optional[str] = None
    end: Optional[str] = None
    limit: int = 100

class WebhookRequest(BaseModel):
    name: str
    url: str
    active: bool = True

# Create FastAPI app
app = FastAPI(
    title="Alpaca MCP Tool",
    description="Model Context Protocol Tool for Alpaca API access",
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
session = None
redis_client = None
alpaca_api_key = None
alpaca_api_secret = None
alpaca_base_url = None
alpaca_data_url = None
paper_trading = True
websocket_connections = []

# Tool functions
async def get_account() -> Dict[str, Any]:
    """Get account information from Alpaca."""
    global session, alpaca_api_key, alpaca_api_secret, alpaca_base_url
    
    if not session:
        raise HTTPException(status_code=503, detail="HTTP session not initialized")
    
    try:
        headers = {
            "APCA-API-KEY-ID": alpaca_api_key,
            "APCA-API-SECRET-KEY": alpaca_api_secret
        }
        
        async with session.get(f"{alpaca_base_url}/v2/account", headers=headers) as response:
            if response.status == 200:
                account_data = await response.json()
                
                # Store in Redis if available
                if redis_client:
                    await redis_client.set("alpaca:account_info", json.dumps(account_data))
                
                return account_data
            
            error_text = await response.text()
            logger.error(f"Error getting account info: {response.status} - {error_text}")
            raise HTTPException(status_code=response.status, detail=error_text)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_positions() -> List[Dict[str, Any]]:
    """Get all positions from Alpaca."""
    global session, alpaca_api_key, alpaca_api_secret, alpaca_base_url
    
    if not session:
        raise HTTPException(status_code=503, detail="HTTP session not initialized")
    
    try:
        headers = {
            "APCA-API-KEY-ID": alpaca_api_key,
            "APCA-API-SECRET-KEY": alpaca_api_secret
        }
        
        async with session.get(f"{alpaca_base_url}/v2/positions", headers=headers) as response:
            if response.status == 200:
                positions = await response.json()
                
                # Store in Redis if available
                if redis_client:
                    await redis_client.set("alpaca:positions", json.dumps(positions))
                    
                    # Also store individual positions
                    for position in positions:
                        position_id = position.get("asset_id")
                        symbol = position.get("symbol")
                        
                        if position_id:
                            # Create position data for the trading system
                            position_data = {
                                "position_id": position_id,
                                "symbol": symbol,
                                "quantity": float(position.get("qty", 0)),
                                "entry_price": float(position.get("avg_entry_price", 0)),
                                "current_price": float(position.get("current_price", 0)),
                                "market_value": float(position.get("market_value", 0)),
                                "cost_basis": float(position.get("cost_basis", 0)),
                                "unrealized_pl": float(position.get("unrealized_pl", 0)),
                                "unrealized_plpc": float(position.get("unrealized_plpc", 0)),
                                "side": position.get("side", ""),
                                "updated_at": datetime.now().isoformat()
                            }
                            
                            # Store in Redis
                            await redis_client.set(
                                f"position:{position_id}:active",
                                json.dumps(position_data)
                            )
                            
                            # Publish to position data channel
                            await redis_client.publish(
                                "channel:position_data",
                                json.dumps({
                                    "type": "position_update",
                                    "data": position_data
                                })
                            )
                
                return positions
            
            error_text = await response.text()
            logger.error(f"Error getting positions: {response.status} - {error_text}")
            raise HTTPException(status_code=response.status, detail=error_text)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_position(symbol: str) -> Dict[str, Any]:
    """Get a specific position from Alpaca."""
    global session, alpaca_api_key, alpaca_api_secret, alpaca_base_url
    
    if not session:
        raise HTTPException(status_code=503, detail="HTTP session not initialized")
    
    try:
        headers = {
            "APCA-API-KEY-ID": alpaca_api_key,
            "APCA-API-SECRET-KEY": alpaca_api_secret
        }
        
        async with session.get(f"{alpaca_base_url}/v2/positions/{symbol}", headers=headers) as response:
            if response.status == 200:
                position = await response.json()
                
                # Process and store in Redis if available
                if redis_client:
                    position_id = position.get("asset_id")
                    if position_id:
                        # Create position data for the trading system
                        position_data = {
                            "position_id": position_id,
                            "symbol": symbol,
                            "quantity": float(position.get("qty", 0)),
                            "entry_price": float(position.get("avg_entry_price", 0)),
                            "current_price": float(position.get("current_price", 0)),
                            "market_value": float(position.get("market_value", 0)),
                            "cost_basis": float(position.get("cost_basis", 0)),
                            "unrealized_pl": float(position.get("unrealized_pl", 0)),
                            "unrealized_plpc": float(position.get("unrealized_plpc", 0)),
                            "side": position.get("side", ""),
                            "updated_at": datetime.now().isoformat()
                        }
                        
                        # Store in Redis
                        await redis_client.set(
                            f"position:{position_id}:active",
                            json.dumps(position_data)
                        )
                        
                        # Publish to position data channel
                        await redis_client.publish(
                            "channel:position_data",
                            json.dumps({
                                "type": "position_update",
                                "data": position_data
                            })
                        )
                
                return position
            
            error_text = await response.text()
            if response.status == 404:
                return {"symbol": symbol, "error": "Position not found"}
            
            logger.error(f"Error getting position for {symbol}: {response.status} - {error_text}")
            raise HTTPException(status_code=response.status, detail=error_text)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting position for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_orders() -> List[Dict[str, Any]]:
    """Get all orders from Alpaca."""
    global session, alpaca_api_key, alpaca_api_secret, alpaca_base_url
    
    if not session:
        raise HTTPException(status_code=503, detail="HTTP session not initialized")
    
    try:
        headers = {
            "APCA-API-KEY-ID": alpaca_api_key,
            "APCA-API-SECRET-KEY": alpaca_api_secret
        }
        
        # Get all orders, including filled ones
        status = "open,filled,partially_filled,canceled,expired,replaced,pending_cancel,pending_replace"
        async with session.get(f"{alpaca_base_url}/v2/orders?status={status}", headers=headers) as response:
            if response.status == 200:
                orders = await response.json()
                
                # Store in Redis if available
                if redis_client:
                    await redis_client.set("alpaca:orders", json.dumps(orders))
                    
                    # Process individual orders
                    for order in orders:
                        order_id = order.get("id")
                        if order_id:
                            # Create order data
                            order_data = {
                                "order_id": order_id,
                                "client_order_id": order.get("client_order_id"),
                                "symbol": order.get("symbol"),
                                "side": order.get("side"),
                                "quantity": float(order.get("qty", 0)),
                                "filled_quantity": float(order.get("filled_qty", 0)),
                                "type": order.get("type"),
                                "status": order.get("status"),
                                "created_at": order.get("created_at"),
                                "updated_at": order.get("updated_at"),
                                "submitted_at": order.get("submitted_at"),
                                "filled_at": order.get("filled_at"),
                                "expired_at": order.get("expired_at"),
                                "canceled_at": order.get("canceled_at"),
                                "limit_price": order.get("limit_price"),
                                "stop_price": order.get("stop_price"),
                                "time_in_force": order.get("time_in_force")
                            }
                            
                            # Store in Redis
                            await redis_client.set(
                                f"order:{order_id}",
                                json.dumps(order_data)
                            )
                
                return orders
            
            error_text = await response.text()
            logger.error(f"Error getting orders: {response.status} - {error_text}")
            raise HTTPException(status_code=response.status, detail=error_text)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def submit_order(order_request: OrderRequest) -> Dict[str, Any]:
    """Submit an order to Alpaca."""
    global session, alpaca_api_key, alpaca_api_secret, alpaca_base_url
    
    if not session:
        raise HTTPException(status_code=503, detail="HTTP session not initialized")
    
    try:
        headers = {
            "APCA-API-KEY-ID": alpaca_api_key,
            "APCA-API-SECRET-KEY": alpaca_api_secret,
            "Content-Type": "application/json"
        }
        
        # Build order payload
        order_data = {
            "symbol": order_request.symbol,
            "qty": str(order_request.quantity),
            "side": order_request.side,
            "type": order_request.order_type,
            "time_in_force": order_request.time_in_force,
            "extended_hours": order_request.extended_hours
        }
        
        # Add optional parameters if provided
        if order_request.limit_price:
            order_data["limit_price"] = str(order_request.limit_price)
        
        if order_request.stop_price:
            order_data["stop_price"] = str(order_request.stop_price)
        
        if order_request.client_order_id:
            order_data["client_order_id"] = order_request.client_order_id
        
        # Submit order
        async with session.post(f"{alpaca_base_url}/v2/orders", headers=headers, json=order_data) as response:
            if response.status == 200 or response.status == 201:
                order_result = await response.json()
                
                # Process and store in Redis if available
                if redis_client and order_result:
                    order_id = order_result.get("id")
                    if order_id:
                        # Create order data
                        processed_order = {
                            "order_id": order_id,
                            "client_order_id": order_result.get("client_order_id"),
                            "symbol": order_result.get("symbol"),
                            "side": order_result.get("side"),
                            "quantity": float(order_result.get("qty", 0)),
                            "filled_quantity": float(order_result.get("filled_qty", 0)),
                            "type": order_result.get("type"),
                            "status": order_result.get("status"),
                            "created_at": order_result.get("created_at"),
                            "updated_at": order_result.get("updated_at"),
                            "submitted_at": order_result.get("submitted_at"),
                            "filled_at": order_result.get("filled_at"),
                            "expired_at": order_result.get("expired_at"),
                            "canceled_at": order_result.get("canceled_at"),
                            "limit_price": order_result.get("limit_price"),
                            "stop_price": order_result.get("stop_price"),
                            "time_in_force": order_result.get("time_in_force")
                        }
                        
                        # Store in Redis
                        await redis_client.set(
                            f"order:{order_id}",
                            json.dumps(processed_order)
                        )
                        
                        # Publish to order channel
                        await redis_client.publish(
                            "channel:order_status",
                            json.dumps({
                                "type": "order_created",
                                "data": processed_order
                            })
                        )
                
                return order_result
            
            error_text = await response.text()
            logger.error(f"Error submitting order: {response.status} - {error_text}")
            raise HTTPException(status_code=response.status, detail=error_text)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def cancel_order(order_id: str) -> Dict[str, Any]:
    """Cancel an order in Alpaca."""
    global session, alpaca_api_key, alpaca_api_secret, alpaca_base_url
    
    if not session:
        raise HTTPException(status_code=503, detail="HTTP session not initialized")
    
    try:
        headers = {
            "APCA-API-KEY-ID": alpaca_api_key,
            "APCA-API-SECRET-KEY": alpaca_api_secret
        }
        
        # Submit cancellation
        async with session.delete(f"{alpaca_base_url}/v2/orders/{order_id}", headers=headers) as response:
            if response.status == 200 or response.status == 204:
                # Check if we have any response content
                try:
                    result = await response.json()
                except:
                    result = {"status": "success", "order_id": order_id}
                
                # Update in Redis if available
                if redis_client:
                    order_data = await redis_client.get(f"order:{order_id}")
                    if order_data:
                        order = json.loads(order_data)
                        order["status"] = "canceled"
                        order["canceled_at"] = datetime.now(timezone.utc).isoformat()
                        
                        await redis_client.set(f"order:{order_id}", json.dumps(order))
                        
                        # Publish update
                        await redis_client.publish(
                            "channel:order_status",
                            json.dumps({
                                "type": "order_canceled",
                                "data": order
                            })
                        )
                
                return result
            
            error_text = await response.text()
            logger.error(f"Error canceling order {order_id}: {response.status} - {error_text}")
            raise HTTPException(status_code=response.status, detail=error_text)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error canceling order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_market_data(symbols: List[str], timeframe: str = "1Min", 
                         start: Optional[str] = None, end: Optional[str] = None, 
                         limit: int = 100) -> Dict[str, Any]:
    """Get market data from Alpaca."""
    global session, alpaca_api_key, alpaca_api_secret, alpaca_data_url
    
    if not session:
        raise HTTPException(status_code=503, detail="HTTP session not initialized")
    
    try:
        headers = {
            "APCA-API-KEY-ID": alpaca_api_key,
            "APCA-API-SECRET-KEY": alpaca_api_secret
        }
        
        # Build query parameters
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "limit": limit
        }
        
        if start:
            params["start"] = start
        
        if end:
            params["end"] = end
        
        # Get market data
        async with session.get(f"{alpaca_data_url}/v2/stocks/bars", headers=headers, params=params) as response:
            if response.status == 200:
                market_data = await response.json()
                
                # Process and store in Redis if available
                if redis_client:
                    for symbol, bars in market_data.get("bars", {}).items():
                        if bars and len(bars) > 0:
                            latest_bar = bars[-1]
                            
                            # Create market data entry
                            processed_data = {
                                "symbol": symbol,
                                "price": float(latest_bar.get("c", 0)),
                                "open": float(latest_bar.get("o", 0)),
                                "high": float(latest_bar.get("h", 0)),
                                "low": float(latest_bar.get("l", 0)),
                                "close": float(latest_bar.get("c", 0)),
                                "volume": int(latest_bar.get("v", 0)),
                                "timestamp": latest_bar.get("t"),
                                "updated_at": datetime.now().isoformat()
                            }
                            
                            # Store in Redis
                            await redis_client.set(
                                f"market_data:{symbol}:latest",
                                json.dumps(processed_data)
                            )
                            
                            # Publish to market data channel
                            await redis_client.publish(
                                "channel:market_data",
                                json.dumps({
                                    "type": "market_data_update",
                                    "data": processed_data
                                })
                            )
                
                return market_data
            
            error_text = await response.text()
            logger.error(f"Error getting market data: {response.status} - {error_text}")
            raise HTTPException(status_code=response.status, detail=error_text)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_quote(symbol: str) -> Dict[str, Any]:
    """Get latest quote for a symbol from Alpaca."""
    global session, alpaca_api_key, alpaca_api_secret, alpaca_data_url
    
    if not session:
        raise HTTPException(status_code=503, detail="HTTP session not initialized")
    
    try:
        headers = {
            "APCA-API-KEY-ID": alpaca_api_key,
            "APCA-API-SECRET-KEY": alpaca_api_secret
        }
        
        # Get quote
        async with session.get(f"{alpaca_data_url}/v2/stocks/{symbol}/quotes/latest", headers=headers) as response:
            if response.status == 200:
                quote_data = await response.json()
                
                # Process and store in Redis if available
                if redis_client and quote_data.get("quote"):
                    quote = quote_data["quote"]
                    
                    # Create quote data entry
                    processed_data = {
                        "symbol": symbol,
                        "askprice": float(quote.get("ap", 0)),
                        "asksize": int(quote.get("as", 0)),
                        "bidprice": float(quote.get("bp", 0)),
                        "bidsize": int(quote.get("bs", 0)),
                        "timestamp": quote.get("t"),
                        "updated_at": datetime.now().isoformat()
                    }
                    
                    # Store in Redis
                    await redis_client.set(
                        f"market_data:{symbol}:quote",
                        json.dumps(processed_data)
                    )
                    
                    # Update latest market data with bid/ask
                    latest_data = await redis_client.get(f"market_data:{symbol}:latest")
                    if latest_data:
                        market_data = json.loads(latest_data)
                        market_data.update({
                            "askprice": processed_data["askprice"],
                            "asksize": processed_data["asksize"],
                            "bidprice": processed_data["bidprice"],
                            "bidsize": processed_data["bidsize"],
                            "updated_at": datetime.now().isoformat()
                        })
                        
                        await redis_client.set(
                            f"market_data:{symbol}:latest",
                            json.dumps(market_data)
                        )
                        
                        # Publish update
                        await redis_client.publish(
                            "channel:market_data",
                            json.dumps({
                                "type": "market_data_update",
                                "data": market_data
                            })
                        )
                
                return quote_data
            
            error_text = await response.text()
            logger.error(f"Error getting quote for {symbol}: {response.status} - {error_text}")
            raise HTTPException(status_code=response.status, detail=error_text)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_market_clock() -> Dict[str, Any]:
    """Get market clock information from Alpaca."""
    global session, alpaca_api_key, alpaca_api_secret, alpaca_base_url
    
    if not session:
        raise HTTPException(status_code=503, detail="HTTP session not initialized")
    
    try:
        headers = {
            "APCA-API-KEY-ID": alpaca_api_key,
            "APCA-API-SECRET-KEY": alpaca_api_secret
        }
        
        # Get market clock
        async with session.get(f"{alpaca_base_url}/v2/clock", headers=headers) as response:
            if response.status == 200:
                clock_data = await response.json()
                
                # Store in Redis if available
                if redis_client:
                    await redis_client.set("alpaca:market_clock", json.dumps(clock_data))
                
                return clock_data
            
            error_text = await response.text()
            logger.error(f"Error getting market clock: {response.status} - {error_text}")
            raise HTTPException(status_code=response.status, detail=error_text)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market clock: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_portfolio() -> Dict[str, Any]:
    """Get portfolio information including account and positions."""
    try:
        # Get account information
        account = await get_account()
        
        # Get positions
        positions = await get_positions()
        
        # Get open orders
        orders = await get_orders()
        
        # Combine into portfolio
        portfolio = {
            "account": account,
            "positions": positions,
            "orders": orders,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Calculate portfolio statistics
        if positions:
            portfolio["statistics"] = {
                "total_positions": len(positions),
                "long_positions": sum(1 for p in positions if p.get("side") == "long"),
                "short_positions": sum(1 for p in positions if p.get("side") == "short"),
                "total_market_value": sum(float(p.get("market_value", 0)) for p in positions),
                "total_cost_basis": sum(float(p.get("cost_basis", 0)) for p in positions),
                "total_unrealized_pl": sum(float(p.get("unrealized_pl", 0)) for p in positions)
            }
        
        return portfolio
    
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
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
        "name": "Alpaca MCP Tool",
        "version": "1.0.0",
        "status": "running",
        "alpaca_connected": session is not None,
        "redis_connected": redis_client is not None,
        "paper_trading": paper_trading,
        "tools": [
            "get_account",
            "get_positions",
            "get_position",
            "get_orders",
            "submit_order",
            "cancel_order",
            "get_market_data",
            "get_quote",
            "get_market_clock",
            "get_portfolio"
        ],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/execute_tool/{tool_name}", dependencies=[Depends(verify_api_key)])
async def execute_tool(tool_name: str, request: Request):
    """Execute a tool with the provided arguments."""
    try:
        arguments = await request.json()
        
        if tool_name == "get_account":
            return await get_account()
        
        elif tool_name == "get_positions":
            return await get_positions()
        
        elif tool_name == "get_position":
            symbol = arguments.get("symbol")
            if not symbol:
                raise HTTPException(status_code=400, detail="Symbol is required")
            return await get_position(symbol)
        
        elif tool_name == "get_orders":
            return await get_orders()
        
        elif tool_name == "submit_order":
            order_request = OrderRequest(**arguments)
            return await submit_order(order_request)
        
        elif tool_name == "cancel_order":
            order_id = arguments.get("order_id")
            if not order_id:
                raise HTTPException(status_code=400, detail="Order ID is required")
            return await cancel_order(order_id)
        
        elif tool_name == "get_market_data":
            symbols = arguments.get("symbols")
            if not symbols:
                raise HTTPException(status_code=400, detail="Symbols are required")
            
            timeframe = arguments.get("timeframe", "1Min")
            start = arguments.get("start")
            end = arguments.get("end")
            limit = arguments.get("limit", 100)
            
            return await get_market_data(symbols, timeframe, start, end, limit)
        
        elif tool_name == "get_quote":
            symbol = arguments.get("symbol")
            if not symbol:
                raise HTTPException(status_code=400, detail="Symbol is required")
            return await get_quote(symbol)
        
        elif tool_name == "get_market_clock":
            return await get_market_clock()
        
        elif tool_name == "get_portfolio":
            return await get_portfolio()
        
        else:
            raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing tool: {str(e)}")

@app.get("/account", dependencies=[Depends(verify_api_key)])
async def account_endpoint():
    """Get account information."""
    return await get_account()

@app.get("/positions", dependencies=[Depends(verify_api_key)])
async def positions_endpoint():
    """Get all positions."""
    return await get_positions()

@app.get("/positions/{symbol}", dependencies=[Depends(verify_api_key)])
async def position_endpoint(symbol: str):
    """Get position for a specific symbol."""
    return await get_position(symbol)

@app.get("/orders", dependencies=[Depends(verify_api_key)])
async def orders_endpoint():
    """Get all orders."""
    return await get_orders()

@app.post("/orders", dependencies=[Depends(verify_api_key)])
async def create_order_endpoint(order: OrderRequest):
    """Create a new order."""
    return await submit_order(order)

@app.delete("/orders/{order_id}", dependencies=[Depends(verify_api_key)])
async def cancel_order_endpoint(order_id: str):
    """Cancel an order."""
    return await cancel_order(order_id)

@app.get("/market_data", dependencies=[Depends(verify_api_key)])
async def market_data_endpoint(symbols: str, timeframe: str = "1Min", start: Optional[str] = None, 
                              end: Optional[str] = None, limit: int = 100):
    """Get market data."""
    symbol_list = symbols.split(",")
    return await get_market_data(symbol_list, timeframe, start, end, limit)

@app.get("/quotes/{symbol}", dependencies=[Depends(verify_api_key)])
async def quote_endpoint(symbol: str):
    """Get latest quote for a symbol."""
    return await get_quote(symbol)

@app.get("/clock", dependencies=[Depends(verify_api_key)])
async def clock_endpoint():
    """Get market clock information."""
    return await get_market_clock()

@app.get("/portfolio", dependencies=[Depends(verify_api_key)])
async def portfolio_endpoint():
    """Get portfolio information."""
    return await get_portfolio()

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
            channels = ["channel:market_data", "channel:order_status", "channel:position_data"]
        
        # Subscribe to channels
        for channel in channels:
            await pubsub.subscribe(channel)
        
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
                        await pubsub.subscribe(channel)
                        
                elif cmd.get("type") == "unsubscribe":
                    for channel in cmd.get("channels", []):
                        await pubsub.unsubscribe(channel)
                
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
    global config, session, redis_client, alpaca_api_key, alpaca_api_secret, alpaca_base_url, alpaca_data_url, paper_trading
    
    parser = argparse.ArgumentParser(description='Alpaca MCP Tool')
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
        
        # Initialize HTTP session
        session = aiohttp.ClientSession()
        
        # Redis configuration
        redis_host = os.getenv("REDIS_HOST", config.get("redis", {}).get("host", "localhost"))
        redis_port = int(os.getenv("REDIS_PORT", config.get("redis", {}).get("port", "6379")))
        redis_db = int(os.getenv("REDIS_DB", config.get("redis", {}).get("db", "0")))
        redis_password = os.getenv("REDIS_PASSWORD", config.get("redis", {}).get("password"))
        
        # Alpaca configuration
        alpaca_api_key = os.getenv("ALPACA_API_KEY", config.get("alpaca", {}).get("api_key", ""))
        alpaca_api_secret = os.getenv("ALPACA_API_SECRET", config.get("alpaca", {}).get("api_secret", ""))
        paper_trading = os.getenv("PAPER_TRADING", "").lower() in ["true", "1", "yes", "y"]
        if paper_trading is None:
            paper_trading = config.get("alpaca", {}).get("paper_trading", True)
        
        # Set Alpaca URLs based on paper trading flag
        if paper_trading:
            alpaca_base_url = "https://paper-api.alpaca.markets"
        else:
            alpaca_base_url = "https://api.alpaca.markets"
        
        alpaca_data_url = "https://data.alpaca.markets"
        
        # Connect to Redis
        redis_client = aioredis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True
        )
        
        # Test Redis connection
        await redis_client.ping()
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        
        # Test Alpaca connection by getting account info
        account = await get_account()
        logger.info(f"Connected to Alpaca API for account: {account.get('account_number')}")
        logger.info(f"Alpaca trading mode: {'Paper' if paper_trading else 'Live'}")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    global session, redis_client
    
    # Close all WebSocket connections
    for ws in websocket_connections:
        try:
            await ws.close()
        except:
            pass
    
    # Close HTTP session
    if session:
        await session.close()
        session = None
    
    # Close Redis connection
    if redis_client:
        await redis_client.close()
        redis_client = None
    
    logger.info("Alpaca MCP Tool shutdown complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Alpaca MCP Tool')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8010, help='Port to bind the server to')
    parser.add_argument('--config', type=str, default='config/llm_config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)