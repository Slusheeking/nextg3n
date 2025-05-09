"""
Polygon WebSocket FastAPI Server.
Standard API for real-time market data from Polygon WebSocket API.
"""

import os
import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from dotenv import load_dotenv
import websockets
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("polygon_websocket_api")

# Load environment variables
load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("POLYGON_API_KEY", "")
WEBSOCKET_URL = os.getenv("POLYGON_WEBSOCKET_URL", "wss://socket.polygon.io/stocks")
BUFFER_SIZE = int(os.getenv("POLYGON_BUFFER_SIZE", "1000"))
MAX_SESSION_TIME = int(os.getenv("POLYGON_MAX_SESSION_TIME", "14400"))
RATE_LIMIT = int(os.getenv("POLYGON_RATE_LIMIT", "60"))

# --- Data Models ---
class RealtimeDataRequest(BaseModel):
    symbols: List[str]
    channels: List[str] = ["T", "Q"]
    duration_seconds: int = 60
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("At least one symbol must be provided")
        if len(v) > 100:
            raise ValueError("Maximum of 100 symbols allowed")
        return v
    
    @validator('channels')
    def validate_channels(cls, v):
        allowed_channels = ["T", "Q", "A", "AM"]
        for channel in v:
            if channel not in allowed_channels:
                raise ValueError(f"Channel {channel} not supported. Allowed: {allowed_channels}")
        return v

class StreamRequest(BaseModel):
    symbols: List[str]
    channels: List[str] = ["T", "Q"]
    session_id: Optional[str] = None
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("At least one symbol must be provided")
        if len(v) > 100:
            raise ValueError("Maximum of 100 symbols allowed")
        return v

class StopStreamRequest(BaseModel):
    session_id: str

class SnapshotRequest(BaseModel):
    session_id: str
    max_messages: Optional[int] = 1000

# --- Streaming Session Management ---
active_streams: Dict[str, Dict[str, Any]] = {}

async def polygon_ws_stream(symbols: List[str], channels: List[str], duration: int, buffer: List[dict]):
    """Connect to Polygon WebSocket API and stream market data."""
    if not symbols or not channels:
        logger.error("Symbols or channels list is empty")
        raise HTTPException(status_code=400, detail="Symbols and channels must not be empty")
        
    url = WEBSOCKET_URL
    api_key = API_KEY
    
    if not api_key:
        logger.error("Polygon API key not configured")
        raise HTTPException(status_code=500, detail="Polygon API key not configured")
    
    max_retries = 5
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                logger.debug(f"Connected to Polygon WebSocket at {url} (attempt {attempt+1})")
                
                # Authenticate
                await ws.send(json.dumps({"action": "auth", "params": api_key}))
                auth_response = await asyncio.wait_for(ws.recv(), timeout=5)
                auth_data = json.loads(auth_response)
                
                if isinstance(auth_data, list) and auth_data[0].get("status") == "auth_success":
                    logger.info("Authentication with Polygon WebSocket successful")
                else:
                    logger.error(f"Authentication failed: {auth_data}")
                    raise Exception("Authentication failed")
                
                # Subscribe to channels
                subscription_params = ",".join(f"{ch}.{sym}" for ch in channels for sym in symbols)
                logger.debug(f"Subscribing to: {subscription_params}")
                await ws.send(json.dumps({"action": "subscribe", "params": subscription_params}))
                
                # Start receiving data
                start_time = time.time()
                buffer_limit = BUFFER_SIZE
                
                while time.time() - start_time < duration:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        
                        # Manage buffer size
                        if len(buffer) >= buffer_limit:
                            del buffer[:len(buffer) - buffer_limit + 1]
                        buffer.append(data)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed unexpectedly")
                        break
                
                # Unsubscribe and close
                await ws.send(json.dumps({"action": "unsubscribe", "params": subscription_params}))
                logger.debug("Sent unsubscribe request to Polygon WebSocket")
                break
        except Exception as e:
            logger.error(f"WebSocket connection error (attempt {attempt+1}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Retrying connection in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed to connect after {max_retries} attempts")
                raise

# --- Process Market Data ---
def process_market_data(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Process raw websocket data into structured market data."""
    if not data:
        logger.warning("No data provided for processing")
        return {}
        
    processed = {}
    symbol_data = {}
    
    # Group data by symbol
    for msg in data:
        if isinstance(msg, list):
            for item in msg:
                if "sym" in item:
                    symbol = item["sym"]
                    if symbol not in symbol_data:
                        symbol_data[symbol] = []
                    symbol_data[symbol].append(item)
        elif isinstance(msg, dict) and "sym" in msg:
            symbol = msg["sym"]
            if symbol not in symbol_data:
                symbol_data[symbol] = []
            symbol_data[symbol].append(msg)
    
    # Process each symbol's data
    for symbol, messages in symbol_data.items():
        trades = [msg for msg in messages if msg.get("ev") == "T"]
        quotes = [msg for msg in messages if msg.get("ev") == "Q"]
        
        if not trades:
            continue
            
        prices = [trade.get("p", 0) for trade in trades if trade.get("p", 0) > 0]
        volumes = [trade.get("s", 0) for trade in trades if trade.get("s", 0) > 0]
        
        if not prices or not volumes:
            continue
            
        open_price = prices[0]
        high_price = max(prices)
        low_price = min(prices)
        close_price = prices[-1]
        volume = sum(volumes)
        vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes) if sum(volumes) > 0 else 0
        
        processed[symbol] = {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "vwap": vwap,
            "timestamp": datetime.now().isoformat()
        }
    
    return processed

# --- FastAPI Server ---
app = FastAPI(
    title="Polygon WebSocket API",
    description="Standard API for real-time market data from Polygon WebSocket",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Polygon WebSocket API starting up")
    if not API_KEY:
        logger.warning("Polygon API key not configured - service will not function correctly")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Polygon WebSocket API shutting down")
    
    for session_id, stream_data in list(active_streams.items()):
        logger.info(f"Cancelling active stream {session_id}")
        try:
            stream_data["task"].cancel()
        except Exception as e:
            logger.error(f"Error cancelling stream {session_id}: {str(e)}")

@app.get("/api/info")
async def get_api_info():
    """Get information about the API."""
    return {
        "name": "Polygon WebSocket API",
        "version": "1.0.0",
        "description": "Standard API for real-time market data from Polygon WebSocket",
        "endpoints": [
            "/api/market/realtime",
            "/api/stream/start",
            "/api/stream/stop",
            "/api/stream/snapshot",
            "/api/stream/active"
        ]
    }

@app.post("/api/market/realtime")
async def fetch_realtime_data(req: RealtimeDataRequest):
    """Fetch real-time market data for specified symbols."""
    logger.info(f"Received realtime data request for symbols: {req.symbols}")
    
    try:
        buffer = []
        await polygon_ws_stream(req.symbols, req.channels, req.duration_seconds, buffer)
        
        # Process the raw data
        processed_data = process_market_data(buffer)
        
        logger.info(f"Completed realtime data request, received {len(buffer)} messages")
        
        return {
            "success": True,
            "raw_data": buffer,
            "processed_data": processed_data,
            "symbols": req.symbols,
            "channels": req.channels,
            "timestamp": datetime.utcnow().isoformat(),
            "message_count": len(buffer)
        }
    except Exception as e:
        logger.error(f"Error processing realtime data request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching realtime data: {str(e)}")

@app.post("/api/stream/start")
async def start_stream(req: StreamRequest):
    """Start a persistent WebSocket stream for market data."""
    if not API_KEY:
        logger.error("Polygon API key not configured")
        raise HTTPException(status_code=500, detail="Polygon API key not configured")
        
    if len(req.symbols) > 100:
        logger.warning(f"Too many symbols requested: {len(req.symbols)}, limiting to 100")
        req.symbols = req.symbols[:100]
        
    session_id = req.session_id or f"stream_{int(time.time())}"
    logger.info(f"Starting stream session {session_id} for symbols: {req.symbols}")
    
    # Check if session already exists
    if session_id in active_streams:
        logger.warning(f"Session ID {session_id} already exists")
        raise HTTPException(status_code=400, detail=f"Session ID {session_id} already exists")
    
    buffer = []
    max_session_time = MAX_SESSION_TIME
    task = asyncio.create_task(polygon_ws_stream(req.symbols, req.channels, max_session_time, buffer))
    
    session_data = {
        "task": task,
        "buffer": buffer,
        "symbols": req.symbols,
        "channels": req.channels,
        "start_time": datetime.utcnow().isoformat(),
        "last_accessed": time.time()
    }
    
    active_streams[session_id] = session_data
    
    logger.info(f"Stream session {session_id} started successfully")
    
    return {
        "success": True,
        "session_id": session_id,
        "symbols": req.symbols,
        "channels": req.channels,
        "start_time": session_data["start_time"],
        "max_session_time": f"{max_session_time//3600} hours"
    }

@app.post("/api/stream/stop")
async def stop_stream(req: StopStreamRequest):
    """Stop an active WebSocket stream."""
    session_id = req.session_id
    logger.info(f"Stopping stream session {session_id}")
    
    if session_id not in active_streams:
        logger.warning(f"Session ID {session_id} not found")
        raise HTTPException(status_code=404, detail=f"Session ID {session_id} not found")
    
    task = active_streams[session_id]["task"]
    task.cancel()
    
    del active_streams[session_id]
    
    logger.info(f"Stream session {session_id} stopped successfully")
    
    return {
        "success": True,
        "session_id": session_id,
        "message": f"Stream {session_id} stopped successfully"
    }

@app.post("/api/stream/snapshot")
async def get_snapshot(req: SnapshotRequest):
    """Get a snapshot of data from an active stream."""
    session_id = req.session_id
    logger.info(f"Getting snapshot for stream session {session_id}")
    
    if session_id not in active_streams:
        logger.warning(f"Session ID {session_id} not found")
        raise HTTPException(status_code=404, detail=f"Session ID {session_id} not found")
    
    buffer = active_streams[session_id]["buffer"]
    active_streams[session_id]["last_accessed"] = time.time()
    
    max_messages = req.max_messages or 1000
    if len(buffer) > max_messages:
        logger.info(f"Limiting snapshot response to {max_messages} messages (total: {len(buffer)})")
        response_data = buffer[-max_messages:]
    else:
        response_data = buffer
    
    # Process the raw data
    processed_data = process_market_data(response_data)
    
    logger.info(f"Returning snapshot with {len(response_data)} messages for session {session_id}")
    
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "raw_data": response_data,
        "processed_data": processed_data,
        "total_messages": len(buffer),
        "returned_messages": len(response_data),
        "session_id": session_id
    }

@app.get("/api/stream/active")
async def get_active_sessions():
    """Get a list of all active streaming sessions."""
    try:
        sessions = []
        for session_id, session_data in active_streams.items():
            sessions.append({
                "session_id": session_id,
                "symbols": session_data["symbols"],
                "channels": session_data["channels"],
                "start_time": session_data["start_time"],
                "buffer_size": len(session_data["buffer"])
            })
        return {
            "success": True,
            "count": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"Error getting active sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving active sessions: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Check the health status of the API."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "active_streams": len(active_streams),
        "api_key_configured": bool(API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)