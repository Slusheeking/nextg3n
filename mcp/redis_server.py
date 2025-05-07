"""
Redis MCP FastAPI Server for LLM integration (production).
Provides key-value, hash, list, set, and pub/sub operations on Redis.
All configuration is contained in this file.
"""

import os
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date
from dotenv import load_dotenv
from monitor.logging_utils import get_logger
from storage.redis_cluster import RedisClusterManager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "db": int(os.getenv("REDIS_DB", "0")),
    "password": os.getenv("REDIS_PASSWORD", None),
    "pool_size": int(os.getenv("REDIS_POOL_SIZE", "10")),
    "connection_timeout": float(os.getenv("REDIS_CONNECTION_TIMEOUT", "5.0")),
    "retry_on_timeout": os.getenv("REDIS_RETRY_ON_TIMEOUT", "True").lower() == "true",
    "max_retries": int(os.getenv("REDIS_MAX_RETRIES", "3"))
}

# Get logger from centralized logging system
logger = get_logger("redis_server")
logger.info("Initializing Redis server")

# --- Redis Async Client ---
try:
    import redis.asyncio as aioredis
except ImportError:
    logger.error("redis[asyncio] package not installed")
    raise ImportError("redis[asyncio] must be installed for async Redis integration.")

try:
    redis_client = aioredis.Redis(
        host=CONFIG["host"],
        port=CONFIG["port"],
        db=CONFIG["db"],
        password=CONFIG["password"],
        decode_responses=True,
        socket_timeout=CONFIG["connection_timeout"],
        socket_connect_timeout=CONFIG["connection_timeout"],
        retry_on_timeout=CONFIG["retry_on_timeout"],
        health_check_interval=30,
        max_connections=CONFIG["pool_size"]
    )
    logger.info(f"Connected to Redis at {CONFIG['host']}:{CONFIG['port']} (DB: {CONFIG['db']})")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")
    raise

# --- FastAPI Models ---

class SetValueRequest(BaseModel):
    key: str
    value: Any
    expiry: Optional[int] = None

# Stock Candidate Models
class StockCandidate(BaseModel):
    symbol: str
    last_price: float
    volume: int
    rel_volume: float
    price_change_pct: float
    atr: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    ma_50_200_cross: Optional[bool] = None
    screening_score: Optional[float] = None

class RiskAssessedCandidate(StockCandidate):
    stop_price: float
    profit_target: float
    position_size: int
    max_shares: int
    estimated_cost: float
    risk_amount: float
    reward_risk_ratio: float
    portfolio_fit_score: Optional[float] = None
    risk_score: Optional[float] = None

class StockCandidatePoolRequest(BaseModel):
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    source: str
    candidates: List[StockCandidate]

class RiskAssessedPoolRequest(BaseModel):
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    candidates: List[RiskAssessedCandidate]

# Trade Position Models
class TradePosition(BaseModel):
    symbol: str
    entry_price: float
    current_price: float
    quantity: int
    stop_price: float
    profit_target: float
    entry_time: str
    unrealized_pnl: float
    unrealized_pnl_pct: float

class ClosedTradePosition(BaseModel):
    symbol: str
    entry_price: float
    exit_price: float
    quantity: int
    stop_price: float
    profit_target: float
    entry_time: str
    exit_time: str
    realized_pnl: float
    realized_pnl_pct: float
    exit_reason: str

class ActivePositionsRequest(BaseModel):
    positions: List[TradePosition]

class ClosedPositionsRequest(BaseModel):
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    positions: List[ClosedTradePosition]

class GetValueRequest(BaseModel):
    key: str
    parse_json: bool = True

class DeleteKeysRequest(BaseModel):
    keys: List[str]

class HashSetRequest(BaseModel):
    key: str
    fields: Dict[str, Any]
    expiry: Optional[int] = None

class HashGetRequest(BaseModel):
    key: str
    fields: Optional[List[str]] = None
    parse_json: bool = True

class ListOperationRequest(BaseModel):
    key: str
    operation: str
    values: Optional[List[Any]] = None
    side: str = "right"
    start: int = 0
    end: int = -1
    parse_json: bool = True

class PublishMessageRequest(BaseModel):
    channel: str
    message: Any

class KeyOperationRequest(BaseModel):
    operation: str
    keys: List[str]
    seconds: Optional[int] = None
    new_key: Optional[str] = None

# --- FastAPI Server ---

app = FastAPI(title="Redis MCP Server for LLM (Production)")

@app.on_event("startup")
async def startup_event():
    logger.info("Redis server starting up")

    # Store configuration in app state
    app.state.config = {"storage": {"redis": CONFIG}}
    
    # Initialize Redisearch index
    try:
        # Create and store RedisClusterManager for use in endpoints
        app.state.redis_cluster_manager = RedisClusterManager(config=app.state.config)
        await app.state.redis_cluster_manager.initialize_index()
        logger.info("Redisearch index initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Redisearch index: {e}")
        # Don't raise here - allow the service to start even if index setup fails
        # Index setup can be retried later

    # Test Redis connection
    try:
        ping_result = await redis_client.ping()
        logger.info(f"Redis connection test: {ping_result}")
        # Set app state to track Redis health
        app.state.redis_healthy = True
    except Exception as e:
        logger.error(f"Redis connection test failed: {str(e)}")
        app.state.redis_healthy = False

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Redis server shutting down")
    # Close Redis connection
    await redis_client.close()
    logger.info("Redis connection closed")
    
    # Shutdown RedisClusterManager if it exists
    if hasattr(app.state, "redis_cluster_manager"):
        app.state.redis_cluster_manager.shutdown()
        logger.info("RedisClusterManager shut down")

@app.get("/server_info")
async def get_server_info():
    """Get server information and available tools."""
    redis_version = "unknown"
    try:
        # Try to get Redis version
        info = await redis_client.info("server")
        redis_version = info.get("redis_version", "unknown")
    except Exception as e:
        logger.warning(f"Could not get Redis version: {e}")
    
    return {
        "name": "redis",
        "version": "1.1.0",  # Updated version
        "redis_version": redis_version,
        "description": "Production MCP Server for Redis Integration",
        "tools": [
            "set_value", "get_value", "delete_keys", "hash_set", "hash_get", "list_operations", "publish_message", "key_operations",
            "store_stock_candidates", "get_stock_candidates", "store_risk_assessed_candidates", "get_risk_assessed_candidates",
            "store_active_positions", "get_active_positions", "store_closed_positions", "get_closed_positions",
            "create_trade_position_schema", "health", "metrics"
        ],
        "health": getattr(app.state, "redis_healthy", False),
        "config": {k: v for k, v in CONFIG.items() if k != "password"}  # Exclude password from response
    }

@app.post("/set_value")
async def api_set_value(req: SetValueRequest):
    logger.info(f"Setting value for key: {req.key}")
    try:
        value = req.value if isinstance(req.value, str) else json.dumps(req.value)
        
        if req.expiry is not None:
            logger.debug(f"Setting key {req.key} with expiry: {req.expiry}s")
            success = await redis_client.setex(req.key, req.expiry, value)
        else:
            success = await redis_client.set(req.key, value)
            
        logger.info(f"Successfully set value for key: {req.key}")
        return {"success": bool(success)}
    except Exception as e:
        logger.error(f"Error setting value for key {req.key}: {str(e)}")
        raise

@app.post("/get_value")
async def api_get_value(req: GetValueRequest):
    logger.info(f"Getting value for key: {req.key}")
    try:
        value = await redis_client.get(req.key)
        
        if value is None:
            logger.debug(f"Key not found: {req.key}")
            return {"value": None, "exists": False}
            
        if req.parse_json:
            try:
                value = json.loads(value)
                logger.debug(f"Successfully parsed JSON value for key: {req.key}")
            except Exception as e:
                logger.debug(f"Failed to parse JSON for key {req.key}: {str(e)}")
                
        logger.info(f"Successfully retrieved value for key: {req.key}")
        return {"value": value, "exists": True}
    except Exception as e:
        logger.error(f"Error getting value for key {req.key}: {str(e)}")
        raise

@app.post("/delete_keys")
async def api_delete_keys(req: DeleteKeysRequest):
    logger.info(f"Deleting keys: {req.keys}")
    try:
        deleted_count = await redis_client.delete(*req.keys)
        logger.info(f"Successfully deleted {deleted_count} keys")
        return {"deleted_count": deleted_count}
    except Exception as e:
        logger.error(f"Error deleting keys: {str(e)}")
        raise

@app.post("/hash_set")
async def api_hash_set(req: HashSetRequest):
    logger.info(f"Setting hash fields for key: {req.key}")
    try:
        mapping = {k: v if isinstance(v, str) else json.dumps(v) for k, v in req.fields.items()}
        await redis_client.hset(req.key, mapping=mapping)
        
        if req.expiry is not None:
            logger.debug(f"Setting expiry for hash key {req.key}: {req.expiry}s")
            await redis_client.expire(req.key, req.expiry)
            
        logger.info(f"Successfully set hash fields for key: {req.key}")
        return {"success": True, "fields_count": len(req.fields)}
    except Exception as e:
        logger.error(f"Error setting hash fields for key {req.key}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting hash: {str(e)}")

@app.post("/hash_get")
async def api_hash_get(req: HashGetRequest):
    logger.info(f"Getting hash fields for key: {req.key}" + (f", fields: {req.fields}" if req.fields else ", all fields"))
    try:
        if req.fields:
            values = await redis_client.hmget(req.key, req.fields)
            result = dict(zip(req.fields, values))
        else:
            result = await redis_client.hgetall(req.key)
            
        if req.parse_json:
            for k, v in result.items():
                if v is not None:
                    try:
                        result[k] = json.loads(v)
                    except Exception as e:
                        logger.debug(f"Failed to parse JSON for field {k} in hash {req.key}: {str(e)}")
                        
        exists = bool(result)
        logger.info(f"Retrieved hash for key {req.key}: exists={exists}, fields_count={len(result) if exists else 0}")
        return {"values": result, "exists": exists}
    except Exception as e:
        logger.error(f"Error getting hash fields for key {req.key}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving hash: {str(e)}")

@app.post("/list_operations")
async def api_list_operations(req: ListOperationRequest):
    logger.info(f"List operation: {req.operation} for key: {req.key}, side: {req.side}")
    try:
        if req.operation == "push":
            vals = [v if isinstance(v, str) else json.dumps(v) for v in (req.values or [])]
            if req.side == "left":
                length = await redis_client.lpush(req.key, *vals)
                logger.debug(f"Pushed {len(vals)} items to left of list {req.key}")
            else:
                length = await redis_client.rpush(req.key, *vals)
                logger.debug(f"Pushed {len(vals)} items to right of list {req.key}")
            result = vals
            
        elif req.operation == "pop":
            if req.side == "left":
                value = await redis_client.lpop(req.key)
                logger.debug(f"Popped item from left of list {req.key}")
            else:
                value = await redis_client.rpop(req.key)
                logger.debug(f"Popped item from right of list {req.key}")
                
            if value is not None and req.parse_json:
                try:
                    value = json.loads(value)
                except Exception as e:
                    logger.debug(f"Failed to parse JSON for popped value from list {req.key}: {str(e)}")
                    
            result = value
            length = await redis_client.llen(req.key)
            
        elif req.operation == "range":
            values = await redis_client.lrange(req.key, req.start, req.end)
            logger.debug(f"Range query on list {req.key}: start={req.start}, end={req.end}, found={len(values)} items")
            
            if req.parse_json:
                parsed = []
                for v in values:
                    try:
                        parsed.append(json.loads(v))
                    except Exception as e:
                        logger.debug(f"Failed to parse JSON for item in list {req.key}: {str(e)}")
                        parsed.append(v)
                result = parsed
            else:
                result = values
                
            length = await redis_client.llen(req.key)
            
        else:
            logger.warning(f"Invalid list operation requested: {req.operation}")
            raise HTTPException(status_code=400, detail=f"Invalid list operation: {req.operation}")
            
        logger.info(f"Successfully executed list operation {req.operation} on key {req.key}")
        return {"result": result, "length": length}
        
    except Exception as e:
        logger.error(f"Error performing list operation {req.operation} on key {req.key}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error with list operation: {str(e)}")

@app.post("/publish_message")
async def api_publish_message(req: PublishMessageRequest):
    logger.info(f"Publishing message to channel: {req.channel}")
    try:
        message = req.message if isinstance(req.message, str) else json.dumps(req.message)
        receivers = await redis_client.publish(req.channel, message)
        logger.info(f"Message published to channel {req.channel}, reached {receivers} receivers")
        return {"receivers": receivers, "channel": req.channel, "success": True}
    except Exception as e:
        logger.error(f"Error publishing message to channel {req.channel}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error publishing message: {str(e)}")

@app.post("/key_operations")
async def api_key_operations(req: KeyOperationRequest):
    """Perform various operations on Redis keys."""
    logger.info(f"Key operation: {req.operation} for keys: {req.keys}")
    try:
        if req.operation == "exists":
            result = await redis_client.exists(*req.keys)
            logger.debug(f"Checked existence of {len(req.keys)} keys, result: {result}")
            
        elif req.operation == "expire":
            if len(req.keys) != 1 or req.seconds is None:
                logger.warning(f"Invalid expire operation params: keys={req.keys}, seconds={req.seconds}")
                raise HTTPException(status_code=400, detail="Expire requires one key and seconds")
                
            result = await redis_client.expire(req.keys[0], req.seconds)
            logger.debug(f"Set expiry of {req.seconds}s on key '{req.keys[0]}', result: {result}")
            
        elif req.operation == "ttl":
            if len(req.keys) != 1:
                logger.warning(f"Invalid TTL operation, multiple keys provided: {req.keys}")
                raise HTTPException(status_code=400, detail="TTL requires one key")
                
            result = await redis_client.ttl(req.keys[0])
            logger.debug(f"TTL for key '{req.keys[0]}': {result}")
            
        elif req.operation == "type":
            if len(req.keys) != 1:
                logger.warning(f"Invalid TYPE operation, multiple keys provided: {req.keys}")
                raise HTTPException(status_code=400, detail="Type requires one key")
                
            result = await redis_client.type(req.keys[0])
            logger.debug(f"Type of key '{req.keys[0]}': {result}")
            
        elif req.operation == "rename":
            if len(req.keys) != 1 or not req.new_key:
                logger.warning(f"Invalid RENAME operation: keys={req.keys}, new_key={req.new_key}")
                raise HTTPException(status_code=400, detail="Rename requires one key and new_key")
                
            result = await redis_client.rename(req.keys[0], req.new_key)
            logger.debug(f"Renamed key '{req.keys[0]}' to '{req.new_key}', result: {result}")
            
        else:
            logger.warning(f"Invalid key operation requested: {req.operation}")
            raise HTTPException(status_code=400, detail=f"Invalid key operation: {req.operation}")
            
        logger.info(f"Successfully executed {req.operation} operation on {len(req.keys)} keys")
        return {"success": True, "operation": req.operation, "result": result}
        
    except HTTPException:
        # Pass through HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error performing {req.operation} on keys {req.keys}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error with key operation: {str(e)}")

# --- Stock Candidate Pool Operations ---
# Added consistent error handling for all operations

@app.post("/store_stock_candidates")
async def api_store_stock_candidates(req: StockCandidatePoolRequest):
    """Store stock candidates in the pool for a specific date and source."""
    logger.info(f"Storing {len(req.candidates)} stock candidates for date {req.date} from source {req.source}")
    try:
        # Define key structure: stock_pool:{date}:{source}
        key = f"stock_pool:{req.date}:{req.source}"
        
        # Define JSON structure for candidate stocks
        value = {
            "timestamp": datetime.now().isoformat(),
            "source": req.source,
            "candidates": [candidate.dict() for candidate in req.candidates]
        }
        
        success = await redis_client.set(key, json.dumps(value))
        
        # Implement TTL for automatic data expiration (7 days)
        await redis_client.expire(key, 60 * 60 * 24 * 7)
        
        logger.info(f"Successfully stored stock candidates with key: {key}")
        return {"success": bool(success), "key": key, "count": len(req.candidates)}
    except Exception as e:
        logger.error(f"Error storing stock candidates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing stock candidates: {str(e)}")

@app.get("/get_stock_candidates")
async def api_get_stock_candidates(date: str = Query(default=datetime.now().strftime("%Y-%m-%d")),
                                  source: Optional[str] = None):
    """Get stock candidates from the pool for a specific date and optionally a specific source."""
    logger.info(f"Getting stock candidates for date {date}" + (f" from source {source}" if source else ""))
    try:
        result = {}
        
        if source:
            # Get candidates from a specific source
            key = f"stock_pool:{date}:{source}"
            value = await redis_client.get(key)
            if value:
                result[source] = json.loads(value)
        else:
            # Get candidates from all sources
            keys = await redis_client.keys(f"stock_pool:{date}:*")
            for key in keys:
                value = await redis_client.get(key)
                if value:
                    source = key.split(":")[-1]
                    result[source] = json.loads(value)
        
        logger.info(f"Retrieved stock candidates from {len(result)} sources")
        return {"success": True, "date": date, "sources": list(result.keys()), "data": result}
    except Exception as e:
        logger.error(f"Error getting stock candidates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stock candidates: {str(e)}")

@app.post("/store_risk_assessed_candidates")
async def api_store_risk_assessed_candidates(req: RiskAssessedPoolRequest):
    """Store risk-assessed stock candidates."""
    logger.info(f"Storing {len(req.candidates)} risk-assessed candidates for date {req.date}")
    try:
        key = f"stock_pool:{req.date}:risk_assessed"
        value = {
            "timestamp": datetime.now().isoformat(),
            "candidates": [candidate.dict() for candidate in req.candidates]
        }
        
        success = await redis_client.set(key, json.dumps(value))
        # Set TTL to 7 days
        await redis_client.expire(key, 60 * 60 * 24 * 7)
        
        logger.info(f"Successfully stored risk-assessed candidates with key: {key}")
        return {"success": bool(success), "key": key, "count": len(req.candidates)}
    except Exception as e:
        logger.error(f"Error storing risk-assessed candidates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing risk-assessed candidates: {str(e)}")

@app.get("/get_risk_assessed_candidates")
async def api_get_risk_assessed_candidates(date: str = Query(default=datetime.now().strftime("%Y-%m-%d"))):
    """Get risk-assessed stock candidates for a specific date."""
    logger.info(f"Getting risk-assessed candidates for date {date}")
    try:
        key = f"stock_pool:{date}:risk_assessed"
        value = await redis_client.get(key)
        
        if value:
            data = json.loads(value)
            logger.info(f"Retrieved {len(data.get('candidates', []))} risk-assessed candidates")
            return {"success": True, "date": date, "data": data}
        else:
            logger.info(f"No risk-assessed candidates found for date {date}")
            return {"success": False, "date": date, "data": None}
    except Exception as e:
        logger.error(f"Error getting risk-assessed candidates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting risk-assessed candidates: {str(e)}")

# --- Trade Position Operations ---

@app.post("/store_active_positions")
async def api_store_active_positions(req: ActivePositionsRequest):
    """Store active trade positions."""
    logger.info(f"Storing {len(req.positions)} active trade positions")
    try:
        key = "trade_positions:active"
        value = {
            "timestamp": datetime.now().isoformat(),
            "positions": [position.dict() for position in req.positions]
        }
        
        success = await redis_client.set(key, json.dumps(value))
        
        logger.info(f"Successfully stored active positions")
        return {"success": bool(success), "count": len(req.positions)}
    except Exception as e:
        logger.error(f"Error storing active positions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing active positions: {str(e)}")

@app.get("/get_active_positions")
async def api_get_active_positions():
    """Get active trade positions."""
    logger.info("Getting active trade positions")
    try:
        key = "trade_positions:active"
        value = await redis_client.get(key)
        
        if value:
            data = json.loads(value)
            logger.info(f"Retrieved {len(data.get('positions', []))} active positions")
            return {"success": True, "data": data}
        else:
            logger.info("No active positions found")
            return {"success": False, "data": {"timestamp": datetime.now().isoformat(), "positions": []}}
    except Exception as e:
        logger.error(f"Error getting active positions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting active positions: {str(e)}")

def calculate_unrealized_pnl(position: TradePosition) -> float:
    """Calculates the unrealized profit/loss for a given trade position."""
    try:
        return (position.current_price - position.entry_price) * position.quantity
    except Exception as e:
        logger.error(f"Error calculating unrealized PnL for {position.symbol}: {str(e)}")
        return 0.0

async def update_trade_position(position: TradePosition, current_price: float):
    """Updates the current price of a trade position and recalculates the unrealized P&L."""
    try:
        position.current_price = current_price
        position.unrealized_pnl = calculate_unrealized_pnl(position)
        
        # Avoid division by zero
        if position.entry_price <= 0 or position.quantity <= 0:
            logger.warning(f"Invalid entry price or quantity for {position.symbol}")
            position.unrealized_pnl_pct = 0
        else:
            position.unrealized_pnl_pct = (position.unrealized_pnl / (position.entry_price * position.quantity)) * 100
            
        return position
    except Exception as e:
        logger.error(f"Error updating trade position for {position.symbol}: {str(e)}")
        return position

@app.post("/store_closed_positions")
async def api_store_closed_positions(req: ClosedPositionsRequest):
    """Store closed trade positions for a specific date."""
    logger.info(f"Storing {len(req.positions)} closed trade positions for date {req.date}")
    try:
        key = f"trade_positions:history:{req.date}"
        value = {
            "timestamp": datetime.now().isoformat(),
            "positions": [position.dict() for position in req.positions]
        }
        
        success = await redis_client.set(key, json.dumps(value))
        # Set TTL to 30 days
        await redis_client.expire(key, 60 * 60 * 24 * 30)
        
        logger.info(f"Successfully stored closed positions with key: {key}")
        return {"success": bool(success), "key": key, "count": len(req.positions)}
    except Exception as e:
        logger.error(f"Error storing closed positions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing closed positions: {str(e)}")

@app.get("/get_closed_positions")
async def api_get_closed_positions(date: str = Query(default=datetime.now().strftime("%Y-%m-%d"))):
    """Get closed trade positions for a specific date."""
    logger.info(f"Getting closed trade positions for date {date}")
    try:
        key = f"trade_positions:history:{date}"
        value = await redis_client.get(key)
        
        if value:
            data = json.loads(value)
            logger.info(f"Retrieved {len(data.get('positions', []))} closed positions")
            return {"success": True, "date": date, "data": data}
        else:
            logger.info(f"No closed positions found for date {date}")
            return {"success": False, "date": date, "data": None}
    except Exception as e:
        logger.error(f"Error getting closed positions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting closed positions: {str(e)}")

@app.get("/resource/{resource_uri:path}")
async def get_resource(resource_uri: str):
    """Access various Redis resources by URI path."""
    logger.info(f"Resource request for URI: {resource_uri}")
    
    try:
        if resource_uri == "info":
            logger.debug("Fetching Redis server info")
            info = await redis_client.info()
            logger.info("Successfully retrieved Redis server info")
            return {
                "success": True,
                "info": info,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        elif resource_uri == "stats":
            logger.debug("Fetching Redis server stats")
            stats = {
                "memory": await redis_client.info("memory"),
                "clients": await redis_client.info("clients"),
                "stats": await redis_client.info("stats"),
                "keyspace": await redis_client.info("keyspace")
            }
            logger.info("Successfully retrieved Redis server stats")
            return {
                "success": True,
                "stats": stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        elif resource_uri.startswith("keys/"):
            pattern = resource_uri[5:]
            logger.debug(f"Fetching keys matching pattern: {pattern}")
            
            # Implement pagination for large key sets
            try:
                cursor = 0
                batch_size = 1000
                all_keys = []
                
                # Use scan instead of keys for production safety with large datasets
                while True:
                    cursor, keys = await redis_client.scan(cursor=cursor, match=pattern, count=batch_size)
                    all_keys.extend(keys)
                    
                    if cursor == 0:
                        break
                        
                logger.info(f"Found {len(all_keys)} keys matching pattern: {pattern}")
                return {
                    "success": True,
                    "keys": all_keys,
                    "count": len(all_keys),
                    "pattern": pattern,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Error scanning keys with pattern {pattern}: {str(e)}")
                # Fallback to keys command if scan fails
                logger.debug(f"Falling back to keys command for pattern: {pattern}")
                keys = await redis_client.keys(pattern)
                logger.info(f"Found {len(keys)} keys matching pattern: {pattern}")
                return {
                    "success": True,
                    "keys": keys,
                    "count": len(keys),
                    "pattern": pattern,
                    "timestamp": datetime.utcnow().isoformat(),
                    "note": "Used fallback keys command instead of scan"
                }
                
        elif resource_uri == "cluster_info":
            if hasattr(app.state, "redis_cluster_manager") and app.state.redis_cluster_manager is not None:
                logger.debug("Fetching RedisClusterManager info")
                return {
                    "success": True,
                    "initialized": True,
                    "config": app.state.redis_cluster_manager.config,
                    "indices": ["risk_assessed_idx", "trade_positions_idx"],
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                logger.warning("RedisClusterManager not initialized")
                return {
                    "success": False,
                    "initialized": False,
                    "error": "RedisClusterManager not initialized",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
        else:
            logger.warning(f"Unknown resource URI requested: {resource_uri}")
            raise HTTPException(status_code=404, detail=f"Unknown resource URI: {resource_uri}")
            
    except HTTPException:
        # Pass through HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error accessing resource {resource_uri}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error accessing resource: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Check Redis connection
        ping_result = await redis_client.ping()
        redis_info = await redis_client.info("server")
        
        # Check RedisClusterManager
        cluster_manager_healthy = hasattr(app.state, "redis_cluster_manager") and app.state.redis_cluster_manager.redis is not None
        
        # Determine overall health
        is_healthy = ping_result and cluster_manager_healthy
        app.state.redis_healthy = is_healthy
        
        health_details = {
            "status": "healthy" if is_healthy else "unhealthy",
            "redis_ping": ping_result,
            "cluster_manager": cluster_manager_healthy,
            "redis_version": redis_info.get("redis_version", "unknown"),
            "uptime_seconds": redis_info.get("uptime_in_seconds", 0),
            "connected_clients": redis_info.get("connected_clients", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.debug(f"Health check: {health_details['status']}")
        
        status_code = 200 if is_healthy else 503
        return health_details
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        app.state.redis_healthy = False
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/create_trade_position_schema")
async def api_create_trade_position_schema():
    """Create Redis schema for trade positions."""
    logger.info("Creating trade position schema")
    try:
        # Use the stored RedisClusterManager from app state
        if not hasattr(app.state, "redis_cluster_manager") or app.state.redis_cluster_manager is None:
            logger.error("RedisClusterManager not initialized")
            raise HTTPException(status_code=500, detail="RedisClusterManager not initialized")
            
        index_name = "trade_positions_idx"
        redis_key = "trade_positions:active:*"
        app.state.redis_cluster_manager.create_index_trade_positions(index_name, redis_key)
        logger.info(f"Trade position schema with index name '{index_name}' created successfully")
        
        return {
            "success": True,
            "message": "Trade position schema created successfully",
            "index": index_name,
            "key_pattern": redis_key
        }
    except Exception as e:
        logger.error(f"Error creating trade position schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get Redis server metrics for monitoring."""
    logger.info("Fetching Redis metrics")
    try:
        # Get basic Redis stats
        info = await redis_client.info()
        
        # Process metrics we're interested in
        metrics = {
            "connected_clients": info.get("clients", {}).get("connected_clients", 0),
            "blocked_clients": info.get("clients", {}).get("blocked_clients", 0),
            "memory_used_bytes": info.get("memory", {}).get("used_memory", 0),
            "memory_peak_bytes": info.get("memory", {}).get("used_memory_peak", 0),
            "total_commands_processed": info.get("stats", {}).get("total_commands_processed", 0),
            "total_connections_received": info.get("stats", {}).get("total_connections_received", 0),
            "rejected_connections": info.get("stats", {}).get("rejected_connections", 0),
            "expired_keys": info.get("stats", {}).get("expired_keys", 0),
            "evicted_keys": info.get("stats", {}).get("evicted_keys", 0),
            "keyspace_hits": info.get("stats", {}).get("keyspace_hits", 0),
            "keyspace_misses": info.get("stats", {}).get("keyspace_misses", 0),
            "uptime_in_seconds": info.get("server", {}).get("uptime_in_seconds", 0),
        }
        
        # Add hit rate if we have both hits and misses
        hits = metrics["keyspace_hits"]
        misses = metrics["keyspace_misses"]
        total_operations = hits + misses
        
        if total_operations > 0:
            metrics["cache_hit_rate"] = hits / total_operations
        else:
            metrics["cache_hit_rate"] = 0
            
        # Add memory utilization percentage if peak is available
        if metrics["memory_peak_bytes"] > 0:
            metrics["memory_usage_pct"] = (metrics["memory_used_bytes"] / metrics["memory_peak_bytes"]) * 100
            
        logger.info("Successfully retrieved Redis metrics")
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error fetching Redis metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {str(e)}")
