"""
Redis MCP FastAPI Server for LLM integration (production).
Provides key-value, hash, list, set, and pub/sub operations on Redis.
All configuration is contained in this file.
"""


import os
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date
from dotenv import load_dotenv
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Query, Request, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
import yaml
from pathlib import Path
from copy import deepcopy
import logging
from storage.redis_cluster import RedisClusterManager

# Fallback logger
try:
    from monitor.logging_utils import get_logger
except ImportError:
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

# Load environment variables
load_dotenv()

# --- Configuration ---
def validate_config():
    config_path = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'redis_config.yaml'))
    default_config = {
        "redis": {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": int(os.getenv("REDIS_DB", "0")),
            "password": os.getenv("REDIS_PASSWORD", None),
            "pool_size": 10,
            "connection_timeout": 5.0,
            "retry_on_timeout": True,
            "max_retries": 3,
            "ttl_stock_candidates": 7 * 24 * 60 * 60,  # 7 days
            "ttl_closed_positions": 30 * 24 * 60 * 60,  # 30 days
            "ttl_metrics": 300  # 5 minutes
        },
        "security": {
            "enable_auth": True
        }
    }
    try:
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f) or {}
            def deep_merge(default: dict, update: dict) -> dict:
                merged = deepcopy(default)
                for key, value in update.items():
                    if isinstance(value, dict) and key in merged:
                        merged[key] = deep_merge(merged[key], value)
                    else:
                        merged[key] = value
                return merged
            config = deep_merge(default_config, file_config)
        else:
            logger.warning(f"Config file not found at {config_path}. Using default configuration.")
            config = default_config

        # Override with environment variables
        config["redis"]["host"] = os.getenv("REDIS_HOST", config["redis"]["host"])
        config["redis"]["port"] = int(os.getenv("REDIS_PORT", config["redis"]["port"]))
        config["redis"]["db"] = int(os.getenv("REDIS_DB", config["redis"]["db"]))
        config["redis"]["password"] = os.getenv("REDIS_PASSWORD", config["redis"]["password"])
        config["redis"]["pool_size"] = int(os.getenv("REDIS_POOL_SIZE", config["redis"]["pool_size"]))
        config["redis"]["connection_timeout"] = float(os.getenv("REDIS_CONNECTION_TIMEOUT", config["redis"]["connection_timeout"]))
        config["redis"]["retry_on_timeout"] = os.getenv("REDIS_RETRY_ON_TIMEOUT", str(config["redis"]["retry_on_timeout"])).lower() == "true"
        config["redis"]["max_retries"] = int(os.getenv("REDIS_MAX_RETRIES", config["redis"]["max_retries"]))
        config["redis"]["ttl_stock_candidates"] = int(os.getenv("REDIS_TTL_STOCK_CANDIDATES", config["redis"]["ttl_stock_candidates"]))
        config["redis"]["ttl_closed_positions"] = int(os.getenv("REDIS_TTL_CLOSED_POSITIONS", config["redis"]["ttl_closed_positions"]))
        config["redis"]["ttl_metrics"] = int(os.getenv("REDIS_TTL_METRICS", config["redis"]["ttl_metrics"]))
        config["security"]["enable_auth"] = os.getenv("REDIS_ENABLE_AUTH", str(config["security"]["enable_auth"])).lower() == "true"

        # Validate configuration
        if config["redis"]["port"] <= 0:
            logger.warning(f"Invalid redis_port: {config['redis']['port']}. Using default: 6379")
            config["redis"]["port"] = 6379
        if config["redis"]["db"] < 0:
            logger.warning(f"Invalid redis_db: {config['redis']['db']}. Using default: 0")
            config["redis"]["db"] = 0
        if config["redis"]["pool_size"] <= 0:
            logger.warning(f"Invalid pool_size: {config['redis']['pool_size']}. Using default: 10")
            config["redis"]["pool_size"] = 10
        if config["redis"]["connection_timeout"] <= 0:
            logger.warning(f"Invalid connection_timeout: {config['redis']['connection_timeout']}. Using default: 5.0")
            config["redis"]["connection_timeout"] = 5.0
        if config["redis"]["max_retries"] < 0:
            logger.warning(f"Invalid max_retries: {config['redis']['max_retries']}. Using default: 3")
            config["redis"]["max_retries"] = 3
        if config["redis"]["ttl_stock_candidates"] <= 0:
            logger.warning(f"Invalid ttl_stock_candidates: {config['redis']['ttl_stock_candidates']}. Using default: 7 days")
            config["redis"]["ttl_stock_candidates"] = 7 * 24 * 60 * 60
        if config["redis"]["ttl_closed_positions"] <= 0:
            logger.warning(f"Invalid ttl_closed_positions: {config['redis']['ttl_closed_positions']}. Using default: 30 days")
            config["redis"]["ttl_closed_positions"] = 30 * 24 * 60 * 60
        if config["redis"]["ttl_metrics"] <= 0:
            logger.warning(f"Invalid ttl_metrics: {config['redis']['ttl_metrics']}. Using default: 300 seconds")
            config["redis"]["ttl_metrics"] = 300
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}. Using default configuration.")
        return default_config

CONFIG = validate_config()

logger = get_logger("redis_server")
logger.info("Initializing Redis server")

# --- Redis Async Client ---
redis_client = None

async def connect_redis(max_retries=3, retry_delay=5):
    global redis_client
    for attempt in range(max_retries):
        try:
            redis_client = aioredis.Redis(
                host=CONFIG["redis"]["host"],
                port=CONFIG["redis"]["port"],
                db=CONFIG["redis"]["db"],
                password=CONFIG["redis"]["password"],
                decode_responses=True,
                socket_timeout=CONFIG["redis"]["connection_timeout"],
                socket_connect_timeout=CONFIG["redis"]["connection_timeout"],
                retry_on_timeout=CONFIG["redis"]["retry_on_timeout"],
                health_check_interval=30,
                max_connections=CONFIG["redis"]["pool_size"]
            )
            await redis_client.ping()
            logger.info(f"Connected to Redis at {CONFIG['redis']['host']}:{CONFIG['redis']['port']} (DB: {CONFIG['redis']['db']})")
            return
        except Exception as e:
            logger.error(f"Redis connection attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
    logger.error("Failed to connect to Redis after retries")
    raise RuntimeError("Failed to connect to Redis")

# --- Rate Limiter ---
class RateLimiter:
    def __init__(self, calls_limit: int = 60, time_window: int = 60):
        self.calls_limit = calls_limit
        self.time_window = time_window
        self.redis_key_prefix = "redis_rate_limit"
    
    async def check_rate_limit(self, request: Request):
        if not redis_client:
            logger.warning("Redis unavailable, skipping rate limiting")
            return True
        client_ip = request.client.host
        key = f"{self.redis_key_prefix}:{client_ip}"
        now = int(time.time())
        try:
            async with redis_client.pipeline() as pipe:
                pipe.zremrangebyscore(key, 0, now - self.time_window)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, self.time_window)
                _, _, count, _ = await pipe.execute()
            if count > self.calls_limit:
                logger.warning(f"Rate limit exceeded for IP {client_ip}: {count} requests in last {self.time_window}s")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking rate limit in Redis: {e}")
            return True

rate_limiter = RateLimiter()

# --- Authentication ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not CONFIG["security"]["enable_auth"]:
        return
    valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")
    if not api_key or api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# --- Utility Functions ---
def serialize_value(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)

def deserialize_value(value: str, parse_json: bool = True) -> Any:
    if value is None:
        return None
    if parse_json:
        try:
            return json.loads(value)
        except Exception as e:
            logger.debug(f"Failed to parse JSON: {str(e)}")
    return value

# --- FastAPI Models ---
class SetValueRequest(BaseModel):
    key: str = Field(..., min_length=1)
    value: Any
    expiry: Optional[int] = Field(None, ge=1)

class StockCandidate(BaseModel):
    symbol: str = Field(..., min_length=1)
    last_price: float = Field(..., ge=0)
    volume: int = Field(..., ge=0)
    rel_volume: float = Field(..., ge=0)
    price_change_pct: float
    atr: Optional[float] = Field(None, ge=0)
    rsi: Optional[float] = Field(None, ge=0, le=100)
    macd: Optional[float] = None
    ma_50_200_cross: Optional[bool] = None
    screening_score: Optional[float] = Field(None, ge=0)

class RiskAssessedCandidate(StockCandidate):
    stop_price: float = Field(..., ge=0)
    profit_target: float = Field(..., ge=0)
    position_size: int = Field(..., ge=0)
    max_shares: int = Field(..., ge=0)
    estimated_cost: float = Field(..., ge=0)
    risk_amount: float = Field(..., ge=0)
    reward_risk_ratio: float = Field(..., ge=0)
    portfolio_fit_score: Optional[float] = Field(None, ge=0)
    risk_score: Optional[float] = Field(None, ge=0)

class StockCandidatePoolRequest(BaseModel):
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    source: str = Field(..., min_length=1)
    candidates: List[StockCandidate]

class RiskAssessedPoolRequest(BaseModel):
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    candidates: List[RiskAssessedCandidate]

class TradePosition(BaseModel):
    symbol: str = Field(..., min_length=1)
    entry_price: float = Field(..., ge=0)
    current_price: float = Field(..., ge=0)
    quantity: int = Field(..., ge=0)
    stop_price: float = Field(..., ge=0)
    profit_target: float = Field(..., ge=0)
    entry_time: str
    unrealized_pnl: float
    unrealized_pnl_pct: float

class ClosedTradePosition(BaseModel):
    symbol: str = Field(..., min_length=1)
    entry_price: float = Field(..., ge=0)
    exit_price: float = Field(..., ge=0)
    quantity: int = Field(..., ge=0)
    stop_price: float = Field(..., ge=0)
    profit_target: float = Field(..., ge=0)
    entry_time: str
    exit_time: str
    realized_pnl: float
    realized_pnl_pct: float
    exit_reason: str = Field(..., min_length=1)

class ActivePositionsRequest(BaseModel):
    positions: List[TradePosition]

class ClosedPositionsRequest(BaseModel):
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    positions: List[ClosedTradePosition]

class GetValueRequest(BaseModel):
    key: str = Field(..., min_length=1)
    parse_json: bool = True

class DeleteKeysRequest(BaseModel):
    keys: List[str] = Field(..., min_items=1)

class HashSetRequest(BaseModel):
    key: str = Field(..., min_length=1)
    fields: Dict[str, Any]
    expiry: Optional[int] = Field(None, ge=1)

class HashGetRequest(BaseModel):
    key: str = Field(..., min_length=1)
    fields: Optional[List[str]] = None
    parse_json: bool = True

class ListOperationRequest(BaseModel):
    key: str = Field(..., min_length=1)
    operation: str = Field(..., pattern="^(push|pop|range)$")
    values: Optional[List[Any]] = None
    side: str = Field("right", pattern="^(left|right)$")
    start: int = Field(0, ge=0)
    end: int = Field(-1)
    parse_json: bool = True

class PublishMessageRequest(BaseModel):
    channel: str = Field(..., min_length=1)
    message: Any

class KeyOperationRequest(BaseModel):
    operation: str = Field(..., pattern="^(exists|expire|ttl|type|rename)$")
    keys: List[str] = Field(..., min_items=1)
    seconds: Optional[int] = Field(None, ge=1)
    new_key: Optional[str] = Field(None, min_length=1)

# --- FastAPI Server ---
class RateLimitMiddleware:
    async def dispatch(self, request: Request, call_next):
        if not await rate_limiter.check_rate_limit(request):
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
        return await call_next(request)

app = FastAPI(
    title="Redis MCP Server for LLM (Production)",
    description="Production MCP Server for Redis Integration",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "General", "description": "Server information and health checks"},
        {"name": "Key-Value", "description": "Basic key-value operations"},
        {"name": "Hash", "description": "Hash operations"},
        {"name": "List", "description": "List operations"},
        {"name": "PubSub", "description": "Publish/subscribe operations"},
        {"name": "Stock", "description": "Stock candidate operations"},
        {"name": "Trade", "description": "Trade position operations"}
    ]
)

app.add_middleware(RateLimitMiddleware)

@app.on_event("startup")
async def startup_event():
    logger.info("Redis server starting up")
    await connect_redis(max_retries=CONFIG["redis"]["max_retries"])
    app.state.config = {"storage": {"redis": CONFIG}}
    try:
        app.state.redis_cluster_manager = RedisClusterManager(config=app.state.config)
        if await redis_client.ping():
            await app.state.redis_cluster_manager.initialize_index()
            logger.info("Redisearch index initialized successfully")
        app.state.redis_healthy = True
    except Exception as e:
        logger.error(f"Failed to initialize Redisearch index: {str(e)}")
        app.state.redis_healthy = False

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Redis server shutting down")
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")
    if hasattr(app.state, "redis_cluster_manager"):
        app.state.redis_cluster_manager.shutdown()
        logger.info("RedisClusterManager shut down")

@app.get("/server_info", tags=["General"])
async def get_server_info():
    redis_version = "unknown"
    try:
        info = await redis_client.info("server")
        redis_version = info.get("redis_version", "unknown")
    except Exception as e:
        logger.warning(f"Could not get Redis version: {str(e)}")
    return {
        "name": "redis",
        "version": "1.1.0",
        "redis_version": redis_version,
        "description": "Production MCP Server for Redis Integration",
        "tools": [
            "set_value", "get_value", "delete_keys", "hash_set", "hash_get", "list_operations", "publish_message", "key_operations",
            "store_stock_candidates", "get_stock_candidates", "store_risk_assessed_candidates", "get_risk_assessed_candidates",
            "store_active_positions", "get_active_positions", "store_closed_positions", "get_closed_positions",
            "create_trade_position_schema", "health", "metrics"
        ],
        "health": getattr(app.state, "redis_healthy", False),
        "config": {k: v for k, v in CONFIG["redis"].items() if k != "password"}
    }

@app.post("/set_value", tags=["Key-Value"], dependencies=[Depends(verify_api_key)])
async def api_set_value(req: SetValueRequest):
    logger.info(f"Setting value for key: {req.key}")
    try:
        value = serialize_value(req.value)
        if req.expiry is not None:
            success = await redis_client.setex(req.key, req.expiry, value)
        else:
            success = await redis_client.set(req.key, value)
        logger.info(f"Successfully set value for key: {req.key}")
        return {"success": bool(success)}
    except Exception as e:
        logger.error(f"Error setting value for key {req.key}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting value: {str(e)}")

@app.post("/get_value", tags=["Key-Value"], dependencies=[Depends(verify_api_key)])
async def api_get_value(req: GetValueRequest):
    logger.info(f"Getting value for key: {req.key}")
    try:
        value = await redis_client.get(req.key)
        if value is None:
            logger.debug(f"Key not found: {req.key}")
            return {"value": None, "exists": False}
        value = deserialize_value(value, req.parse_json)
        logger.info(f"Successfully retrieved value for key: {req.key}")
        return {"value": value, "exists": True}
    except Exception as e:
        logger.error(f"Error getting value for key {req.key}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting value: {str(e)}")

@app.post("/delete_keys", tags=["Key-Value"], dependencies=[Depends(verify_api_key)])
async def api_delete_keys(req: DeleteKeysRequest):
    logger.info(f"Deleting keys: {req.keys}")
    try:
        deleted_count = await redis_client.delete(*req.keys)
        logger.info(f"Successfully deleted {deleted_count} keys")
        return {"deleted_count": deleted_count}
    except Exception as e:
        logger.error(f"Error deleting keys: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting keys: {str(e)}")

@app.post("/hash_set", tags=["Hash"], dependencies=[Depends(verify_api_key)])
async def api_hash_set(req: HashSetRequest):
    logger.info(f"Setting hash fields for key: {req.key}")
    try:
        mapping = {k: serialize_value(v) for k, v in req.fields.items()}
        await redis_client.hset(req.key, mapping=mapping)
        if req.expiry is not None:
            await redis_client.expire(req.key, req.expiry)
        logger.info(f"Successfully set hash fields for key: {req.key}")
        return {"success": True, "fields_count": len(req.fields)}
    except Exception as e:
        logger.error(f"Error setting hash fields for key {req.key}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting hash: {str(e)}")

@app.post("/hash_get", tags=["Hash"], dependencies=[Depends(verify_api_key)])
async def api_hash_get(req: HashGetRequest):
    logger.info(f"Getting hash fields for key: {req.key}")
    try:
        if req.fields:
            values = await redis_client.hmget(req.key, req.fields)
            result = dict(zip(req.fields, [deserialize_value(v, req.parse_json) for v in values]))
        else:
            result = await redis_client.hgetall(req.key)
            result = {k: deserialize_value(v, req.parse_json) for k, v in result.items()}
        exists = bool(result)
        logger.info(f"Retrieved hash for key {req.key}: exists={exists}, fields_count={len(result)}")
        return {"values": result, "exists": exists}
    except Exception as e:
        logger.error(f"Error getting hash fields for key {req.key}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving hash: {str(e)}")

@app.post("/list_operations", tags=["List"], dependencies=[Depends(verify_api_key)])
async def api_list_operations(req: ListOperationRequest):
    logger.info(f"List operation: {req.operation} for key: {req.key}")
    try:
        if req.operation == "push":
            if not req.values:
                raise HTTPException(status_code=400, detail="Values required for push operation")
            vals = [serialize_value(v) for v in req.values]
            if req.side == "left":
                length = await redis_client.lpush(req.key, *vals)
            else:
                length = await redis_client.rpush(req.key, *vals)
            result = req.values
        elif req.operation == "pop":
            value = await redis_client.lpop(req.key) if req.side == "left" else await redis_client.rpop(req.key)
            result = deserialize_value(value, req.parse_json)
            length = await redis_client.llen(req.key)
        elif req.operation == "range":
            values = await redis_client.lrange(req.key, req.start, req.end)
            result = [deserialize_value(v, req.parse_json) for v in values]
            length = await redis_client.llen(req.key)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid list operation: {req.operation}")
        logger.info(f"Successfully executed list operation {req.operation} on key {req.key}")
        return {"result": result, "length": length}
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error performing list operation {req.operation} on key {req.key}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error with list operation: {str(e)}")

@app.post("/publish_message", tags=["PubSub"], dependencies=[Depends(verify_api_key)])
async def api_publish_message(req: PublishMessageRequest):
    logger.info(f"Publishing message to channel: {req.channel}")
    try:
        message = serialize_value(req.message)
        receivers = await redis_client.publish(req.channel, message)
        logger.info(f"Message published to channel {req.channel}, reached {receivers} receivers")
        return {"receivers": receivers, "channel": req.channel, "success": True}
    except Exception as e:
        logger.error(f"Error publishing message to channel {req.channel}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error publishing message: {str(e)}")

@app.post("/key_operations", tags=["Key-Value"], dependencies=[Depends(verify_api_key)])
async def api_key_operations(req: KeyOperationRequest):
    logger.info(f"Key operation: {req.operation} for keys: {req.keys}")
    try:
        if req.operation == "exists":
            result = await redis_client.exists(*req.keys)
        elif req.operation == "expire":
            if len(req.keys) != 1 or req.seconds is None:
                raise HTTPException(status_code=400, detail="Expire requires one key and seconds")
            result = await redis_client.expire(req.keys[0], req.seconds)
        elif req.operation == "ttl":
            if len(req.keys) != 1:
                raise HTTPException(status_code=400, detail="TTL requires one key")
            result = await redis_client.ttl(req.keys[0])
        elif req.operation == "type":
            if len(req.keys) != 1:
                raise HTTPException(status_code=400, detail="Type requires one key")
            result = await redis_client.type(req.keys[0])
        elif req.operation == "rename":
            if len(req.keys) != 1 or not req.new_key:
                raise HTTPException(status_code=400, detail="Rename requires one key and new_key")
            result = await redis_client.rename(req.keys[0], req.new_key)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid key operation: {req.operation}")
        logger.info(f"Successfully executed {req.operation} operation on {len(req.keys)} keys")
        return {"success": True, "operation": req.operation, "result": result}
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error performing {req.operation} on keys {req.keys}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error with key operation: {str(e)}")

@app.post("/store_stock_candidates", tags=["Stock"], dependencies=[Depends(verify_api_key)])
async def api_store_stock_candidates(req: StockCandidatePoolRequest):
    logger.info(f"Storing {len(req.candidates)} stock candidates for date {req.date} from source {req.source}")
    try:
        key = f"stock_pool:{req.date}:{req.source}"
        value = {
            "timestamp": datetime.now().isoformat(),
            "source": req.source,
            "candidates": [candidate.dict() for candidate in req.candidates]
        }
        success = await redis_client.set(key, json.dumps(value))
        await redis_client.expire(key, CONFIG["redis"]["ttl_stock_candidates"])
        logger.info(f"Successfully stored stock candidates with key: {key}")
        return {"success": bool(success), "key": key, "count": len(req.candidates)}
    except Exception as e:
        logger.error(f"Error storing stock candidates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing stock candidates: {str(e)}")

@app.get("/get_stock_candidates", tags=["Stock"], dependencies=[Depends(verify_api_key)])
async def api_get_stock_candidates(date: str = Query(default=datetime.now().strftime("%Y-%m-%d")), source: Optional[str] = None):
    logger.info(f"Getting stock candidates for date {date}")
    try:
        result = {}
        if source:
            key = f"stock_pool:{date}:{source}"
            value = await redis_client.get(key)
            if value:
                result[source] = json.loads(value)
        else:
            cursor = 0
            while True:
                cursor, keys = await redis_client.scan(cursor, match=f"stock_pool:{date}:*", count=100)
                for key in keys:
                    value = await redis_client.get(key)
                    if value:
                        source = key.split(":")[-1]
                        result[source] = json.loads(value)
                if cursor == 0:
                    break
        logger.info(f"Retrieved stock candidates from {len(result)} sources")
        return {"success": True, "date": date, "sources": list(result.keys()), "data": result}
    except Exception as e:
        logger.error(f"Error getting stock candidates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stock candidates: {str(e)}")

@app.post("/store_risk_assessed_candidates", tags=["Stock"], dependencies=[Depends(verify_api_key)])
async def api_store_risk_assessed_candidates(req: RiskAssessedPoolRequest):
    logger.info(f"Storing {len(req.candidates)} risk-assessed candidates for date {req.date}")
    try:
        key = f"stock_pool:{req.date}:risk_assessed"
        value = {
            "timestamp": datetime.now().isoformat(),
            "candidates": [candidate.dict() for candidate in req.candidates]
        }
        success = await redis_client.set(key, json.dumps(value))
        await redis_client.expire(key, CONFIG["redis"]["ttl_stock_candidates"])
        logger.info(f"Successfully stored risk-assessed candidates with key: {key}")
        return {"success": bool(success), "key": key, "count": len(req.candidates)}
    except Exception as e:
        logger.error(f"Error storing risk-assessed candidates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing risk-assessed candidates: {str(e)}")

@app.get("/get_risk_assessed_candidates", tags=["Stock"], dependencies=[Depends(verify_api_key)])
async def api_get_risk_assessed_candidates(date: str = Query(default=datetime.now().strftime("%Y-%m-%d"))):
    logger.info(f"Getting risk-assessed candidates for date {date}")
    try:
        key = f"stock_pool:{date}:risk_assessed"
        value = await redis_client.get(key)
        if value:
            data = json.loads(value)
            logger.info(f"Retrieved {len(data.get('candidates', []))} risk-assessed candidates")
            return {"success": True, "date": date, "data": data}
        logger.info(f"No risk-assessed candidates found for date {date}")
        return {"success": False, "date": date, "data": None}
    except Exception as e:
        logger.error(f"Error getting risk-assessed candidates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting risk-assessed candidates: {str(e)}")

@app.post("/store_active_positions", tags=["Trade"], dependencies=[Depends(verify_api_key)])
async def api_store_active_positions(req: ActivePositionsRequest):
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

@app.get("/get_active_positions", tags=["Trade"], dependencies=[Depends(verify_api_key)])
async def api_get_active_positions():
    logger.info("Getting active trade positions")
    try:
        key = "trade_positions:active"
        value = await redis_client.get(key)
        if value:
            data = json.loads(value)
            logger.info(f"Retrieved {len(data.get('positions', []))} active positions")
            return {"success": True, "data": data}
        logger.info("No active positions found")
        return {"success": False, "data": {"timestamp": datetime.now().isoformat(), "positions": []}}
    except Exception as e:
        logger.error(f"Error getting active positions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting active positions: {str(e)}")

def calculate_unrealized_pnl(position: TradePosition) -> float:
    try:
        if position.entry_price <= 0 or position.quantity <= 0:
            logger.warning(f"Invalid entry price or quantity for {position.symbol}")
            return 0.0
        return (position.current_price - position.entry_price) * position.quantity
    except Exception as e:
        logger.error(f"Error calculating unrealized PnL for {position.symbol}: {str(e)}")
        return 0.0

async def update_trade_position(position: TradePosition, current_price: float):
    try:
        if current_price < 0:
            logger.warning(f"Invalid current price for {position.symbol}: {current_price}")
            return position
        position.current_price = current_price
        position.unrealized_pnl = calculate_unrealized_pnl(position)
        if position.entry_price <= 0 or position.quantity <= 0:
            position.unrealized_pnl_pct = 0
        else:
            position.unrealized_pnl_pct = (position.unrealized_pnl / (position.entry_price * position.quantity)) * 100
        return position
    except Exception as e:
        logger.error(f"Error updating trade position for {position.symbol}: {str(e)}")
        return position

@app.post("/store_closed_positions", tags=["Trade"], dependencies=[Depends(verify_api_key)])
async def api_store_closed_positions(req: ClosedPositionsRequest):
    logger.info(f"Storing {len(req.positions)} closed trade positions for date {req.date}")
    try:
        key = f"trade_positions:history:{req.date}"
        value = {
            "timestamp": datetime.now().isoformat(),
            "positions": [position.dict() for position in req.positions]
        }
        success = await redis_client.set(key, json.dumps(value))
        await redis_client.expire(key, CONFIG["redis"]["ttl_closed_positions"])
        logger.info(f"Successfully stored closed positions with key: {key}")
        return {"success": bool(success), "key": key, "count": len(req.positions)}
    except Exception as e:
        logger.error(f"Error storing closed positions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing closed positions: {str(e)}")

@app.get("/get_closed_positions", tags=["Trade"], dependencies=[Depends(verify_api_key)])
async def api_get_closed_positions(date: str = Query(default=datetime.now().strftime("%Y-%m-%d"))):
    logger.info(f"Getting closed trade positions for date {date}")
    try:
        key = f"trade_positions:history:{date}"
        value = await redis_client.get(key)
        if value:
            data = json.loads(value)
            logger.info(f"Retrieved {len(data.get('positions', []))} closed positions")
            return {"success": True, "date": date, "data": data}
        logger.info(f"No closed positions found for date {date}")
        return {"success": False, "date": date, "data": None}
    except Exception as e:
        logger.error(f"Error getting closed positions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting closed positions: {str(e)}")

@app.get("/resource/{resource_uri:path}", tags=["General"], dependencies=[Depends(verify_api_key)])
async def get_resource(resource_uri: str):
    logger.info(f"Resource request for URI: {resource_uri}")
    try:
        if resource_uri == "info":
            info = await redis_client.info()
            logger.info("Successfully retrieved Redis server info")
            return {"success": True, "info": info, "timestamp": datetime.utcnow().isoformat()}
        elif resource_uri == "stats":
            stats = {
                "memory": await redis_client.info("memory"),
                "clients": await redis_client.info("clients"),
                "stats": await redis_client.info("stats"),
                "keyspace": await redis_client.info("keyspace")
            }
            logger.info("Successfully retrieved Redis server stats")
            return {"success": True, "stats": stats, "timestamp": datetime.utcnow().isoformat()}
        elif resource_uri.startswith("keys/"):
            pattern = resource_uri[5:]
            logger.debug(f"Fetching keys matching pattern: {pattern}")
            cursor = 0
            all_keys = []
            while True:
                cursor, keys = await redis_client.scan(cursor, match=pattern, count=1000)
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
        elif resource_uri == "cluster_info":
            if hasattr(app.state, "redis_cluster_manager"):
                return {
                    "success": True,
                    "initialized": True,
                    "config": app.state.redis_cluster_manager.config,
                    "indices": ["risk_assessed_idx", "trade_positions_idx"],
                    "timestamp": datetime.utcnow().isoformat()
                }
            raise HTTPException(status_code=500, detail="RedisClusterManager not initialized")
        else:
            raise HTTPException(status_code=404, detail=f"Unknown resource URI: {resource_uri}")
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error accessing resource {resource_uri}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error accessing resource: {str(e)}")

@app.get("/health", tags=["General"])
async def health_check():
    try:
        ping_result = await redis_client.ping()
        redis_info = await redis_client.info("server")
        cluster_manager_healthy = hasattr(app.state, "redis_cluster_manager") and app.state.redis_cluster_manager.redis is not None
        index_healthy = cluster_manager_healthy and await app.state.redis_cluster_manager.check_index_exists("trade_positions_idx")
        is_healthy = ping_result and cluster_manager_healthy and index_healthy
        app.state.redis_healthy = is_healthy
        health_details = {
            "status": "healthy" if is_healthy else "unhealthy",
            "redis_ping": ping_result,
            "cluster_manager": cluster_manager_healthy,
            "index_healthy": index_healthy,
            "redis_version": redis_info.get("redis_version", "unknown"),
            "uptime_seconds": redis_info.get("uptime_in_seconds", 0),
            "connected_clients": redis_info.get("connected_clients", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.debug(f"Health check: {health_details['status']}")
        return health_details
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        app.state.redis_healthy = False
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/create_trade_position_schema", tags=["Trade"], dependencies=[Depends(verify_api_key)])
async def api_create_trade_position_schema():
    logger.info("Creating trade position schema")
    try:
        if not hasattr(app.state, "redis_cluster_manager"):
            raise HTTPException(status_code=500, detail="RedisClusterManager not initialized")
        index_name = "trade_positions_idx"
        redis_key = "trade_positions:active:*"
        if await app.state.redis_cluster_manager.check_index_exists(index_name):
            logger.info(f"Trade position schema '{index_name}' already exists")
            return {"success": True, "message": "Schema already exists", "index": index_name, "key_pattern": redis_key}
        app.state.redis_cluster_manager.create_index_trade_positions(index_name, redis_key)
        if await app.state.redis_cluster_manager.check_index_exists(index_name):
            logger.info(f"Trade position schema with index name '{index_name}' created successfully")
            return {"success": True, "message": "Schema created successfully", "index": index_name, "key_pattern": redis_key}
        raise HTTPException(status_code=500, detail="Failed to create schema")
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error creating trade position schema: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating schema: {str(e)}")

@app.get("/metrics", tags=["General"], dependencies=[Depends(verify_api_key)])
async def get_metrics():
    logger.info("Fetching Redis metrics")
    cache_key = "redis:metrics"
    if redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Error reading Redis cache: {str(e)}")
    try:
        info = await redis_client.info()
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
        hits = metrics["keyspace_hits"]
        misses = metrics["keyspace_misses"]
        total_operations = hits + misses
        metrics["cache_hit_rate"] = hits / total_operations if total_operations > 0 else 0
        metrics["memory_usage_pct"] = (metrics["memory_used_bytes"] / metrics["memory_peak_bytes"]) * 100 if metrics["memory_peak_bytes"] > 0 else 0
        response = {"success": True, "timestamp": datetime.utcnow().isoformat(), "metrics": metrics}
        if redis_client:
            try:
                await redis_client.set(cache_key, json.dumps(response), ex=CONFIG["redis"]["ttl_metrics"])
                logger.debug(f"Cached response for {cache_key}")
            except Exception as e:
                logger.warning(f"Error writing to Redis cache: {str(e)}")
        logger.info("Successfully retrieved Redis metrics")
        return response
    except Exception as e:
        logger.error(f"Error fetching Redis metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {str(e)}")