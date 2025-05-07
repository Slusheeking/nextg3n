"""
Unusual Whales MCP FastAPI Server for LLM integration (production).

Provides options flow data, alerts, open interest, and analysis from Unusual Whales API.
This server exposes endpoints for accessing financial market data, particularly options trading
activity with unusual patterns that may indicate significant market moves.

Features:
- Options flow data retrieval with filtering
- Trading alerts with customizable parameters
- Open interest data access
- ML-based analysis to identify unusual market activity
- Options chain visualization
- Redis-backed caching and storage of analysis results

Configuration is loaded from environment variables with sensible defaults.
See the CONFIG dictionary for all configurable parameters.

Dependencies:
- FastAPI for API server
- Redis for caching and persistent storage
- HDBSCAN for machine learning clustering
- Matplotlib for data visualization
"""


import os
import aiohttp
import hdbscan
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
import matplotlib
matplotlib.use('Agg')  # Thread-safe backend
import matplotlib.pyplot as plt
import io
import base64
import yaml
from pathlib import Path
from copy import deepcopy
import logging
import re
import json
import tempfile

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
    config_path = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'unusual_whales_config.yaml'))
    default_config = {
        "unusual_whales": {
            "api_key": os.getenv("UNUSUAL_WHALES_API_KEY", ""),
            "rate_limit_per_minute": 60,
            "base_url": "https://unusual_whales.com/api",
            "min_premium": 10000.0,
            "min_volume": 100
        },
        "redis": {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": int(os.getenv("REDIS_DB", "0")),
            "password": os.getenv("REDIS_PASSWORD", None),
            "max_retries": 3
        },
        "cache": {
            "use_cache": True,
            "cache_ttl": 300
        },
        "ttl": {
            "analysis_summary": 7 * 24 * 60 * 60,  # 7 days
            "analysis_full": 24 * 60 * 60,  # 24 hours
            "alerts": 7 * 24 * 60 * 60  # 7 days
        },
        "security": {
            "enable_auth": True,
            "enable_admin_auth": True
        },
        "environment": {
            "mode": os.getenv("ENVIRONMENT", "production"),
            "log_level": os.getenv("LOG_LEVEL", "INFO")
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
        config["unusual_whales"]["api_key"] = os.getenv("UNUSUAL_WHALES_API_KEY", config["unusual_whales"]["api_key"])
        config["unusual_whales"]["rate_limit_per_minute"] = int(os.getenv("UNUSUAL_WHALES_RATE_LIMIT", config["unusual_whales"]["rate_limit_per_minute"]))
        config["unusual_whales"]["base_url"] = os.getenv("UNUSUAL_WHALES_API_URL", config["unusual_whales"]["base_url"])
        config["unusual_whales"]["min_premium"] = float(os.getenv("MIN_PREMIUM", config["unusual_whales"]["min_premium"]))
        config["unusual_whales"]["min_volume"] = int(os.getenv("MIN_VOLUME", config["unusual_whales"]["min_volume"]))
        config["redis"]["host"] = os.getenv("REDIS_HOST", config["redis"]["host"])
        config["redis"]["port"] = int(os.getenv("REDIS_PORT", config["redis"]["port"]))
        config["redis"]["db"] = int(os.getenv("REDIS_DB", config["redis"]["db"]))
        config["redis"]["password"] = os.getenv("REDIS_PASSWORD", config["redis"]["password"])
        config["redis"]["max_retries"] = int(os.getenv("REDIS_MAX_RETRIES", config["redis"]["max_retries"]))
        config["cache"]["use_cache"] = os.getenv("USE_CACHE", str(config["cache"]["use_cache"])).lower() == "true"
        config["cache"]["cache_ttl"] = int(os.getenv("CACHE_TTL", config["cache"]["cache_ttl"]))
        config["security"]["enable_auth"] = os.getenv("UNUSUAL_WHALES_ENABLE_AUTH", str(config["security"]["enable_auth"])).lower() == "true"
        config["security"]["enable_admin_auth"] = os.getenv("UNUSUAL_WHALES_ENABLE_ADMIN_AUTH", str(config["security"]["enable_admin_auth"])).lower() == "true"
        config["environment"]["mode"] = os.getenv("ENVIRONMENT", config["environment"]["mode"])
        config["environment"]["log_level"] = os.getenv("LOG_LEVEL", config["environment"]["log_level"])

        # Validate configuration
        if config["unusual_whales"]["rate_limit_per_minute"] <= 0:
            logger.warning(f"Invalid rate_limit_per_minute: {config['unusual_whales']['rate_limit_per_minute']}. Using default: 60")
            config["unusual_whales"]["rate_limit_per_minute"] = 60
        if config["unusual_whales"]["min_premium"] <= 0:
            logger.warning(f"Invalid min_premium: {config['unusual_whales']['min_premium']}. Using default: 10000")
            config["unusual_whales"]["min_premium"] = 10000.0
        if config["unusual_whales"]["min_volume"] <= 0:
            logger.warning(f"Invalid min_volume: {config['unusual_whales']['min_volume']}. Using default: 100")
            config["unusual_whales"]["min_volume"] = 100
        if config["redis"]["port"] <= 0:
            logger.warning(f"Invalid redis_port: {config['redis']['port']}. Using default: 6379")
            config["redis"]["port"] = 6379
        if config["redis"]["db"] < 0:
            logger.warning(f"Invalid redis_db: {config['redis']['db']}. Using default: 0")
            config["redis"]["db"] = 0
        if config["redis"]["max_retries"] < 0:
            logger.warning(f"Invalid max_retries: {config['redis']['max_retries']}. Using default: 3")
            config["redis"]["max_retries"] = 3
        if config["cache"]["cache_ttl"] <= 0:
            logger.warning(f"Invalid cache_ttl: {config['cache']['cache_ttl']}. Using default: 300")
            config["cache"]["cache_ttl"] = 300
        if config["ttl"]["analysis_summary"] <= 0:
            logger.warning(f"Invalid ttl_analysis_summary: {config['ttl']['analysis_summary']}. Using default: 7 days")
            config["ttl"]["analysis_summary"] = 7 * 24 * 60 * 60
        if config["ttl"]["analysis_full"] <= 0:
            logger.warning(f"Invalid ttl_analysis_full: {config['ttl']['analysis_full']}. Using default: 24 hours")
            config["ttl"]["analysis_full"] = 24 * 60 * 60
        if config["ttl"]["alerts"] <= 0:
            logger.warning(f"Invalid ttl_alerts: {config['ttl']['alerts']}. Using default: 7 days")
            config["ttl"]["alerts"] = 7 * 24 * 60 * 60
        if config["environment"]["log_level"] not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            logger.warning(f"Invalid log_level: {config['environment']['log_level']}. Using default: INFO")
            config["environment"]["log_level"] = "INFO"
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}. Using default configuration.")
        return default_config

CONFIG = validate_config()

logger = get_logger("unusual_whales_server")
logging.getLogger().setLevel(getattr(logging, CONFIG["environment"]["log_level"]))
logger.info("Initializing Unusual Whales server")

# --- Redis Client ---
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
                max_connections=20
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
    def __init__(self, calls_limit: int, time_window: int = 60):
        self.calls_limit = calls_limit
        self.time_window = time_window
        self.redis_key_prefix = "unusual_whales_rate_limit"
    
    async def check_rate_limit(self, request: Request):
        if not redis_client:
            logger.warning("Redis unavailable, skipping rate limiting")
            return True
        client_ip = request.client.host
        key = f"{self.redis_key_prefix}:{client_ip}"
        now = int(datetime.now().timestamp())
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
            logger.error(f"Error checking rate limit in Redis: {str(e)}")
            return True

rate_limiter = RateLimiter(calls_limit=CONFIG["unusual_whales"]["rate_limit_per_minute"])

# --- Authentication ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
admin_api_key_header = APIKeyHeader(name="X-Admin-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not CONFIG["security"]["enable_auth"]:
        return
    valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")
    if not api_key or api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

async def verify_admin_api_key(api_key: str = Security(admin_api_key_header)):
    if not CONFIG["security"]["enable_admin_auth"]:
        return
    admin_api_key = os.getenv("ADMIN_API_KEY", "")
    if not admin_api_key:
        raise HTTPException(status_code=501, detail="Admin API key not configured on server")
    if api_key != admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin API key")
    return api_key

# --- FastAPI Models ---
class OptionsFlowRequest(BaseModel):
    symbols: Optional[List[str]] = Field(None, description="List of ticker symbols to filter by", example=["AAPL", "TSLA", "SPY"])
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format", example="2025-01-01")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format", example="2025-01-15")
    min_premium: Optional[float] = Field(CONFIG["unusual_whales"]["min_premium"], description="Minimum premium amount in dollars", gt=0)
    min_volume: Optional[int] = Field(CONFIG["unusual_whales"]["min_volume"], description="Minimum trading volume", gt=0)
    option_type: Optional[str] = Field("both", description="Type of options to filter by", example="both")
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if v is not None:
            if not all(isinstance(s, str) and s.strip() and len(s) <= 10 for s in v):
                raise ValueError("All symbols must be non-empty strings with max length 10")
        return [s.upper() for s in v] if v else v
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        if v is not None:
            date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
            if not date_pattern.match(v):
                raise ValueError("Date must be in YYYY-MM-DD format")
            try:
                year, month, day = map(int, v.split('-'))
                date(year, month, day)
            except ValueError:
                raise ValueError("Invalid date value")
        return v
    
    @validator('option_type')
    def validate_option_type(cls, v):
        valid_types = ["call", "put", "both"]
        if v.lower() not in valid_types:
            raise ValueError(f"Option type must be one of {valid_types}")
        return v.lower()

class AlertsRequest(BaseModel):
    symbols: Optional[List[str]] = Field(None, description="List of ticker symbols to filter by", example=["AAPL", "TSLA", "SPY"])
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format", example="2025-01-01")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format", example="2025-01-15")
    alert_type: Optional[str] = Field("all", description="Type of alerts to filter by", example="all")
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if v is not None:
            if not all(isinstance(s, str) and s.strip() for s in v):
                raise ValueError("All symbols must be non-empty strings")
        return [s.upper() for s in v] if v else v
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        if v is not None:
            date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
            if not date_pattern.match(v):
                raise ValueError("Date must be in YYYY-MM-DD format")
            try:
                year, month, day = map(int, v.split('-'))
                date(year, month, day)
            except ValueError:
                raise ValueError("Invalid date value")
        return v

class OpenInterestRequest(BaseModel):
    symbol: str = Field(..., description="The ticker symbol to fetch data for", example="SPY")
    expiration_date: Optional[str] = Field(None, description="Options expiration date in YYYY-MM-DD format", example="2025-01-17")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v.strip():
            raise ValueError("Symbol must not be empty")
        if len(v) > 10:
            raise ValueError("Symbol must be 10 characters or less")
        return v.upper()
    
    @validator('expiration_date')
    def validate_date_format(cls, v):
        if v is not None:
            date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
            if not date_pattern.match(v):
                raise ValueError("Date must be in YYYY-MM-DD format")
            try:
                year, month, day = map(int, v.split('-'))
                date(year, month, day)
            except ValueError:
                raise ValueError("Invalid date value")
        return v

class AnalyzeOptionsFlowRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of ticker symbols to analyze", example=["AAPL", "TSLA", "SPY"])
    days: int = Field(1, description="Number of days of historical data to analyze", ge=1, le=30)
    min_premium: float = Field(CONFIG["unusual_whales"]["min_premium"], description="Minimum premium amount in dollars", gt=0)
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("At least one symbol must be provided")
        if not all(isinstance(s, str) and s.strip() for s in v):
            raise ValueError("All symbols must be non-empty strings")
        return [s.upper() for s in v]

# --- Helper Functions ---
def sanitize_key_component(component: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', component.strip())

async def fetch_unusual_whales(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not CONFIG["unusual_whales"]["api_key"]:
        logger.error("Unusual Whales API key not configured")
        raise HTTPException(status_code=500, detail="Unusual Whales API key not configured")
    
    cache_key = f"unusual_whales:{endpoint}:{json.dumps(params, sort_keys=True)}"
    if CONFIG["cache"]["use_cache"] and redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                app.state.cache_hits += 1
                return json.loads(cached)
            app.state.cache_misses += 1
        except Exception as e:
            logger.warning(f"Error reading Redis cache: {str(e)}")
    
    headers = {"Authorization": f"Bearer {CONFIG['unusual_whales']['api_key']}"}
    url = f"{CONFIG['unusual_whales']['base_url']}/{endpoint}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers, timeout=30) as resp:
                if resp.status == 401:
                    logger.error("Invalid or unauthorized API key")
                    raise HTTPException(status_code=401, detail="Invalid or unauthorized API key")
                if resp.status == 429:
                    logger.warning("API rate limit exceeded")
                    raise HTTPException(status_code=429, detail="Unusual Whales API rate limit exceeded")
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Unusual Whales API error: {resp.status} - {text}")
                    raise HTTPException(status_code=resp.status, detail=text)
                response_data = await resp.json()
                if CONFIG["cache"]["use_cache"] and redis_client:
                    try:
                        await redis_client.setex(cache_key, CONFIG["cache"]["cache_ttl"], json.dumps(response_data))
                        logger.debug(f"Cached response for {cache_key}")
                    except Exception as e:
                        logger.warning(f"Error writing to Redis cache: {str(e)}")
                logger.debug(f"Received successful response from {endpoint}")
                return response_data
    except asyncio.TimeoutError:
        logger.error(f"Timeout when connecting to Unusual Whales API: {endpoint}")
        raise HTTPException(status_code=504, detail="Request to Unusual Whales API timed out")
    except aiohttp.ClientError as e:
        logger.error(f"Network error when connecting to Unusual Whales API: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error fetching data from Unusual Whales API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# --- FastAPI Server ---
class RateLimitMiddleware:
    async def dispatch(self, request: Request, call_next):
        if not await rate_limiter.check_rate_limit(request):
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
        return await call_next(request)

app = FastAPI(
    title="Unusual Whales MCP Server for LLM",
    description="Production MCP Server for accessing Unusual Whales options flow data and analytics",
    version="1.0.1",
    docs_url=None if CONFIG["environment"]["mode"] == "production" else "/api/docs",
    redoc_url=None if CONFIG["environment"]["mode"] == "production" else "/api/redoc",
    openapi_tags=[
        {"name": "Options Data", "description": "Endpoints for fetching options trading data"},
        {"name": "Alerts", "description": "Endpoints for trading alerts and signals"},
        {"name": "Analysis", "description": "Machine learning-based analysis of options data"},
        {"name": "Visualization", "description": "Data visualization endpoints"},
        {"name": "Resources", "description": "Resource access endpoints"},
        {"name": "System", "description": "System status and information endpoints"},
        {"name": "Admin", "description": "Administrative endpoints for server management"},
    ]
)

app.add_middleware(RateLimitMiddleware)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_details = [{"loc": error.get("loc", []), "msg": error.get("msg", ""), "type": error.get("type", "")} for error in exc.errors()]
    logger.warning(f"Validation error: {error_details}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation error",
            "details": error_details,
            "message": "Please check your request parameters and try again."
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    error_id = f"error_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    logger.exception(f"Unhandled exception ({error_id}): {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_id": error_id,
            "message": f"An unexpected error occurred. Reference ID: {error_id}"
        },
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP exception {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": str(exc.detail),
            "status_code": exc.status_code
        },
    )

@app.on_event("startup")
async def startup_event():
    logger.info(f"Unusual Whales server starting up in {CONFIG['environment']['mode']} environment")
    startup_info = {
        "environment": CONFIG["environment"]["mode"],
        "version": "1.0.1",
        "redis_host": CONFIG["redis"]["host"],
        "cache_enabled": CONFIG["cache"]["use_cache"],
        "rate_limit": CONFIG["unusual_whales"]["rate_limit_per_minute"]
    }
    logger.info(f"Unusual Whales server starting up: {json.dumps(startup_info)}")
    await connect_redis(max_retries=CONFIG["redis"]["max_retries"])
    if not CONFIG["unusual_whales"]["api_key"]:
        logger.warning("UNUSUAL_WHALES_API_KEY not set - API functionality will be limited")
    app.state.cache_hits = 0
    app.state.cache_misses = 0

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Unusual Whales server shutting down")
    if redis_client:
        try:
            await redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {str(e)}")

@app.get("/server_info", summary="Get server information", tags=["System"], dependencies=[Depends(verify_api_key)])
async def get_server_info():
    return {
        "name": "unusual_whales",
        "version": "1.0.1",
        "description": "Production MCP Server for Unusual Whales Options Flow Integration",
        "tools": [
            "fetch_options_flow", "fetch_alerts", "fetch_open_interest", "analyze_options_flow",
            "visualize_options_chain", "get_unusual_activity_alerts"
        ],
        "resources": [
            "options_flow", "alerts", "open_interest", "visualizations", "status"
        ],
        "config": {
            "rate_limit_per_minute": CONFIG["unusual_whales"]["rate_limit_per_minute"],
            "min_premium": CONFIG["unusual_whales"]["min_premium"],
            "min_volume": CONFIG["unusual_whales"]["min_volume"],
            "base_url": CONFIG["unusual_whales"]["base_url"],
            "environment": CONFIG["environment"]["mode"]
        }
    }

@app.post("/fetch_options_flow", summary="Fetch options flow data", tags=["Options Data"], dependencies=[Depends(verify_api_key)])
async def api_fetch_options_flow(req: OptionsFlowRequest):
    logger.info(f"Fetching options flow data for symbols: {req.symbols}")
    try:
        params = {
            "min_premium": req.min_premium,
            "min_volume": req.min_volume,
        }
        if req.symbols:
            params["symbols"] = ",".join(req.symbols)
        if req.start_date:
            params["start_date"] = req.start_date
        if req.end_date:
            params["end_date"] = req.end_date
        if req.option_type and req.option_type != "both":
            params["option_type"] = req.option_type
        data = await fetch_unusual_whales("options/flow", params)
        result_count = len(data.get("data", [])) if isinstance(data.get("data"), list) else 0
        logger.info(f"Successfully fetched options flow data, found {result_count} results")
        return {"success": True, "data": data, "result_count": result_count}
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error fetching options flow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching options flow: {str(e)}")

@app.post("/fetch_alerts", summary="Fetch trading alerts", tags=["Alerts"], dependencies=[Depends(verify_api_key)])
async def api_fetch_alerts(req: AlertsRequest):
    logger.info(f"Fetching alerts for symbols: {req.symbols}, alert type: {req.alert_type}")
    try:
        params = {}
        if req.symbols:
            params["symbols"] = ",".join(req.symbols)
        if req.start_date:
            params["start_date"] = req.start_date
        if req.end_date:
            params["end_date"] = req.end_date
        if req.alert_type and req.alert_type != "all":
            params["alert_type"] = req.alert_type
        data = await fetch_unusual_whales("options/alerts", params)
        result_count = len(data.get("data", [])) if isinstance(data.get("data"), list) else 0
        logger.info(f"Successfully fetched alerts, found {result_count} results")
        return {"success": True, "data": data, "result_count": result_count}
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error fetching alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {str(e)}")

@app.post("/fetch_open_interest", summary="Fetch options open interest", tags=["Options Data"], dependencies=[Depends(verify_api_key)])
async def api_fetch_open_interest(req: OpenInterestRequest):
    logger.info(f"Fetching open interest for symbol: {req.symbol}")
    try:
        params = {"symbol": req.symbol}
        if req.expiration_date:
            params["expiration_date"] = req.expiration_date
        data = await fetch_unusual_whales("options/open_interest", params)
        logger.info(f"Successfully fetched open interest data for {req.symbol}")
        return {"success": True, "data": data}
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error fetching open interest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching open interest: {str(e)}")

async def analyze_options_flow_with_ml(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("Analyzing options flow data with ML models")
    try:
        if not data:
            logger.warning("No options flow data provided for analysis")
            return []
        features = []
        for item in data:
            required_keys = ["premium", "volume", "open_interest", "price_change", "put_call_ratio", "implied_volatility"]
            if not all(key in item for key in required_keys):
                logger.debug(f"Skipping item with missing features: {item.get('symbol', 'unknown')}")
                continue
            feature_values = [
                item.get("premium", 0),
                item.get("volume", 0),
                item.get("open_interest", 0),
                item.get("price_change", 0),
                item.get("put_call_ratio", 0),
                item.get("implied_volatility", 0)
            ]
            feature_values = [float(v) if v is not None else 0.0 for v in feature_values]
            features.append(feature_values)
        if len(features) < 5:
            logger.warning(f"Insufficient data for clustering: {len(features)} samples")
            return data
        min_cluster_size = max(5, len(features) // 10)
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, cluster_selection_epsilon=0.5)
        cluster_labels = hdbscan_model.fit_predict(features)
        unusual_indices = [i for i, label in enumerate(cluster_labels) if label == -1]
        unusual_activity = [data[i] for i in unusual_indices]
        logger.info(f"Identified {len(unusual_activity)} unusual activities out of {len(data)} total records")
        return unusual_activity
    except Exception as e:
        logger.error(f"Error during options flow analysis with ML: {str(e)}")
        return []

@app.post("/analyze_options_flow", summary="Analyze options flow with ML", tags=["Analysis"], dependencies=[Depends(verify_api_key)])
async def api_analyze_options_flow(req: AnalyzeOptionsFlowRequest):
    logger.info(f"Analyzing options flow for symbols: {req.symbols}, days: {req.days}")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=req.days)
        params = {
            "symbols": ",".join(req.symbols),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "min_premium": req.min_premium
        }
        data = await fetch_unusual_whales("options/flow", params)
        unusual_activity = await analyze_options_flow_with_ml(data.get("data", []))
        analysis_result = {
            "results": data,
            "unusual_activity": unusual_activity,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_version": "1.0.1"
        }
        await store_options_flow_analysis_in_redis(req.symbols, req.days, analysis_result)
        if unusual_activity:
            await store_unusual_activity_alert_in_redis(req.symbols, unusual_activity)
        logger.info(f"Successfully analyzed options flow, found {len(unusual_activity)} unusual activities")
        return {"success": True, "data": analysis_result}
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error analyzing options flow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing options flow: {str(e)}")

@app.get("/resource/{resource_uri:path}", summary="Access resource by URI", tags=["Resources"], dependencies=[Depends(verify_api_key)])
async def get_resource(resource_uri: str):
    logger.info(f"Resource request for URI: {resource_uri}")
    try:
        if resource_uri == "signals/bullish":
            raise HTTPException(status_code=400, detail="Bullish signals require parameters: ?symbols=AAPL,TSLA&days=7")
        elif resource_uri == "signals/bearish":
            raise HTTPException(status_code=400, detail="Bearish signals require parameters: ?symbols=AAPL,TSLA&days=7")
        elif resource_uri == "signals/unusual":
            raise HTTPException(status_code=400, detail="Unusual signals require parameters: ?symbols=AAPL,TSLA&days=7")
        elif resource_uri == "status":
            return {
                "status": "operational",
                "rate_limit_per_minute": CONFIG["unusual_whales"]["rate_limit_per_minute"],
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Unknown resource URI: {resource_uri}")
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error in get_resource: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error accessing resource: {str(e)}")

async def visualize_options_chain(options_data: List[Dict[str, Any]]) -> str:
    logger.info("Generating options chain visualization")
    try:
        if not options_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No options data available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14)
            ax.set_title("Options Chain Visualization")
            ax.axis('off')
        else:
            call_options = [opt for opt in options_data if opt.get("option_type") == "call"]
            put_options = [opt for opt in options_data if opt.get("option_type") == "put"]
            call_options.sort(key=lambda x: float(x.get("strike", 0)))
            put_options.sort(key=lambda x: float(x.get("strike", 0)))
            call_strikes = [float(opt.get("strike", 0)) for opt in call_options]
            put_strikes = [float(opt.get("strike", 0)) for opt in put_options]
            call_volumes = [int(opt.get("volume", 0)) for opt in call_options]
            put_volumes = [int(opt.get("volume", 0)) for opt in put_options]
            all_strikes = sorted(set(call_strikes + put_strikes))
            fig, ax = plt.subplots(figsize=(12, 6))
            if call_volumes:
                ax.plot(call_strikes, call_volumes, 'g-', label="Calls", marker='o')
            if put_volumes:
                ax.plot(put_strikes, put_volumes, 'r-', label="Puts", marker='o')
            ax.set_xlabel("Strike Price ($)")
            ax.set_ylabel("Volume (contracts)")
            symbol = options_data[0].get("symbol", "Unknown") if options_data else "Unknown"
            ax.set_title(f"Options Chain Visualization - {symbol}")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.2f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
            ax.legend()
            ax.grid(True)
            if all_strikes:
                padding = (max(all_strikes) - min(all_strikes)) * 0.05
                ax.set_xlim(min(all_strikes) - padding, max(all_strikes) + padding)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.savefig(tmp_file.name, format="png", dpi=100)
            tmp_file.close()
            with open(tmp_file.name, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            os.unlink(tmp_file.name)
        
        plt.close(fig)
        logger.info("Options chain visualization generated successfully")
        return img_base64
    except Exception as e:
        logger.error(f"Error generating options chain visualization: {str(e)}")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'Error generating visualization: {str(e)}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
        ax.set_title("Error in Options Chain Visualization")
        ax.axis('off')
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.savefig(tmp_file.name, format="png")
            tmp_file.close()
            with open(tmp_file.name, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            os.unlink(tmp_file.name)
        plt.close(fig)
        return img_base64

@app.post("/visualize_options_chain", summary="Generate options chain visualization", tags=["Visualization"], dependencies=[Depends(verify_api_key)])
async def api_visualize_options_chain(req: AnalyzeOptionsFlowRequest):
    logger.info(f"Visualizing options chain for symbols: {req.symbols}, days: {req.days}")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=req.days)
        params = {
            "symbols": ",".join(req.symbols),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        }
        data = await fetch_unusual_whales("options/flow", params)
        img_base64 = await visualize_options_chain(data.get("data", []))
        logger.info("Successfully visualized options chain")
        return {"success": True, "image": img_base64}
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error visualizing options chain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error visualizing options chain: {str(e)}")

async def store_options_flow_analysis_in_redis(symbols: List[str], days: int, analysis_data: Dict[str, Any]):
    try:
        sanitized_symbols = [sanitize_key_component(s) for s in symbols]
        key = f"unusual_whales:options_analysis:{'_'.join(sanitized_symbols)}:{days}d:{datetime.now().strftime('%Y%m%d')}"
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "days": days,
            "unusual_count": len(analysis_data.get("unusual_activity", [])),
            "total_count": len(analysis_data.get("results", {}).get("data", [])),
        }
        await redis_client.setex(f"{key}:summary", CONFIG["ttl"]["analysis_summary"], json.dumps(summary_data))
        await redis_client.setex(key, CONFIG["ttl"]["analysis_full"], json.dumps(analysis_data))
        logger.info(f"Stored options flow analysis in Redis with key: {key}")
        return key
    except Exception as e:
        logger.error(f"Error storing options flow analysis in Redis: {str(e)}")
        return None

async def store_unusual_activity_alert_in_redis(symbols: List[str], unusual_activity: List[Dict[str, Any]]):
    try:
        timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
        sanitized_symbols = [sanitize_key_component(s) for s in symbols]
        alert_key = f"unusual_whales:alert:{'_'.join(sanitized_symbols)}:{timestamp_str}"
        sample_activity = [
            {
                "symbol": activity.get("symbol"),
                "price": activity.get("price"),
                "strike": activity.get("strike"),
                "premium": activity.get("premium"),
                "volume": activity.get("volume"),
                "option_type": activity.get("option_type"),
                "expiration": activity.get("expiration")
            }
            for activity in unusual_activity[:5]
        ]
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "activity_count": len(unusual_activity),
            "sample_activity": sample_activity
        }
        await redis_client.setex(alert_key, CONFIG["ttl"]["alerts"], json.dumps(alert_data))
        score = datetime.now().timestamp()
        await redis_client.zadd("unusual_whales:alerts_by_time", {alert_key: score})
        logger.info(f"Stored unusual activity alert in Redis with key: {alert_key}")
        return alert_key
    except Exception as e:
        logger.error(f"Error storing unusual activity alert in Redis: {str(e)}")
        return None

@app.post("/get_unusual_activity_alerts_from_redis", summary="Retrieve stored unusual activity alerts", tags=["Alerts"], dependencies=[Depends(verify_api_key)])
async def api_get_unusual_activity_alerts_from_redis(symbols: Optional[List[str]] = None, limit: int = 50, offset: int = 0):
    logger.info(f"Retrieving unusual activity alerts for symbols: {symbols}, limit: {limit}, offset: {offset}")
    try:
        alerts = []
        if symbols and len(symbols) > 0:
            sanitized_symbols = [sanitize_key_component(s.upper()) for s in symbols]
            alert_keys = await redis_client.zrevrange("unusual_whales:alerts_by_time", 0, -1)
            processed = 0
            skipped = 0
            for key in alert_keys:
                if len(alerts) >= limit:
                    break
                alert_data = await redis_client.get(key)
                if not alert_data:
                    continue
                try:
                    alert = json.loads(alert_data)
                    alert_symbols = [s.upper() for s in alert.get("symbols", [])]
                    if any(s in alert_symbols for s in symbols):
                        if skipped < offset:
                            skipped += 1
                        else:
                            alerts.append(alert)
                            processed += 1
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON data in Redis key: {key}")
        else:
            alert_keys = await redis_client.zrevrange("unusual_whales:alerts_by_time", offset, offset + limit - 1)
            for key in alert_keys:
                alert_data = await redis_client.get(key)
                if alert_data:
                    try:
                        alerts.append(json.loads(alert_data))
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON data in Redis key: {key}")
        total_count = await redis_client.zcard("unusual_whales:alerts_by_time")
        logger.info(f"Retrieved {len(alerts)} unusual activity alerts from Redis out of {total_count} total alerts")
        return {
            "success": True,
            "alerts": alerts,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + len(alerts) < total_count
            }
        }
    except Exception as e:
        logger.error(f"Error retrieving unusual activity alerts from Redis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving alerts: {str(e)}")

@app.get("/health", summary="Server health check", tags=["System"], dependencies=[Depends(verify_api_key)])
async def health_check():
    try:
        redis_status = "operational"
        redis_ping_time = None
        try:
            start_time = datetime.now()
            await redis_client.ping()
            redis_ping_time = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            redis_status = f"error: {str(e)}"
        api_key_status = "configured" if CONFIG["unusual_whales"]["api_key"] else "missing"
        return {
            "status": "healthy" if redis_status == "operational" and api_key_status == "configured" else "degraded",
            "version": "1.0.1",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "redis": {
                    "status": redis_status,
                    "ping_time_seconds": redis_ping_time
                },
                "api_key": api_key_status,
                "cache": {
                    "enabled": CONFIG["cache"]["use_cache"],
                    "hits": app.state.cache_hits,
                    "misses": app.state.cache_misses
                },
                "rate_limiting": {
                    "limit_per_minute": CONFIG["unusual_whales"]["rate_limit_per_minute"]
                }
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/admin/clear_cache", summary="Clear Redis cache", tags=["Admin"], dependencies=[Depends(verify_admin_api_key)])
async def clear_cache():
    try:
        cursor = 0
        cleared_entries = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, match="unusual_whales:*", count=1000)
            if keys:
                cleared_entries += await redis_client.delete(*keys)
            if cursor == 0:
                break
        app.state.cache_hits = 0
        app.state.cache_misses = 0
        logger.info(f"Cache cleared, removed {cleared_entries} entries")
        return {"success": True, "cleared_entries": cleared_entries}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")
