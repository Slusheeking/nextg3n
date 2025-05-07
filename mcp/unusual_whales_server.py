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
from datetime import datetime, timedelta
from dotenv import load_dotenv
from monitor.logging_utils import get_logger
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import matplotlib.pyplot as plt
import io
import base64
import redis
import json

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
CONFIG = {
    "api_key": os.getenv("UNUSUAL_WHALES_API_KEY", ""),
    "rate_limit_per_minute": int(os.getenv("UNUSUAL_WHALES_RATE_LIMIT", "60")),
    "use_cache": os.getenv("USE_CACHE", "true").lower() == "true",
    "min_premium": float(os.getenv("MIN_PREMIUM", "10000")),
    "min_volume": int(os.getenv("MIN_VOLUME", "100")),
    "base_url": os.getenv("UNUSUAL_WHALES_API_URL", "https://unusualwhales.com/api"),
    "redis_host": os.getenv("REDIS_HOST", "localhost"),
    "redis_port": int(os.getenv("REDIS_PORT", 6379)),
    "redis_db": int(os.getenv("REDIS_DB", 0)),
    "redis_password": os.getenv("REDIS_PASSWORD", None),
    "environment": os.getenv("ENVIRONMENT", "production"),
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
}

# Get logger from centralized logging system
logger = get_logger("unusual_whales_server")
logger.info("Initializing Unusual Whales server")

# --- FastAPI Models ---
from pydantic import Field, validator
from datetime import date
import re

class OptionsFlowRequest(BaseModel):
    symbols: Optional[List[str]] = Field(
        default=None,
        description="List of ticker symbols to filter by",
        example=["AAPL", "TSLA", "SPY"]
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date in YYYY-MM-DD format",
        example="2025-01-01"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date in YYYY-MM-DD format",
        example="2025-01-15"
    )
    min_premium: Optional[float] = Field(
        default=CONFIG["min_premium"],
        description="Minimum premium amount in dollars",
        gt=0
    )
    min_volume: Optional[int] = Field(
        default=CONFIG["min_volume"],
        description="Minimum trading volume",
        gt=0
    )
    option_type: Optional[str] = Field(
        default="both",
        description="Type of options to filter by",
        example="both"
    )
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if v is not None:
            if not all(isinstance(s, str) and s.strip() and len(s) <= 10 for s in v):
                raise ValueError("All symbols must be non-empty strings with max length 10")
        return v
    
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
    symbols: Optional[List[str]] = Field(
        default=None,
        description="List of ticker symbols to filter by",
        example=["AAPL", "TSLA", "SPY"]
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date in YYYY-MM-DD format",
        example="2025-01-01"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date in YYYY-MM-DD format",
        example="2025-01-15"
    )
    alert_type: Optional[str] = Field(
        default="all",
        description="Type of alerts to filter by",
        example="all"
    )
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if v is not None:
            if not all(isinstance(s, str) and s.strip() for s in v):
                raise ValueError("All symbols must be non-empty strings")
        return v
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        if v is not None:
            date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
            if not date_pattern.match(v):
                raise ValueError("Date must be in YYYY-MM-DD format")
        return v

class OpenInterestRequest(BaseModel):
    symbol: str = Field(
        ...,  # Required field
        description="The ticker symbol to fetch data for",
        example="SPY"
    )
    expiration_date: Optional[str] = Field(
        default=None,
        description="Options expiration date in YYYY-MM-DD format",
        example="2025-01-17"
    )
    
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
        return v

class AnalyzeOptionsFlowRequest(BaseModel):
    symbols: List[str] = Field(
        ...,  # Required field
        description="List of ticker symbols to analyze",
        example=["AAPL", "TSLA", "SPY"]
    )
    days: int = Field(
        default=1,
        description="Number of days of historical data to analyze",
        ge=1,  # greater than or equal to 1
        le=30  # less than or equal to 30
    )
    min_premium: float = Field(
        default=CONFIG["min_premium"],
        description="Minimum premium amount in dollars",
        gt=0
    )
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("At least one symbol must be provided")
        if not all(isinstance(s, str) and s.strip() for s in v):
            raise ValueError("All symbols must be non-empty strings")
        return [s.upper() for s in v]  # Convert symbols to uppercase

# Cache implementation
CACHE = {}
CACHE_TTL = 300  # 5 minutes in seconds

# Initialize Redis client with connection pool for production environments
redis_pool = redis.ConnectionPool(
    host=CONFIG["redis_host"],
    port=CONFIG["redis_port"],
    db=CONFIG["redis_db"],
    password=CONFIG["redis_password"],
    max_connections=20,
    decode_responses=False  # Keep raw bytes for proper serialization
)
redis_client = redis.Redis(connection_pool=redis_pool)

# Rate limiting implementation
request_counts = {}
RATE_LIMIT_RESET_INTERVAL = 60  # seconds
last_reset_time = datetime.now()

def check_rate_limit(endpoint: str) -> bool:
    """Check if the request is within rate limits."""
    global last_reset_time
    
    # Reset counters if reset interval has passed
    current_time = datetime.now()
    if (current_time - last_reset_time).total_seconds() > RATE_LIMIT_RESET_INTERVAL:
        request_counts.clear()
        last_reset_time = current_time
    
    # Check and update request count
    current_count = request_counts.get(endpoint, 0)
    if current_count >= CONFIG["rate_limit_per_minute"]:
        return False
    
    request_counts[endpoint] = current_count + 1
    return True

# --- Helper Functions ---
async def fetch_unusual_whales(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not CONFIG["api_key"]:
        logger.error("Unusual Whales API key not configured")
        raise HTTPException(status_code=500, detail="Unusual Whales API key not configured")
    
    # Check rate limit
    if not check_rate_limit(endpoint):
        logger.warning(f"Rate limit exceeded for endpoint: {endpoint}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    
    # Check cache if enabled
    if CONFIG["use_cache"]:
        cache_key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        cached_data = CACHE.get(cache_key)
        if cached_data:
            cache_timestamp = cached_data.get("timestamp")
            current_time = datetime.now().timestamp()
            
            # Return cached data if it's still valid
            if current_time - cache_timestamp < CACHE_TTL:
                logger.debug(f"Cache hit for {endpoint}")
                app.state.cache_hits += 1
                return cached_data.get("data", {})
                
        app.state.cache_misses += 1
    
    headers = {"Authorization": f"Bearer {CONFIG['api_key']}"}
    url = f"{CONFIG['base_url']}/{endpoint}"
    
    logger.debug(f"Making request to Unusual Whales API: {endpoint} with params: {params}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers, timeout=30) as resp:
                if resp.status == 429:
                    logger.warning("API rate limit exceeded on Unusual Whales side")
                    raise HTTPException(status_code=429, detail="Unusual Whales API rate limit exceeded")
                elif resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Unusual Whales API error: {resp.status} - {text}")
                    raise HTTPException(status_code=resp.status, detail=text)
                
                response_data = await resp.json()
                logger.debug(f"Received successful response from {endpoint}")
                
                # Cache the result if caching is enabled
                if CONFIG["use_cache"]:
                    CACHE[cache_key] = {
                        "data": response_data,
                        "timestamp": datetime.now().timestamp()
                    }
                
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

app = FastAPI(
    title="Unusual Whales MCP Server for LLM",
    description="Production MCP Server for accessing Unusual Whales options flow data and analytics",
    version="1.0.1",
    docs_url="/api/docs" if CONFIG["environment"] != "production" else None,  # Disable docs in production
    redoc_url="/api/redoc" if CONFIG["environment"] != "production" else None,  # Disable redoc in production
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

# Add CORS middleware for production security
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Add your allowed origins here for production
        # For example: "https://api.yourcompany.com"
        # In development, you might want to allow all origins with "*"
        "*" if CONFIG["environment"] != "production" else "",
    ],
    allow_credentials=True,
    allow_methods=["*"] if CONFIG["environment"] != "production" else ["GET", "POST"],
    allow_headers=["*"],
)

# --- Exception handlers ---

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors in API requests with user-friendly messages"""
    error_details = []
    for error in exc.errors():
        error_details.append({
            "loc": error.get("loc", []),
            "msg": error.get("msg", ""),
            "type": error.get("type", "")
        })
    
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
    """Handle unhandled exceptions with a graceful response"""
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

# Custom HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom handler for HTTP exceptions with improved error response format"""
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
    logger.info(f"Unusual Whales server starting up in {CONFIG['environment']} environment")
    # Log startup in structured format for better monitoring
    startup_info = {
        "environment": CONFIG["environment"],
        "version": "1.0.1",
        "redis_host": CONFIG["redis_host"],
        "cache_enabled": CONFIG["use_cache"],
        "rate_limit": CONFIG["rate_limit_per_minute"]
    }
    logger.info(f"Unusual Whales server starting up: {json.dumps(startup_info)}")
    
    # Check API key availability
    if not CONFIG["api_key"]:
        logger.warning("UNUSUAL_WHALES_API_KEY not set - API functionality will be limited")
    
    # Check Redis connectivity
    try:
        redis_client.ping()
        logger.info(f"Connected to Redis at {CONFIG['redis_host']}:{CONFIG['redis_port']}")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
    
    # Setup rate limiting reset task
    asyncio.create_task(reset_rate_limits_periodically())
    
    # Initialize cache hit/miss tracking
    app.state.cache_hits = 0
    app.state.cache_misses = 0

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Unusual Whales server shutting down")
    # Clean up any connections
    try:
        redis_pool.disconnect()
        logger.info("Redis connection pool closed")
    except Exception as e:
        logger.warning(f"Error closing Redis connections: {e}")

async def reset_rate_limits_periodically():
    """Task to periodically reset rate limits"""
    global request_counts, last_reset_time
    while True:
        await asyncio.sleep(RATE_LIMIT_RESET_INTERVAL)
        request_counts.clear()
        last_reset_time = datetime.now()
        logger.debug("Rate limit counters reset")

@app.get("/server_info", summary="Get server information", tags=["System"])
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
            "rate_limit_per_minute": CONFIG["rate_limit_per_minute"],
            "min_premium": CONFIG["min_premium"],
            "min_volume": CONFIG["min_volume"],
            "base_url": CONFIG["base_url"],
            "environment": CONFIG["environment"]
        }
    }

@app.post("/fetch_options_flow", summary="Fetch options flow data", tags=["Options Data"])
async def api_fetch_options_flow(req: OptionsFlowRequest):
    """
    Fetches options flow data from Unusual Whales API with the specified filters.
    
    Options flow data shows unusual trading activity in options contracts that
    may indicate informed positioning by market participants.
    
    Parameters:
    - **symbols**: Optional list of ticker symbols to filter by
    - **start_date**: Optional start date (YYYY-MM-DD)
    - **end_date**: Optional end date (YYYY-MM-DD)
    - **min_premium**: Minimum premium amount to filter by (default: 10000)
    - **min_volume**: Minimum trading volume to filter by (default: 100)
    - **option_type**: Filter by option type: "call", "put", or "both" (default)
    
    Returns:
    - JSON object with matching options flow data
    """
    logger.info(f"Fetching options flow data for symbols: {req.symbols}")
    
    try:
        params = {
            "min_premium": req.min_premium,
            "min_volume": req.min_volume,
        }
        
        if req.symbols:
            params["symbols"] = ",".join(req.symbols)
            logger.debug(f"Filtering by symbols: {req.symbols}")
            
        if req.start_date:
            params["start_date"] = req.start_date
            logger.debug(f"Using start date: {req.start_date}")
            
        if req.end_date:
            params["end_date"] = req.end_date
            logger.debug(f"Using end date: {req.end_date}")
            
        if req.option_type and req.option_type != "both":
            params["option_type"] = req.option_type
            logger.debug(f"Filtering by option type: {req.option_type}")
            
        data = await fetch_unusual_whales("options/flow", params)
        
        result_count = len(data.get("data", [])) if isinstance(data.get("data"), list) else "unknown"
        logger.info(f"Successfully fetched options flow data, found {result_count} results")
        
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"Error fetching options flow: {str(e)}")
        raise

@app.post("/fetch_alerts", summary="Fetch trading alerts", tags=["Alerts"])
async def api_fetch_alerts(req: AlertsRequest):
    """
    Fetches trading alerts from Unusual Whales API with the specified filters.
    
    Alerts represent trading signals based on unusual activity that may require
    attention.
    
    Parameters:
    - **symbols**: Optional list of ticker symbols to filter by
    - **start_date**: Optional start date (YYYY-MM-DD)
    - **end_date**: Optional end date (YYYY-MM-DD)
    - **alert_type**: Type of alert to filter by; "all" returns all types (default)
    
    Returns:
    - JSON object with matching alerts data
    """
    logger.info(f"Fetching alerts for symbols: {req.symbols}, alert type: {req.alert_type}")
    
    try:
        params = {}
        
        if req.symbols:
            params["symbols"] = ",".join(req.symbols)
            logger.debug(f"Filtering by symbols: {req.symbols}")
            
        if req.start_date:
            params["start_date"] = req.start_date
            logger.debug(f"Using start date: {req.start_date}")
            
        if req.end_date:
            params["end_date"] = req.end_date
            logger.debug(f"Using end date: {req.end_date}")
            
        if req.alert_type and req.alert_type != "all":
            params["alert_type"] = req.alert_type
            logger.debug(f"Filtering by alert type: {req.alert_type}")
            
        data = await fetch_unusual_whales("options/alerts", params)
        
        result_count = len(data.get("data", [])) if isinstance(data.get("data"), list) else "unknown"
        logger.info(f"Successfully fetched alerts, found {result_count} results")
        
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"Error fetching alerts: {str(e)}")
        raise

@app.post("/fetch_open_interest", summary="Fetch options open interest", tags=["Options Data"])
async def api_fetch_open_interest(req: OpenInterestRequest):
    """
    Fetches open interest data for options of a specific symbol.
    
    Open interest represents the total number of outstanding derivative contracts,
    providing insight into the liquidity and interest in specific options.
    
    Parameters:
    - **symbol**: The ticker symbol to fetch open interest data for (required)
    - **expiration_date**: Optional specific expiration date (YYYY-MM-DD)
    
    Returns:
    - JSON object with open interest data for the requested symbol
    """
    logger.info(f"Fetching open interest for symbol: {req.symbol}")
    
    try:
        params = {"symbol": req.symbol}
        
        if req.expiration_date:
            params["expiration_date"] = req.expiration_date
            logger.debug(f"Using expiration date: {req.expiration_date}")
            
        data = await fetch_unusual_whales("options/open_interest", params)
        
        logger.info(f"Successfully fetched open interest data for {req.symbol}")
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"Error fetching open interest: {str(e)}")
        raise

# Initial analyze_options_flow handler removed to avoid duplicate route
# This was replaced by the more advanced version below that uses ML analysis

async def analyze_options_flow_with_ml(data: List[Dict[str, Any]]):
    """
    Analyzes options flow data using AI/ML models (HDBSCAN).
    
    This function applies HDBSCAN clustering to options flow data to detect unusual
    patterns and outliers that may indicate significant market activity. It works by:
    1. Extracting numerical features from options data (premium, volume, etc.)
    2. Applying HDBSCAN clustering to group similar options activity
    3. Identifying outliers (points marked as noise by HDBSCAN) as unusual activity
    
    HDBSCAN is particularly suitable for this analysis as it:
    - Does not require specifying the number of clusters in advance
    - Handles noise points well (which become our unusual activity markers)
    - Works with varying density clusters in the feature space
    
    Args:
        data: List of dictionaries containing options data with fields like
             premium, volume, open_interest, price_change, etc.
             
    Returns:
        List of dictionaries containing options data identified as unusual
    """
    logger.info("Analyzing options flow data with ML models")
    
    try:
        # Skip analysis if data is empty
        if not data:
            logger.warning("No options flow data provided for analysis")
            return []
            
        # 1. Data Preparation: Extract relevant features from options flow data
        features = []
        for item in data:
            # Extract features like premium, volume, open_interest, etc.
            premium = item.get("premium", 0)
            volume = item.get("volume", 0)
            open_interest = item.get("open_interest", 0)
            price_change = item.get("price_change", 0)
            put_call_ratio = item.get("put_call_ratio", 0)
            implied_volatility = item.get("implied_volatility", 0)
            
            # Check for NaN or None values and replace with 0
            feature_values = [premium, volume, open_interest, price_change, put_call_ratio, implied_volatility]
            feature_values = [0 if v is None else v for v in feature_values]
            
            features.append(feature_values)
        
        # Check if we have enough data for clustering
        if len(features) < 5:
            logger.warning(f"Insufficient data for clustering: {len(features)} samples")
            return data  # Return all as potentially unusual if not enough data
            
        # 2. HDBSCAN Clustering: Cluster the options flow data
        min_cluster_size = max(5, len(features) // 10)  # Dynamic cluster size based on data amount
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                      min_samples=1,
                                      cluster_selection_epsilon=0.5)
        cluster_labels = hdbscan_model.fit_predict(features)
        
        # Identify outliers as unusual activity
        unusual_indices = [i for i, label in enumerate(cluster_labels) if label == -1]
        unusual_activity = [data[i] for i in unusual_indices]
        
        logger.info(f"Identified {len(unusual_activity)} unusual activities out of {len(data)} total records.")
        
        return unusual_activity
    except Exception as e:
        logger.error(f"Error during options flow analysis with ML: {str(e)}")
        # In production, return empty list on error rather than raising exception
        # This allows the API to continue functioning even if analysis fails
        return []

@app.post("/analyze_options_flow", summary="Analyze options flow with ML", tags=["Analysis"])
async def api_analyze_options_flow(req: AnalyzeOptionsFlowRequest):
    """
    Analyzes options flow data using machine learning to identify unusual activity.
    
    This endpoint fetches options flow data and applies HDBSCAN clustering to
    detect outliers and patterns that may indicate market-moving events.
    Results are stored in Redis for later retrieval.
    
    Parameters:
    - **symbols**: List of ticker symbols to analyze (required)
    - **days**: Number of days of historical data to analyze (default: 1)
    - **min_premium**: Minimum premium threshold for filtering (default: 10000)
    
    Returns:
    - JSON object containing analysis results including identified unusual activity
    """
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
        
        logger.debug(f"Date range: {params['start_date']} to {params['end_date']}")
        
        data = await fetch_unusual_whales("options/flow", params)
        
        # Analyze the options flow data with ML models
        unusual_activity = await analyze_options_flow_with_ml(data.get("data", []))
        
        logger.info(f"Successfully analyzed options flow, found {len(unusual_activity)} unusual activities")
        
        # Store analysis and alerts in Redis
        analysis_result = {
            "results": data,
            "unusual_activity": unusual_activity,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_version": "1.0.1"
        }
        await store_options_flow_analysis_in_redis(req.symbols, req.days, analysis_result)
        if unusual_activity:
            await store_unusual_activity_alert_in_redis(req.symbols, unusual_activity)
            
        return {"success": True, "data": analysis_result}
    except Exception as e:
        logger.error(f"Error analyzing options flow: {str(e)}")
        raise

@app.get("/resource/{resource_uri:path}", summary="Access resource by URI", tags=["Resources"])
async def get_resource(resource_uri: str):
    """
    Access a specific resource by URI path.
    
    Available resources:
    - **signals/bullish**: Bullish trading signals (requires filtering parameters)
    - **signals/bearish**: Bearish trading signals (requires filtering parameters)
    - **signals/unusual**: Unusual activity signals (requires filtering parameters)
    - **status**: Server operational status
    
    Parameters:
    - **resource_uri**: Path to the requested resource
    
    Returns:
    - JSON object containing the requested resource data or an appropriate error
    """
    logger.info(f"Resource request for URI: {resource_uri}")
    
    try:
        if resource_uri == "signals/bullish":
            logger.warning("Bullish signals endpoint called without parameters")
            raise HTTPException(status_code=400, detail="Provide filtering parameters for production use.")
        elif resource_uri == "signals/bearish":
            logger.warning("Bearish signals endpoint called without parameters")
            raise HTTPException(status_code=400, detail="Provide filtering parameters for production use.")
        elif resource_uri == "signals/unusual":
            logger.warning("Unusual signals endpoint called without parameters")
            raise HTTPException(status_code=400, detail="Provide filtering parameters for production use.")
        elif resource_uri == "status":
            logger.debug("Returning operational status")
            return {
                "status": "operational",
                "rate_limit_per_minute": CONFIG["rate_limit_per_minute"],
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            logger.warning(f"Unknown resource URI requested: {resource_uri}")
            raise HTTPException(status_code=404, detail=f"Unknown resource URI: {resource_uri}")
    except Exception as e:
        logger.error(f"Error in get_resource: {str(e)}")
        raise

# --- Options Chain Visualization ---

async def visualize_options_chain(options_data: List[Dict[str, Any]]) -> str:
    """
    Generates a visualization of the options chain data using Matplotlib.

    Args:
        options_data: A list of dictionaries containing options data.

    Returns:
        A base64 encoded string of the plot.
    """
    logger.info("Generating options chain visualization")
    
    try:
        # Validation
        if not options_data:
            logger.warning("No options data provided for visualization")
            # Create an empty plot with a message
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No options data available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title("Options Chain Visualization")
            ax.axis('off')
            
            # Convert the plot to a base64 encoded string
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="png")
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
            plt.close(fig)
            return img_base64
            
        # Extract and validate data for plotting
        call_options = [opt for opt in options_data if opt.get("option_type") == "call"]
        put_options = [opt for opt in options_data if opt.get("option_type") == "put"]
        
        # Sort by strike price
        call_options.sort(key=lambda x: float(x.get("strike", 0)))
        put_options.sort(key=lambda x: float(x.get("strike", 0)))
        
        call_strikes = [float(opt.get("strike", 0)) for opt in call_options]
        put_strikes = [float(opt.get("strike", 0)) for opt in put_options]
        call_volumes = [int(opt.get("volume", 0)) for opt in call_options]
        put_volumes = [int(opt.get("volume", 0)) for opt in put_options]
        
        # Get a common set of strike prices for x-axis alignment
        all_strikes = sorted(set(call_strikes + put_strikes))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot data if available
        if call_volumes:
            ax.plot(call_strikes, call_volumes, 'g-', label="Calls", marker='o')
        if put_volumes:
            ax.plot(put_strikes, put_volumes, 'r-', label="Puts", marker='o')

        # Customize the plot
        ax.set_xlabel("Strike Price ($)")
        ax.set_ylabel("Volume (contracts)")
        
        # Get the symbol from the data if available
        symbol = options_data[0].get("symbol", "Unknown") if options_data else "Unknown"
        ax.set_title(f"Options Chain Visualization - {symbol}")
        
        # Format the x-axis labels for currency
        import matplotlib.ticker as mtick
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.2f}'))
        
        # Add commas to y-axis values for readability
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        
        ax.legend()
        ax.grid(True)
        
        # Set reasonable limits if there's data
        if all_strikes:
            padding = (max(all_strikes) - min(all_strikes)) * 0.05  # 5% padding
            ax.set_xlim(min(all_strikes) - padding, max(all_strikes) + padding)

        # Convert the plot to a base64 encoded string
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png", dpi=100)
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")

        # Clear the plot for the next use
        plt.close(fig)
        
        logger.info("Options chain visualization generated successfully")
        return img_base64
    except Exception as e:
        logger.error(f"Error generating options chain visualization: {str(e)}")
        # Return a simple error plot instead of raising an exception
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'Error generating visualization: {str(e)}',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red')
        ax.set_title("Error in Options Chain Visualization")
        ax.axis('off')
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
        plt.close(fig)
        return img_base64

@app.post("/visualize_options_chain", summary="Generate options chain visualization", tags=["Visualization"])
async def api_visualize_options_chain(req: AnalyzeOptionsFlowRequest):
    """
    Generates a visualization of options chain data for the specified symbols.
    
    Creates a graphical representation of options chain showing call and put volumes
    across different strike prices, which helps identify areas of high interest.
    
    Parameters:
    - **symbols**: List of ticker symbols to visualize (required)
    - **days**: Number of days of historical data to include (default: 1)
    
    Returns:
    - JSON object containing a base64 encoded image of the visualization
    """
    logger.info(f"Visualizing options chain for symbols: {req.symbols}, days: {req.days}")
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=req.days)
        
        params = {
            "symbols": ",".join(req.symbols),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        }
        
        logger.debug(f"Date range: {params['start_date']} to {params['end_date']}")
        
        data = await fetch_unusual_whales("options/flow", params)
        
        # Generate the options chain visualization
        img_base64 = await visualize_options_chain(data.get("data", []))
        
        logger.info("Successfully visualized options chain")
        
        return {"success": True, "image": img_base64}
    except Exception as e:
        logger.error(f"Error visualizing options chain: {str(e)}")
        raise

# --- Redis Integration for Options Analysis ---

async def store_options_flow_analysis_in_redis(symbols: List[str], days: int, analysis_data: Dict[str, Any]):
    """Stores the options flow analysis data in Redis."""
    try:
        # Sanitize symbols for key creation
        sanitized_symbols = ['_'.join(s.split()) for s in symbols]
        key = f"unusual_whales:options_analysis:{'_'.join(sanitized_symbols)}:{days}d:{datetime.now().strftime('%Y%m%d')}"
        
        # Create a summary version with less data to conserve space
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "days": days,
            "unusual_count": len(analysis_data.get("unusual_activity", [])),
            "total_count": len(analysis_data.get("results", {}).get("data", [])),
        }
        
        # Store summary data with longer expiration
        await asyncio.to_thread(
            redis_client.setex,
            f"{key}:summary",
            timedelta(days=7),  # Keep summary for a week
            json.dumps(summary_data)
        )
        
        # Store full data with shorter expiration
        await asyncio.to_thread(
            redis_client.setex,
            key,
            timedelta(hours=24),  # Keep full data for a day
            json.dumps(analysis_data)
        )
        
        logger.info(f"Stored options flow analysis in Redis with key: {key}")
        return key
    except redis.RedisError as e:
        logger.error(f"Redis error storing options flow analysis: {e}")
    except Exception as e:
        logger.error(f"Error storing options flow analysis in Redis: {e}")

async def store_unusual_activity_alert_in_redis(symbols: List[str], unusual_activity: List[Dict[str, Any]]):
    """Stores an alert for unusual activity in Redis."""
    try:
        timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
        # Sanitize symbols for key creation
        sanitized_symbols = ['_'.join(s.split()) for s in symbols]
        alert_key = f"unusual_whales:alert:{'_'.join(sanitized_symbols)}:{timestamp_str}"
        
        # Extract the most important fields for each activity to save space
        sample_activity = []
        for activity in unusual_activity[:5]:  # Store up to 5 samples
            sample = {
                "symbol": activity.get("symbol"),
                "price": activity.get("price"),
                "strike": activity.get("strike"),
                "premium": activity.get("premium"),
                "volume": activity.get("volume"),
                "option_type": activity.get("option_type"),
                "expiration": activity.get("expiration")
            }
            sample_activity.append(sample)
            
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "activity_count": len(unusual_activity),
            "sample_activity": sample_activity
        }
        
        # Store as JSON string, expire after 7 days
        await asyncio.to_thread(redis_client.setex, alert_key, timedelta(days=7), json.dumps(alert_data))
        
        # Also add to a sorted set for faster retrieval by timestamp
        score = datetime.now().timestamp()
        await asyncio.to_thread(redis_client.zadd, "unusual_whales:alerts_by_time", {alert_key: score})
        
        logger.info(f"Stored unusual activity alert in Redis with key: {alert_key}")
        return alert_key
    except redis.RedisError as e:
        logger.error(f"Redis error storing unusual activity alert: {e}")
    except Exception as e:
        logger.error(f"Error storing unusual activity alert in Redis: {e}")

@app.post("/get_unusual_activity_alerts_from_redis", summary="Retrieve stored unusual activity alerts", tags=["Alerts"])
async def api_get_unusual_activity_alerts_from_redis(
    symbols: Optional[List[str]] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    Retrieves previously stored unusual activity alerts from Redis.
    
    Provides access to historical alerts with pagination support. Alerts can be
    filtered by symbol or retrieved for all available symbols.
    
    Parameters:
    - **symbols**: Optional list of symbols to filter alerts by
    - **limit**: Maximum number of alerts to return (default: 50)
    - **offset**: Number of alerts to skip for pagination (default: 0)
    
    Returns:
    - JSON object containing matching alerts with pagination metadata
    """
    try:
        logger.info(f"Retrieving unusual activity alerts for symbols: {symbols}, limit: {limit}, offset: {offset}")
        
        if symbols and len(symbols) > 0:
            # More efficient retrieval using the sorted set by timestamp
            # For symbol filtering, we still need to examine each alert
            alerts = []
            # Get all keys from the sorted set (most recent first)
            alert_keys = await asyncio.to_thread(
                redis_client.zrevrange,
                "unusual_whales:alerts_by_time",
                0,
                -1
            )
            
            # Process keys with pagination
            processed = 0
            skipped = 0
            
            for key_bytes in alert_keys:
                if len(alerts) >= limit:
                    break
                    
                key = key_bytes.decode('utf-8')
                alert_json = await asyncio.to_thread(redis_client.get, key)
                
                if not alert_json:
                    continue
                    
                try:
                    alert_data = json.loads(alert_json)
                    
                    # Filter by symbols
                    alert_symbols = alert_data.get("symbols", [])
                    if any(s in alert_symbols for s in symbols):
                        if skipped < offset:
                            skipped += 1
                        else:
                            alerts.append(alert_data)
                            processed += 1
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON data in Redis key: {key}")
                    continue
        else:
            # No symbol filtering - use the sorted set directly with pagination
            alert_keys = await asyncio.to_thread(
                redis_client.zrevrange,
                "unusual_whales:alerts_by_time",
                offset,
                offset + limit - 1
            )
            
            alerts = []
            for key_bytes in alert_keys:
                key = key_bytes.decode('utf-8')
                alert_json = await asyncio.to_thread(redis_client.get, key)
                if alert_json:
                    try:
                        alert_data = json.loads(alert_json)
                        alerts.append(alert_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON data in Redis key: {key}")
                        continue
        
        # Get the total count for pagination info
        total_count = await asyncio.to_thread(redis_client.zcard, "unusual_whales:alerts_by_time")
        
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
    except redis.RedisError as e:
        logger.error(f"Redis error retrieving unusual activity alerts: {e}")
        raise HTTPException(status_code=503, detail=f"Redis service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Error retrieving unusual activity alerts from Redis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Health check and diagnostics ---

@app.get("/health", summary="Server health check", tags=["System"])
async def health_check():
    """
    Health check endpoint for monitoring systems.
    
    Provides detailed status information about server components:
    - Redis connection status
    - API key configuration status
    - Cache status and statistics
    - Rate limiting information
    
    Returns:
    - JSON object containing health status and component details
    """
    try:
        # Check Redis connection
        redis_status = "operational"
        redis_ping_time = None
        try:
            start_time = datetime.now()
            await asyncio.to_thread(redis_client.ping)
            redis_ping_time = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            redis_status = f"error: {str(e)}"
        
        # Check API key
        api_key_status = "configured" if CONFIG["api_key"] else "missing"
        
        return {
            "status": "healthy",
            "version": "1.0.1",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "redis": {
                    "status": redis_status,
                    "ping_time_seconds": redis_ping_time
                },
                "api_key": api_key_status,
                "cache": {
                    "enabled": CONFIG["use_cache"],
                    "entries": len(CACHE),
                    "ttl_seconds": CACHE_TTL
                },
                "rate_limiting": {
                    "limit_per_minute": CONFIG["rate_limit_per_minute"],
                    "current_requests": len(request_counts)
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

# --- Admin utilities and monitoring ---

# Admin endpoint security checker
from fastapi import Security, Depends
from fastapi.security import APIKeyHeader

API_KEY_NAME = "X-Admin-API-Key"
API_KEY = os.getenv("ADMIN_API_KEY", "")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def verify_admin_api_key(api_key: str = Security(api_key_header)):
    """Verify admin API key for protected endpoints"""
    if not API_KEY:
        logger.warning("Admin API key not configured, denying access")
        raise HTTPException(
            status_code=501,
            detail="Admin API key not configured on server"
        )
    if api_key != API_KEY:
        logger.warning(f"Access attempt with invalid API key: {api_key[:5]}...")
        raise HTTPException(
            status_code=403,
            detail="Invalid admin API key"
        )
    return api_key

@app.post("/admin/clear_cache", summary="Clear in-memory cache", tags=["Admin"],
          dependencies=[Depends(verify_admin_api_key)])
async def clear_cache():
    """
    Administrative endpoint to clear the in-memory cache.
    
    Requires admin API key authentication.
    
    Removes all cached API responses, forcing subsequent requests to fetch
    fresh data from the Unusual Whales API. This can be useful after
    configuration changes or when troubleshooting data issues.
    
    Returns:
    - JSON object indicating success and the number of cleared cache entries
    """
    try:
        current_entries = len(CACHE)
        CACHE.clear()
        app.state.cache_hits = 0
        app.state.cache_misses = 0
        logger.info(f"Cache cleared, removed {current_entries} entries")
        return {"success": True, "cleared_entries": current_entries}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
