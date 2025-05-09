"""
Unusual Whales FastAPI Server.
Provides options flow data, alerts, open interest, and analysis from Unusual Whales API.
"""

import os
import aiohttp
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
import logging
import re

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("unusual_whales_api")

# Load environment variables
load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("UNUSUAL_WHALES_API_KEY", "")
BASE_URL = os.getenv("UNUSUAL_WHALES_API_URL", "https://unusual_whales.com/api")
MIN_PREMIUM = float(os.getenv("MIN_PREMIUM", "10000.0"))
MIN_VOLUME = int(os.getenv("MIN_VOLUME", "100"))
RATE_LIMIT = int(os.getenv("UNUSUAL_WHALES_RATE_LIMIT", "60"))
ENABLE_AUTH = os.getenv("UNUSUAL_WHALES_ENABLE_AUTH", "true").lower() == "true"

# --- API Models ---
class OptionsFlowRequest(BaseModel):
    symbols: Optional[List[str]] = Field(None, description="List of ticker symbols to filter by", example=["AAPL", "TSLA", "SPY"])
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format", example="2025-01-01")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format", example="2025-01-15")
    min_premium: Optional[float] = Field(MIN_PREMIUM, description="Minimum premium amount in dollars", gt=0)
    min_volume: Optional[int] = Field(MIN_VOLUME, description="Minimum trading volume", gt=0)
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

# --- Rate Limiter ---
class RateLimiter:
    def __init__(self, calls_limit: int, time_window: int = 60):
        self.calls_limit = calls_limit
        self.time_window = time_window
        self.timestamps = []
    
    async def check_rate_limit(self, request: Request):
        client_ip = request.client.host
        now = datetime.now().timestamp()
        
        # Remove timestamps older than the time window
        self.timestamps = [ts for ts in self.timestamps if now - ts < self.time_window]
        
        # Check if limit exceeded
        if len(self.timestamps) >= self.calls_limit:
            logger.warning(f"Rate limit exceeded for IP {client_ip}: {len(self.timestamps)} requests in last {self.time_window}s")
            return False
        
        # Add current timestamp
        self.timestamps.append(now)
        return True

rate_limiter = RateLimiter(calls_limit=RATE_LIMIT)

# --- Authentication ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not ENABLE_AUTH:
        return
    valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")
    if not api_key or api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# --- Helper Functions ---
async def fetch_unusual_whales(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch data from the Unusual Whales API."""
    if not API_KEY:
        logger.error("Unusual Whales API key not configured")
        raise HTTPException(status_code=500, detail="Unusual Whales API key not configured")
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"{BASE_URL}/{endpoint}"
    
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

# --- Rate Limiting Middleware ---
class RateLimitMiddleware:
    async def __call__(self, request: Request, call_next):
        if not await rate_limiter.check_rate_limit(request):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."}
            )
        return await call_next(request)

# --- FastAPI Server ---
app = FastAPI(
    title="Unusual Whales API",
    description="Standard API for accessing Unusual Whales options flow data",
    version="1.0.0"
)

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

@app.on_event("startup")
async def startup_event():
    logger.info("Unusual Whales API starting up")
    if not API_KEY:
        logger.warning("UNUSUAL_WHALES_API_KEY not set - API functionality will be limited")

@app.get("/api/info")
async def get_api_info():
    """Get information about the API."""
    return {
        "name": "Unusual Whales API",
        "version": "1.0.0",
        "description": "Standard API for Unusual Whales Options Flow Data",
        "endpoints": [
            "/api/options/flow",
            "/api/options/alerts", 
            "/api/options/open_interest"
        ]
    }

@app.post("/api/options/flow", dependencies=[Depends(verify_api_key)])
async def fetch_options_flow(req: OptionsFlowRequest):
    """Fetch options flow data."""
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching options flow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching options flow: {str(e)}")

@app.post("/api/options/alerts", dependencies=[Depends(verify_api_key)])
async def fetch_alerts(req: AlertsRequest):
    """Fetch trading alerts."""
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {str(e)}")

@app.post("/api/options/open_interest", dependencies=[Depends(verify_api_key)])
async def fetch_open_interest(req: OpenInterestRequest):
    """Fetch options open interest."""
    logger.info(f"Fetching open interest for symbol: {req.symbol}")
    
    try:
        params = {"symbol": req.symbol}
        
        if req.expiration_date:
            params["expiration_date"] = req.expiration_date
            
        data = await fetch_unusual_whales("options/open_interest", params)
        
        logger.info(f"Successfully fetched open interest data for {req.symbol}")
        return {"success": True, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching open interest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching open interest: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Check the health status of the API."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "api_key_configured": bool(API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)