"""
Polygon REST API FastAPI server.
Handles market data retrieval from Polygon's REST API.
"""

import os
import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("polygon_rest_api")

# Load environment variables
load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("POLYGON_API_KEY", "")
BASE_URL = "https://api.polygon.io"
RATE_LIMIT_PER_MINUTE = int(os.getenv("POLYGON_RATE_LIMIT", "5"))

# --- Global state ---
http_session: Optional[aiohttp.ClientSession] = None
request_timestamps: List[datetime] = []

async def get_http_session():
    """Get or create an HTTP session for API requests."""
    global http_session
    if http_session is None or http_session.closed:
        http_session = aiohttp.ClientSession()
    return http_session

async def wait_for_rate_limit():
    """Simple in-memory rate limiter for API requests."""
    global request_timestamps
    now = datetime.now()
    request_timestamps = [ts for ts in request_timestamps if now - ts < timedelta(minutes=1)]
    
    if len(request_timestamps) >= RATE_LIMIT_PER_MINUTE:
        wait_time = (request_timestamps[0] + timedelta(minutes=1)) - now
        if wait_time.total_seconds() > 0:
            logger.warning(f"Rate limit reached. Waiting for {wait_time.total_seconds():.2f}s")
            await asyncio.sleep(wait_time.total_seconds())
            now = datetime.now()
            request_timestamps = [ts for ts in request_timestamps if now - ts < timedelta(minutes=1)]
            
    request_timestamps.append(now)

async def polygon_request(endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make a request to the Polygon API with rate limiting."""
    await wait_for_rate_limit()
    session = await get_http_session()
    
    if not API_KEY:
        error_message = "Polygon API key not configured. Set POLYGON_API_KEY environment variable."
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    
    final_params = params.copy() if params else {}
    final_params["apiKey"] = API_KEY
    
    url = f"{BASE_URL}{endpoint}"
    logger.debug(f"Making Polygon API request to {url}")
    
    try:
        async with session.get(url, params=final_params) as response:
            if response.status == 429:
                retry_after = int(response.headers.get("Retry-After", "60"))
                logger.warning(f"Polygon API rate limit hit (429). Retrying after {retry_after}s.")
                await asyncio.sleep(retry_after)
                return await polygon_request(endpoint, params)
                
            response_json = await response.json()
            
            if response.status != 200:
                error_detail = response_json.get("message") or response_json.get("error") or str(response_json)
                logger.error(f"Polygon API error: {response.status} - {error_detail}")
                raise HTTPException(status_code=response.status, detail=error_detail)
                
            return response_json
    except aiohttp.ClientError as e:
        logger.error(f"HTTP client error for Polygon endpoint {endpoint}: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")
    except Exception as e:
        logger.error(f"Unexpected error for Polygon endpoint {endpoint}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- API Models ---
class AggregatesRequest(BaseModel):
    ticker: str
    multiplier: int
    timespan: str
    from_date: str
    to_date: str
    adjusted: bool = True
    sort: str = "asc"
    limit: int = 5000
    
    @validator('timespan')
    def validate_timespan(cls, v):
        allowed_timespans = ["minute", "hour", "day", "week", "month", "quarter", "year"]
        if v not in allowed_timespans:
            raise ValueError(f"Timespan {v} not supported. Allowed: {allowed_timespans}")
        return v

class TickerDetailsRequest(BaseModel):
    ticker: str

class MarketStatusRequest(BaseModel):
    pass

class TickerNewsRequest(BaseModel):
    ticker: str
    limit: int = 10
    sort: str = "published_utc"
    order: str = "desc"

# --- API Endpoints ---
async def fetch_aggregates(
    ticker: str, multiplier: int, timespan: str, from_date: str, to_date: str,
    adjusted: bool = True, sort: str = "asc", limit: int = 5000
) -> Dict[str, Any]:
    """Fetch aggregated market data from Polygon."""
    endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {"adjusted": str(adjusted).lower(), "sort": sort, "limit": str(limit)}
    
    response = await polygon_request(endpoint, params)
    
    if response.get("status") == "OK" and "results" in response:
        return {
            "success": True, 
            "ticker": response.get("ticker"), 
            "results": response["results"]
        }
    
    error_msg = response.get("message", response.get("error", "Unknown error fetching aggregates"))
    logger.error(f"Failed to fetch aggregates for {ticker}: {error_msg}")
    return {"success": False, "error": error_msg, "ticker": ticker}

async def fetch_ticker_details(ticker: str) -> Dict[str, Any]:
    """Fetch details about a ticker symbol."""
    endpoint = f"/v3/reference/tickers/{ticker}"
    
    response = await polygon_request(endpoint)
    
    if response.get("status") == "OK" and "results" in response:
        return {"success": True, "results": response["results"]}
    
    error_msg = response.get("message", response.get("error", "Unknown error fetching ticker details"))
    logger.error(f"Failed to fetch ticker details for {ticker}: {error_msg}")
    return {"success": False, "error": error_msg, "ticker": ticker}

async def fetch_market_status() -> Dict[str, Any]:
    """Get current market status."""
    endpoint = "/v1/marketstatus/now"
    
    response = await polygon_request(endpoint)
    
    if "market" in response:
        return {"success": True, "status": response}
    
    error_msg = response.get("message", response.get("error", "Unknown error fetching market status"))
    logger.error(f"Failed to fetch market status: {error_msg}")
    return {"success": False, "error": error_msg}

async def fetch_ticker_news(ticker: str, limit: int = 10, sort: str = "published_utc", order: str = "desc") -> Dict[str, Any]:
    """Fetch recent news for a ticker symbol."""
    endpoint = f"/v2/reference/news"
    params = {"ticker": ticker, "limit": str(limit), "order": order, "sort": sort}
    
    response = await polygon_request(endpoint, params)
    
    if response.get("status") == "OK" and "results" in response:
        return {"success": True, "results": response["results"]}
    
    error_msg = response.get("message", response.get("error", "Unknown error fetching news"))
    logger.error(f"Failed to fetch news for {ticker}: {error_msg}")
    return {"success": False, "error": error_msg, "ticker": ticker}

# --- FastAPI Server ---
app = FastAPI(
    title="Polygon REST API",
    description="Standard API for market data from Polygon REST API",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Polygon REST API starting up")
    if not API_KEY:
        logger.warning("Polygon API key not configured - service will not function correctly")
    await get_http_session()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Polygon REST API shutting down")
    global http_session
    if http_session and not http_session.closed:
        await http_session.close()

@app.get("/api/info")
async def get_api_info():
    """Get information about the API."""
    return {
        "name": "Polygon REST API",
        "version": "1.0.0",
        "description": "Standard API for market data from Polygon REST API",
        "endpoints": [
            "/api/market/aggregates",
            "/api/ticker/details",
            "/api/market/status",
            "/api/ticker/news"
        ]
    }

@app.post("/api/market/aggregates")
async def api_fetch_aggregates(req: AggregatesRequest):
    """Fetch historical market data aggregates."""
    logger.info(f"Fetching aggregates for {req.ticker} from {req.from_date} to {req.to_date}")
    return await fetch_aggregates(
        req.ticker, req.multiplier, req.timespan, 
        req.from_date, req.to_date, req.adjusted, req.sort, req.limit
    )

@app.post("/api/ticker/details")
async def api_ticker_details(req: TickerDetailsRequest):
    """Get details about a ticker symbol."""
    logger.info(f"Fetching ticker details for {req.ticker}")
    return await fetch_ticker_details(req.ticker)

@app.post("/api/market/status")
async def api_market_status(req: MarketStatusRequest):
    """Get current market status."""
    logger.info("Fetching market status")
    return await fetch_market_status()

@app.post("/api/ticker/news")
async def api_ticker_news(req: TickerNewsRequest):
    """Get recent news for a ticker symbol."""
    logger.info(f"Fetching news for {req.ticker}")
    return await fetch_ticker_news(req.ticker, req.limit, req.sort, req.order)

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
    uvicorn.run(app, host="0.0.0.0", port=8001)