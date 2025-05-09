"""
Yahoo Finance FastAPI Server.
Provides financial news, quotes, chart data, and summary from Yahoo Finance.
"""

import os
import json
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
import yfinance
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("yahoo_finance_api")

# Load environment variables
load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("YAHOO_FINANCE_API_KEY", "")
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
DEFAULT_NEWS_COUNT = int(os.getenv("DEFAULT_NEWS_COUNT", "20"))
DEFAULT_CHART_INTERVAL = os.getenv("DEFAULT_CHART_INTERVAL", "1d")
DEFAULT_CHART_RANGE = os.getenv("DEFAULT_CHART_RANGE", "1mo")
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.5"))  # Seconds between yfinance calls
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
ENABLE_AUTH = os.getenv("YAHOO_FINANCE_ENABLE_AUTH", "true").lower() == "true"

# --- Rate Limiter ---
class RateLimiter:
    def __init__(self, calls_limit: int, time_window: int = 60):
        self.calls_limit = calls_limit
        self.time_window = time_window
        self.timestamps = []
    
    async def check_rate_limit(self, request: Request) -> bool:
        now = time.time()
        # Remove timestamps older than the time window
        self.timestamps = [ts for ts in self.timestamps if now - ts < self.time_window]
        
        # Check if limit exceeded
        if len(self.timestamps) >= self.calls_limit:
            logger.warning(f"Rate limit exceeded: {len(self.timestamps)} requests in last {self.time_window}s")
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

# --- Utility Functions ---
def serialize_pandas_data(data: Any) -> Any:
    """Convert pandas objects to JSON-serializable format."""
    import pandas as pd
    import numpy as np
    
    if isinstance(data, pd.Series):
        return {k: serialize_pandas_data(v) for k, v in data.to_dict().items()}
    elif isinstance(data, pd.DataFrame):
        return {k: serialize_pandas_data(v) for k, v in data.to_dict(orient="index").items()}
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif isinstance(data, (list, dict)):
        return json.loads(json.dumps(data, default=str))
    elif data is None or isinstance(data, (int, float, str, bool)):
        return data
    return str(data)

# --- API Models ---
class NewsRequest(BaseModel):
    symbols: List[str] = []
    count: int = Field(DEFAULT_NEWS_COUNT, ge=1, le=100)

class QuotesRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=1, max_items=50)

class ChartRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    interval: str = Field(DEFAULT_CHART_INTERVAL)
    range: str = Field(DEFAULT_CHART_RANGE)
    
    @validator('interval')
    def validate_interval(cls, v):
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if v not in valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of: {', '.join(valid_intervals)}")
        return v
    
    @validator('range')
    def validate_range(cls, v):
        valid_ranges = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        if v not in valid_ranges:
            raise ValueError(f"Invalid range. Must be one of: {', '.join(valid_ranges)}")
        return v

class SummaryRequest(BaseModel):
    symbol: str = Field(..., min_length=1)

# --- Rate Limiting Middleware ---
class RateLimitMiddleware:
    async def __call__(self, request: Request, call_next):
        if not await rate_limiter.check_rate_limit(request):
            return HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        return await call_next(request)

# --- FastAPI Server ---
app = FastAPI(
    title="Yahoo Finance API",
    description="Standard API for Yahoo Finance data",
    version="1.0.0"
)

app.add_middleware(RateLimitMiddleware)

@app.on_event("startup")
async def startup_event():
    logger.info("Yahoo Finance API starting up")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Yahoo Finance API shutting down")

@app.get("/api/info")
async def get_api_info():
    """Get information about the API."""
    return {
        "name": "Yahoo Finance API",
        "version": "1.0.0",
        "description": "Standard API for Yahoo Finance data",
        "endpoints": [
            "/api/news",
            "/api/quotes", 
            "/api/chart", 
            "/api/summary"
        ]
    }

@app.post("/api/news", dependencies=[Depends(verify_api_key)])
async def fetch_news(req: NewsRequest):
    """Fetch financial news for specified symbols."""
    logger.info(f"Fetching news for symbols: {req.symbols}")
    
    try:
        symbols_list = req.symbols if req.symbols else []
        count = req.count if req.count else DEFAULT_NEWS_COUNT
        
        all_articles = []
        tickers_to_fetch = symbols_list if symbols_list else ["^GSPC"]  # Use S&P 500 as default
        
        for ticker_str in tickers_to_fetch:
            try:
                ticker_obj = yfinance.Ticker(ticker_str)
                news_items = ticker_obj.news
                
                if not news_items:
                    logger.warning(f"No news found for {ticker_str}")
                    continue
                    
                for article in news_items[:count if len(tickers_to_fetch) == 1 else count // len(tickers_to_fetch) or 1]:
                    processed_article = {
                        "id": str(article.get("uuid", "")),
                        "title": article.get("title", ""),
                        "publisher": article.get("publisher", ""),
                        "link": article.get("link", ""),
                        "published_at": article.get("providerPublishTime", ""),
                        "summary": article.get("summary", ""),
                        "tickers": article.get("relatedTickers", [ticker_str]),
                        "type": article.get("type", "")
                    }
                    all_articles.append(processed_article)
                    
                await asyncio.sleep(REQUEST_DELAY)
                
            except Exception as e:
                logger.warning(f"Error fetching news for {ticker_str}: {str(e)}")
        
        # Remove duplicate articles
        unique_articles = []
        seen_links = set()
        for art in all_articles:
            if art["link"] not in seen_links:
                unique_articles.append(art)
                seen_links.add(art["link"])
                
        logger.info(f"Fetched {len(unique_articles)} unique news articles")
        return {"success": True, "articles": unique_articles[:count]}
        
    except Exception as e:
        logger.error(f"Error in fetch_news: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")

@app.post("/api/quotes", dependencies=[Depends(verify_api_key)])
async def fetch_quotes(req: QuotesRequest):
    """Fetch current quotes for specified symbols."""
    logger.info(f"Fetching quotes for symbols: {req.symbols}")
    
    try:
        data = {}
        
        for symbol in req.symbols:
            try:
                ticker = yfinance.Ticker(symbol)
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    data[symbol] = serialize_pandas_data(hist.iloc[-1])
                else:
                    data[symbol] = {"error": "No data found"}
                    
                await asyncio.sleep(REQUEST_DELAY)
                
            except Exception as e:
                logger.warning(f"Error fetching quote for {symbol}: {str(e)}")
                data[symbol] = {"error": str(e)}
                
        logger.info(f"Successfully fetched quotes for {len(req.symbols)} symbols")
        return {"success": True, "quotes": data, "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Error in fetch_quotes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching quotes: {str(e)}")

@app.post("/api/chart", dependencies=[Depends(verify_api_key)])
async def fetch_chart_data(req: ChartRequest):
    """Fetch historical chart data for a symbol."""
    logger.info(f"Fetching chart data for {req.symbol} with interval {req.interval} and range {req.range}")
    
    try:
        ticker = yfinance.Ticker(req.symbol)
        hist = ticker.history(period=req.range, interval=req.interval)
        
        if hist.empty:
            logger.warning(f"No chart data found for {req.symbol}")
            return {"success": False, "error": "No chart data found"}
            
        # Convert datetime index to string for JSON serialization
        hist.index = hist.index.strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Successfully fetched chart data for {req.symbol}")
        return {"success": True, "chart_data": serialize_pandas_data(hist)}
        
    except Exception as e:
        logger.error(f"Error in fetch_chart_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching chart data: {str(e)}")

@app.post("/api/summary", dependencies=[Depends(verify_api_key)])
async def fetch_summary(req: SummaryRequest):
    """Fetch company summary information."""
    logger.info(f"Fetching summary for {req.symbol}")
    
    try:
        ticker = yfinance.Ticker(req.symbol)
        info = ticker.info
        
        summary_data = {
            "longName": info.get("longName"),
            "symbol": info.get("symbol"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "website": info.get("website"),
            "marketCap": info.get("marketCap"),
            "previousClose": info.get("previousClose"),
            "open": info.get("open"),
            "dayLow": info.get("dayLow"),
            "dayHigh": info.get("dayHigh"),
            "volume": info.get("volume"),
            "averageVolume": info.get("averageVolume"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "beta": info.get("beta"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "dividendYield": info.get("dividendYield"),
            "description": info.get("longBusinessSummary", ""),
            "employees": info.get("fullTimeEmployees"),
            "exchange": info.get("exchange"),
        }
        
        logger.info(f"Successfully fetched summary for {req.symbol}")
        return {"success": True, "summary": summary_data}
        
    except Exception as e:
        logger.error(f"Error in fetch_summary: {str(e)}")
        if "No fundamentals found" in str(e) or "No data found for symbol" in str(e):
            return {"success": False, "error": f"No summary data found for symbol {req.symbol}"}
        raise HTTPException(status_code=500, detail=f"Error fetching summary: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Check the health status of the API."""
    try:
        # Check YFinance connectivity
        test_ticker = yfinance.Ticker("AAPL")
        info = test_ticker.info
        yfinance_status = "healthy" if info and "symbol" in info else "unhealthy"
        
        return {
            "status": "healthy" if yfinance_status == "healthy" else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "yfinance": yfinance_status
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)