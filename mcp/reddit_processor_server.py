"""
Reddit MCP FastAPI Server for LLM integration (production).
Provides social sentiment data from Reddit API.
All configuration is contained in this file.
"""


import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
import asyncpraw
import asyncio
import time
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import torch
import redis.asyncio as aioredis
from starlette.middleware.base import BaseHTTPMiddleware
import yaml
from pathlib import Path
from copy import deepcopy
import logging
from fastapi.security import APIKeyHeader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification, pipeline

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
    config_path = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'reddit_config.yaml'))
    default_config = {
        "reddit": {
            "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
            "client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
            "user_agent": os.getenv("REDDIT_USER_AGENT", "NextG3N Trading Bot v1.0"),
            "username": os.getenv("REDDIT_USERNAME", ""),
            "password": os.getenv("REDDIT_PASSWORD", ""),
            "rate_limit_calls": 30,
            "rate_limit_window": 60,
            "max_retries": 3,
            "retry_delay": 1.5
        },
        "redis": {
            "host": os.getenv("REDDIT_REDIS_HOST", os.getenv("REDIS_HOST", "localhost")),
            "port": int(os.getenv("REDDIT_REDIS_PORT", os.getenv("REDIS_PORT", "6379"))),
            "db": int(os.getenv("REDDIT_REDIS_DB", os.getenv("REDIS_DB", "1"))),
            "password": os.getenv("REDDIT_REDIS_PASSWORD", os.getenv("REDIS_PASSWORD", None))
        },
        "cache": {
            "use_cache": True
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
        config["reddit"]["client_id"] = os.getenv("REDDIT_CLIENT_ID", config["reddit"]["client_id"])
        config["reddit"]["client_secret"] = os.getenv("REDDIT_CLIENT_SECRET", config["reddit"]["client_secret"])
        config["reddit"]["user_agent"] = os.getenv("REDDIT_USER_AGENT", config["reddit"]["user_agent"])
        config["reddit"]["username"] = os.getenv("REDDIT_USERNAME", config["reddit"]["username"])
        config["reddit"]["password"] = os.getenv("REDDIT_PASSWORD", config["reddit"]["password"])
        config["reddit"]["rate_limit_calls"] = int(os.getenv("REDDIT_RATE_LIMIT", config["reddit"]["rate_limit_calls"]))
        config["reddit"]["rate_limit_window"] = int(os.getenv("REDDIT_RATE_WINDOW", config["reddit"]["rate_limit_window"]))
        config["reddit"]["max_retries"] = int(os.getenv("REDDIT_MAX_RETRIES", config["reddit"]["max_retries"]))
        config["reddit"]["retry_delay"] = float(os.getenv("REDDIT_RETRY_DELAY", config["reddit"]["retry_delay"]))
        config["redis"]["host"] = os.getenv("REDDIT_REDIS_HOST", os.getenv("REDIS_HOST", config["redis"]["host"]))
        config["redis"]["port"] = int(os.getenv("REDDIT_REDIS_PORT", os.getenv("REDIS_PORT", config["redis"]["port"])))
        config["redis"]["db"] = int(os.getenv("REDDIT_REDIS_DB", os.getenv("REDIS_DB", config["redis"]["db"])))
        config["redis"]["password"] = os.getenv("REDDIT_REDIS_PASSWORD", os.getenv("REDIS_PASSWORD", config["redis"]["password"]))
        config["cache"]["use_cache"] = os.getenv("REDDIT_USE_CACHE", str(config["cache"]["use_cache"])).lower() == "true"
        config["security"]["enable_auth"] = os.getenv("REDDIT_ENABLE_AUTH", str(config["security"]["enable_auth"])).lower() == "true"

        # Validate configuration
        if not config["reddit"]["client_id"] or not config["reddit"]["client_secret"]:
            logger.warning("Reddit API credentials not set. API requests will fail.")
        if config["reddit"]["rate_limit_calls"] <= 0:
            logger.warning(f"Invalid rate_limit_calls: {config['reddit']['rate_limit_calls']}. Using default: 30")
            config["reddit"]["rate_limit_calls"] = 30
        if config["reddit"]["rate_limit_window"] <= 0:
            logger.warning(f"Invalid rate_limit_window: {config['reddit']['rate_limit_window']}. Using default: 60")
            config["reddit"]["rate_limit_window"] = 60
        if config["reddit"]["max_retries"] < 0:
            config["reddit"]["max_retries"] = 3
        if config["reddit"]["retry_delay"] <= 0:
            config["reddit"]["retry_delay"] = 1.5
        if config["redis"]["port"] <= 0:
            logger.warning(f"Invalid redis_port: {config['redis']['port']}. Using default: 6379")
            config["redis"]["port"] = 6379
        if config["redis"]["db"] < 0:
            logger.warning(f"Invalid redis_db: {config['redis']['db']}. Using default: 1")
            config["redis"]["db"] = 1
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}. Using default configuration.")
        return default_config

CONFIG = validate_config()

logger = get_logger("reddit_processor_server")
logger.info("Initializing Reddit Processor server")

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
                decode_responses=True
            )
            await redis_client.ping()
            logger.info(f"Connected to Redis at {CONFIG['redis']['host']}:{CONFIG['redis']['port']} (DB: {CONFIG['redis']['db']})")
            return
        except Exception as e:
            logger.error(f"Redis connection attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
    logger.warning("Failed to connect to Redis after retries. Proceeding without Redis.")
    redis_client = None

async def get_redis_health() -> bool:
    if not redis_client:
        return False
    try:
        await redis_client.ping()
        return True
    except Exception:
        return False

# --- Reddit API Helper ---
reddit_client = None

async def init_reddit_client():
    global reddit_client
    logger.debug("Creating Reddit API client")
    if not CONFIG["reddit"]["client_id"] or not CONFIG["reddit"]["client_secret"]:
        logger.error("Reddit API credentials not configured")
        raise ValueError("Reddit API credentials not configured")
    reddit_client = asyncpraw.Reddit(
        client_id=CONFIG["reddit"]["client_id"],
        client_secret=CONFIG["reddit"]["client_secret"],
        user_agent=CONFIG["reddit"]["user_agent"],
        username=CONFIG["reddit"]["username"] or None,
        password=CONFIG["reddit"]["password"] or None,
        requestor_kwargs={"timeout": 30}
    )

async def reddit_api_with_retry(api_call):
    max_retries = CONFIG["reddit"]["max_retries"]
    retry_delay = CONFIG["reddit"]["retry_delay"]
    attempt = 0
    while attempt < max_retries:
        try:
            return await api_call()
        except asyncpraw.exceptions.RedditAPIException as e:
            if any(isinstance(error, asyncpraw.exceptions.APIException) and error.error_type == "RATELIMIT" for error in e.items):
                retry_after = 60  # Default retry-after
                for error in e.items:
                    if hasattr(error, "message") and "try again in" in error.message:
                        try:
                            retry_after = int(error.message.split("try again in ")[1].split(" ")[0]) * 60
                        except:
                            pass
                logger.warning(f"Reddit API rate limit hit, retrying in {retry_after}s")
                await asyncio.sleep(retry_after)
                attempt += 1
                continue
            raise
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                logger.error(f"Reddit API call failed after {max_retries} attempts: {str(e)}")
                raise
            wait_time = retry_delay * (2 ** (attempt - 1))
            logger.warning(f"Reddit API call failed (attempt {attempt}/{max_retries}), retrying in {wait_time}s: {str(e)}")
            await asyncio.sleep(wait_time)

# --- Rate Limiter ---
class RateLimiter:
    def __init__(self, calls_limit: int, time_window: int):
        self.calls_limit = calls_limit
        self.time_window = time_window
        self.redis_key_prefix = "reddit_rate_limit"
    
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

rate_limiter = RateLimiter(
    calls_limit=CONFIG["reddit"]["rate_limit_calls"],
    time_window=CONFIG["reddit"]["rate_limit_window"]
)

# --- Authentication ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not CONFIG["security"]["enable_auth"]:
        return
    valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")
    if not api_key or api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# --- FastAPI Models ---
class SubredditPostsRequest(BaseModel):
    subreddits: List[str]
    sort: str = Field("hot", description="Sort method: hot, new, top, rising")
    time_filter: str = Field("day", description="Time filter: hour, day, week, month, year, all")
    limit: int = Field(100, ge=1, le=500, description="Maximum number of posts to retrieve")
    
    @validator('subreddits')
    def validate_subreddits(cls, v):
        if not v or not all(isinstance(s, str) and s.strip() for s in v):
            raise ValueError("Subreddits must be non-empty strings")
        return v
    
    @validator('sort')
    def validate_sort(cls, v):
        allowed_values = ['hot', 'new', 'top', 'rising', 'controversial']
        if v not in allowed_values:
            raise ValueError(f"Sort must be one of {allowed_values}")
        return v
    
    @validator('time_filter')
    def validate_time_filter(cls, v):
        allowed_values = ['hour', 'day', 'week', 'month', 'year', 'all']
        if v not in allowed_values:
            raise ValueError(f"Time filter must be one of {allowed_values}")
        return v

class PostCommentsRequest(BaseModel):
    subreddit: str
    post_id: str = Field(..., description="Reddit post ID")
    sort: str = Field("confidence", description="Sort method: confidence, top, new, controversial, old")
    limit: int = Field(100, ge=1, le=500, description="Maximum number of comments to retrieve")
    
    @validator('subreddit')
    def validate_subreddit(cls, v):
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("Subreddit must be a non-empty string")
        return v
    
    @validator('post_id')
    def validate_post_id(cls, v):
        if not v or not isinstance(v, str) or len(v) < 5:
            raise ValueError("Post ID must be a valid Reddit post identifier")
        return v
    
    @validator('sort')
    def validate_sort(cls, v):
        allowed_values = ['confidence', 'top', 'new', 'controversial', 'old', 'qa']
        if v not in allowed_values:
            raise ValueError(f"Sort must be one of {allowed_values}")
        return v

class SearchRedditRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Search query string")
    subreddits: List[str] = Field(..., min_items=1, description="Subreddits to search")
    sort: str = Field("relevance", description="Sort method: relevance, hot, new, top, comments")
    time_filter: str = Field("week", description="Time filter: hour, day, week, month, year, all")
    limit: int = Field(100, ge=1, le=500, description="Maximum number of results to retrieve")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Search query must be at least 2 characters long")
        return v.strip()
    
    @validator('subreddits')
    def validate_subreddits(cls, v):
        if not v or not all(isinstance(s, str) and s.strip() for s in v):
            raise ValueError("Subreddits must be non-empty strings")
        return v
    
    @validator('sort')
    def validate_sort(cls, v):
        allowed_values = ['relevance', 'hot', 'new', 'top', 'comments']
        if v not in allowed_values:
            raise ValueError(f"Sort must be one of {allowed_values}")
        return v
    
    @validator('time_filter')
    def validate_time_filter(cls, v):
        allowed_values = ['hour', 'day', 'week', 'month', 'year', 'all']
        if v not in allowed_values:
            raise ValueError(f"Time filter must be one of {allowed_values}")
        return v

class AnalyzeTickerSentimentRequest(BaseModel):
    tickers: List[str] = Field(..., min_items=1, max_items=10, description="List of ticker symbols to analyze")
    subreddits: List[str] = Field(..., min_items=1, description="Subreddits to analyze")
    time_filter: str = Field("week", description="Time filter: hour, day, week, month, year, all")
    limit_per_ticker: int = Field(50, ge=1, le=100, description="Maximum posts to fetch per ticker")
    
    @validator('tickers')
    def validate_tickers(cls, v):
        if not v or not all(isinstance(t, str) and t.strip() for t in v):
            raise ValueError("Tickers must be non-empty strings")
        if len(v) > 10:
            raise ValueError("Maximum 10 tickers can be analyzed at once")
        return [t.strip().upper() for t in v]
    
    @validator('subreddits')
    def validate_subreddits(cls, v):
        if not v or not all(isinstance(s, str) and s.strip() for s in v):
            raise ValueError("Subreddits must be non-empty strings")
        return v
    
    @validator('time_filter')
    def validate_time_filter(cls, v):
        allowed_values = ['hour', 'day', 'week', 'month', 'year', 'all']
        if v not in allowed_values:
            raise ValueError(f"Time filter must be one of {allowed_values}")
        return v

class EntitySentimentRequest(BaseModel):
    entity_text: str = Field(..., min_length=2, description="Entity to analyze sentiment for")
    subreddits: List[str] = Field(..., min_items=1, description="Subreddits to analyze")
    time_filter: str = Field("week", description="Time filter: hour, day, week, month, year, all")
    limit: int = Field(20, ge=1, le=100, description="Maximum posts to fetch")
    
    @validator('entity_text')
    def validate_entity_text(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Entity text must be at least 2 characters long")
        return v.strip()
    
    @validator('subreddits')
    def validate_subreddits(cls, v):
        if not v or not all(isinstance(s, str) and s.strip() for s in v):
            raise ValueError("Subreddits must be non-empty strings")
        return v
    
    @validator('time_filter')
    def validate_time_filter(cls, v):
        allowed_values = ['hour', 'day', 'week', 'month', 'year', 'all']
        if v not in allowed_values:
            raise ValueError(f"Time filter must be one of {allowed_values}")
        return v

class SentimentTrendRequest(BaseModel):
    ticker: str = Field(..., min_length=1, description="Ticker symbol to analyze trend for")
    days_history: int = Field(30, ge=1, le=90, description="Number of past days to analyze")
    
    @validator('ticker')
    def validate_ticker(cls, v):
        if not v or not v.strip():
            raise ValueError("Ticker cannot be empty")
        return v.strip().upper()

# --- FastAPI Server ---
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = datetime.utcnow()
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        method = request.method
        logger.info(f"Request: {method} {path} from {client_ip}")
        try:
            response = await call_next(request)
            process_time = (datetime.utcnow() - start_time).total_seconds()
            status_code = response.status_code
            logger.info(f"Response: {status_code} for {method} {path} ({process_time:.3f}s)")
            return response
        except Exception as e:
            logger.error(f"Unhandled exception: {str(e)} for {method} {path}")
            raise

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/fetch_") or path.startswith("/search_") or path.startswith("/analyze_"):
            if not await rate_limiter.check_rate_limit(request):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please try again later."}
                )
        return await call_next(request)

app = FastAPI(
    title="Reddit MCP Server for LLM",
    description="Production MCP Server for Reddit Social Sentiment Integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "General", "description": "General server information and status"},
        {"name": "Data", "description": "Endpoints for fetching Reddit posts and comments"},
        {"name": "Analysis", "description": "Endpoints for sentiment and trend analysis"},
        {"name": "Models", "description": "Endpoints for model health checks"}
    ]
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(GZipMiddleware)

@app.on_event("startup")
async def startup_event():
    logger.info("Reddit Processor server starting up")
    await connect_redis()
    await init_reddit_client()
    finbert_tone_model.load_model()
    roberta_sec_model.load_model()
    financial_bert_ner_model.load_model()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Reddit Processor server shutting down")
    if redis_client:
        await redis_client.close()
        logger.info("Closed Redis connection")

@app.get("/server_info", tags=["General"])
async def get_server_info():
    safe_config = {k: {sk: sv for sk, sv in v.items() if sk not in ["client_id", "client_secret", "password"]} for k, v in CONFIG.items()}
    return {
        "name": "reddit_processor",
        "version": "1.0.0",
        "description": "Production MCP Server for Reddit Social Sentiment Integration",
        "tools": [
            "fetch_subreddit_posts", "fetch_post_comments", "search_reddit",
            "analyze_ticker_sentiment", "get_historical_sentiment_trend", "get_entity_sentiment_analysis"
        ],
        "models": ["FinBERT-tone", "RoBERTa-SEC", "FinancialBERT-NER"],
        "config": safe_config
    }

@app.post("/fetch_subreddit_posts", tags=["Data"], dependencies=[Depends(verify_api_key)])
async def api_fetch_subreddit_posts(req: SubredditPostsRequest):
    logger.info(f"Fetching posts from subreddits: {req.subreddits}, sort: {req.sort}")
    cache_key = f"reddit:posts:{','.join(sorted(req.subreddits))}:{req.sort}:{req.time_filter}:{req.limit}"
    if CONFIG["cache"]["use_cache"] and redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Error reading Redis cache: {e}")
    try:
        posts_data = []
        async def fetch_from_subreddit(subreddit_name):
            results = []
            post_count = 0
            subreddit = await reddit_client.subreddit(subreddit_name)
            sort_method = getattr(subreddit, req.sort)
            async for post_item in sort_method(limit=req.limit, time_filter=req.time_filter if hasattr(sort_method, "__call__") else None):
                results.append({
                    "id": post_item.id,
                    "title": post_item.title,
                    "score": post_item.score,
                    "created_utc": post_item.created_utc,
                    "num_comments": post_item.num_comments,
                    "permalink": post_item.permalink,
                    "author": str(post_item.author) if post_item.author else "unknown",
                    "subreddit": subreddit_name,
                    "url": post_item.url,
                    "selftext": post_item.selftext if hasattr(post_item, 'selftext') else ""
                })
                post_count += 1
            logger.debug(f"Retrieved {post_count} posts from {subreddit_name}")
            return results
        for subreddit_name in req.subreddits:
            try:
                subreddit_posts = await reddit_api_with_retry(lambda: fetch_from_subreddit(subreddit_name))
                posts_data.extend(subreddit_posts)
            except Exception as e:
                logger.error(f"Failed to fetch posts from subreddit {subreddit_name}: {str(e)}")
        response = {"success": True, "posts": posts_data}
        if CONFIG["cache"]["use_cache"] and redis_client and posts_data:
            try:
                await redis_client.set(cache_key, json.dumps(response), ex=3600)
                logger.debug(f"Cached response for {cache_key}")
            except Exception as e:
                logger.warning(f"Error writing to Redis cache: {e}")
        logger.info(f"Retrieved {len(posts_data)} posts from {len(req.subreddits)} subreddits")
        return response
    except Exception as e:
        logger.error(f"Error fetching subreddit posts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fetch_post_comments", tags=["Data"], dependencies=[Depends(verify_api_key)])
async def api_fetch_post_comments(req: PostCommentsRequest):
    logger.info(f"Fetching comments for post {req.post_id} in subreddit {req.subreddit}")
    cache_key = f"reddit:comments:{req.subreddit}:{req.post_id}:{req.sort}:{req.limit}"
    if CONFIG["cache"]["use_cache"] and redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Error reading Redis cache: {e}")
    try:
        async def fetch_comments():
            comments_data = []
            submission = await reddit_client.submission(id=req.post_id)
            await submission.load()
            logger.debug(f"Replacing more comments for post {req.post_id}")
            await submission.comments.replace_more(limit=None)
            comment_limit = req.limit if req.limit > 0 else None
            for comment_item in submission.comments.list()[:comment_limit]:
                comments_data.append({
                    "id": comment_item.id,
                    "body": comment_item.body,
                    "score": comment_item.score,
                    "created_utc": comment_item.created_utc,
                    "author": str(comment_item.author) if comment_item.author else "unknown",
                    "parent_id": comment_item.parent_id
                })
            return comments_data
        comments_data = await reddit_api_with_retry(fetch_comments)
        response = {"success": True, "comments": comments_data}
        if CONFIG["cache"]["use_cache"] and redis_client and comments_data:
            try:
                await redis_client.set(cache_key, json.dumps(response), ex=3600)
                logger.debug(f"Cached response for {cache_key}")
            except Exception as e:
                logger.warning(f"Error writing to Redis cache: {e}")
        logger.info(f"Retrieved {len(comments_data)} comments for post {req.post_id}")
        return response
    except Exception as e:
        logger.error(f"Error fetching post comments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_reddit", tags=["Data"], dependencies=[Depends(verify_api_key)])
async def api_search_reddit(req: SearchRedditRequest):
    logger.info(f"Searching Reddit for query: '{req.query}' in subreddits: {req.subreddits}")
    cache_key = f"reddit:search:{req.query}:{','.join(sorted(req.subreddits))}:{req.sort}:{req.time_filter}:{req.limit}"
    if CONFIG["cache"]["use_cache"] and redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Error reading Redis cache: {e}")
    try:
        async def perform_search():
            search_results = []
            post_count = 0
            target_search_location = "+".join(req.subreddits)
            search_location_obj = await reddit_client.subreddit(target_search_location)
            async for post_item in search_location_obj.search(req.query, sort=req.sort, time_filter=req.time_filter, limit=req.limit):
                search_results.append({
                    "id": post_item.id,
                    "title": post_item.title,
                    "score": post_item.score,
                    "created_utc": post_item.created_utc,
                    "num_comments": post_item.num_comments,
                    "permalink": post_item.permalink,
                    "author": str(post_item.author) if post_item.author else "unknown",
                    "subreddit": str(post_item.subreddit),
                    "url": post_item.url,
                    "selftext": post_item.selftext if hasattr(post_item, 'selftext') else ""
                })
                post_count += 1
            logger.debug(f"Found {post_count} results in {target_search_location}")
            return search_results
        search_results_posts = await reddit_api_with_retry(perform_search)
        response = {"success": True, "posts": search_results_posts}
        if CONFIG["cache"]["use_cache"] and redis_client and search_results_posts:
            try:
                await redis_client.set(cache_key, json.dumps(response), ex=3600)
                logger.debug(f"Cached response for {cache_key}")
            except Exception as e:
                logger.warning(f"Error writing to Redis cache: {e}")
        logger.info(f"Found {len(search_results_posts)} results for query: '{req.query}'")
        return response
    except Exception as e:
        logger.error(f"Error searching Reddit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_ticker_sentiment", tags=["Analysis"], dependencies=[Depends(verify_api_key)])
async def api_analyze_ticker_sentiment(req: AnalyzeTickerSentimentRequest):
    logger.info(f"Analyzing sentiment for tickers: {req.tickers} in subreddits: {req.subreddits}")
    cache_key = f"reddit:ticker_sentiment:{','.join(sorted(req.tickers))}:{','.join(sorted(req.subreddits))}:{req.time_filter}:{req.limit_per_ticker}"
    if CONFIG["cache"]["use_cache"] and redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Error reading Redis cache: {e}")
    try:
        analysis_results = {
            "success": True,
            "tickers": {},
            "summary": {}
        }
        all_posts_content = []
        for ticker in req.tickers:
            logger.debug(f"Analyzing sentiment for ticker: {ticker}")
            ticker_posts = []
            search_query = f"\"{ticker}\" OR \"${ticker}\""
            async def search_for_ticker(subreddit_name):
                content_list = []
                subreddit = await reddit_client.subreddit(subreddit_name)
                async for post_item in subreddit.search(search_query, sort="relevance", time_filter=req.time_filter, limit=req.limit_per_ticker):
                    content = post_item.title
                    if hasattr(post_item, 'selftext') and post_item.selftext:
                        content += ". " + post_item.selftext
                    content_list.append({"content": content, "score": post_item.score})
                return content_list
            for subreddit_name in req.subreddits:
                try:
                    subreddit_content = await reddit_api_with_retry(lambda: search_for_ticker(subreddit_name))
                    ticker_posts.extend(subreddit_content)
                except Exception as e:
                    logger.warning(f"Could not search in subreddit {subreddit_name} for {ticker}: {str(e)}")
            analysis_results["tickers"][ticker] = {
                "mention_count": len(ticker_posts),
                "sentiment": {"dominant": "neutral", "scores": {"positive": 0.0, "negative": 0.0, "neutral": 0.5}, "positive_mentions": 0, "negative_mentions": 0, "neutral_mentions": 0},
                "related_entities": []
            }
            logger.debug(f"Found {len(ticker_posts)} mentions of {ticker}")
            if not ticker_posts:
                continue
            all_posts_content.extend([p["content"] for p in ticker_posts])
            sentiments = []
            entities = []
            total_score_weight = sum(p["score"] for p in ticker_posts if p["score"] > 0) or 1
            for post in ticker_posts:
                text_content = post["content"]
                weight = post["score"] / total_score_weight if post["score"] > 0 else 1 / len(ticker_posts)
                try:
                    sentiment_result = finbert_tone_model.predict(text_content)
                    if sentiment_result:
                        sentiments.append({"result": sentiment_result[0], "weight": weight})
                except Exception as e:
                    logger.error(f"Error during FinBERT-tone prediction for '{text_content[:50]}...': {e}")
                    sentiments.append({"result": {"label": "NEUTRAL", "score": 0.5}, "weight": weight})
                try:
                    extracted_entities = financial_bert_ner_model.predict(text_content)
                    entities.extend(extracted_entities)
                except Exception as e:
                    logger.error(f"Error during FinancialBERT-NER prediction for '{text_content[:50]}...': {e}")
            if sentiments:
                avg_sentiment = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                label_counts = {"positive": 0, "negative": 0, "neutral": 0}
                for s in sentiments:
                    label = s["result"]["label"].upper()
                    score = s["result"]["score"] * s["weight"]
                    if label == "POSITIVE":
                        avg_sentiment["positive"] += score
                        label_counts["positive"] += 1
                    elif label == "NEGATIVE":
                        avg_sentiment["negative"] += score
                        label_counts["negative"] += 1
                    else:
                        avg_sentiment["neutral"] += score
                        label_counts["neutral"] += 1
                dominant = "neutral"
                if avg_sentiment["positive"] > avg_sentiment["negative"] and avg_sentiment["positive"] > avg_sentiment["neutral"]:
                    dominant = "positive"
                elif avg_sentiment["negative"] > avg_sentiment["positive"] and avg_sentiment["negative"] > avg_sentiment["neutral"]:
                    dominant = "negative"
                analysis_results["tickers"][ticker]["sentiment"] = {
                    "dominant": dominant,
                    "scores": avg_sentiment,
                    "positive_mentions": label_counts["positive"],
                    "negative_mentions": label_counts["negative"],
                    "neutral_mentions": label_counts["neutral"]
                }
            if entities:
                unique_entities = []
                seen = set()
                for entity in entities:
                    if entity["text"] not in seen:
                        unique_entities.append(entity)
                        seen.add(entity["text"])
                analysis_results["tickers"][ticker]["related_entities"] = unique_entities[:15]
            if redis_client and analysis_results["tickers"][ticker]["mention_count"] > 0:
                current_ts = datetime.utcnow()
                redis_key = f"reddit_sentiment:{ticker}:{current_ts.strftime('%Y%m%d')}"
                daily_data = {
                    "date": current_ts.strftime('%Y-%m-%d'),
                    "sentiment": analysis_results["tickers"][ticker]["sentiment"],
                    "mention_count": analysis_results["tickers"][ticker]["mention_count"]
                }
                try:
                    await redis_client.zadd(f"reddit_sentiment:{ticker}", {json.dumps(daily_data): current_ts.timestamp()})
                    await redis_client.expire(f"reddit_sentiment:{ticker}", 60 * 60 * 24 * 90)
                    logger.debug(f"Stored sentiment for {ticker} in Redis: {redis_key}")
                except Exception as e:
                    logger.error(f"Error storing sentiment for {ticker} in Redis: {e}")
        total_mentions = sum(d["mention_count"] for d in analysis_results["tickers"].values())
        analysis_results["summary"] = {
            "ticker_count": len(req.tickers),
            "total_mentions": total_mentions,
            "timestamp": datetime.utcnow().isoformat()
        }
        if total_mentions > 0:
            most_mentioned = max(analysis_results["tickers"].items(), key=lambda x: x[1]["mention_count"])[0]
            analysis_results["summary"]["most_mentioned_ticker"] = most_mentioned
            logger.info(f"Most mentioned ticker: {most_mentioned}")
        if CONFIG["cache"]["use_cache"] and redis_client:
            try:
                await redis_client.set(cache_key, json.dumps(analysis_results), ex=3600)
                logger.debug(f"Cached response for {cache_key}")
            except Exception as e:
                logger.warning(f"Error writing to Redis cache: {e}")
        return analysis_results
    except Exception as e:
        logger.error(f"Error analyzing ticker sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_historical_sentiment_trend", tags=["Analysis"], dependencies=[Depends(verify_api_key)])
async def api_get_historical_sentiment_trend(req: SentimentTrendRequest):
    logger.info(f"Fetching historical sentiment trend for ticker: {req.ticker} over {req.days_history} days")
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis client not available")
    cache_key = f"reddit:sentiment_trend:{req.ticker}:{req.days_history}"
    if CONFIG["cache"]["use_cache"]:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Error reading Redis cache: {e}")
    try:
        trends = []
        now = datetime.utcnow()
        for i in range(req.days_history - 1, -1, -1):
            day = now - timedelta(days=i)
            redis_key = f"reddit_sentiment:{req.ticker}"
            entries = await redis_client.zrangebyscore(redis_key, day.timestamp(), (day + timedelta(days=1)).timestamp())
            if entries:
                daily_sentiments = []
                daily_mentions = 0
                for entry in entries:
                    data = json.loads(entry)
                    daily_sentiments.append(data["sentiment"]["scores"])
                    daily_mentions += data["sentiment"].get("positive_mentions", 0) + data["sentiment"].get("negative_mentions", 0) + data["sentiment"].get("neutral_mentions", 0)
                if daily_sentiments:
                    avg_daily_pos = sum(s["positive"] for s in daily_sentiments) / len(daily_sentiments)
                    avg_daily_neg = sum(s["negative"] for s in daily_sentiments) / len(daily_sentiments)
                    avg_daily_neu = sum(s["neutral"] for s in daily_sentiments) / len(daily_sentiments)
                    trends.append({
                        "date": day.strftime('%Y-%m-%d'),
                        "avg_positive_score": avg_daily_pos,
                        "avg_negative_score": avg_daily_neg,
                        "avg_neutral_score": avg_daily_neu,
                        "total_mentions": daily_mentions
                    })
        response = {"success": True, "ticker": req.ticker, "trend_data": trends}
        if not trends:
            response["message"] = "No historical sentiment data found for the given period"
        if CONFIG["cache"]["use_cache"] and redis_client:
            try:
                await redis_client.set(cache_key, json.dumps(response), ex=3600)
                logger.debug(f"Cached response for {cache_key}")
            except Exception as e:
                logger.warning(f"Error writing to Redis cache: {e}")
        return response
    except Exception as e:
        logger.error(f"Error fetching historical sentiment trend for {req.ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_entity_sentiment_analysis", tags=["Analysis"], dependencies=[Depends(verify_api_key)])
async def api_get_entity_sentiment_analysis(req: EntitySentimentRequest):
    logger.info(f"Fetching sentiment for entity: '{req.entity_text}' in subreddits: {req.subreddits}")
    cache_key = f"reddit:entity_sentiment:{req.entity_text}:{','.join(sorted(req.subreddits))}:{req.time_filter}:{req.limit}"
    if CONFIG["cache"]["use_cache"] and redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Error reading Redis cache: {e}")
    try:
        posts_content = []
        for subreddit_name in req.subreddits:
            subreddit = await reddit_client.subreddit(subreddit_name)
            try:
                async for post_item in subreddit.search(f"\"{req.entity_text}\"", sort="relevance", time_filter=req.time_filter, limit=req.limit):
                    content = post_item.title
                    if hasattr(post_item, 'selftext') and post_item.selftext:
                        content += ". " + post_item.selftext
                    posts_content.append({"content": content, "score": post_item.score})
            except Exception as e:
                logger.warning(f"Could not search for entity '{req.entity_text}' in {subreddit_name}: {str(e)}")
        if not posts_content:
            response = {"success": True, "entity": req.entity_text, "analysis": {"mention_count": 0, "sentiment": {"dominant": "neutral", "scores": {"positive": 0.0, "negative": 0.0, "neutral": 0.5}, "positive_mentions": 0, "negative_mentions": 0, "neutral_mentions": 0}, "relevant_entities": [], "related_entities": []}}
            return response
        sentiments = []
        entities = []
        total_score_weight = sum(p["score"] for p in posts_content if p["score"] > 0) or 1
        for post in posts_content:
            text_content = post["content"]
            weight = post["score"] / total_score_weight if post["score"] > 0 else 1 / len(posts_content)
            try:
                sentiment_result = finbert_tone_model.predict(text_content)
                if sentiment_result:
                    sentiments.append({"result": sentiment_result[0], "weight": weight})
            except Exception as e:
                logger.error(f"Error during FinBERT-tone prediction for '{text_content[:50]}...': {e}")
                sentiments.append({"result": {"label": "NEUTRAL", "score": 0.5}, "weight": weight})
            try:
                extracted_entities = financial_bert_ner_model.predict(text_content)
                entities.extend(extracted_entities)
            except Exception as e:
                logger.error(f"Error during FinancialBERT-NER prediction for '{text_content[:50]}...': {e}")
        final_sentiment = {"dominant": "neutral", "scores": {"positive": 0.0, "negative": 0.0, "neutral": 0.5}, "positive_mentions": 0, "negative_mentions": 0, "neutral_mentions": 0, "total_mentions": len(posts_content)}
        if sentiments:
            avg_sentiment = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            label_counts = {"positive": 0, "negative": 0, "neutral": 0}
            for s in sentiments:
                label = s["result"]["label"].upper()
                score = s["result"]["score"] * s["weight"]
                if label == "POSITIVE":
                    avg_sentiment["positive"] += score
                    label_counts["positive"] += 1
                elif label == "NEGATIVE":
                    avg_sentiment["negative"] += score
                    label_counts["negative"] += 1
                else:
                    avg_sentiment["neutral"] += score
                    label_counts["neutral"] += 1
            dominant = "neutral"
            if avg_sentiment["positive"] > avg_sentiment["negative"] and avg_sentiment["positive"] > avg_sentiment["neutral"]:
                dominant = "positive"
            elif avg_sentiment["negative"] > avg_sentiment["positive"] and avg_sentiment["negative"] > avg_sentiment["neutral"]:
                dominant = "negative"
            final_sentiment = {
                "dominant": dominant,
                "scores": avg_sentiment,
                "positive_mentions": label_counts["positive"],
                "negative_mentions": label_counts["negative"],
                "neutral_mentions": label_counts["neutral"],
                "total_mentions": len(posts_content)
            }
        relevant_entities = []
        entity_freq = {}
        for entity in entities:
            entity_text = entity["text"].lower()
            if req.entity_text.lower() in entity_text and entity["score"] >= 0.75:
                relevant_entities.append(entity)
            if entity_text in entity_freq:
                entity_freq[entity_text] += 1
            else:
                entity_freq[entity_text] = 1
        related_entities = [
            {"entity": entity, "frequency": freq}
            for entity, freq in sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)
            if entity != req.entity_text.lower() and freq > 1
        ][:10]
        response = {
            "success": True,
            "entity": req.entity_text,
            "analysis": {
                "mention_count": len(posts_content),
                "sentiment": final_sentiment,
                "relevant_entities": relevant_entities[:10],
                "related_entities": related_entities
            }
        }
        if redis_client and len(posts_content) > 0:
            current_ts = datetime.utcnow()
            redis_key = f"reddit_entity:{req.entity_text}:{current_ts.strftime('%Y%m%d')}"
            entity_data = {
                "date": current_ts.strftime('%Y-%m-%d'),
                "sentiment": final_sentiment,
                "mention_count": len(posts_content)
            }
            try:
                await redis_client.zadd(f"reddit_entity:{req.entity_text}", {json.dumps(entity_data): current_ts.timestamp()})
                await redis_client.expire(f"reddit_entity:{req.entity_text}", 60 * 60 * 24 * 60)
                logger.debug(f"Stored entity sentiment for '{req.entity_text}' in Redis")
            except Exception as e:
                logger.error(f"Error storing entity sentiment for '{req.entity_text}' in Redis: {e}")
        if CONFIG["cache"]["use_cache"] and redis_client:
            try:
                await redis_client.set(cache_key, json.dumps(response), ex=3600)
                logger.debug(f"Cached response for {cache_key}")
            except Exception as e:
                logger.warning(f"Error writing to Redis cache: {e}")
        return response
    except Exception as e:
        logger.error(f"Error analyzing entity sentiment for '{req.entity_text}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/resource/{resource_uri:path}", tags=["General"], dependencies=[Depends(verify_api_key)])
async def get_resource(resource_uri: str):
    logger.info(f"Resource request for URI: {resource_uri}")
    try:
        params = {}
        if "?" in resource_uri:
            base_uri, query_params = resource_uri.split("?", 1)
            resource_uri = base_uri
            for param in query_params.split("&"):
                if "=" in param:
                    key, value = param.split("=", 1)
                    params[key] = value
        if resource_uri == "sentiment/trending":
            days = int(params.get("days", "7"))
            limit = int(params.get("limit", "5"))
            subreddits = params.get("subreddits", ",".join(["wallstreetbets", "stocks", "investing"])).split(",")
            if not redis_client:
                raise HTTPException(status_code=503, detail="Redis unavailable for trend data")
            trending_data = await get_trending_sentiment(subreddits, days, limit)
            return {
                "success": True,
                "trending_sentiment": trending_data,
                "parameters": {
                    "days": days,
                    "limit": limit,
                    "subreddits": subreddits
                }
            }
        elif resource_uri == "tickers/trending":
            days = int(params.get("days", "1"))
            limit = int(params.get("limit", "10"))
            if not redis_client:
                raise HTTPException(status_code=503, detail="Redis unavailable for trend data")
            trending_tickers = await get_trending_tickers(days, limit)
            return {
                "success": True,
                "trending_tickers": trending_tickers,
                "parameters": {
                    "days": days,
                    "limit": limit
                }
            }
        elif resource_uri == "topics/trending":
            days = int(params.get("days", "1"))
            limit = int(params.get("limit", "10"))
            if not redis_client:
                raise HTTPException(status_code=503, detail="Redis unavailable for trend data")
            trending_topics = await get_trending_topics(days, limit)
            return {
                "success": True,
                "trending_topics": trending_topics,
                "parameters": {
                    "days": days,
                    "limit": limit
                }
            }
        elif resource_uri == "status":
            redis_status = "connected" if redis_client and await redis_client.ping() else "disconnected"
            models_status = {
                "FinBERT-tone": finbert_tone_model.model is not None,
                "RoBERTa-SEC": roberta_sec_model.model is not None,
                "FinancialBERT-NER": financial_bert_ner_model.model is not None
            }
            return {
                "status": "operational",
                "redis_status": redis_status,
                "models_status": models_status,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            logger.warning(f"Unknown resource URI requested: {resource_uri}")
            raise HTTPException(status_code=404, detail=f"Unknown resource URI: {resource_uri}")
    except Exception as e:
        logger.error(f"Error in get_resource: {str(e)}")
        if resource_uri == "status":
            return {"status": "operational", "redis_status": "error during ping", "timestamp": datetime.utcnow().isoformat()}
        raise HTTPException(status_code=500, detail=str(e))

async def get_trending_sentiment(subreddits: List[str], days: int, limit: int) -> List[Dict[str, Any]]:
    cache_key = f"reddit:trending_sentiment:{','.join(sorted(subreddits))}:{days}:{limit}"
    if CONFIG["cache"]["use_cache"] and redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Error reading Redis cache: {e}")
    try:
        ticker_data = []
        now = datetime.utcnow()
        for ticker in await redis_client.zrange("reddit_tickers", 0, -1):
            total_mentions = 0
            avg_sentiments = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            sentiment_counts = 0
            for i in range(days):
                day = now - timedelta(days=i)
                entries = await redis_client.zrangebyscore(f"reddit_sentiment:{ticker}", day.timestamp(), (day + timedelta(days=1)).timestamp())
                for entry in entries:
                    data = json.loads(entry)
                    scores = data.get("sentiment", {}).get("scores", {})
                    for sentiment_type, score in scores.items():
                        avg_sentiments[sentiment_type] += score
                    total_mentions += data.get("sentiment", {}).get("positive_mentions", 0) + data.get("sentiment", {}).get("negative_mentions", 0) + data.get("sentiment", {}).get("neutral_mentions", 0)
                    sentiment_counts += 1
            if sentiment_counts > 0:
                for sentiment_type in avg_sentiments:
                    avg_sentiments[sentiment_type] /= sentiment_counts
                dominant = "neutral"
                if avg_sentiments["positive"] > avg_sentiments["negative"] and avg_sentiments["positive"] > avg_sentiments["neutral"]:
                    dominant = "positive"
                elif avg_sentiments["negative"] > avg_sentiments["positive"] and avg_sentiments["negative"] > avg_sentiments["neutral"]:
                    dominant = "negative"
                ticker_data.append({
                    "ticker": ticker,
                    "total_mentions": total_mentions,
                    "avg_sentiment": avg_sentiments,
                    "dominant_sentiment": dominant
                })
        trending = sorted(ticker_data, key=lambda x: x["total_mentions"], reverse=True)[:limit]
        if CONFIG["cache"]["use_cache"] and redis_client:
            try:
                await redis_client.set(cache_key, json.dumps(trending), ex=3600)
                logger.debug(f"Cached response for {cache_key}")
            except Exception as e:
                logger.warning(f"Error writing to Redis cache: {e}")
        return trending
    except Exception as e:
        logger.error(f"Error getting trending sentiment: {str(e)}")
        return []

async def get_trending_tickers(days: int, limit: int) -> List[Dict[str, Any]]:
    cache_key = f"reddit:trending_tickers:{days}:{limit}"
    if CONFIG["cache"]["use_cache"] and redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Error reading Redis cache: {e}")
    try:
        tickers = await redis_client.zrange("reddit_tickers", 0, -1, desc=True, withscores=True)
        trending_tickers = [
            {"ticker": ticker, "mentions": int(score)}
            for ticker, score in tickers
        ][:limit]
        if CONFIG["cache"]["use_cache"] and redis_client:
            try:
                await redis_client.set(cache_key, json.dumps(trending_tickers), ex=3600)
                logger.debug(f"Cached response for {cache_key}")
            except Exception as e:
                logger.warning(f"Error writing to Redis cache: {e}")
        return trending_tickers
    except Exception as e:
        logger.error(f"Error getting trending tickers: {str(e)}")
        return []

async def get_trending_topics(days: int, limit: int) -> List[Dict[str, Any]]:
    cache_key = f"reddit:trending_topics:{days}:{limit}"
    if CONFIG["cache"]["use_cache"] and redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Error reading Redis cache: {e}")
    try:
        entities = await redis_client.zrange("reddit_entities", 0, -1, desc=True, withscores=True)
        trending_topics = [
            {"entity": entity, "mentions": int(score)}
            for entity, score in entities
        ][:limit]
        if CONFIG["cache"]["use_cache"] and redis_client:
            try:
                await redis_client.set(cache_key, json.dumps(trending_topics), ex=3600)
                logger.debug(f"Cached response for {cache_key}")
            except Exception as e:
                logger.warning(f"Error writing to Redis cache: {e}")
        return trending_topics
    except Exception as e:
        logger.error(f"Error getting trending topics: {str(e)}")
        return []

class FinBERTtoneModel:
    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"FinBERT-tone will run on {self.device}")

    def load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info("FinBERT-tone model loaded")
        except Exception as e:
            logger.error(f"Error loading FinBERT-tone model ({self.model_name}): {str(e)}")
            raise

    def predict(self, text: str) -> List[Dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            self.load_model()
        try:
            if not text or len(text.strip()) == 0:
                logger.warning("Empty text provided to FinBERT-tone model")
                return [{"label": "NEUTRAL", "score": 0.5}]
            text = text[:1024]  # Truncate to avoid token limit
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            labels = self.model.config.id2label
            return [{"label": labels[i], "score": float(score)} for i, score in enumerate(scores)]
        except Exception as e:
            logger.error(f"FinBERT-tone prediction error: {str(e)}")
            return [{"label": "NEUTRAL", "score": 0.5}]

class RoBERTaSECModel:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"RoBERTa-SEC will run on {self.device}")

    def load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info("RoBERTa-SEC model loaded")
        except Exception as e:
            logger.error(f"Error loading RoBERTa-SEC model ({self.model_name}): {str(e)}")
            raise

    def predict(self, text: str) -> List[Dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            self.load_model()
        try:
            if not text or len(text.strip()) == 0:
                logger.warning("Empty text provided to RoBERTa-SEC model")
                return [{"label": "neutral", "score": 0.5}]
            text = text[:1024]
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            labels = self.model.config.id2label
            return [{"label": labels[i], "score": float(score)} for i, score in enumerate(scores)]
        except Exception as e:
            logger.error(f"RoBERTa-SEC prediction error: {str(e)}")
            return [{"label": "neutral", "score": 0.5}]

class FinancialBERTNERModel:
    def __init__(self):
        self.model_name = "dslim/bert-base-NER"
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"FinancialBERT-NER will run on {self.device} using {self.model_name}")

    def load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info("FinancialBERT-NER model loaded")
        except Exception as e:
            logger.error(f"Error loading FinancialBERT-NER model ({self.model_name}): {str(e)}")
            raise

    def predict(self, text: str) -> List[Dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            self.load_model()
        try:
            if not text or len(text.strip()) == 0:
                logger.warning("Empty text provided to FinancialBERT-NER model")
                return []
            text = text[:1024]
            ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                aggregation_strategy="simple"
            )
            ner_results = ner_pipeline(text)
            return [
                {
                    "type": entity["entity_group"],
                    "text": entity["word"],
                    "score": float(entity["score"]),
                    "start": int(entity["start"]),
                    "end": int(entity["end"])
                }
                for entity in ner_results if entity["score"] >= 0.75
            ]
        except Exception as e:
            logger.error(f"FinancialBERT-NER prediction error: {str(e)}")
            return []

finbert_tone_model = FinBERTtoneModel()
roberta_sec_model = RoBERTaSECModel()
financial_bert_ner_model = FinancialBERTNERModel()

def check_models_loaded() -> Dict[str, bool]:
    return {
        "finbert_tone": finbert_tone_model.model is not None,
        "roberta_sec": roberta_sec_model.model is not None,
        "financial_bert_ner": financial_bert_ner_model.model is not None
    }

@app.get("/models/status", tags=["Models"])
async def api_models_status():
    model_status = check_models_loaded()
    all_loaded = all(model_status.values())
    if not all_loaded:
        if not model_status["finbert_tone"]:
            try:
                finbert_tone_model.load_model()
                model_status["finbert_tone"] = finbert_tone_model.model is not None
            except Exception as e:
                logger.error(f"Failed to reload FinBERT-tone model: {str(e)}")
        if not model_status["roberta_sec"]:
            try:
                roberta_sec_model.load_model()
                model_status["roberta_sec"] = roberta_sec_model.model is not None
            except Exception as e:
                logger.error(f"Failed to reload RoBERTa-SEC model: {str(e)}")
        if not model_status["financial_bert_ner"]:
            try:
                financial_bert_ner_model.load_model()
                model_status["financial_bert_ner"] = financial_bert_ner_model.model is not None
            except Exception as e:
                logger.error(f"Failed to reload FinancialBERT-NER model: {str(e)}")
    return {
        "status": "operational" if all(model_status.values()) else "degraded",
        "models": model_status,
        "device": {
            "finbert_tone": finbert_tone_model.device,
            "roberta_sec": roberta_sec_model.device,
            "financial_bert_ner": financial_bert_ner_model.device
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    