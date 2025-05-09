"""
Reddit FastAPI Server.
Provides social data from Reddit API.
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from dotenv import load_dotenv
import asyncpraw
import asyncio
import time
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
import logging
from fastapi.security import APIKeyHeader

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("reddit_api")

# Load environment variables
load_dotenv()

# --- Configuration ---
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "Trading Bot v1.0")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME", "")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD", "")
RATE_LIMIT_CALLS = int(os.getenv("REDDIT_RATE_LIMIT", "30"))
RATE_LIMIT_WINDOW = int(os.getenv("REDDIT_RATE_WINDOW", "60"))
MAX_RETRIES = int(os.getenv("REDDIT_MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("REDDIT_RETRY_DELAY", "1.5"))
ENABLE_AUTH = os.getenv("REDDIT_ENABLE_AUTH", "true").lower() == "true"

# --- Reddit API Client ---
reddit_client = None

async def init_reddit_client():
    """Initialize the Reddit API client."""
    global reddit_client
    logger.debug("Creating Reddit API client")
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        logger.error("Reddit API credentials not configured")
        raise ValueError("Reddit API credentials not configured")
    reddit_client = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        username=REDDIT_USERNAME or None,
        password=REDDIT_PASSWORD or None,
        requestor_kwargs={"timeout": 30}
    )

async def reddit_api_with_retry(api_call):
    """Execute a Reddit API call with retry logic."""
    attempt = 0
    while attempt < MAX_RETRIES:
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
            if attempt >= MAX_RETRIES:
                logger.error(f"Reddit API call failed after {MAX_RETRIES} attempts: {str(e)}")
                raise
            wait_time = RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning(f"Reddit API call failed (attempt {attempt}/{MAX_RETRIES}), retrying in {wait_time}s: {str(e)}")
            await asyncio.sleep(wait_time)

# --- API Models ---
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

# --- Rate Limiter ---
class RateLimiter:
    def __init__(self, calls_limit: int, time_window: int):
        self.calls_limit = calls_limit
        self.time_window = time_window
        self.timestamps = []
    
    async def check_rate_limit(self, request: Request):
        client_ip = request.client.host
        now = time.time()
        
        # Remove timestamps older than the time window
        self.timestamps = [ts for ts in self.timestamps if now - ts < self.time_window]
        
        # Check if limit exceeded
        if len(self.timestamps) >= self.calls_limit:
            logger.warning(f"Rate limit exceeded for IP {client_ip}: {len(self.timestamps)} requests in last {self.time_window}s")
            return False
        
        # Add current timestamp
        self.timestamps.append(now)
        return True

rate_limiter = RateLimiter(
    calls_limit=RATE_LIMIT_CALLS,
    time_window=RATE_LIMIT_WINDOW
)

# --- Authentication ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not ENABLE_AUTH:
        return
    valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")
    if not api_key or api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# --- Rate Limiting Middleware ---
class RateLimitMiddleware:
    async def __call__(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/reddit/"):
            if not await rate_limiter.check_rate_limit(request):
                return HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
        return await call_next(request)

# --- FastAPI Server ---
app = FastAPI(
    title="Reddit API",
    description="Standard API for Reddit social data",
    version="1.0.0"
)

app.add_middleware(GZipMiddleware)

@app.on_event("startup")
async def startup_event():
    logger.info("Reddit API server starting up")
    await init_reddit_client()

@app.get("/api/info")
async def get_api_info():
    """Get information about the API."""
    return {
        "name": "Reddit API",
        "version": "1.0.0",
        "description": "Standard API for Reddit social data",
        "endpoints": [
            "/api/reddit/posts",
            "/api/reddit/comments",
            "/api/reddit/search"
        ]
    }

@app.post("/api/reddit/posts", dependencies=[Depends(verify_api_key)])
async def fetch_subreddit_posts(req: SubredditPostsRequest):
    """Fetch posts from specified subreddits."""
    logger.info(f"Fetching posts from subreddits: {req.subreddits}, sort: {req.sort}")
    
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
                
        logger.info(f"Retrieved {len(posts_data)} posts from {len(req.subreddits)} subreddits")
        return {"success": True, "posts": posts_data}
        
    except Exception as e:
        logger.error(f"Error fetching subreddit posts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reddit/comments", dependencies=[Depends(verify_api_key)])
async def fetch_post_comments(req: PostCommentsRequest):
    """Fetch comments for a specific Reddit post."""
    logger.info(f"Fetching comments for post {req.post_id} in subreddit {req.subreddit}")
    
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
        logger.info(f"Retrieved {len(comments_data)} comments for post {req.post_id}")
        return {"success": True, "comments": comments_data}
        
    except Exception as e:
        logger.error(f"Error fetching post comments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reddit/search", dependencies=[Depends(verify_api_key)])
async def search_reddit(req: SearchRedditRequest):
    """Search Reddit for posts matching a query."""
    logger.info(f"Searching Reddit for query: '{req.query}' in subreddits: {req.subreddits}")
    
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
        logger.info(f"Found {len(search_results_posts)} results for query: '{req.query}'")
        return {"success": True, "posts": search_results_posts}
        
    except Exception as e:
        logger.error(f"Error searching Reddit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Check the health status of the API."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "api_key_configured": bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)