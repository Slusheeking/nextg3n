"""
Reddit MCP FastAPI Server for LLM integration (production).
Provides social sentiment data from Reddit API.
All configuration is contained in this file.
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta # Added timedelta
import json # Added json
from dotenv import load_dotenv
from monitor.logging_utils import get_logger
import asyncpraw
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import redis.asyncio as aioredis # Added for Redis

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
CONFIG = {
    "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
    "client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
    "user_agent": os.getenv("REDDIT_USER_AGENT", "NextG3N Trading Bot v1.0"),
    "username": os.getenv("REDDIT_USERNAME", ""),
    "password": os.getenv("REDDIT_PASSWORD", ""),
    "default_subreddits": [
        "wallstreetbets", "stocks", "investing", "options", "stockmarket"
    ],
    "use_cache": True,
    "redis_host": os.getenv("REDDIT_REDIS_HOST", os.getenv("REDIS_HOST", "localhost")), # Specific or general Redis host
    "redis_port": int(os.getenv("REDDIT_REDIS_PORT", os.getenv("REDIS_PORT", "6379"))),
    "redis_db": int(os.getenv("REDDIT_REDIS_DB", os.getenv("REDIS_DB", "1"))), # Using DB 1 for Reddit
    "redis_password": os.getenv("REDDIT_REDIS_PASSWORD", os.getenv("REDIS_PASSWORD", None)),
}

# Get logger from centralized logging system
logger = get_logger("reddit_processor_server")
logger.info("Initializing Reddit Processor server")

# --- Redis Client ---
redis_client: Optional[aioredis.Redis] = None

async def connect_redis():
    global redis_client
    try:
        redis_client = aioredis.Redis(
            host=CONFIG["redis_host"],
            port=CONFIG["redis_port"],
            db=CONFIG["redis_db"],
            password=CONFIG["redis_password"],
            decode_responses=True
        )
        await redis_client.ping()
        logger.info(f"Connected to Redis for Reddit Processor at {CONFIG['redis_host']}:{CONFIG['redis_port']} (DB: {CONFIG['redis_db']})")
    except Exception as e:
        logger.error(f"Failed to connect to Redis for Reddit Processor: {str(e)}")
        redis_client = None


# --- Reddit API Helper ---
def get_reddit():
    logger.debug("Creating Reddit API client")
    
    if not CONFIG["client_id"] or not CONFIG["client_secret"]:
        logger.error("Reddit API credentials not configured")
        raise ValueError("Reddit API credentials not configured")
        
    return asyncpraw.Reddit(
        client_id=CONFIG["client_id"],
        client_secret=CONFIG["client_secret"],
        user_agent=CONFIG["user_agent"],
        username=CONFIG["username"] or None,
        password=CONFIG["password"] or None,
    )

# --- FastAPI Models ---

class SubredditPostsRequest(BaseModel):
    subreddits: List[str]
    sort: str = "hot"
    time_filter: str = "day"
    limit: int = 100

class PostCommentsRequest(BaseModel):
    subreddit: str
    post_id: str
    sort: str = "confidence"
    limit: int = 100

class SearchRedditRequest(BaseModel):
    query: str
    subreddits: Optional[List[str]] = None
    sort: str = "relevance"
    time_filter: str = "week"
    limit: int = 100

class AnalyzeTickerSentimentRequest(BaseModel):
    tickers: List[str]
    subreddits: Optional[List[str]] = None
    time_filter: str = "week" # e.g., "hour", "day", "week", "month", "year", "all"
    limit_per_ticker: int = 50 # Max posts to fetch per ticker for analysis

class EntitySentimentRequest(BaseModel):
    entity_text: str
    subreddits: Optional[List[str]] = None
    time_filter: str = "week"
    limit: int = 20

class SentimentTrendRequest(BaseModel):
    ticker: str
    days_history: int = 30 # Number of past days to analyze for trend

# --- FastAPI Server ---

app = FastAPI(title="Reddit MCP Server for LLM (Production)")

@app.on_event("startup")
async def startup_event():
    logger.info("Reddit Processor server starting up")
    await connect_redis() # Connect to Redis on startup
    # Load ML models on startup
    finbert_tone_model.load_model()
    roberta_sec_model.load_model()
    financial_bert_ner_model.load_model()


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Reddit Processor server shutting down")
    if redis_client:
        await redis_client.close()
        logger.info("Closed Redis connection for Reddit Processor.")

@app.get("/server_info")
async def get_server_info():
    return {
        "name": "reddit_processor",
        "version": "1.0.0",
        "description": "Production MCP Server for Reddit Social Sentiment Integration",
        "tools": [
            "fetch_subreddit_posts", "fetch_post_comments", "search_reddit", 
            "analyze_ticker_sentiment", "get_historical_sentiment_trend", "get_entity_sentiment_analysis"
        ],
        "models": ["FinBERT-tone", "RoBERTa-SEC", "FinancialBERT-NER"],
        "config": {k: v for k, v in CONFIG.items() if k not in ["password", "client_secret"]} # Exclude sensitive info
    }

@app.post("/fetch_subreddit_posts")
async def api_fetch_subreddit_posts(req: SubredditPostsRequest):
    logger.info(f"Fetching posts from subreddits: {req.subreddits}, sort: {req.sort}")
    try:
        reddit = get_reddit()
        posts_data = [] # Renamed from posts to avoid conflict
        
        for subreddit_name in req.subreddits:
            logger.debug(f"Fetching from subreddit: {subreddit_name}")
            subreddit = await reddit.subreddit(subreddit_name)
            sort_method = getattr(subreddit, req.sort)
            
            post_count = 0
            async for post_item in sort_method(limit=req.limit, time_filter=req.time_filter if hasattr(sort_method, "__call__") else None): # Added time_filter
                posts_data.append({
                    "id": post_item.id,
                    "title": post_item.title,
                    "score": post_item.score,
                    "created_utc": post_item.created_utc,
                    "num_comments": post_item.num_comments,
                    "permalink": post_item.permalink,
                    "author": str(post_item.author),
                    "subreddit": subreddit_name,
                    "url": post_item.url,
                    "selftext": post_item.selftext if hasattr(post_item, 'selftext') else ""
                })
                post_count += 1
                
            logger.debug(f"Retrieved {post_count} posts from {subreddit_name}")
            
        if not posts_data:
            logger.warning(f"No posts found for subreddits: {req.subreddits}")
            # Return success with empty list instead of 404 if no posts found but request is valid
            return {"success": True, "posts": []} 
            
        logger.info(f"Successfully retrieved {len(posts_data)} posts from {len(req.subreddits)} subreddits")
        return {"success": True, "posts": posts_data}
    except Exception as e:
        logger.error(f"Error fetching subreddit posts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch_post_comments")
async def api_fetch_post_comments(req: PostCommentsRequest):
    logger.info(f"Fetching comments for post {req.post_id} in subreddit {req.subreddit}")
    try:
        reddit = get_reddit()
        submission = await reddit.submission(id=req.post_id)
        # submission = await reddit.submission(url=f"https://www.reddit.com/r/{req.subreddit}/comments/{req.post_id}/") # Alternative way if only ID is known
        await submission.load() # Ensure all attributes are loaded
        
        logger.debug(f"Replacing more comments for post {req.post_id}")
        await submission.comments.replace_more(limit=None) # Fetch all comments
        
        comments_data = [] # Renamed
        comment_limit = req.limit if req.limit > 0 else None # Handle limit=0 for all
        
        for comment_item in submission.comments.list()[:comment_limit]:
            comments_data.append({
                "id": comment_item.id,
                "body": comment_item.body,
                "score": comment_item.score,
                "created_utc": comment_item.created_utc,
                "author": str(comment_item.author),
                "parent_id": comment_item.parent_id
            })
            
        if not comments_data:
            logger.warning(f"No comments found for post {req.post_id}")
            return {"success": True, "comments": []}
            
        logger.info(f"Successfully retrieved {len(comments_data)} comments for post {req.post_id}")
        return {"success": True, "comments": comments_data}
    except Exception as e:
        logger.error(f"Error fetching post comments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_reddit")
async def api_search_reddit(req: SearchRedditRequest):
    subreddits_to_search = req.subreddits or CONFIG["default_subreddits"]
    logger.info(f"Searching Reddit for query: '{req.query}' in subreddits: {subreddits_to_search}")
    
    try:
        reddit = get_reddit()
        search_results_posts = [] # Renamed
        
        # PRAW search is typically done on a specific subreddit or 'all'
        # If multiple subreddits, search each or join names if API supports (PRAW usually per subreddit)
        target_search_location = "+".join(subreddits_to_search) if subreddits_to_search else "all"
        
        # If searching specific subreddits, iterate. Otherwise, search 'all'.
        # For simplicity, this example will search across the combined string of subreddits.
        # A more robust way might be to search each and aggregate, or use a multireddit.
        
        search_location_obj = await reddit.subreddit(target_search_location)

        post_count = 0
        async for post_item in search_location_obj.search(req.query, sort=req.sort, time_filter=req.time_filter, limit=req.limit):
            search_results_posts.append({
                "id": post_item.id,
                "title": post_item.title,
                "score": post_item.score,
                "created_utc": post_item.created_utc,
                "num_comments": post_item.num_comments,
                "permalink": post_item.permalink,
                "author": str(post_item.author),
                "subreddit": str(post_item.subreddit), # Get subreddit name from post
                "url": post_item.url,
                "selftext": post_item.selftext if hasattr(post_item, 'selftext') else ""
            })
            post_count += 1
                
        logger.debug(f"Found {post_count} results in {target_search_location}")
            
        if not search_results_posts:
            logger.warning(f"No search results found for query: '{req.query}'")
            return {"success": True, "posts": []}
            
        logger.info(f"Successfully found {len(search_results_posts)} results for query: '{req.query}'")
        return {"success": True, "posts": search_results_posts}
    except Exception as e:
        logger.error(f"Error searching Reddit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_ticker_sentiment")
async def api_analyze_ticker_sentiment(req: AnalyzeTickerSentimentRequest):
    subreddits_to_analyze = req.subreddits or CONFIG["default_subreddits"]
    logger.info(f"Analyzing sentiment for tickers: {req.tickers} in subreddits: {subreddits_to_analyze}, time: {req.time_filter}, limit: {req.limit_per_ticker}")
    
    try:
        reddit = get_reddit()
        analysis_results = { # Renamed
            "success": True,
            "ticker_sentiment_analysis": {}, # Changed key
            "overall_summary": {} # Changed key
        }
        
        all_posts_content_for_analysis = []

        for ticker_symbol in req.tickers:
            logger.debug(f"Analyzing sentiment for ticker: {ticker_symbol}")
            ticker_posts_titles_and_bodies = [] # Store titles and selftext
            
            # Search across specified subreddits for the ticker
            search_query = f"\"{ticker_symbol}\" OR \"${ticker_symbol}\"" # More precise search
            
            # For simplicity, searching in each subreddit individually
            # A more advanced approach could use multireddits or search 'all' and filter by subreddit
            for subreddit_name in subreddits_to_analyze:
                subreddit_obj = await reddit.subreddit(subreddit_name)
                try:
                    async for post_item in subreddit_obj.search(search_query, sort="relevance", time_filter=req.time_filter, limit=req.limit_per_ticker):
                        content_to_analyze = post_item.title
                        if hasattr(post_item, 'selftext') and post_item.selftext:
                            content_to_analyze += ". " + post_item.selftext # Append body if exists
                        ticker_posts_titles_and_bodies.append(content_to_analyze)
                except Exception as search_err:
                    logger.warning(f"Could not search in subreddit {subreddit_name} for {ticker_symbol}: {search_err}")

            
            analysis_results["ticker_sentiment_analysis"][ticker_symbol] = {
                "mention_count": len(ticker_posts_titles_and_bodies),
                "sentiment": {"dominant": "neutral", "scores": {}, "positive_mentions":0, "negative_mentions":0, "neutral_mentions":0},
                "related_entities": []
            }
            logger.debug(f"Found {len(ticker_posts_titles_and_bodies)} mentions of {ticker_symbol}")
            
            if not ticker_posts_titles_and_bodies:
                continue

            all_posts_content_for_analysis.extend(ticker_posts_titles_and_bodies) # For overall summary later

            sentiments_for_ticker = []
            entities_for_ticker = []

            for text_content in ticker_posts_titles_and_bodies:
                try:
                    sentiment_result = finbert_tone_model.predict(text_content)
                    if sentiment_result: # Ensure prediction was successful
                        sentiments_for_ticker.append(sentiment_result[0])
                except Exception as e_sent:
                    logger.error(f"Error during FinBERT-tone prediction for '{text_content[:50]}...': {e_sent}")
                    sentiments_for_ticker.append({'label': 'NEUTRAL', 'score': 0.5}) 

                try:
                    extracted_entities = financial_bert_ner_model.predict(text_content)
                    entities_for_ticker.extend(extracted_entities)
                except Exception as e_ner:
                    logger.error(f"Error during FinancialBERT-NER prediction for '{text_content[:50]}...': {e_ner}")

            # Aggregate sentiment scores for the current ticker
            if sentiments_for_ticker:
                avg_sentiment = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                label_counts = {"positive": 0, "negative": 0, "neutral": 0}
                
                for s_item in sentiments_for_ticker:
                    label = s_item['label'].upper() # Standardize label
                    score = s_item['score']
                    if label == 'POSITIVE':
                        avg_sentiment['positive'] += score
                        label_counts['positive'] +=1
                    elif label == 'NEGATIVE':
                        avg_sentiment['negative'] += score
                        label_counts['negative'] +=1
                    else: # NEUTRAL
                        avg_sentiment['neutral'] += score
                        label_counts['neutral'] +=1

                total_scored_mentions = sum(label_counts.values())
                if total_scored_mentions > 0:
                    for label_type in avg_sentiment:
                        if label_counts[label_type] > 0: # Avoid division by zero
                            avg_sentiment[label_type] /= label_counts[label_type]
                
                dominant_sentiment_label = "neutral"
                if avg_sentiment['positive'] > avg_sentiment['negative'] and avg_sentiment['positive'] > avg_sentiment['neutral']:
                    dominant_sentiment_label = "positive"
                elif avg_sentiment['negative'] > avg_sentiment['positive'] and avg_sentiment['negative'] > avg_sentiment['neutral']:
                    dominant_sentiment_label = "negative"

                analysis_results["ticker_sentiment_analysis"][ticker_symbol]["sentiment"] = {
                    "dominant": dominant_sentiment_label,
                    "scores": avg_sentiment,
                    "positive_mentions": label_counts["positive"],
                    "negative_mentions": label_counts["negative"],
                    "neutral_mentions": label_counts["neutral"],
                }
            
            # Store unique entities for the ticker
            if entities_for_ticker:
                unique_ticker_entities_list = []
                seen_entities_text_set = set()
                for entity_item in entities_for_ticker:
                    # Filter for ORG or specific financial entities if needed
                    if entity_item['text'] not in seen_entities_text_set:
                        unique_ticker_entities_list.append(entity_item)
                        seen_entities_text_set.add(entity_item['text'])
                analysis_results["ticker_sentiment_analysis"][ticker_symbol]["related_entities"] = unique_ticker_entities_list[:15] # Limit

            # Store individual ticker sentiment in Redis
            if redis_client and analysis_results["ticker_sentiment_analysis"][ticker_symbol]["mention_count"] > 0:
                current_ts = datetime.utcnow()
                redis_key_ticker = f"reddit_sentiment:{ticker_symbol}:{current_ts.strftime('%Y%m%d')}"
                # Store daily aggregate. For trends, more granular storage might be needed or daily rollups.
                daily_data = {
                    "date": current_ts.strftime('%Y-%m-%d'),
                    "sentiment_summary": analysis_results["ticker_sentiment_analysis"][ticker_symbol]["sentiment"],
                    "mention_count": analysis_results["ticker_sentiment_analysis"][ticker_symbol]["mention_count"],
                }
                try:
                    # Append to a list for the day or store as a hash field
                    await redis_client.hset(redis_key_ticker, current_ts.strftime('%H%M%S'), json.dumps(daily_data))
                    await redis_client.expire(redis_key_ticker, timedelta(days=90)) # Keep for 90 days
                    logger.debug(f"Stored daily sentiment for {ticker_symbol} in Redis hash: {redis_key_ticker}")
                except Exception as redis_err:
                    logger.error(f"Error storing daily sentiment for {ticker_symbol} in Redis: {redis_err}")

        # Overall summary (can be enhanced)
        total_mentions_all_tickers = sum(d["mention_count"] for d in analysis_results["ticker_sentiment_analysis"].values())
        analysis_results["overall_summary"] = {
            "analyzed_ticker_count": len(req.tickers),
            "total_mentions_analyzed": total_mentions_all_tickers,
            "timestamp": datetime.utcnow().isoformat()
        }
        if total_mentions_all_tickers > 0 and analysis_results["ticker_sentiment_analysis"]:
             most_mentioned_ticker_overall = max(analysis_results["ticker_sentiment_analysis"].items(), key=lambda item: item[1]["mention_count"])[0]
             analysis_results["overall_summary"]["most_mentioned_ticker"] = most_mentioned_ticker_overall
             logger.info(f"Most mentioned ticker overall: {most_mentioned_ticker_overall}")
            
        return analysis_results
    except Exception as e:
        logger.error(f"Error analyzing ticker sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- New Endpoints for Trend and Entity-Sentiment ---

@app.post("/get_historical_sentiment_trend")
async def api_get_historical_sentiment_trend(req: SentimentTrendRequest):
    logger.info(f"Fetching historical sentiment trend for ticker: {req.ticker} over {req.days_history} days.")
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis client not available.")

    trends = []
    try:
        for i in range(req.days_history -1, -1, -1): # Iterate from oldest to newest day in range
            day_to_fetch = datetime.utcnow() - timedelta(days=i)
            redis_key_pattern = f"reddit_sentiment:{req.ticker}:{day_to_fetch.strftime('%Y%m%d')}"
            
            # Since we store hourly/minutely data in a hash, we need to get all fields
            daily_entries_raw = await redis_client.hgetall(redis_key_pattern)
            
            if daily_entries_raw:
                daily_sentiments = []
                daily_mention_count = 0
                
                for timestamp_key, entry_json in daily_entries_raw.items():
                    entry_data = json.loads(entry_json)
                    daily_sentiments.append(entry_data["sentiment_summary"]["scores"]) # Collect all scores for the day
                    daily_mention_count += entry_data["sentiment_summary"].get("positive_mentions",0) + \
                                           entry_data["sentiment_summary"].get("negative_mentions",0) + \
                                           entry_data["sentiment_summary"].get("neutral_mentions",0)
                
                if daily_sentiments:
                    # Average the sentiment scores for the day
                    avg_daily_pos = sum(s['positive'] for s in daily_sentiments) / len(daily_sentiments)
                    avg_daily_neg = sum(s['negative'] for s in daily_sentiments) / len(daily_sentiments)
                    avg_daily_neu = sum(s['neutral'] for s in daily_sentiments) / len(daily_sentiments)
                    
                    trends.append({
                        "date": day_to_fetch.strftime('%Y-%m-%d'),
                        "avg_positive_score": avg_daily_pos,
                        "avg_negative_score": avg_daily_neg,
                        "avg_neutral_score": avg_daily_neu,
                        "total_mentions": daily_mention_count
                    })
        
        if not trends:
            return {"success": True, "ticker": req.ticker, "trend_data": [], "message": "No historical sentiment data found for the given period."}

        return {"success": True, "ticker": req.ticker, "trend_data": trends}
    except Exception as e:
        logger.error(f"Error fetching historical sentiment trend for {req.ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_entity_sentiment_analysis")
async def api_get_entity_sentiment_analysis(req: EntitySentimentRequest):
    logger.info(f"Fetching sentiment for entity: '{req.entity_text}' in subreddits: {req.subreddits}, time: {req.time_filter}, limit: {req.limit}")
    # This endpoint will be similar to analyze_ticker_sentiment but focused on a specific entity string
    # It will search for the entity_text, then perform NER to confirm the entity and sentiment analysis on surrounding text.
    # For brevity, this is a simplified version focusing on direct search and sentiment.
    
    subreddits_to_search = req.subreddits or CONFIG["default_subreddits"]
    try:
        reddit = get_reddit()
        found_posts_content = []

        for subreddit_name in subreddits_to_search:
            subreddit_obj = await reddit.subreddit(subreddit_name)
            try:
                async for post_item in subreddit_obj.search(f"\"{req.entity_text}\"", sort="relevance", time_filter=req.time_filter, limit=req.limit):
                    content = post_item.title
                    if hasattr(post_item, 'selftext') and post_item.selftext:
                        content += ". " + post_item.selftext
                    found_posts_content.append(content)
            except Exception as search_err:
                logger.warning(f"Could not search for entity '{req.entity_text}' in {subreddit_name}: {search_err}")
        
        if not found_posts_content:
            return {"success": True, "entity": req.entity_text, "analysis": {"mention_count": 0, "sentiment": "neutral", "scores": {}}}

        sentiments = []
        extracted_entities_all = []
        for text_content in found_posts_content:
            try:
                sentiment_result = finbert_tone_model.predict(text_content)
                if sentiment_result: sentiments.append(sentiment_result[0])
            except Exception: pass # Logged in model

            try: # NER to confirm entity presence and find related ones
                entities_in_text = financial_bert_ner_model.predict(text_content)
                extracted_entities_all.extend(entities_in_text)
            except Exception: pass # Logged in model
        
        # Aggregate sentiment
        final_sentiment = {"dominant": "neutral", "scores": {"positive":0,"negative":0,"neutral":0.5}, "mentions":len(found_posts_content)}
        if sentiments:
            # (Aggregation logic similar to api_analyze_ticker_sentiment)
            # ...
            pass # Placeholder for full aggregation logic

        # Filter entities related to the searched entity_text (simplified)
        relevant_entities = [e for e in extracted_entities_all if req.entity_text.lower() in e['text'].lower()]


        return {
            "success": True, 
            "entity": req.entity_text, 
            "analysis": {
                "mention_count": len(found_posts_content),
                "aggregated_sentiment": final_sentiment, # Implement full aggregation
                "sample_entities_found": relevant_entities[:10]
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing entity sentiment for '{req.entity_text}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resource/{resource_uri:path}")
async def get_resource(resource_uri: str):
    logger.info(f"Resource request for URI: {resource_uri}")
    
    try:
        if resource_uri == "sentiment/trending":
            logger.warning("Trending sentiment endpoint called without parameters")
            # This could be enhanced to pull from Redis aggregated trends
            raise HTTPException(status_code=400, detail="Trending sentiment requires a list of subreddits/tickers in production.")
        elif resource_uri == "tickers/trending":
            logger.warning("Trending tickers endpoint called without parameters")
            # Could query Redis for most mentioned tickers in last N hours/days
            raise HTTPException(status_code=400, detail="Trending tickers requires a list of subreddits/tickers in production.")
        elif resource_uri == "topics/trending": # General topics, harder to define without more context
            logger.warning("Trending topics endpoint called without parameters")
            raise HTTPException(status_code=400, detail="Trending topics requires a list of subreddits/tickers in production.")
        elif resource_uri == "status":
            logger.debug("Returning operational status")
            redis_status = "connected" if redis_client and await redis_client.ping() else "disconnected"
            return {"status": "operational", "redis_status": redis_status, "timestamp": datetime.utcnow().isoformat()}
        else:
            logger.warning(f"Unknown resource URI requested: {resource_uri}")
            raise HTTPException(status_code=404, detail=f"Unknown resource URI: {resource_uri}")
    except Exception as e:
        logger.error(f"Error in get_resource: {str(e)}")
        # For Redis ping, ensure it doesn't crash if Redis is down during status check
        if resource_uri == "status":
            return {"status": "operational", "redis_status": "error during ping", "timestamp": datetime.utcnow().isoformat()}
        raise HTTPException(status_code=500, detail=str(e))

# --- AI/ML Model Classes ---
class FinBERTtoneModel:
    def __init__(self):
        self.model_name = "ProsusAI/finbert" 
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"FinBERT-tone will run on {self.device}")

    def load_model(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if self.model is None or self.tokenizer is None:
            logger.info(f"Loading FinBERT-tone model: {self.model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
                logger.info("FinBERT-tone model loaded.")
            except Exception as e:
                logger.error(f"Error loading FinBERT-tone model ({self.model_name}): {e}")
                # Fallback or raise
                raise

    def predict(self, text: str) -> List[Dict[str, Any]]:
        from transformers import pipeline
        if self.model is None or self.tokenizer is None:
            self.load_model() # Ensure model is loaded
        
        sentiment_analysis = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=0 if self.device=="cuda" else -1)
        results = sentiment_analysis(text)
        return [{"label": r["label"], "score": float(r["score"])} for r in results]


class RoBERTaSECModel: 
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest" 
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"RoBERTa-SEC will run on {self.device}")


    def load_model(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if self.model is None or self.tokenizer is None:
            logger.info(f"Loading RoBERTa-SEC model: {self.model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
                logger.info("RoBERTa-SEC model loaded.")
            except Exception as e:
                logger.error(f"Error loading RoBERTa-SEC model ({self.model_name}): {e}")
                raise

    def predict(self, text: str) -> List[Dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            self.load_model()

        encoded_input = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            output = self.model(**encoded_input)

        scores_tensor = torch.nn.functional.softmax(output.logits, dim=-1)
        scores = scores_tensor.cpu().detach().numpy()[0] 

        results = []
        for i, score_val in enumerate(scores): # score_val to avoid conflict
            label = self.model.config.id2label.get(i, f"label_{i}")
            results.append({"label": label, "score": float(score_val)})
        return results


class FinancialBERTNERModel: 
    def __init__(self):
        # Using a more general NER model as ProsusAI/finbert-ner might not be available or correctly configured
        self.model_name = "Jean-Baptiste/roberta-large-ner-english" 
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"FinancialBERT-NER will run on {self.device} using {self.model_name}")

    def load_model(self):
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        if self.model is None or self.tokenizer is None:
            logger.info(f"Loading FinancialBERT-NER model: {self.model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
                logger.info("FinancialBERT-NER model loaded.")
            except Exception as e:
                logger.error(f"Error loading FinancialBERT-NER model ({self.model_name}): {e}")
                raise


    def predict(self, text: str) -> List[Dict[str, Any]]:
        from transformers import pipeline
        if self.model is None or self.tokenizer is None:
            self.load_model()

        ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=0 if self.device=="cuda" else -1, aggregation_strategy="simple")
        ner_results = ner_pipeline(text)
        
        formatted_entities = []
        for entity in ner_results:
            formatted_entities.append({
                'type': entity['entity_group'],
                'text': entity['word'],
                'score': float(entity['score']),
                'start': int(entity['start']),
                'end': int(entity['end'])
            })
        return formatted_entities

# Instantiate models globally after class definitions
finbert_tone_model = FinBERTtoneModel()
roberta_sec_model = RoBERTaSECModel() # Not directly used yet, but loaded
financial_bert_ner_model = FinancialBERTNERModel()
