"""
Polygon REST API FastAPI server for LLM integration.
Handles historical market data retrieval and processing from Polygon's REST API,
including feature engineering and pattern classification.
"""


import aiohttp
import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Union, Tuple
import datetime
from datetime import timedelta
import os
import pandas as pd
import numpy as np
from urllib.parse import urlencode
from dotenv import load_dotenv
import redis.asyncio as aioredis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import logging
import time
import yaml
from pathlib import Path
from copy import deepcopy

# Fallback logger if monitor.logging_utils is unavailable
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

# Configuration
def validate_config():
    config_path = Path("config/polygon_config.yaml")
    default_config = {
        "polygon": {
            "api_key": os.getenv("POLYGON_API_KEY", ""),
            "rate_limit_per_minute": int(os.getenv("POLYGON_RATE_LIMIT", 5)),
            "use_cache": os.getenv("POLYGON_USE_CACHE", "True").lower() == "true"
        },
        "redis": {
            "host": os.getenv("POLYGON_REDIS_HOST", os.getenv("REDIS_HOST", "localhost")),
            "port": int(os.getenv("POLYGON_REDIS_PORT", os.getenv("REDIS_PORT", "6379"))),
            "db": int(os.getenv("POLYGON_REDIS_DB", os.getenv("REDIS_DB", "3"))),
            "password": os.getenv("POLYGON_REDIS_PASSWORD", os.getenv("REDIS_PASSWORD", None))
        },
        "feature_engineering": {
            "sma_periods": [10, 20, 50],
            "rsi_period": 14,
            "atr_period": 14,
            "rolling_volatility_window": 20
        }
    }
    
    try:
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f) or {}
            
            # Deep merge default and file config
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
        
        # Override with environment variables if set
        config["polygon"]["api_key"] = os.getenv("POLYGON_API_KEY", config["polygon"]["api_key"])
        config["polygon"]["rate_limit_per_minute"] = int(os.getenv("POLYGON_RATE_LIMIT", config["polygon"]["rate_limit_per_minute"]))
        config["polygon"]["use_cache"] = os.getenv("POLYGON_USE_CACHE", str(config["polygon"]["use_cache"])).lower() == "true"
        config["redis"]["host"] = os.getenv("POLYGON_REDIS_HOST", os.getenv("REDIS_HOST", config["redis"]["host"]))
        config["redis"]["port"] = int(os.getenv("POLYGON_REDIS_PORT", os.getenv("REDIS_PORT", config["redis"]["port"])))
        config["redis"]["db"] = int(os.getenv("POLYGON_REDIS_DB", os.getenv("REDIS_DB", config["redis"]["db"])))
        config["redis"]["password"] = os.getenv("POLYGON_REDIS_PASSWORD", os.getenv("REDIS_PASSWORD", config["redis"]["password"]))
        
        # Validate configuration
        if not config["polygon"]["api_key"]:
            logger.warning("POLYGON_API_KEY is not set. API requests will fail.")
        if config["polygon"]["rate_limit_per_minute"] <= 0:
            logger.warning(f"Invalid rate_limit_per_minute: {config['polygon']['rate_limit_per_minute']}. Using default: 5")
            config["polygon"]["rate_limit_per_minute"] = 5
        if not all(isinstance(p, int) and p > 0 for p in config["feature_engineering"]["sma_periods"]):
            logger.warning("Invalid SMA periods. Using default: [10, 20, 50]")
            config["feature_engineering"]["sma_periods"] = [10, 20, 50]
        if config["feature_engineering"]["rsi_period"] <= 0:
            config["feature_engineering"]["rsi_period"] = 14
        if config["feature_engineering"]["atr_period"] <= 0:
            config["feature_engineering"]["atr_period"] = 14
        if config["feature_engineering"]["rolling_volatility_window"] <= 0:
            config["feature_engineering"]["rolling_volatility_window"] = 20
        
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}. Using default configuration.")
        return default_config

CONFIG = validate_config()
BASE_URL = "https://api.polygon.io"
RATE_LIMIT_PER_MINUTE = CONFIG["polygon"]["rate_limit_per_minute"]
USE_CACHE = CONFIG["polygon"]["use_cache"]

logger = get_logger("polygon_rest_server")
logger.info("Initializing Polygon REST API server with feature engineering capabilities.")

# Global state
redis_client: Optional[aioredis.Redis] = None
http_session: Optional[aiohttp.ClientSession] = None
request_timestamps: List[datetime.datetime] = []

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
                await asyncio.sleep(retry_delay)
    logger.warning("Failed to connect to Redis after retries. Proceeding without Redis.")
    redis_client = None

async def get_http_session():
    global http_session
    if http_session is None or http_session.closed:
        http_session = aiohttp.ClientSession()
    return http_session

async def wait_for_rate_limit():
    if not redis_client:
        global request_timestamps
        now = datetime.datetime.now()
        request_timestamps = [ts for ts in request_timestamps if now - ts < datetime.timedelta(minutes=1)]
        if len(request_timestamps) >= RATE_LIMIT_PER_MINUTE:
            wait_time = (request_timestamps[0] + datetime.timedelta(minutes=1)) - now
            if wait_time.total_seconds() > 0:
                logger.warning(f"Polygon API rate limit reached. Waiting for {wait_time.total_seconds():.2f}s")
                await asyncio.sleep(wait_time.total_seconds())
                now = datetime.datetime.now()
                request_timestamps = [ts for ts in request_timestamps if now - ts < datetime.timedelta(minutes=1)]
        request_timestamps.append(now)
        return
    
    key = "polygon:rate_limit"
    now = int(time.time())
    window = 60
    try:
        async with redis_client.pipeline() as pipe:
            pipe.zremrangebyscore(key, 0, now - window)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, window)
            _, _, count, _ = await pipe.execute()
        if count >= RATE_LIMIT_PER_MINUTE:
            oldest = await redis_client.zrange(key, 0, 0)
            if oldest:
                wait_time = (int(oldest[0]) + window - now)
                if wait_time > 0:
                    logger.warning(f"Polygon API rate limit reached. Waiting for {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
    except Exception as e:
        logger.error(f"Error managing rate limit in Redis: {e}. Falling back to no rate limiting.")

async def polygon_request(endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    await wait_for_rate_limit()
    session = await get_http_session()
    api_key = CONFIG["polygon"]["api_key"]
    if not api_key:
        error_message = "Polygon API key not configured. Set POLYGON_API_KEY environment variable."
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    
    final_params = params.copy() if params else {}
    final_params["apiKey"] = api_key
    cache_key = None
    if USE_CACHE and redis_client:
        sorted_params_str = urlencode(sorted(final_params.items()))
        cache_key = f"polygon:{endpoint.replace('/', '_')}:{sorted_params_str}"
        try:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                logger.debug(f"Redis cache hit for Polygon endpoint {endpoint}")
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Error reading Redis cache for {endpoint}: {e}")
    
    url = f"{BASE_URL}{endpoint}"
    logger.debug(f"Making Polygon API request to {url} with params {final_params}")
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
            if USE_CACHE and redis_client and cache_key:
                try:
                    await redis_client.set(cache_key, json.dumps(response_json), ex=3600)
                    logger.debug(f"Cached Polygon response in Redis for {endpoint}")
                except Exception as e:
                    logger.warning(f"Error writing to Redis cache for {endpoint}: {e}")
            return response_json
    except aiohttp.ClientError as e:
        logger.error(f"AIOHTTP ClientError for Polygon endpoint {endpoint}: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")
    except Exception as e:
        logger.error(f"Unexpected error for Polygon endpoint {endpoint}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_aggregates(
    ticker: str, multiplier: int, timespan: str, from_date: str, to_date: str,
    adjusted: bool = True, sort: str = "asc", limit: int = 5000
) -> Dict[str, Any]:
    endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {"adjusted": str(adjusted).lower(), "sort": sort, "limit": str(limit)}
    response = await polygon_request(endpoint, params)
    if response.get("status") == "OK" and "results" in response:
        return {"success": True, "ticker": response.get("ticker"), "results": response["results"]}
    error_msg = response.get("message", response.get("error", "Unknown error fetching aggregates"))
    logger.error(f"Failed to fetch aggregates for {ticker}: {error_msg}")
    return {"success": False, "error": error_msg, "ticker": ticker}

# --- Feature Engineering ---
def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def extract_features_from_aggregates(df_agg: pd.DataFrame) -> pd.DataFrame:
    if df_agg.empty:
        logger.warning("Empty DataFrame provided for feature engineering")
        return pd.DataFrame()
    required_columns = ['o', 'h', 'l', 'c', 'v', 't']
    if not all(col in df_agg.columns for col in required_columns):
        logger.error(f"Missing required columns in aggregates DataFrame: {set(required_columns) - set(df_agg.columns)}")
        return pd.DataFrame()
    try:
        df = df_agg.copy()
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df = df.set_index('t')
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        fe_config = CONFIG["feature_engineering"]
        for period in fe_config["sma_periods"]:
            df[f'sma{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df['rsi'] = calculate_rsi(df['close'], period=fe_config["rsi_period"])
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], period=fe_config["atr_period"])
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['bb_sma'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_sma'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_sma'] - (df['bb_std'] * 2)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_returns'].rolling(window=fe_config["rolling_volatility_window"]).std() * np.sqrt(fe_config["rolling_volatility_window"])
        df['relative_volume'] = df['volume'] / df['volume'].rolling(window=50).mean()
        df['volume_change'] = df['volume'].pct_change()
        df['volume_sma5'] = df['volume'].rolling(window=5).mean()
        df['volume_sma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume_sma5'] / df['volume_sma20']
        return df.dropna()
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        return pd.DataFrame()

def identify_simple_support_resistance(df: pd.DataFrame, window_size: int = 10, num_levels: int = 3) -> Tuple[List[float], List[float]]:
    if df.empty or len(df) < window_size * 2:
        logger.warning("Not enough data to identify support and resistance levels")
        return [], []
    required_columns = ['high', 'low']
    if not all(col in df.columns for col in required_columns):
        logger.warning("Missing high or low columns in DataFrame")
        return [], []
    
    df_copy = df.copy()
    resistance_points = []
    support_points = []
    
    highs = df_copy['high'].rolling(window=2*window_size+1, center=True).apply(
        lambda x: x[window_size] if x[window_size] == x.max() else np.nan, raw=False
    ).dropna()
    lows = df_copy['low'].rolling(window=2*window_size+1, center=True).apply(
        lambda x: x[window_size] if x[window_size] == x.min() else np.nan, raw=False
    ).dropna()
    
    resistance_points = highs.tolist()
    support_points = lows.tolist()
    
    def cluster_price_levels(levels: List[float]) -> List[float]:
        if not levels:
            return []
        levels = sorted(levels)
        clustered = []
        cluster = [levels[0]]
        for i in range(1, len(levels)):
            if levels[i] < cluster[0] * 1.01:
                cluster.append(levels[i])
            else:
                clustered.append(sum(cluster) / len(cluster))
                cluster = [levels[i]]
        if cluster:
            clustered.append(sum(cluster) / len(cluster))
        return sorted(clustered)[:num_levels]
    
    return (cluster_price_levels(support_points), cluster_price_levels(resistance_points))

# --- Pattern Classification Model ---
class PatternClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        if X.empty or y.empty:
            logger.error("Cannot train PatternClassifier: Input data or labels are empty")
            return
        if not all(col in X.columns for col in ['close', 'rsifiber', 'atr', 'macd']):
            logger.error("Missing required features in training data")
            return
        self.feature_columns = X.columns.tolist()
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            logger.info(f"PatternClassifier trained with {len(self.feature_columns)} features")
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "pattern_classifier")
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(self.model, os.path.join(model_dir, "pattern_classifier_rf.joblib"))
            joblib.dump(self.scaler, os.path.join(model_dir, "pattern_scaler.joblib"))
            joblib.dump(self.feature_columns, os.path.join(model_dir, "pattern_features.joblib"))
            logger.info(f"Saved trained model to {model_dir}")
        except Exception as e:
            logger.error(f"Error training PatternClassifier: {e}")
            self.is_trained = False
    
    def predict(self, X_new: pd.DataFrame) -> List[str]:
        if not self.is_trained or not self.feature_columns:
            logger.warning("PatternClassifier not trained. Returning default predictions")
            return ["unknown_pattern"] * len(X_new)
        if X_new.empty:
            return []
        try:
            X_new_aligned = X_new.reindex(columns=self.feature_columns, fill_value=0)
            X_new_scaled = self.scaler.transform(X_new_aligned)
            predictions = self.model.predict(X_new_scaled)
            return predictions.tolist()
        except Exception as e:
            logger.error(f"Error predicting with PatternClassifier: {e}")
            return ["unknown_pattern"] * len(X_new)

pattern_classifier = PatternClassifier()

# --- Historical Pattern Analysis ---
async def analyze_historical_patterns_with_ml(symbol: str, start_date_str: str, end_date_str: str, timeframes: List[str] = None):
    try:
        datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
        datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Invalid date format for {symbol}. Use YYYY-MM-DD")
        return {"success": False, "error": "Invalid date format. Use YYYY-MM-DD"}
    
    if timeframes is None:
        timeframes = ["1day"]
    analysis_results = {"symbol": symbol, "period": f"{start_date_str}_to_{end_date_str}", "analysis_by_timeframe": {}}
    
    for tf_full in timeframes:
        multiplier = 1
        timespan_unit = tf_full
        match = re.match(r"(\d+)([a-zA-Z]+)", tf_full)
        if match:
            multiplier = int(match.group(1))
            timespan_unit = match.group(2).lower()
            if timespan_unit.endswith('s'):
                timespan_unit = timespan_unit[:-1]
            if timespan_unit == "min":
                timespan_unit = "minute"
        
        agg_data = await fetch_aggregates(symbol, multiplier, timespan_unit, start_date_str, end_date_str)
        current_tf_analysis = {
            "raw_bar_count": 0,
            "feature_count": 0,
            "patterns_detected": [],
            "support": [],
            "resistance": [],
            "trend_strength": "N/A",
            "classified_pattern": "N/A"
        }
        
        if agg_data.get("success") and agg_data.get("results"):
            df_agg = pd.DataFrame(agg_data["results"])
            current_tf_analysis["raw_bar_count"] = len(df_agg)
            if not df_agg.empty:
                df_features = extract_features_from_aggregates(df_agg)
                current_tf_analysis["feature_count"] = len(df_features)
                if not df_features.empty:
                    if len(df_features) >= 5:
                        recent_features = df_features.tail(5)
                        classified_patterns = pattern_classifier.predict(recent_features)
                        current_tf_analysis["patterns_detected"] = classified_patterns
                        current_tf_analysis["classified_pattern"] = classified_patterns[-1]
                    else:
                        classified_patterns = pattern_classifier.predict(df_features.iloc[[-1]])
                        current_tf_analysis["classified_pattern"] = classified_patterns[0] if classified_patterns else "N/A"
                
                sup, res = identify_simple_support_resistance(
                    df_agg.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"})
                )
                current_tf_analysis["support"] = sup
                current_tf_analysis["resistance"] = res
                if 'c' in df_agg.columns and len(df_agg) > 50:
                    sma50 = df_agg['c'].rolling(window=50).mean()
                    if not sma50.empty and not pd.isna(sma50.iloc[-1]):
                        current_tf_analysis["trend_strength"] = "up" if df_agg['c'].iloc[-1] > sma50.iloc[-1] else "down"
        
        analysis_results["analysis_by_timeframe"][tf_full] = current_tf_analysis
    
    if redis_client:
        redis_key = f"polygon:ml_pattern_analysis:{symbol}:{start_date_str}-{end_date_str}"
        try:
            await redis_client.set(redis_key, json.dumps(analysis_results), ex=timedelta(hours=6))
            logger.info(f"Stored ML pattern analysis for {symbol} in Redis: {redis_key}")
        except Exception as e:
            logger.error(f"Error storing ML pattern analysis in Redis for {symbol}: {e}")
    
    return {"success": True, "analysis": analysis_results}

# --- FastAPI Models ---
class AggregatesRequest(BaseModel):
    ticker: str
    multiplier: int
    timespan: str
    from_date: str
    to_date: str
    adjusted: bool = True
    sort: str = "asc"
    limit: int = 5000

class PatternAnalysisToolRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    timeframes: Optional[List[str]] = Field(default=["1day", "1hour"])

class CustomTimeframeAggregatesRequest(BaseModel):
    ticker: str
    multiplier: int
    timespan: str
    from_date: str
    to_date: str
    adjusted |bool = True
    sort: str = "asc"
    limit: int = 5000

# --- FastAPI Server ---
app = FastAPI(
    title="Polygon REST MCP Server (Production)",
    description="Production-ready server for Polygon REST API, providing historical market data, feature engineering, and pattern analysis.",
    version="1.2.0"
)

# Authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")
    if not api_key or api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

@app.on_event("startup")
async def startup_event():
    logger.info("Polygon REST API server starting up...")
    if not CONFIG["polygon"]["api_key"]:
        logger.warning("WARNING: Polygon API key is not configured. Set POLYGON_API_KEY environment variable.")
    else:
        logger.info("Polygon API key is configured.")
    
    await connect_redis()
    await get_http_session()
    
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "pattern_classifier")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "pattern_classifier_rf.joblib")
    scaler_path = os.path.join(model_dir, "pattern_scaler.joblib")
    features_path = os.path.join(model_dir, "pattern_features.joblib")
    
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
            pattern_classifier.model = joblib.load(model_path)
            pattern_classifier.scaler = joblib.load(scaler_path)
            pattern_classifier.feature_columns = joblib.load(features_path)
            pattern_classifier.is_trained = True
            logger.info(f"Loaded pre-trained PatternClassifier model from {model_path}")
        else:
            logger.warning("No pre-trained PatternClassifier model found. Pattern predictions will use defaults.")
    except Exception as e:
        logger.error(f"Error loading pre-trained PatternClassifier model: {e}. Pattern predictions will use defaults.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Polygon REST API server shutting down.")
    global http_session
    if http_session and not http_session.closed:
        await http_session.close()
    if redis_client:
        await redis_client.close()

@app.get("/server_info", tags=["General"])
async def get_server_info():
    safe_config = {k: v for k, v in CONFIG["polygon"].items() if k != "api_key"}
    return {
        "name": "polygon_rest",
        "version": "1.2.0",
        "description": "MCP Server for Polygon REST API, providing historical data, feature engineering, and pattern analysis.",
        "tools": [
            "fetch_aggregates",
            "analyze_historical_patterns_with_ml",
            "get_custom_timeframe_aggregates"
        ],
        "config": safe_config
    }

@app.post("/fetch_aggregates", tags=["Data"], dependencies=[Depends(verify_api_key)])
async def api_fetch_aggregates(req: AggregatesRequest):
    """Fetch aggregated market data from Polygon."""
    return await fetch_aggregates(req.ticker, req.multiplier, req.timespan, req.from_date, req.to_date, req.adjusted, req.sort, req.limit)

@app.post("/analyze_historical_patterns_with_ml", tags=["Analysis"], dependencies=[Depends(verify_api_key)])
async def api_analyze_historical_patterns_ml(req: PatternAnalysisToolRequest):
    """Analyze historical price patterns using machine learning."""
    return await analyze_historical_patterns_with_ml(req.symbol, req.start_date, req.end_date, req.timeframes)

@app.post("/get_custom_timeframe_aggregates", tags=["Data"], dependencies=[Depends(verify_api_key)])
async def api_get_custom_timeframe_aggregates(req: CustomTimeframeAggregatesRequest):
    """Fetch aggregated market data for a custom timeframe."""
    return await fetch_aggregates(req.ticker, req.multiplier, req.timespan, req.from_date, req.to_date, req.adjusted, req.sort, req.limit)