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
from monitor.logging_utils import get_logger
import redis.asyncio as aioredis
from sklearn.ensemble import RandomForestClassifier # Using RandomForest as an example
from sklearn.model_selection import train_test_split # For potential future training
from sklearn.preprocessing import StandardScaler
import joblib  # For model persistence

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
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
BASE_URL = "https://api.polygon.io"
RATE_LIMIT_PER_MINUTE = CONFIG["polygon"]["rate_limit_per_minute"]
USE_CACHE = CONFIG["polygon"]["use_cache"]
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "polygon_rest")
os.makedirs(CACHE_DIR, exist_ok=True)

logger = get_logger("polygon_rest_server")
logger.info("Initializing Polygon REST API server with feature engineering capabilities.")

# Redis Client & HTTP Session (same as before)
redis_client: Optional[aioredis.Redis] = None
http_session: Optional[aiohttp.ClientSession] = None
request_timestamps: List[datetime.datetime] = []

async def connect_redis():
    global redis_client
    if redis_client is None:
        try:
            redis_client = aioredis.Redis(
                host=CONFIG["redis"]["host"], port=CONFIG["redis"]["port"],
                db=CONFIG["redis"]["db"], password=CONFIG["redis"]["password"],
                decode_responses=True
            )
            await redis_client.ping()
            logger.info(f"Polygon REST server connected to Redis at {CONFIG['redis']['host']}:{CONFIG['redis']['port']} (DB: {CONFIG['redis']['db']})")
        except Exception as e:
            logger.error(f"Polygon REST server failed to connect to Redis: {e}")
            redis_client = None

async def get_http_session():
    global http_session
    if http_session is None or http_session.closed:
        http_session = aiohttp.ClientSession()
    return http_session

async def wait_for_rate_limit():
    global request_timestamps
    now = datetime.datetime.now()
    request_timestamps = [ts for ts in request_timestamps if now - ts < datetime.timedelta(minutes=1)]
    if len(request_timestamps) >= RATE_LIMIT_PER_MINUTE:
        wait_time = (request_timestamps[0] + datetime.timedelta(minutes=1)) - now
        if wait_time.total_seconds() > 0:
            logger.warning(f"Polygon API rate limit nearly reached. Waiting for {wait_time.total_seconds():.2f}s")
            await asyncio.sleep(wait_time.total_seconds())
            now = datetime.datetime.now()
            request_timestamps = [ts for ts in request_timestamps if now - ts < datetime.timedelta(minutes=1)]
    request_timestamps.append(now)

async def polygon_request(endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    # (polygon_request implementation remains the same as previously provided)
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
    if USE_CACHE:
        sorted_params_str = urlencode(sorted(final_params.items()))
        cache_key = f"{endpoint.replace('/', '_')}_{sorted_params_str}.json"
        cache_path = os.path.join(CACHE_DIR, cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f: cache_data = json.load(f)
                logger.debug(f"Cache hit for Polygon endpoint {endpoint}")
                return cache_data
            except Exception as e: logger.warning(f"Error reading Polygon cache for {endpoint}: {e}")
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
            if USE_CACHE and cache_key:
                try:
                    with open(os.path.join(CACHE_DIR, cache_key), 'w') as f: json.dump(response_json, f)
                    logger.debug(f"Cached Polygon response for {endpoint}")
                except Exception as e: logger.warning(f"Error writing to Polygon cache for {endpoint}: {e}")
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
    else:
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
    if df_agg.empty: return pd.DataFrame()
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
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_sma'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_sma'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_sma'] - (df['bb_std'] * 2)
    
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_returns'].rolling(window=fe_config["rolling_volatility_window"]).std() * np.sqrt(fe_config["rolling_volatility_window"]) # Annualized if daily

    # Relative Volume implementation
    df['relative_volume'] = df['volume'] / df['volume'].rolling(window=50).mean()
    
    # Additional volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_sma5'] = df['volume'].rolling(window=5).mean()
    df['volume_sma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume_sma5'] / df['volume_sma20']
    
    # Drop NaN created by rolling functions at the beginning
    return df.dropna()

def identify_simple_support_resistance(df: pd.DataFrame, window_size: int = 10, num_levels: int = 3) -> Tuple[List[float], List[float]]:
    """
    Identify simple support and resistance levels using local minima and maxima.
    
    Args:
        df: DataFrame with OHLCV data
        window_size: Size of the window to look for local minima and maxima
        num_levels: Number of support and resistance levels to return
        
    Returns:
        Tuple of (support_levels, resistance_levels) as lists of float values
    """
    if df.empty or len(df) < window_size * 2:
        logger.warning("Not enough data to identify support and resistance levels")
        return [], []
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Ensure we have high and low columns
    if 'high' not in df_copy.columns or 'low' not in df_copy.columns:
        logger.warning("Missing high and low columns in dataframe")
        return [], []
    
    # Find local maxima and minima
    resistance_points = []
    support_points = []
    
    for i in range(window_size, len(df_copy) - window_size):
        # Check if this point is a local maximum (potential resistance)
        if all(df_copy['high'].iloc[i] >= df_copy['high'].iloc[i-j] for j in range(1, window_size+1)) and \
           all(df_copy['high'].iloc[i] >= df_copy['high'].iloc[i+j] for j in range(1, window_size+1)):
            resistance_points.append(df_copy['high'].iloc[i])
        
        # Check if this point is a local minimum (potential support)
        if all(df_copy['low'].iloc[i] <= df_copy['low'].iloc[i-j] for j in range(1, window_size+1)) and \
           all(df_copy['low'].iloc[i] <= df_copy['low'].iloc[i+j] for j in range(1, window_size+1)):
            support_points.append(df_copy['low'].iloc[i])
    
    # Cluster close levels (averaging points within 1% of each other)
    def cluster_price_levels(levels: List[float]) -> List[float]:
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        cluster = [levels[0]]
        
        for i in range(1, len(levels)):
            # If this level is within 1% of the previous level, add to the same cluster
            if levels[i] < cluster[0] * 1.01:
                cluster.append(levels[i])
            else:
                # Complete the current cluster and start a new one
                clustered.append(sum(cluster) / len(cluster))
                cluster = [levels[i]]
        
        # Add the last cluster
        if cluster:
            clustered.append(sum(cluster) / len(cluster))
        
        return sorted(clustered)
    
    # Cluster and limit the number of levels
    support_levels = cluster_price_levels(support_points)
    resistance_levels = cluster_price_levels(resistance_points)
    
    # Return the strongest levels (most frequently observed)
    return (support_levels[:num_levels], resistance_levels[:num_levels])

# --- Pattern Classification Model (Example with RandomForest) ---
class PatternClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [] # To be defined during training

    def train(self, X: pd.DataFrame, y: pd.Series):
        # This is a simplified training method. Production training would be more involved.
        if X.empty or y.empty:
            logger.error("Cannot train PatternClassifier: Input data or labels are empty.")
            return
        self.feature_columns = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info(f"PatternClassifier trained with {len(self.feature_columns)} features.")
        # Save the trained model and scaler
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "pattern_classifier")
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(model_dir, "pattern_classifier_rf.joblib"))
        joblib.dump(self.scaler, os.path.join(model_dir, "pattern_scaler.joblib"))
        joblib.dump(self.feature_columns, os.path.join(model_dir, "pattern_features.joblib"))
        logger.info(f"Saved trained model to {model_dir}")


    def predict(self, X_new: pd.DataFrame) -> List[str]:
        if not self.is_trained or not self.feature_columns:
            logger.warning("PatternClassifier not trained or feature columns not set. Returning default predictions.")
            # Return a default or empty prediction if not trained
            return ["unknown_pattern"] * len(X_new) 
        
        if X_new.empty: return []
        
        # Ensure X_new has the same columns as training data
        X_new_aligned = X_new.reindex(columns=self.feature_columns, fill_value=0)
        X_new_scaled = self.scaler.transform(X_new_aligned)
        predictions = self.model.predict(X_new_scaled)
        # Assuming predictions are class labels (e.g., "bull_flag", "consolidation")
        return predictions.tolist() 

# Global instance (or manage via dependency injection)
pattern_classifier = PatternClassifier()
# pattern_classifier.train(X_train_data, y_train_labels) # This would be done offline or in a separate training script

# --- Historical Pattern Analysis Component ---
async def analyze_historical_patterns_with_ml(symbol: str, start_date_str: str, end_date_str: str, timeframes: List[str] = None):
    if timeframes is None: timeframes = ["1day"] # Default to daily, using Polygon's timespan format
    analysis_results = {"symbol": symbol, "period": f"{start_date_str}_to_{end_date_str}", "analysis_by_timeframe": {}}

    for tf_full in timeframes:
        multiplier = 1
        timespan_unit = tf_full
        match = re.match(r"(\d+)([a-zA-Z]+)", tf_full)
        if match:
            multiplier = int(match.group(1))
            timespan_unit = match.group(2).lower()
            if timespan_unit.endswith('s'): timespan_unit = timespan_unit[:-1]
            if timespan_unit == "min": timespan_unit = "minute" # Align with Polygon

        agg_data = await fetch_aggregates(symbol, multiplier, timespan_unit, start_date_str, end_date_str)
        
        current_tf_analysis = {"raw_bar_count": 0, "feature_count": 0, "patterns_detected": [], "support": [], "resistance": [], "trend_strength": "N/A", "classified_pattern": "N/A"}
        if agg_data.get("success") and agg_data.get("results"):
            df_agg = pd.DataFrame(agg_data["results"])
            current_tf_analysis["raw_bar_count"] = len(df_agg)
            if not df_agg.empty:
                df_features = extract_features_from_aggregates(df_agg)
                current_tf_analysis["feature_count"] = len(df_features)
                if not df_features.empty:
                    # Use the last row of features for classification of the most recent state
                    # For classifying segments, one would typically take windows of features.
                    # This is a simplification for demonstrating endpoint.
                    last_features = df_features.iloc[[-1]] 
                    classified_patterns = pattern_classifier.predict(last_features)
                    current_tf_analysis["classified_pattern"] = classified_patterns[0] if classified_patterns else "N/A"
                
                # Simple S/R and trend from raw data (can be enhanced)
                sup, res = identify_simple_support_resistance(df_agg.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"}))
                current_tf_analysis["support"] = sup
                current_tf_analysis["resistance"] = res
                if 'c' in df_agg.columns and len(df_agg) > 50: # Need enough data for SMA
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
    ticker: str; multiplier: int; timespan: str; from_date: str; to_date: str
    adjusted: bool = True; sort: str = "asc"; limit: int = 5000

class PatternAnalysisToolRequest(BaseModel):
    symbol: str; start_date: str; end_date: str
    timeframes: Optional[List[str]] = Field(default=["1day", "1hour"]) # Use Polygon compatible timespans

class CustomTimeframeAggregatesRequest(BaseModel): # Same as AggregatesRequest, for tool distinction
    ticker: str; multiplier: int; timespan: str; from_date: str; to_date: str
    adjusted: bool = True; sort: str = "asc"; limit: int = 5000

# --- FastAPI Server & Endpoints ---
app = FastAPI(title="Polygon REST MCP Server (Production)")

@app.on_event("startup")
async def startup_event():
    logger.info("Polygon REST API server starting up...")
    
    # Check for Polygon API key at startup
    api_key = CONFIG["polygon"]["api_key"]
    if not api_key:
        logger.warning("WARNING: Polygon API key is not configured. Set POLYGON_API_KEY environment variable.")
    else:
        logger.info("Polygon API key is configured.")
        
    await connect_redis()
    await get_http_session()
    
    # Load pre-trained model if available
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
    if http_session and not http_session.closed: await http_session.close()
    if redis_client: await redis_client.close()

@app.get("/server_info")
async def get_server_info():
    return {
        "name": "polygon_rest", "version": "1.2.0", # Version bump
        "description": "MCP Server for Polygon REST API, providing historical data, feature engineering, and pattern analysis.",
        "tools": [
            "fetch_aggregates", 
            "analyze_historical_patterns_with_ml", # Updated analysis tool
            "get_custom_timeframe_aggregates" 
        ],
        "config": {k: v for k, v in CONFIG["polygon"].items() if k != "api_key"}
    }

@app.post("/fetch_aggregates")
async def api_fetch_aggregates(req: AggregatesRequest):
    return await fetch_aggregates(req.ticker, req.multiplier, req.timespan, req.from_date, req.to_date, req.adjusted, req.sort, req.limit)

@app.post("/analyze_historical_patterns_with_ml")
async def api_analyze_historical_patterns_ml(req: PatternAnalysisToolRequest):
    return await analyze_historical_patterns_with_ml(req.symbol, req.start_date, req.end_date, req.timeframes)

@app.post("/get_custom_timeframe_aggregates")
async def api_get_custom_timeframe_aggregates(req: CustomTimeframeAggregatesRequest):
    return await fetch_aggregates(req.ticker, req.multiplier, req.timespan, req.from_date, req.to_date, req.adjusted, req.sort, req.limit)
