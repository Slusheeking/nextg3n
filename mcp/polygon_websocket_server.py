"""
Polygon WebSocket MCP FastAPI Server for LLM integration (production).
Provides real-time market data from Polygon WebSocket API.
All configuration is contained in this file.
"""


import os
import asyncio
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dotenv import load_dotenv
import websockets
import aiohttp
import redis.asyncio as aioredis
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from functools import lru_cache
import yaml
from pathlib import Path
from copy import deepcopy
import logging

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
@lru_cache()
def load_system_config():
    """Load configuration from system_config.yaml with caching."""
    config_path = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'system_config.yaml'))
    default_config = {
        "services": {
            "polygon": {
                "api_key": os.getenv("POLYGON_API_KEY", ""),
                "buffer_size": 1000,
                "websocket_url": "wss://socket.polygon.io/stocks",
                "min_volume": 2000000,
                "min_rel_volume": 1.5,
                "min_price_change": 0.03,
                "min_atr": 0.25,
                "model_dir": "models/pretrained",
                "rate_limit": 60,
                "max_session_time": 14400
            },
            "stock_screening": {
                "update_interval_minutes": 15
            }
        },
        "redis": {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": int(os.getenv("REDIS_DB", "0")),
            "password": os.getenv("REDIS_PASSWORD", None)
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
        return config
    except Exception as e:
        logger.error(f"Failed to load system config: {str(e)}. Using default configuration.")
        return default_config

def get_config():
    """Get merged configuration."""
    config = load_system_config()
    merged_config = {
        "api_key": config["services"]["polygon"]["api_key"],
        "buffer_size": config["services"]["polygon"]["buffer_size"],
        "websocket_url": config["services"]["polygon"]["websocket_url"],
        "redis_host": config["redis"]["host"],
        "redis_port": config["redis"]["port"],
        "redis_db": config["redis"]["db"],
        "redis_password": config["redis"]["password"],
        "min_volume": config["services"]["polygon"]["min_volume"],
        "min_rel_volume": config["services"]["polygon"]["min_rel_volume"],
        "min_price_change": config["services"]["polygon"]["min_price_change"],
        "min_atr": config["services"]["polygon"]["min_atr"],
        "model_dir": config["services"]["polygon"]["model_dir"],
        "rate_limit": config["services"]["polygon"]["rate_limit"],
        "max_session_time": config["services"]["polygon"]["max_session_time"],
        "update_interval_minutes": config["services"]["stock_screening"]["update_interval_minutes"]
    }
    return merged_config

CONFIG = get_config()

logger = get_logger("polygon_websocket_server")
logger.info("Initializing Polygon WebSocket server")

# --- Redis Client ---
redis_client = None

async def connect_redis(max_retries=3, retry_delay=5):
    global redis_client
    for attempt in range(max_retries):
        try:
            redis_client = aioredis.Redis(
                host=CONFIG["redis_host"],
                port=CONFIG["redis_port"],
                db=CONFIG["redis_db"],
                password=CONFIG["redis_password"],
                decode_responses=True
            )
            await redis_client.ping()
            logger.info(f"Connected to Redis at {CONFIG['redis_host']}:{CONFIG['redis_port']} (DB: {CONFIG['redis_db']})")
            return
        except Exception as e:
            logger.error(f"Redis connection attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
    logger.warning("Failed to connect to Redis after retries. Proceeding without Redis.")
    redis_client = None

# --- AI/ML Models ---
class DeepARModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size-1) * dilation,
            dilation=dilation
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.conv1(x)))

class TCNModel(nn.Module):
    def __init__(self, input_dim: int, num_channels: List[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.network(x)
        return self.fc(x[:, :, -1].squeeze())

class InformerModel(nn.Module):
    def __init__(self, enc_in: int, d_model: int, n_heads: int = 8):
        super().__init__()
        self.embedding = nn.Linear(enc_in, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.transpose(0, 1)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.transpose(0, 1)
        return self.fc(attn_output[:, -1, :])

# --- Stock Screening ---
class StockScreener:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_volume = config.get("min_volume", 2000000)
        self.min_rel_volume = config.get("min_rel_volume", 1.5)
        self.min_price_change = config.get("min_price_change", 0.03)
        self.min_atr = config.get("min_atr", 0.25)
        self.models = {}
        self.model_dir = config.get("model_dir", "models/pretrained")
        self._load_models()
    
    def _load_models(self):
        try:
            self.models["deepar"] = DeepARModel(input_dim=10, hidden_dim=64)
            model_path = os.path.join(self.model_dir, "deepar_stocks.pth")
            if os.path.exists(model_path):
                self.models["deepar"].load_state_dict(torch.load(model_path))
                logger.info(f"Loaded DeepAR model from {model_path}")
            self.models["deepar"].eval()
            self.models["tcn"] = TCNModel(input_dim=10, num_channels=[32, 64, 64, 32])
            model_path = os.path.join(self.model_dir, "tcn_anomaly.pth")
            if os.path.exists(model_path):
                self.models["tcn"].load_state_dict(torch.load(model_path))
                logger.info(f"Loaded TCN model from {model_path}")
            self.models["tcn"].eval()
            self.models["informer"] = InformerModel(enc_in=10, d_model=64)
            model_path = os.path.join(self.model_dir, "informer_orderbook.pth")
            if os.path.exists(model_path):
                self.models["informer"].load_state_dict(torch.load(model_path))
                logger.info(f"Loaded Informer model from {model_path}")
            self.models["informer"].eval()
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def preprocess_data(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not data:
            logger.warning("No data provided for preprocessing")
            return {}
        processed = {}
        symbol_data = {}
        for msg in data:
            if "sym" in msg:
                symbol = msg["sym"]
                if symbol not in symbol_data:
                    symbol_data[symbol] = []
                symbol_data[symbol].append(msg)
        for symbol, messages in symbol_data.items():
            trades = [msg for msg in messages if msg.get("ev") == "T"]
            quotes = [msg for msg in messages if msg.get("ev") == "Q"]
            if not trades:
                continue
            prices = [trade.get("p", 0) for trade in trades if trade.get("p", 0) > 0]
            volumes = [trade.get("s", 0) for trade in trades if trade.get("s", 0) > 0]
            if not prices or not volumes:
                continue
            open_price = prices[0]
            high_price = max(prices)
            low_price = min(prices)
            close_price = prices[-1]
            volume = sum(volumes)
            vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes) if sum(volumes) > 0 else 0
            price_change = close_price - open_price
            price_change_pct = price_change / open_price if open_price > 0 else 0
            atr = high_price - low_price
            processed[symbol] = {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "vwap": vwap,
                "price_change": price_change,
                "price_change_pct": abs(price_change_pct),
                "atr": atr,
                "trades": trades,
                "quotes": quotes
            }
        return processed
    
    async def get_historical_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        historical = {}
        try:
            async with aiohttp.ClientSession() as session:
                for symbol in symbols:
                    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
                    params = {"apiKey": CONFIG["api_key"], "adjusted": "true"}
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("results"):
                                result = data["results"][0]
                                historical[symbol] = {
                                    "prev_volume": result.get("v", 0),
                                    "prev_close": result.get("c", 0),
                                    "prev_vwap": result.get("vw", 0)
                                }
                        else:
                            logger.warning(f"Failed to get historical data for {symbol}: {response.status}")
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
        return historical
    
    async def screen_stocks(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not data:
            logger.warning("No data provided for stock screening")
            return []
        processed = self.preprocess_data(data)
        symbols = list(processed.keys())
        historical = await self.get_historical_data(symbols)
        candidates = []
        for symbol, metrics in processed.items():
            if symbol not in historical:
                logger.debug(f"No historical data for {symbol}, skipping")
                continue
            hist = historical[symbol]
            rel_volume = metrics["volume"] / hist["prev_volume"] if hist["prev_volume"] > 0 else 0
            if (metrics["volume"] >= self.min_volume and
                rel_volume >= self.min_rel_volume and
                metrics["price_change_pct"] >= self.min_price_change and
                metrics["atr"] >= self.min_atr):
                prices = [trade.get("p", 0) for trade in metrics["trades"] if trade.get("p", 0) > 0]
                if len(prices) < 14:
                    logger.debug(f"Insufficient price data for {symbol}, skipping RSI/MACD")
                    rsi = 50
                    macd = signal_line = None
                else:
                    price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                    gains = [max(0, change) for change in price_changes[-14:]]
                    losses = [abs(min(0, change)) for change in price_changes[-14:]]
                    avg_gain = sum(gains) / 14 if gains else 0
                    avg_loss = sum(losses) / 14 if losses else 0.001
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    ema12 = self._calculate_ema(prices, 12)
                    ema26 = self._calculate_ema(prices, 26)
                    macd = ema12[-1] - ema26[-1] if ema12 and ema26 else None
                    signal_line = self._calculate_ema([macd] if macd is not None else [0], 9)[0] if macd is not None else None
                sma_20 = sum(prices[-20:]) / 20 if len(prices) >= 20 else None
                sma_50 = sum(prices[-50:]) / 50 if len(prices) >= 50 else None
                candidate = {
                    "symbol": symbol,
                    "last_price": metrics["close"],
                    "volume": metrics["volume"],
                    "rel_volume": rel_volume,
                    "price_change_pct": metrics["price_change_pct"],
                    "atr": metrics["atr"],
                    "rsi": rsi,
                    "macd": macd,
                    "signal_line": signal_line,
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "screening_score": 0.0
                }
                if self.models:
                    try:
                        features = self._extract_features(metrics, hist)
                        with torch.no_grad():
                            deepar_score = self._predict_deepar(features)
                            tcn_score = self._predict_tcn(features)
                            informer_score = self._predict_informer(features)
                            ml_score = (deepar_score + tcn_score + informer_score) / 3
                            candidate["screening_score"] = float(ml_score)
                    except Exception as e:
                        logger.error(f"Error applying ML models for {symbol}: {e}")
                candidates.append(candidate)
        candidates.sort(key=lambda x: x["screening_score"], reverse=True)
        return candidates
    
    def _calculate_ema(self, data: List[float], period: int) -> List[float]:
        if not data:
            return []
        ema = [sum(data[:period]) / period]
        smoothing_constant = 2 / (period + 1)
        for i in range(period, len(data)):
            ema.append((data[i] * smoothing_constant) + (ema[-1] * (1 - smoothing_constant)))
        return ema
    
    def _extract_features(self, metrics: Dict[str, Any], historical: Dict[str, Any]) -> torch.Tensor:
        required_keys = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'price_change', 'atr']
        if not all(key in metrics for key in required_keys) or not all(key in historical for key in ['prev_close', 'prev_volume']):
            logger.error("Missing required features for ML model input")
            return torch.zeros(1, 1, 10, dtype=torch.float32)
        features = [
            metrics["open"], metrics["high"], metrics["low"], metrics["close"],
            metrics["volume"], metrics["vwap"], metrics["price_change"], metrics["atr"],
            historical["prev_close"], historical["prev_volume"]
        ]
        try:
            return torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        except Exception as e:
            logger.error(f"Error creating feature tensor: {e}")
            return torch.zeros(1, 1, 10, dtype=torch.float32)
    
    def _predict_deepar(self, features: torch.Tensor) -> float:
        if "deepar" in self.models:
            output = self.models["deepar"](features)
            return float(torch.sigmoid(output).item())
        return 0.5
    
    def _predict_tcn(self, features: torch.Tensor) -> float:
        if "tcn" in self.models:
            output = self.models["tcn"](features)
            return float(torch.sigmoid(output).item())
        return 0.5
    
    def _predict_informer(self, features: torch.Tensor) -> float:
        if "informer" in self.models:
            output = self.models["informer"](features)
            return float(torch.sigmoid(output).item())
        return 0.5

# --- FastAPI Models ---
class RealtimeDataRequest(BaseModel):
    symbols: List[str]
    channels: List[str] = ["T", "Q"]
    duration_seconds: int = 60
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("At least one symbol must be provided")
        if len(v) > 100:
            raise ValueError("Maximum of 100 symbols allowed")
        return v
    
    @validator('channels')
    def validate_channels(cls, v):
        allowed_channels = ["T", "Q", "A", "AM"]
        for channel in v:
            if channel not in allowed_channels:
                raise ValueError(f"Channel {channel} not supported. Allowed: {allowed_channels}")
        return v

class StartStreamRequest(BaseModel):
    symbols: List[str]
    channels: List[str] = ["T", "Q"]
    session_id: Optional[str] = None
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("At least one symbol must be provided")
        if len(v) > 100:
            raise ValueError("Maximum of 100 symbols allowed")
        return v
    
    @validator('channels')
    def validate_channels(cls, v):
        allowed_channels = ["T", "Q", "A", "AM"]
        for channel in v:
            if channel not in allowed_channels:
                raise ValueError(f"Channel {channel} not supported. Allowed: {allowed_channels}")
        return v

class StopStreamRequest(BaseModel):
    session_id: str

class SnapshotRequest(BaseModel):
    session_id: str
    max_messages: Optional[int] = 1000

class ManualScreenRequest(BaseModel):
    symbols: List[str]
    channels: List[str] = ["T", "Q"]
    duration_seconds: int = 60
    min_volume: Optional[int] = Field(default=None)
    min_rel_volume: Optional[float] = Field(default=None)
    min_price_change: Optional[float] = Field(default=None)
    min_atr: Optional[float] = Field(default=None)
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("At least one symbol must be provided")
        if len(v) > 50:
            raise ValueError("Maximum of 50 symbols allowed for screening")
        return v
    
    @validator('duration_seconds')
    def validate_duration(cls, v):
        if v < 10 or v > 300:
            raise ValueError("Duration must be between 10-300 seconds")
        return v
    
    @validator('min_volume')
    def validate_min_volume(cls, v):
        if v is not None and v < 0:
            raise ValueError("Minimum volume cannot be negative")
        return v
    
    @validator('min_rel_volume')
    def validate_min_rel_volume(cls, v):
        if v is not None and v < 0:
            raise ValueError("Minimum relative volume cannot be negative")
        return v

# --- Streaming Session Management ---
active_streams: Dict[str, Dict[str, Any]] = {}

async def polygon_ws_stream(symbols: List[str], channels: List[str], duration: int, buffer: List[dict]):
    if not symbols or not channels:
        logger.error("Symbols or channels list is empty")
        raise HTTPException(status_code=400, detail="Symbols and channels must not be empty")
    url = CONFIG["websocket_url"]
    api_key = CONFIG["api_key"]
    if not api_key:
        logger.error("Polygon API key not configured")
        raise HTTPException(status_code=500, detail="Polygon API key not configured")
    
    max_retries = 5
    retry_delay = 1
    for attempt in range(max_retries):
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                logger.debug(f"Connected to Polygon WebSocket at {url} (attempt {attempt+1})")
                await ws.send(json.dumps({"action": "auth", "params": api_key}))
                auth_response = await asyncio.wait_for(ws.recv(), timeout=5)
                auth_data = json.loads(auth_response)
                if isinstance(auth_data, list) and auth_data[0].get("status") == "auth_success":
                    logger.info("Authentication with Polygon WebSocket successful")
                else:
                    logger.error(f"Authentication failed: {auth_data}")
                    raise Exception("Authentication failed")
                subscription_params = ",".join(f"{ch}.{sym}" for ch in channels for sym in symbols)
                logger.debug(f"Subscribing to: {subscription_params}")
                await ws.send(json.dumps({"action": "subscribe", "params": subscription_params}))
                start_time = time.time()
                buffer_limit = CONFIG.get("buffer_size", 1000)
                while time.time() - start_time < duration:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        if len(buffer) >= buffer_limit:
                            del buffer[:len(buffer) - buffer_limit + 1]
                        buffer.append(data)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed unexpectedly")
                        break
                await ws.send(json.dumps({"action": "unsubscribe", "params": subscription_params}))
                logger.debug("Sent unsubscribe request to Polygon WebSocket")
                break
        except Exception as e:
            logger.error(f"WebSocket connection error (attempt {attempt+1}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Retrying connection in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed to connect after {max_retries} attempts")
                raise

# --- FastAPI Server ---
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code (before yield)
    logger.info("Polygon WebSocket server starting up")
    await connect_redis()
    if redis_client:
        try:
            ping_result = await redis_client.ping()
            logger.info(f"Redis connection test: {ping_result}")
        except Exception as e:
            logger.error(f"Redis connection test failed: {str(e)}")
    else:
        logger.warning("Redis client not available - some features will be limited")
    if not CONFIG["api_key"]:
        logger.warning("Polygon API key not configured - service will not function correctly")
    screening_task = asyncio.create_task(periodic_screening())
    await rate_limiter.start_cleanup_task()
    
    yield  # This line separates startup from shutdown code
    
    # Shutdown code (after yield)
    logger.info("Polygon WebSocket server shutting down")
    for session_id, stream_data in list(active_streams.items()):
        logger.info(f"Cancelling active stream {session_id}")
        try:
            stream_data["task"].cancel()
        except Exception as e:
            logger.error(f"Error cancelling stream {session_id}: {str(e)}")
    if redis_client:
        await redis_client.close()

app = FastAPI(
    title="Polygon WebSocket MCP Server for LLM (Production)",
    description="Provides real-time market data and stock screening through Polygon WebSocket API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Server Info", "description": "Endpoints for server information and health checks"},
        {"name": "Real-time Data", "description": "Endpoints for accessing real-time market data"},
        {"name": "Stock Screening", "description": "Endpoints for stock screening and analysis"},
        {"name": "Streaming", "description": "Endpoints for managing WebSocket streaming sessions"}
    ],
    lifespan=lifespan
)

class RateLimiter:
    def __init__(self, rate_limit_per_minute: int = 60):
        self.rate_limit = rate_limit_per_minute
        self.redis_key_prefix = "rate_limit"
    
    async def check_rate_limit(self, request: Request):
        if not redis_client:
            logger.warning("Redis unavailable, skipping rate limiting")
            return True
        client_ip = request.client.host
        key = f"{self.redis_key_prefix}:{client_ip}"
        now = int(time.time())
        window = 60
        try:
            async with redis_client.pipeline() as pipe:
                pipe.zremrangebyscore(key, 0, now - window)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, window)
                _, _, count, _ = await pipe.execute()
            if count > self.rate_limit:
                logger.warning(f"Rate limit exceeded for IP {client_ip}: {count} requests in last minute")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking rate limit in Redis: {e}")
            return True
    
    async def start_cleanup_task(self):
        pass

rate_limiter = RateLimiter(CONFIG.get("rate_limit", 60))
startup_time = time.time()
stock_screener = StockScreener(CONFIG)

async def check_rate_limit(request: Request):
    """Middleware function for rate limiting API endpoints.
    Uses the RateLimiter class to implement rate limiting logic."""
    result = await rate_limiter.check_rate_limit(request)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    return result

async def periodic_screening():
    while True:
        try:
            config = get_config()
            screening_interval = config.get("update_interval_minutes", 15)
            if not redis_client:
                logger.warning("Redis unavailable, skipping periodic screening")
                await asyncio.sleep(screening_interval * 60)
                continue
            custom_symbols = await redis_client.get("polygon:watchlist")
            if not custom_symbols:
                logger.warning("No custom watchlist found in Redis, skipping periodic screening")
                await asyncio.sleep(screening_interval * 60)
                continue
            symbols_to_screen = json.loads(custom_symbols)
            logger.info(f"Periodic screening for {len(symbols_to_screen)} symbols")
            req = RealtimeDataRequest(
                symbols=symbols_to_screen[:50],
                channels=["T", "Q"],
                duration_seconds=60
            )
            await api_screen_stocks(req, BackgroundTasks())
            logger.info(f"Periodic stock screening completed. Next run in {screening_interval} minutes.")
            await asyncio.sleep(screening_interval * 60)
        except Exception as e:
            logger.error(f"Error in periodic screening: {str(e)}")
            await asyncio.sleep(300)

@app.get("/server_info", tags=["Server Info"])
async def get_server_info():
    safe_config = {k: v for k, v in CONFIG.items() if k != "api_key"}
    return {
        "name": "polygon_websocket",
        "version": "1.0.0",
        "description": "Production MCP Server for Polygon WebSocket Integration",
        "tools": [
            "fetch_realtime_data", "start_stream", "stop_stream", "get_snapshot",
            "screen_stocks", "get_screened_stocks", "manual_screen_stocks"
        ],
        "config": safe_config,
        "endpoints": [
            "/fetch_realtime_data", "/start_stream", "/stop_stream", "/get_snapshot",
            "/screen_stocks", "/get_screened_stocks", "/manual_screen_stocks"
        ],
        "ai_models": list(stock_screener.models.keys()),
        "screening_criteria": {
            "min_volume": CONFIG["min_volume"],
            "min_rel_volume": CONFIG["min_rel_volume"],
            "min_price_change": CONFIG["min_price_change"],
            "min_atr": CONFIG["min_atr"]
        }
    }

@app.post("/fetch_realtime_data", tags=["Real-time Data"], dependencies=[Depends(check_rate_limit)])
async def api_fetch_realtime_data(req: RealtimeDataRequest):
    logger.info(f"Received realtime data request for symbols: {req.symbols}")
    try:
        buffer = []
        await polygon_ws_stream(req.symbols, req.channels, req.duration_seconds, buffer)
        logger.info(f"Completed realtime data request, received {len(buffer)} messages")
        return {
            "success": True,
            "data": buffer,
            "symbols": req.symbols,
            "channels": req.channels,
            "start_time": datetime.utcnow().isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "duration_seconds": req.duration_seconds
        }
    except Exception as e:
        logger.error(f"Error processing realtime data request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching realtime data: {str(e)}")

@app.post("/screen_stocks", tags=["Stock Screening"], dependencies=[Depends(check_rate_limit)])
async def api_screen_stocks(req: RealtimeDataRequest, background_tasks: BackgroundTasks):
    logger.info(f"Screening stocks for symbols: {req.symbols}")
    try:
        buffer = []
        await polygon_ws_stream(req.symbols, req.channels, req.duration_seconds, buffer)
        logger.info(f"Received {len(buffer)} messages for screening")
        candidates = await stock_screener.screen_stocks(buffer)
        logger.info(f"Found {len(candidates)} stock candidates")
        if redis_client and candidates:
            background_tasks.add_task(store_candidates_in_redis, candidates)
        return {
            "success": True,
            "candidates": candidates,
            "count": len(candidates),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error screening stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error screening stocks: {str(e)}")

@app.get("/get_screened_stocks", tags=["Stock Screening"], dependencies=[Depends(check_rate_limit)])
async def api_get_screened_stocks(source: str = "polygon_websocket", date: str = None):
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Getting screened stocks for date {date} from source {source}")
    try:
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis client not available")
        key = f"stock_pool:{date}:{source}"
        value = await redis_client.get(key)
        if value:
            data = json.loads(value)
            logger.info(f"Retrieved {len(data.get('candidates', []))} screened stocks")
            return {
                "success": True,
                "date": date,
                "source": source,
                "data": data
            }
        else:
            logger.info(f"No screened stocks found for date {date} from source {source}")
            return {
                "success": False,
                "date": date,
                "source": source,
                "data": None
            }
    except Exception as e:
        logger.error(f"Error getting screened stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving screened stocks: {str(e)}")

async def cleanup_inactive_sessions():
    if not redis_client:
        logger.warning("Redis unavailable, skipping session cleanup")
        return
    try:
        while True:
            await asyncio.sleep(600)
            current_time = time.time()
            inactive_threshold = 3600
            sessions = await redis_client.keys("session:*")
            for session_key in sessions:
                session_data = await redis_client.get(session_key)
                if session_data:
                    session = json.loads(session_data)
                    last_accessed = session.get("last_accessed", 0)
                    if current_time - last_accessed > inactive_threshold:
                        logger.info(f"Removing inactive session {session_key}")
                        await redis_client.delete(session_key)
    except asyncio.CancelledError:
        logger.info("Session cleanup task cancelled")
    except Exception as e:
        logger.error(f"Error in session cleanup task: {str(e)}")

@app.get("/active_sessions", tags=["Streaming"])
async def api_get_active_sessions():
    try:
        sessions = []
        if redis_client:
            session_keys = await redis_client.keys("session:*")
            for key in session_keys:
                session_data = await redis_client.get(key)
                if session_data:
                    session = json.loads(session_data)
                    sessions.append({
                        "session_id": key.replace("session:", ""),
                        "symbols": session["symbols"],
                        "channels": session["channels"],
                        "start_time": session["start_time"],
                        "last_accessed": datetime.fromtimestamp(session.get("last_accessed", 0)).isoformat()
                    })
        return {
            "success": True,
            "count": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"Error getting active sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving active sessions: {str(e)}")

@app.get("/health", tags=["Server Info"])
async def api_health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "redis_connected": False,
        "active_streams": len(active_streams)
    }
    if redis_client:
        try:
            ping_result = await redis_client.ping()
            health_status["redis_connected"] = ping_result
        except Exception as e:
            logger.warning(f"Redis health check failed: {str(e)}")
            health_status["redis_error"] = str(e)
    health_status["api_key_configured"] = bool(CONFIG["api_key"])
    return health_status

@app.get("/screened_candidates", tags=["Stock Screening"], dependencies=[Depends(check_rate_limit)])
async def api_screened_candidates(source: str = "polygon_websocket", date: str = None):
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Getting screened candidates for date {date} from source {source}")
    try:
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis client not available")
        key = f"stock_pool:{date}:{source}"
        value = await redis_client.get(key)
        if value:
            data = json.loads(value)
            candidates = data.get("candidates", [])
            logger.info(f"Retrieved {len(candidates)} screened candidates")
            return {
                "success": True,
                "date": date,
                "source": source,
                "candidates": candidates
            }
        else:
            logger.info(f"No screened candidates found for date {date} from source {source}")
            return {
                "success": False,
                "date": date,
                "source": source,
                "candidates": []
            }
    except Exception as e:
        logger.error(f"Error getting screened candidates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving candidates: {str(e)}")

@app.post("/manual_screen_stocks", tags=["Stock Screening"], dependencies=[Depends(check_rate_limit)])
async def api_manual_screen_stocks(req: ManualScreenRequest, background_tasks: BackgroundTasks):
    logger.info(f"Manually screening stocks for symbols: {req.symbols}")
    if not CONFIG["api_key"]:
        raise HTTPException(status_code=503, detail="Polygon API service not available due to missing API key")
    try:
        buffer = []
        await polygon_ws_stream(req.symbols, req.channels, req.duration_seconds, buffer)
        logger.info(f"Received {len(buffer)} messages for manual screening")
        if not buffer:
            logger.warning("No data received from WebSocket, cannot perform screening")
            return JSONResponse(
                status_code=204,
                content={
                    "success": False,
                    "message": "No market data received from Polygon API",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        custom_config = CONFIG.copy()
        if req.min_volume is not None:
            custom_config["min_volume"] = req.min_volume
        if req.min_rel_volume is not None:
            custom_config["min_rel_volume"] = req.min_rel_volume
        if req.min_price_change is not None:
            custom_config["min_price_change"] = req.min_price_change
        if req.min_atr is not None:
            custom_config["min_atr"] = req.min_atr
        stock_screener_custom = StockScreener(custom_config)
        candidates = await stock_screener_custom.screen_stocks(buffer)
        logger.info(f"Found {len(candidates)} stock candidates after manual screening")
        if redis_client and candidates:
            background_tasks.add_task(
                store_custom_candidates_in_redis,
                candidates,
                "manual_screening",
                f"Custom params: vol>{req.min_volume or custom_config['min_volume']}, " +
                f"rel_vol>{req.min_rel_volume or custom_config['min_rel_volume']}, " +
                f"price_chg>{req.min_price_change or custom_config['min_price_change']}"
            )
        return {
            "success": True,
            "candidates": candidates,
            "count": len(candidates),
            "timestamp": datetime.utcnow().isoformat(),
            "parameters": {
                "min_volume": req.min_volume or custom_config["min_volume"],
                "min_rel_volume": req.min_rel_volume or custom_config["min_rel_volume"],
                "min_price_change": req.min_price_change or custom_config["min_price_change"],
                "min_atr": req.min_atr or custom_config["min_atr"]
            }
        }
    except Exception as e:
        logger.error(f"Error manually screening stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing manual screening request: {str(e)}")

@app.post("/start_stream", tags=["Streaming"], dependencies=[Depends(check_rate_limit)])
async def api_start_stream(req: StartStreamRequest):
    if not CONFIG["api_key"]:
        logger.error("Polygon API key not configured")
        raise HTTPException(status_code=500, detail="Polygon API key not configured")
    if len(req.symbols) > 100:
        logger.warning(f"Too many symbols requested: {len(req.symbols)}, limiting to 100")
        req.symbols = req.symbols[:100]
    session_id = req.session_id or f"stream_{int(time.time())}"
    logger.info(f"Starting stream session {session_id} for symbols: {req.symbols}")
    if redis_client:
        session_key = f"session:{session_id}"
        if await redis_client.exists(session_key):
            logger.warning(f"Session ID {session_id} already exists")
            raise HTTPException(status_code=400, detail=f"Session ID {session_id} already exists")
    buffer = []
    max_session_time = 3600 * 4
    task = asyncio.create_task(polygon_ws_stream(req.symbols, req.channels, max_session_time, buffer))
    session_data = {
        "task": task,
        "buffer": buffer,
        "symbols": req.symbols,
        "channels": req.channels,
        "start_time": datetime.utcnow().isoformat(),
        "last_accessed": time.time()
    }
    if redis_client:
        await redis_client.set(session_key, json.dumps({
            "symbols": req.symbols,
            "channels": req.channels,
            "start_time": session_data["start_time"],
            "last_accessed": session_data["last_accessed"]
        }))
    active_streams[session_id] = session_data
    asyncio.create_task(cleanup_inactive_sessions())
    logger.info(f"Stream session {session_id} started successfully")
    return {
        "success": True,
        "session_id": session_id,
        "symbols": req.symbols,
        "channels": req.channels,
        "start_time": session_data["start_time"],
        "max_session_time": f"{max_session_time//3600} hours"
    }

@app.post("/stop_stream", tags=["Streaming"], dependencies=[Depends(check_rate_limit)])
async def api_stop_stream(req: StopStreamRequest):
    session_id = req.session_id
    logger.info(f"Stopping stream session {session_id}")
    if session_id not in active_streams:
        logger.warning(f"Session ID {session_id} not found")
        raise HTTPException(status_code=404, detail=f"Session ID {session_id} not found")
    task = active_streams[session_id]["task"]
    task.cancel()
    if redis_client:
        await redis_client.delete(f"session:{session_id}")
    del active_streams[session_id]
    logger.info(f"Stream session {session_id} stopped successfully")
    return {
        "success": True,
        "session_id": session_id,
        "message": f"Stream {session_id} stopped successfully"
    }

@app.post("/get_snapshot", tags=["Streaming"], dependencies=[Depends(check_rate_limit)])
async def api_get_snapshot(req: SnapshotRequest):
    session_id = req.session_id
    logger.info(f"Getting snapshot for stream session {session_id}")
    if session_id not in active_streams:
        logger.warning(f"Session ID {session_id} not found")
        raise HTTPException(status_code=404, detail=f"Session ID {session_id} not found")
    buffer = active_streams[session_id]["buffer"]
    active_streams[session_id]["last_accessed"] = time.time()
    max_messages = req.max_messages or 1000
    if len(buffer) > max_messages:
        logger.info(f"Limiting snapshot response to {max_messages} messages (total: {len(buffer)})")
        response_data = buffer[-max_messages:]
    else:
        response_data = buffer
    if redis_client:
        await redis_client.set(f"session:{session_id}", json.dumps({
            "symbols": active_streams[session_id]["symbols"],
            "channels": active_streams[session_id]["channels"],
            "start_time": active_streams[session_id]["start_time"],
            "last_accessed": active_streams[session_id]["last_accessed"]
        }))
    logger.info(f"Returning snapshot with {len(response_data)} messages for session {session_id}")
    return {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
        "data": response_data,
        "total_messages": len(buffer),
        "returned_messages": len(response_data),
        "session_id": session_id
    }

@app.get("/resource/{resource_uri:path}", tags=["Server Info"], dependencies=[Depends(check_rate_limit)])
async def get_resource(resource_uri: str):
    logger.info(f"Resource request for URI: {resource_uri}")
    try:
        if resource_uri == "streams/active":
            sessions = []
            if redis_client:
                session_keys = await redis_client.keys("session:*")
                for key in session_keys:
                    session_data = await redis_client.get(key)
                    if session_data:
                        session = json.loads(session_data)
                        sessions.append({
                            "session_id": key.replace("session:", ""),
                            "symbols": session["symbols"],
                            "channels": session["channels"],
                            "start_time": session["start_time"],
                            "buffer_size": len(active_streams.get(key.replace("session:", ""), {}).get("buffer", []))
                        })
            return {
                "active_streams": sessions,
                "count": len(sessions),
                "timestamp": datetime.utcnow().isoformat()
            }
        elif resource_uri == "status":
            return {
                "status": "operational",
                "timestamp": datetime.utcnow().isoformat(),
                "api_key_configured": bool(CONFIG["api_key"]),
                "redis_connected": bool(redis_client)
            }
        elif resource_uri == "metrics":
            return {
                "active_streams": len(active_streams),
                "api_key_configured": bool(CONFIG["api_key"]),
                "redis_connected": bool(redis_client),
                "uptime_seconds": time.time() - startup_time,
                "screening_criteria": {
                    "min_volume": CONFIG["min_volume"],
                    "min_rel_volume": CONFIG["min_rel_volume"],
                    "min_price_change": CONFIG["min_price_change"]
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        elif resource_uri == "version":
            return {
                "version": "1.0.0",
                "build_date": "2025-05-07",
                "api_compatibility": "v1",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            logger.warning(f"Unknown resource URI requested: {resource_uri}")
            raise HTTPException(status_code=404, detail=f"Unknown resource URI: {resource_uri}")
    except Exception as e:
        logger.error(f"Error in get_resource: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving resource: {str(e)}")

async def store_candidates_in_redis(candidates: List[Dict[str, Any]]):
    if not redis_client:
        logger.error("Cannot store candidates: Redis client is not available")
        return
    try:
        date = datetime.now().strftime("%Y-%m-%d")
        source = "polygon_websocket"
        key = f"stock_pool:{date}:{source}"
        value = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "candidates": candidates
        }
        await redis_client.set(key, json.dumps(value))
        await redis_client.expire(key, 60 * 60 * 24 * 7)
        latest_key = "latest_screened_stocks"
        await redis_client.set(latest_key, json.dumps(value))
        await redis_client.expire(latest_key, 60 * 60 * 24)
        logger.info(f"Stored {len(candidates)} stock candidates in Redis with key {key}")
    except Exception as e:
        logger.error(f"Error storing candidates in Redis: {str(e)}")

async def store_custom_candidates_in_redis(candidates: List[Dict[str, Any]], source_tag: str, description: str = ""):
    if not redis_client:
        logger.error("Cannot store candidates: Redis client is not available")
        return
    try:
        date = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now()
        key = f"stock_pool:{date}:{source_tag}:{int(timestamp.timestamp())}"
        value = {
            "timestamp": timestamp.isoformat(),
            "source": source_tag,
            "description": description,
            "candidates": candidates
        }
        await redis_client.set(key, json.dumps(value))
        await redis_client.expire(key, 60 * 60 * 24 * 3)
        await redis_client.sadd("custom_screenings", key)
        logger.info(f"Stored {len(candidates)} custom stock candidates in Redis with key {key}")
    except Exception as e:
        logger.error(f"Error storing custom candidates in Redis: {str(e)}")