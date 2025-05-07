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
from datetime import datetime
from dotenv import load_dotenv
from monitor.logging_utils import get_logger
import websockets
import aiohttp
import redis.asyncio as aioredis
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from functools import lru_cache
import yaml

# --- Configuration ---

# Load environment variables from .env file
load_dotenv()

@lru_cache()
def load_system_config():
    """Load configuration from system_config.yaml file with caching."""
    try:
        config_path = os.path.join(os.environ.get('CONFIG_DIR', '/home/ubuntu/nextg3n/config'), 'system_config.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load system config: {str(e)}")
        return {}

def get_config():
    """Get merged configuration from environment variables and system config."""
    # Base configuration from environment variables
    config = {
        "api_key": os.getenv("POLYGON_API_KEY", ""),
        "buffer_size": int(os.getenv("POLYGON_BUFFER_SIZE", "1000")),
        "websocket_url": os.getenv("POLYGON_WS_URL", "wss://socket.polygon.io/stocks"),
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": int(os.getenv("REDIS_PORT", "6379")),
        "redis_db": int(os.getenv("REDIS_DB", "0")),
        "redis_password": os.getenv("REDIS_PASSWORD", None),
        "min_volume": 2000000,  # 2M shares
        "min_rel_volume": 1.5,  # 1.5x average
        "min_price_change": 0.03,  # 3% change from open
        "min_atr": 0.25,  # Minimum ATR for volatility
        "model_dir": os.getenv("MODEL_DIR", "models/pretrained"),
        "rate_limit": int(os.getenv("API_RATE_LIMIT", "60")),  # requests per minute
        "max_session_time": int(os.getenv("MAX_SESSION_TIME", "14400")),  # 4 hours in seconds
    }
    
    # Override with system config values if available
    system_config = load_system_config()
    if system_config:
        polygon_config = system_config.get("services", {}).get("polygon", {})
        if polygon_config:
            for key, value in polygon_config.items():
                if key in config:
                    config[key] = value
    
    return config

CONFIG = get_config()

# Get logger from centralized logging system
logger = get_logger("polygon_websocket_server")
logger.info("Initializing Polygon WebSocket server")

# --- Redis Client ---
try:
    redis_client = aioredis.Redis(
        host=CONFIG["redis_host"],
        port=CONFIG["redis_port"],
        db=CONFIG["redis_db"],
        password=CONFIG["redis_password"],
        decode_responses=True
    )
    logger.info(f"Connected to Redis at {CONFIG['redis_host']}:{CONFIG['redis_port']} (DB: {CONFIG['redis_db']})")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")
    redis_client = None

# --- AI/ML Models ---

class DeepARModel(nn.Module):
    """DeepAR model for time series forecasting."""
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
    """Temporal Convolutional Network block."""
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
    """Temporal Convolutional Network for anomaly detection."""
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
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        x = self.network(x)
        # Take the last output
        return self.fc(x[:, :, -1].squeeze())

class InformerModel(nn.Module):
    """Informer model for efficient attention on long sequences."""
    def __init__(self, enc_in: int, d_model: int, n_heads: int = 8):
        super().__init__()
        self.embedding = nn.Linear(enc_in, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features]
        x = self.embedding(x)  # [batch, seq_len, d_model]
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.transpose(0, 1)  # [batch, seq_len, d_model]
        return self.fc(attn_output[:, -1, :])

# --- Stock Screening ---

class StockScreener:
    """Stock screener that uses AI/ML models to identify trading opportunities."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_volume = config.get("min_volume", 2000000)
        self.min_rel_volume = config.get("min_rel_volume", 1.5)
        self.min_price_change = config.get("min_price_change", 0.03)
        self.min_atr = config.get("min_atr", 0.25)
        
        # Initialize models
        self.models = {}
        self.model_dir = config.get("model_dir", "models/pretrained")
        
        # Load models if available
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if available."""
        try:
            # DeepAR model
            self.models["deepar"] = DeepARModel(input_dim=10, hidden_dim=64)
            model_path = os.path.join(self.model_dir, "deepar_stocks.pth")
            if os.path.exists(model_path):
                self.models["deepar"].load_state_dict(torch.load(model_path))
                logger.info(f"Loaded DeepAR model from {model_path}")
            self.models["deepar"].eval()
            
            # TCN model
            self.models["tcn"] = TCNModel(input_dim=10, num_channels=[32, 64, 64, 32])
            model_path = os.path.join(self.model_dir, "tcn_anomaly.pth")
            if os.path.exists(model_path):
                self.models["tcn"].load_state_dict(torch.load(model_path))
                logger.info(f"Loaded TCN model from {model_path}")
            self.models["tcn"].eval()
            
            # Informer model
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
        """
        Preprocess raw WebSocket data into features for screening.
        
        Args:
            data: List of WebSocket messages
            
        Returns:
            Dictionary of processed data by symbol
        """
        processed = {}
        
        # Group data by symbol
        symbol_data = {}
        for msg in data:
            if "sym" in msg:
                symbol = msg["sym"]
                if symbol not in symbol_data:
                    symbol_data[symbol] = []
                symbol_data[symbol].append(msg)
        
        # Process each symbol
        for symbol, messages in symbol_data.items():
            # Extract trades
            trades = [msg for msg in messages if msg.get("ev") == "T"]
            quotes = [msg for msg in messages if msg.get("ev") == "Q"]
            
            if not trades:
                continue
                
            # Calculate basic metrics
            prices = [trade.get("p", 0) for trade in trades]
            volumes = [trade.get("s", 0) for trade in trades]
            
            if not prices or not volumes:
                continue
                
            # Calculate OHLC
            open_price = prices[0]
            high_price = max(prices)
            low_price = min(prices)
            close_price = prices[-1]
            
            # Calculate volume
            volume = sum(volumes)
            
            # Calculate VWAP
            vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes) if sum(volumes) > 0 else 0
            
            # Calculate price change
            price_change = close_price - open_price
            price_change_pct = price_change / open_price if open_price > 0 else 0
            
            # Calculate ATR (simplified)
            atr = high_price - low_price
            
            # Store processed data
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
        """
        Get historical data for symbols to calculate relative volume and other metrics.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary of historical data by symbol
        """
        historical = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                for symbol in symbols:
                    # Get previous day's data from Polygon REST API
                    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
                    params = {
                        "apiKey": CONFIG["api_key"],
                        "adjusted": "true"
                    }
                    
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
        """
        Screen stocks based on real-time data and AI/ML models.
        
        Args:
            data: List of WebSocket messages
            
        Returns:
            List of screened stock candidates
        """
        # Preprocess data
        processed = self.preprocess_data(data)
        
        # Get symbols
        symbols = list(processed.keys())
        
        # Get historical data
        historical = await self.get_historical_data(symbols)
        
        # Screen stocks
        candidates = []
        for symbol, metrics in processed.items():
            # Skip if we don't have historical data
            if symbol not in historical:
                continue
            
            # Calculate relative volume
            hist = historical[symbol]
            rel_volume = metrics["volume"] / hist["prev_volume"] if hist["prev_volume"] > 0 else 0
            
            # Apply basic screening criteria
            if (metrics["volume"] >= self.min_volume and
                rel_volume >= self.min_rel_volume and
                metrics["price_change_pct"] >= self.min_price_change and
                metrics["atr"] >= self.min_atr):
                
                # Calculate RSI (simplified)
                price_changes = [metrics["trades"][i]["p"] - metrics["trades"][i-1]["p"] for i in range(1, len(metrics["trades"]))]
                if price_changes:
                    gains = [max(0, change) for change in price_changes]
                    losses = [abs(min(0, change)) for change in price_changes]
                    avg_gain = sum(gains) / len(gains) if gains else 0
                    avg_loss = sum(losses) / len(losses) if losses else 0.001  # Avoid division by zero
                    rs = avg_gain / avg_loss if avg_loss > 0 else 0
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 50  # Default to neutral
                    
                # Get prices from trades for calculations
                prices = [trade.get("p", 0) for trade in metrics["trades"]]
                
                # Calculate MACD (simplified)
                ema_12 = self._calculate_ema(prices, 12)
                ema_26 = self._calculate_ema(prices, 26)
                macd = ema_12 - ema_26
                signal_line = self._calculate_ema([macd], 9)[0] if macd is not None else None

                # Calculate Simple Moving Averages (SMA)
                sma_20 = sum(prices[-20:]) / 20 if len(prices) >= 20 else None
                sma_50 = sum(prices[-50:]) / 50 if len(prices) >= 50 else None
                
                # Prepare candidate
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
                    "screening_score": 0.0  # Will be updated with ML model scores
                }
                
                # Apply ML models if available
                if self.models:
                    try:
                        # Prepare input for models
                        features = self._extract_features(metrics, hist)
                        
                        # Get model predictions
                        with torch.no_grad():
                            deepar_score = self._predict_deepar(features)
                            tcn_score = self._predict_tcn(features)
                            informer_score = self._predict_informer(features)
                            
                            # Combine scores
                            ml_score = (deepar_score + tcn_score + informer_score) / 3
                            candidate["screening_score"] = float(ml_score)
                    except Exception as e:
                        logger.error(f"Error applying ML models: {str(e)}")
                
                candidates.append(candidate)
        
        # Sort by screening score
        candidates.sort(key=lambda x: x["screening_score"], reverse=True)
        
        return candidates

    def _calculate_ema(self, data: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average (EMA)."""
        if not data:
            return []
        
        ema = [sum(data[:period]) / period]  # Initialize with SMA
        smoothing_constant = 2 / (period + 1)
        
        for i in range(period, len(data)):
            ema.append((data[i] * smoothing_constant) + (ema[-1] * (1 - smoothing_constant)))
        
        return ema
    
    def _extract_features(self, metrics: Dict[str, Any], historical: Dict[str, Any]) -> torch.Tensor:
        """Extract features for ML models."""
        # Basic features
        features = [
            metrics["open"],
            metrics["high"],
            metrics["low"],
            metrics["close"],
            metrics["volume"],
            metrics["vwap"],
            metrics["price_change"],
            metrics["atr"],
            historical["prev_close"],
            historical["prev_volume"]
        ]
        
        # Convert to tensor
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    def _predict_deepar(self, features: torch.Tensor) -> float:
        """Get prediction from DeepAR model."""
        if "deepar" in self.models:
            output = self.models["deepar"](features)
            return float(torch.sigmoid(output).item())
        return 0.5
    
    def _predict_tcn(self, features: torch.Tensor) -> float:
        """Get prediction from TCN model."""
        if "tcn" in self.models:
            output = self.models["tcn"](features)
            return float(torch.sigmoid(output).item())
        return 0.5
    
    def _predict_informer(self, features: torch.Tensor) -> float:
        """Get prediction from Informer model."""
        if "informer" in self.models:
            output = self.models["informer"](features)
            return float(torch.sigmoid(output).item())
        return 0.5

# --- FastAPI Models ---

class RealtimeDataRequest(BaseModel):
    symbols: List[str]
    channels: List[str] = ["T", "Q"]
    duration_seconds: int = 60

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

# --- Manual Screening ---

class ManualScreenRequest(BaseModel):
    symbols: List[str]
    channels: List[str] = ["T", "Q"]
    duration_seconds: int = 60
    min_volume: Optional[int] = Field(default=None, description="Minimum volume")
    min_rel_volume: Optional[float] = Field(default=None, description="Minimum relative volume")
    min_price_change: Optional[float] = Field(default=None, description="Minimum price change")
    min_atr: Optional[float] = Field(default=None, description="Minimum ATR")
    
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
    url = CONFIG["websocket_url"]
    api_key = CONFIG["api_key"]
    if not api_key:
        logger.error("Polygon API key not configured")
        raise HTTPException(status_code=500, detail="Polygon API key not configured.")
    
    # Implement exponential backoff for connection retries
    max_retries = 5
    retry_delay = 1  # Start with 1 second delay
    
    for attempt in range(max_retries):
        try:
            async with websockets.connect(url) as ws:
                logger.debug(f"Connecting to Polygon WebSocket at {url} (attempt {attempt+1})")
                
                # Handle authentication
                await ws.send(json.dumps({"action": "auth", "params": api_key}))
                auth_response = await asyncio.wait_for(ws.recv(), timeout=5)
                auth_data = json.loads(auth_response)
                
                # Check if authentication was successful
                if isinstance(auth_data, list) and auth_data[0].get("status") == "auth_success":
                    logger.info("Authentication with Polygon WebSocket successful")
                else:
                    logger.error(f"Authentication failed: {auth_data}")
                    raise Exception("Authentication with Polygon WebSocket failed")
                
                # Subscribe to channels
                subscription_params = ",".join(f"{ch}.{sym}" for ch in channels for sym in symbols)
                logger.debug(f"Subscribing to: {subscription_params}")
                await ws.send(json.dumps({"action": "subscribe", "params": subscription_params}))
                
                # Process messages
                start_time = time.time()
                buffer_limit = CONFIG.get("buffer_size", 1000)
                
                while time.time() - start_time < duration:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)  # 30 second timeout for receiving messages
                        data = json.loads(msg)
                        
                        # Manage buffer size to prevent memory issues
                        if len(buffer) >= buffer_limit:
                            # Remove older items to make room
                            del buffer[:len(buffer) - buffer_limit + 1]
                            
                        buffer.append(data)
                    except asyncio.TimeoutError:
                        # Just a timeout on receive, not fatal
                        continue
                    except Exception as e:
                        logger.error(f"WebSocket message processing error: {e}")
                        break
                
                # Clean disconnect
                try:
                    await ws.send(json.dumps({"action": "unsubscribe", "params": subscription_params}))
                    logger.debug("Sent unsubscribe request to Polygon WebSocket")
                except:
                    pass
                
                break  # Success, exit retry loop
                
        except Exception as e:
            logger.error(f"WebSocket connection error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying connection in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed to connect after {max_retries} attempts")
                raise

# --- FastAPI Server ---

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
    ]
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting implementation
class RateLimiter:
    def __init__(self, rate_limit_per_minute: int = 60):
        self.rate_limit = rate_limit_per_minute
        self.requests = {}
        self.clean_task = None
    
    async def start_cleanup_task(self):
        """Start periodic cleanup of old request records"""
        self.clean_task = asyncio.create_task(self._cleanup_old_requests())
    
    async def _cleanup_old_requests(self):
        """Remove request records older than 2 minutes"""
        while True:
            try:
                current_time = time.time()
                for ip, requests in list(self.requests.items()):
                    self.requests[ip] = [r for r in requests if current_time - r < 120]
                    if not self.requests[ip]:
                        del self.requests[ip]
                await asyncio.sleep(60)  # Clean every minute
            except Exception as e:
                logger.error(f"Error in rate limiter cleanup: {str(e)}")
                await asyncio.sleep(60)
    
    async def check_rate_limit(self, request: Request):
        """Check if request is within rate limits"""
        client_ip = request.client.host
        current_time = time.time()
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Add current request timestamp
        self.requests[client_ip].append(current_time)
        
        # Count requests in the last minute
        minute_ago = current_time - 60
        recent_requests = [r for r in self.requests[client_ip] if r > minute_ago]
        self.requests[client_ip] = recent_requests
        
        # Check limit
        if len(recent_requests) > self.rate_limit:
            logger.warning(f"Rate limit exceeded for IP {client_ip}: {len(recent_requests)} requests in last minute")
            return False
        return True

rate_limiter = RateLimiter(CONFIG.get("rate_limit", 60))

# Dependency for rate limiting
async def check_rate_limit(request: Request):
    if not await rate_limiter.check_rate_limit(request):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )

# Global variable for startup time
startup_time = time.time()

# --- Initialize Stock Screener ---
stock_screener = StockScreener(CONFIG)

@app.on_event("startup")
async def startup_event():
    logger.info("Polygon WebSocket server starting up")
    # Test Redis connection
    if redis_client:
        try:
            ping_result = await redis_client.ping()
            logger.info(f"Redis connection test: {ping_result}")
        except Exception as e:
            logger.error(f"Redis connection test failed: {str(e)}")
    else:
        logger.warning("Redis client not available - some features will be limited")
    
    # Check for API key
    if not CONFIG["api_key"]:
        logger.warning("Polygon API key not configured - service will not function correctly")
    
    # Start background tasks
    asyncio.create_task(periodic_screening())
    await rate_limiter.start_cleanup_task()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Polygon WebSocket server shutting down")
    
    # Cancel all active streams
    for session_id, stream_data in list(active_streams.items()):
        logger.info(f"Cancelling active stream {session_id}")
        try:
            stream_data["task"].cancel()
        except Exception as e:
            logger.error(f"Error cancelling stream {session_id}: {str(e)}")

async def periodic_screening():
    """Periodically screen stocks and store results in Redis."""
    while True:
        try:
            # Load config from file directly
            import yaml
            with open('/home/ubuntu/nextg3n/config/system_config.yaml', 'r') as f:
                system_config = yaml.safe_load(f)
            
            screening_interval = system_config["services"]["stock_screening"]["update_interval_minutes"]
            default_symbols = system_config.get("services", {}).get("stock_screening", {}).get("default_symbols",
                ["AAPL", "MSFT", "GOOG", "TSLA", "NVDA", "AMZN", "META", "AMD", "INTC", "IBM"])
            
            # Get custom symbols list from Redis if available
            symbols_to_screen = default_symbols
            if redis_client:
                try:
                    custom_symbols = await redis_client.get("polygon:watchlist")
                    if custom_symbols:
                        symbols_to_screen = json.loads(custom_symbols)
                        logger.info(f"Using custom watchlist with {len(symbols_to_screen)} symbols")
                except Exception as redis_err:
                    logger.warning(f"Could not load custom symbols from Redis: {str(redis_err)}")
            
            # Create a RealtimeDataRequest object
            req = RealtimeDataRequest(
                symbols=symbols_to_screen[:50],  # Limit to 50 symbols for performance
                channels=["T", "Q"],
                duration_seconds=60
            )

            # Call the api_screen_stocks endpoint
            await api_screen_stocks(req, BackgroundTasks())

            logger.info(f"Periodic stock screening completed. Next run in {screening_interval} minutes.")
            await asyncio.sleep(screening_interval * 60)  # Convert minutes to seconds
        except Exception as e:
            logger.error(f"Error in periodic screening: {str(e)}")
            # Continue running even after errors, but wait before retrying
            await asyncio.sleep(300)  # Wait 5 minutes before retrying after an error

@app.get("/server_info", tags=["Server Info"])
async def get_server_info():
    return {
        "name": "polygon_websocket",
        "version": "1.0.0",
        "description": "Production MCP Server for Polygon WebSocket Integration",
        "tools": [
            "fetch_realtime_data", "start_stream", "stop_stream", "get_snapshot",
            "screen_stocks", "get_screened_stocks"
        ],
        "config": CONFIG,
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
        raise

@app.post("/screen_stocks", tags=["Stock Screening"], dependencies=[Depends(check_rate_limit)])
async def api_screen_stocks(req: RealtimeDataRequest, background_tasks: BackgroundTasks):
    """
    Screen stocks based on real-time data and AI/ML models.
    Stores results in Redis for later retrieval.
    """
    logger.info(f"Screening stocks for symbols: {req.symbols}")
    try:
        # Get real-time data
        buffer = []
        await polygon_ws_stream(req.symbols, req.channels, req.duration_seconds, buffer)
        logger.info(f"Received {len(buffer)} messages for screening")
        
        # Screen stocks
        candidates = await stock_screener.screen_stocks(buffer)
        logger.info(f"Found {len(candidates)} stock candidates")
        
        # Store in Redis
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
        raise

@app.get("/get_screened_stocks", tags=["Stock Screening"], dependencies=[Depends(check_rate_limit)])
async def api_get_screened_stocks(source: str = "polygon_websocket", date: str = None):
    """Get screened stocks from Redis."""
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
        raise

# Add new endpoints and helper functions for better session management

async def cleanup_inactive_sessions():
    """Remove inactive sessions to prevent memory leaks."""
    try:
        # Check every 10 minutes
        while True:
            await asyncio.sleep(600)
            current_time = time.time()
            inactive_threshold = 3600  # 1 hour of inactivity
            
            sessions_to_remove = []
            for session_id, session_data in active_streams.items():
                last_accessed = session_data.get("last_accessed", 0)
                if current_time - last_accessed > inactive_threshold:
                    sessions_to_remove.append(session_id)
            
            # Cancel and remove inactive sessions
            for session_id in sessions_to_remove:
                logger.info(f"Removing inactive session {session_id} (no activity for {inactive_threshold/60} minutes)")
                try:
                    active_streams[session_id]["task"].cancel()
                    del active_streams[session_id]
                except Exception as e:
                    logger.error(f"Error removing session {session_id}: {str(e)}")
    except Exception as e:
        logger.error(f"Error in cleanup task: {str(e)}")

@app.get("/active_sessions", tags=["Streaming"])
async def api_get_active_sessions():
    """Get information about all active streaming sessions."""
    try:
        return {
            "success": True,
            "count": len(active_streams),
            "sessions": [
                {
                    "session_id": session_id,
                    "symbols": info["symbols"],
                    "channels": info["channels"],
                    "start_time": info["start_time"],
                    "buffer_size": len(info["buffer"]),
                    "last_accessed": datetime.fromtimestamp(info.get("last_accessed", 0)).isoformat()
                }
                for session_id, info in active_streams.items()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting active sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving active sessions: {str(e)}")

# Health check endpoint for monitoring services
@app.get("/health", tags=["Server Info"])
async def api_health_check():
    """Health check endpoint for service monitoring."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "redis_connected": False,
        "active_streams": len(active_streams)
    }
    
    # Check Redis connection
    if redis_client:
        try:
            ping_result = await redis_client.ping()
            health_status["redis_connected"] = ping_result
        except Exception as e:
            logger.warning(f"Redis health check failed: {str(e)}")
            health_status["redis_error"] = str(e)
    
    # Check API key configuration
    health_status["api_key_configured"] = bool(CONFIG["api_key"])
    
    return health_status

@app.get("/screened_candidates", tags=["Stock Screening"], dependencies=[Depends(check_rate_limit)])
async def api_screened_candidates(source: str = "polygon_websocket", date: str = None):
    """Get screened candidates from Redis."""
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
    """
    Screen stocks based on real-time data and custom parameters.
    
    This endpoint allows customizing the screening parameters for a one-time scan.
    Results are stored in Redis for later retrieval with a custom source tag.
    
    Parameters:
        req: ManualScreenRequest containing symbols, channels, duration, and custom screening parameters
        
    Returns:
        JSON response with screened stock candidates based on custom parameters
    """
    logger.info(f"Manually screening stocks for symbols: {req.symbols} with custom parameters")

    if not CONFIG["api_key"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Polygon API service not available due to missing API key"
        )

    try:
        # Get real-time data
        buffer = []
        await polygon_ws_stream(req.symbols, req.channels, req.duration_seconds, buffer)
        logger.info(f"Received {len(buffer)} messages for manual screening")

        if not buffer:
            logger.warning("No data received from WebSocket, cannot perform screening")
            return JSONResponse(
                status_code=status.HTTP_204_NO_CONTENT,
                content={
                    "success": False,
                    "message": "No market data received from Polygon API",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        # Update StockScreener config with custom parameters
        custom_config = CONFIG.copy()
        if req.min_volume is not None:
            custom_config["min_volume"] = req.min_volume
        if req.min_rel_volume is not None:
            custom_config["min_rel_volume"] = req.min_rel_volume
        if req.min_price_change is not None:
            custom_config["min_price_change"] = req.min_price_change
        if req.min_atr is not None:
            custom_config["min_atr"] = req.min_atr

        # Create a new StockScreener instance with the custom config
        stock_screener_custom = StockScreener(custom_config)

        # Screen stocks
        candidates = await stock_screener_custom.screen_stocks(buffer)
        logger.info(f"Found {len(candidates)} stock candidates after manual screening")

        # Store in Redis with custom tag
        if redis_client and candidates:
            # Use custom key for manual screening results
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing manual screening request: {str(e)}"
        )

@app.post("/start_stream", tags=["Streaming"], dependencies=[Depends(check_rate_limit)])
async def api_start_stream(req: StartStreamRequest):
    # Validate API key is configured
    if not CONFIG["api_key"]:
        logger.error("Polygon API key not configured")
        raise HTTPException(status_code=500, detail="Polygon API key not configured")
        
    # Check symbol limit
    if len(req.symbols) > 100:
        logger.warning(f"Too many symbols requested: {len(req.symbols)}, limiting to 100")
        req.symbols = req.symbols[:100]
    
    session_id = req.session_id or f"stream_{int(time.time())}"
    logger.info(f"Starting stream session {session_id} for symbols: {req.symbols}")
    
    if session_id in active_streams:
        logger.warning(f"Session ID {session_id} already exists")
        raise HTTPException(status_code=400, detail=f"Session ID {session_id} already exists.")
        
    # Configure a reasonable buffer size for memory management
    buffer: List[dict] = []
    max_session_time = 3600*4  # 4 hour max session for production use
    
    task = asyncio.create_task(polygon_ws_stream(req.symbols, req.channels, max_session_time, buffer))
    active_streams[session_id] = {
        "task": task,
        "buffer": buffer,
        "symbols": req.symbols,
        "channels": req.channels,
        "start_time": datetime.utcnow().isoformat(),
        "last_accessed": time.time()
    }
    
    # Schedule cleanup task for abandoned sessions
    asyncio.create_task(cleanup_inactive_sessions())
    
    logger.info(f"Stream session {session_id} started successfully")
    return {
        "success": True,
        "session_id": session_id,
        "symbols": req.symbols,
        "channels": req.channels,
        "start_time": active_streams[session_id]["start_time"],
        "max_session_time": f"{max_session_time//3600} hours"
    }

@app.post("/stop_stream", tags=["Streaming"], dependencies=[Depends(check_rate_limit)])
async def api_stop_stream(req: StopStreamRequest):
    session_id = req.session_id
    logger.info(f"Stopping stream session {session_id}")
    
    if session_id not in active_streams:
        logger.warning(f"Session ID {session_id} not found")
        raise HTTPException(status_code=404, detail=f"Session ID {session_id} not found.")
        
    task = active_streams[session_id]["task"]
    task.cancel()
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
        raise HTTPException(status_code=404, detail=f"Session ID {session_id} not found.")
        
    buffer = active_streams[session_id]["buffer"]
    
    # Update last accessed time
    active_streams[session_id]["last_accessed"] = time.time()
    
    # Limit response size for large buffers
    max_messages = 1000
    if len(buffer) > max_messages:
        logger.info(f"Limiting snapshot response to {max_messages} messages (total: {len(buffer)})")
        response_data = buffer[-max_messages:]  # Return the most recent messages
    else:
        response_data = buffer
    
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
            active_count = len(active_streams)
            logger.debug(f"Returning information about {active_count} active streams")
            
            return {
                "active_streams": [
                    {
                        "session_id": session_id,
                        "symbols": info["symbols"],
                        "channels": info["channels"],
                        "start_time": info["start_time"],
                        "buffer_size": len(info["buffer"])
                    }
                    for session_id, info in active_streams.items()
                ],
                "count": active_count,
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
            # Collect simple metrics
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
        raise

# --- Helper Functions ---

async def store_candidates_in_redis(candidates: List[Dict[str, Any]]):
    """Store stock candidates in Redis."""
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
        # Set TTL to 7 days
        await redis_client.expire(key, 60 * 60 * 24 * 7)
        
        # Store also the latest candidates for quick access
        latest_key = "latest_screened_stocks"
        await redis_client.set(latest_key, json.dumps(value))
        await redis_client.expire(latest_key, 60 * 60 * 24)  # 1 day TTL for latest
        
        logger.info(f"Stored {len(candidates)} stock candidates in Redis with key {key}")
    except Exception as e:
        logger.error(f"Error storing candidates in Redis: {str(e)}")
        # Even if Redis storage fails, we don't raise the exception as this is a background task

async def store_custom_candidates_in_redis(
    candidates: List[Dict[str, Any]],
    source_tag: str,
    description: str = ""
):
    """Store custom stock candidates in Redis with a specified source tag."""
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
        # Set TTL to 3 days for custom screenings
        await redis_client.expire(key, 60 * 60 * 24 * 3)
        
        # Add to the list of custom screenings
        await redis_client.sadd("custom_screenings", key)
        
        logger.info(f"Stored {len(candidates)} custom stock candidates in Redis with key {key}")
    except Exception as e:
        logger.error(f"Error storing custom candidates in Redis: {str(e)}")
