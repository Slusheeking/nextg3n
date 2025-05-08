"""
Enhanced Market Analysis ML System

This module implements a comprehensive ML-based stock screening and analysis system
that filters from a universe of ~1000 stocks to identify the top 5 trading candidates.
It combines multiple trading strategies, news analysis, and technical indicators
to provide high-conviction trading opportunities.

Optimized for A100 GPU acceleration with CUDA support.
"""

import os
import json
import time
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import aiohttp
import redis.asyncio as aioredis
from yahoo_fin import stock_info as si

# ML Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import joblib
import pickle
import lightgbm as lgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import centralized logging system
from monitor.logging_utils import get_logger
logger = get_logger("market_analysis_ml")

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
DATA_API_BASE_URL = os.getenv("DATA_API_BASE_URL", "http://localhost:8000")
MODEL_PATH = os.getenv("MODEL_PATH", "./models")
MIN_VOLUME = int(os.getenv("MIN_VOLUME", "100000"))
MIN_RELATIVE_VOLUME = float(os.getenv("MIN_RELATIVE_VOLUME", "1.5"))
MIN_PRICE_CHANGE = float(os.getenv("MIN_PRICE_CHANGE", "0.03"))
MIN_ATR = float(os.getenv("MIN_ATR", "0.25"))
MAX_PRESCREEN_STOCKS = int(os.getenv("MAX_PRESCREEN_STOCKS", "100"))
TOP_N_STOCKS = int(os.getenv("TOP_N_STOCKS", "5"))
NEWS_WEIGHT = float(os.getenv("NEWS_WEIGHT", "0.3"))


# --- News Sentiment Analysis ---
class NewsSentimentAnalyzer:
    """Analyzes news sentiment for stocks using pre-trained models."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = device
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained sentiment analysis model."""
        try:
            model_name = "ProsusAI/finbert"  # Pre-trained financial BERT model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Loaded news sentiment model")
        except Exception as e:
            logger.error(f"Error loading news sentiment model: {e}")
            self.model = None
            self.tokenizer = None
    
    async def analyze_stock_news(self, symbol: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Analyze news sentiment for a specific stock."""
        if self.model is None or self.tokenizer is None:
            logger.warning("News sentiment model not available")
            return {
                'score': 0.5,
                'sentiment': 'neutral',
                'news_count': 0
            }
        
        # Fetch recent news
        news = await self._fetch_news(symbol, session)
        if not news:
            logger.info(f"No recent news found for {symbol}")
            return {
                'score': 0.5,
                'sentiment': 'neutral',
                'news_count': 0
            }
        
        try:
            # Analyze sentiment for each news item
            sentiments = []
            for item in news:
                text = f"{item.get('title', '')}. {item.get('summary', '')}"
                
                # Skip if text is too short
                if len(text) < 10:
                    continue
                
                # Tokenize and prepare input
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                
                # Get prediction
                with torch.no_grad():
                    try:
                        outputs = self.model(**inputs)
                        scores = F.softmax(outputs.logits, dim=1)[0].cpu().numpy()
                    except Exception as e:
                        logger.error(f"Error during model inference: {e}")
                        continue
                
                # FinBERT classes: Negative (0), Neutral (1), Positive (2)
                sentiment_scores = {
                    'negative': float(scores[0]),
                    'neutral': float(scores[1]),
                    'positive': float(scores[2])
                }
                
                sentiments.append({
                    'title': item.get('title', ''),
                    'scores': sentiment_scores,
                    'dominant': self._get_dominant_sentiment(scores)
                })
            
            # Calculate aggregate sentiment
            if sentiments:
                avg_negative = sum(item['scores']['negative'] for item in sentiments) / len(sentiments)
                avg_neutral = sum(item['scores']['neutral'] for item in sentiments) / len(sentiments)
                avg_positive = sum(item['scores']['positive'] for item in sentiments) / len(sentiments)
                
                # Convert to a single sentiment score (0 = very negative, 1 = very positive)
                sentiment_score = (avg_positive * 1.0 + avg_neutral * 0.5) / (avg_positive + avg_neutral + avg_negative)
                
                # Get dominant sentiment
                if avg_positive > avg_negative and avg_positive > avg_neutral:
                    dominant = "positive"
                elif avg_negative > avg_positive and avg_negative > avg_neutral:
                    dominant = "negative"
                else:
                    dominant = "neutral"
                
                return {
                    'score': float(sentiment_score),
                    'sentiment': dominant,
                    'news_count': len(sentiments),
                    'details': {
                        'positive': float(avg_positive),
                        'neutral': float(avg_neutral),
                        'negative': float(avg_negative)
                    }
                }
            
            # Default neutral sentiment if no valid news items
            return {
                'score': 0.5,
                'sentiment': 'neutral',
                'news_count': 0
            }
                
        except Exception as e:
            logger.error(f"Error analyzing news sentiment for {symbol}: {e}")
            return {
                'score': 0.5,
                'sentiment': 'neutral',
                'news_count': 0,
                'error': str(e)
            }
    
    async def _fetch_news(self, symbol: str, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """Fetch recent news for a symbol."""
        try:
            async with session.post(
                f"{DATA_API_BASE_URL}/api/news",
                json={"symbols": [symbol], "count": 10}
            ) as response:
                if response.status != 200:
                    logger.error(f"Error fetching news: {await response.text()}")
                    return []
                
                data = await response.json()
                
                if not data.get("success", False):
                    logger.error(f"API returned unsuccessful response for news: {data}")
                    return []
                
                return data.get("articles", [])
                
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def _get_dominant_sentiment(self, scores: List[float]) -> str:
        """Get the dominant sentiment class from scores."""
        max_idx = np.argmax(scores)
        if max_idx == 0:
            return "negative"
        elif max_idx == 1:
            return "neutral"
        else:
            return "positive"


# --- Trading Strategies ---
class TradingStrategy:
    """Base class for trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    def evaluate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the strategy on market data."""
        raise NotImplementedError("Subclasses must implement evaluate method")


class MomentumStrategy(TradingStrategy):
    """Momentum-based trading strategy."""
    
    def __init__(self):
        super().__init__("momentum")
    
    def evaluate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate momentum strategy on market data."""
        if data.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate momentum indicators
        try:
            # RSI (14-period)
            if 'rsi_14' not in df.columns:
                if 'close' in df.columns and len(df) > 14:
                    delta = df['close'].diff()
                    gain = delta.copy()
                    loss = delta.copy()
                    gain[gain < 0] = 0
                    loss[loss > 0] = 0
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.abs().rolling(window=14).mean()
                    rs = avg_gain / avg_loss
                    df['rsi_14'] = 100 - (100 / (1 + rs))
                else:
                    df['rsi_14'] = 50  # Default value
            
            # Rate of Change (10-period)
            if 'roc_10' not in df.columns and 'close' in df.columns:
                df['roc_10'] = df['close'].pct_change(periods=10) * 100
            
            # Moving Average Convergence Divergence (MACD)
            if 'macd' not in df.columns and 'close' in df.columns:
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema12 - ema26
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Price performance (5-day)
            if 'price_perf_5d' not in df.columns and 'close' in df.columns:
                df['price_perf_5d'] = df['close'].pct_change(periods=5) * 100
            
            # Calculate momentum score
            df['momentum_score'] = 0.0
            
            # RSI component (higher RSI = higher score for momentum)
            if 'rsi_14' in df.columns:
                # RSI above 70 is overbought, below 30 is oversold
                # For momentum, we want rising RSI between 40-70
                rsi_score = (df['rsi_14'] - 40) / 30
                rsi_score = rsi_score.clip(0, 1) * 0.3  # 30% weight
                df['momentum_score'] += rsi_score
            
            # ROC component
            if 'roc_10' in df.columns:
                # Normalize ROC to 0-1 range (assuming normal range is -10 to +10)
                roc_score = (df['roc_10'] + 10) / 20
                roc_score = roc_score.clip(0, 1) * 0.3  # 30% weight
                df['momentum_score'] += roc_score
            
            # MACD component
            if 'macd_hist' in df.columns:
                # Positive and rising MACD histogram is good for momentum
                macd_score = (df['macd_hist'] > 0).astype(float) * 0.2  # 20% weight
                df['momentum_score'] += macd_score
            
            # Recent performance component
            if 'price_perf_5d' in df.columns:
                # Normalize 5-day performance to 0-1 range (assuming normal range is -10 to +10)
                perf_score = (df['price_perf_5d'] + 10) / 20
                perf_score = perf_score.clip(0, 1) * 0.2  # 20% weight
                df['momentum_score'] += perf_score
            
            return df
            
        except Exception as e:
            logger.error(f"Error evaluating momentum strategy: {e}")
            df['momentum_score'] = 0.0
            return df


class MeanReversionStrategy(TradingStrategy):
    """Mean reversion trading strategy."""
    
    def __init__(self):
        super().__init__("mean_reversion")
    
    def evaluate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate mean reversion strategy on market data."""
        if data.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate mean reversion indicators
        try:
            # Bollinger Bands
            if 'close' in df.columns and len(df) > 20:
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['bb_std'] = df['close'].rolling(window=20).std()
                df['bb_upper'] = df['sma_20'] + 2 * df['bb_std']
                df['bb_lower'] = df['sma_20'] - 2 * df['bb_std']
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # RSI (14-period) for oversold/overbought
            if 'rsi_14' not in df.columns:
                if 'close' in df.columns and len(df) > 14:
                    delta = df['close'].diff()
                    gain = delta.copy()
                    loss = delta.copy()
                    gain[gain < 0] = 0
                    loss[loss > 0] = 0
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.abs().rolling(window=14).mean()
                    rs = avg_gain / avg_loss
                    df['rsi_14'] = 100 - (100 / (1 + rs))
                else:
                    df['rsi_14'] = 50  # Default value
            
            # Distance from moving averages
            if 'close' in df.columns:
                if 'sma_20' not in df.columns:
                    df['sma_20'] = df['close'].rolling(window=20).mean()
                if 'sma_50' not in df.columns and len(df) > 50:
                    df['sma_50'] = df['close'].rolling(window=50).mean()
                
                df['dist_from_sma20'] = (df['close'] / df['sma_20'] - 1) * 100
                df['dist_from_sma50'] = (df['close'] / df['sma_50'] - 1) * 100 if 'sma_50' in df.columns else 0
            
            # Calculate mean reversion score
            df['mean_reversion_score'] = 0.0
            
            # Bollinger Band component (lower score for extremes)
            if 'bb_position' in df.columns:
                # Best score near the middle (0.5), worst score at extremes (0 or 1)
                bb_score = 1 - abs(df['bb_position'] - 0.5) * 2
                bb_score = bb_score.clip(0, 1) * 0.4  # 40% weight
                df['mean_reversion_score'] += bb_score
            
            # RSI component (higher score for oversold/overbought)
            if 'rsi_14' in df.columns:
                # High score for extreme RSI (either oversold or overbought)
                rsi_oversold = (30 - df['rsi_14']) / 30
                rsi_overbought = (df['rsi_14'] - 70) / 30
                rsi_score = rsi_oversold.clip(0, 1) + rsi_overbought.clip(0, 1)
                rsi_score = rsi_score.clip(0, 1) * 0.3  # 30% weight
                df['mean_reversion_score'] += rsi_score
            
            # Distance from MA component
            if 'dist_from_sma20' in df.columns:
                # Higher score for larger deviations
                dist_score = abs(df['dist_from_sma20']) / 10  # Normalize to 0-1 assuming ±10% is extreme
                dist_score = dist_score.clip(0, 1) * 0.3  # 30% weight
                df['mean_reversion_score'] += dist_score
            
            return df
            
        except Exception as e:
            logger.error(f"Error evaluating mean reversion strategy: {e}")
            df['mean_reversion_score'] = 0.0
            return df


class BreakoutStrategy(TradingStrategy):
    """Breakout trading strategy."""
    
    def __init__(self):
        super().__init__("breakout")
    
    def evaluate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate breakout strategy on market data."""
        if data.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate breakout indicators
        try:
            # Price range (high - low)
            if 'high' in df.columns and 'low' in df.columns:
                df['price_range'] = df['high'] - df['low']
            
            # Average True Range (ATR)
            if 'atr_14' not in df.columns:
                if all(col in df.columns for col in ['high', 'low', 'close']) and len(df) > 14:
                    tr1 = df['high'] - df['low']
                    tr2 = abs(df['high'] - df['close'].shift())
                    tr3 = abs(df['low'] - df['close'].shift())
                    df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    df['atr_14'] = df['true_range'].rolling(window=14).mean()
                else:
                    df['atr_14'] = 0
            
            # Volume surge
            if 'volume' in df.columns:
                df['volume_sma20'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma20']
            
            # Recent high/low levels
            if 'close' in df.columns and 'high' in df.columns and 'low' in df.columns:
                lookback = min(20, len(df))
                if lookback > 0:
                    df['recent_high'] = df['high'].rolling(window=lookback).max()
                    df['recent_low'] = df['low'].rolling(window=lookback).min()
                    
                    # Distance from recent high/low
                    df['dist_from_high'] = (df['recent_high'] - df['close']) / df['close'] * 100
                    df['dist_from_low'] = (df['close'] - df['recent_low']) / df['close'] * 100
                    
                    # Check for breakout (close above recent high or below recent low)
                    df['breakout_up'] = (df['close'] > df['recent_high'].shift())
                    df['breakout_down'] = (df['close'] < df['recent_low'].shift())
            
            # Calculate breakout score
            df['breakout_score'] = 0.0
            
            # Volatility component (higher ATR = better for breakouts)
            if 'atr_14' in df.columns and 'close' in df.columns:
                # Normalize ATR as a percentage of price
                atr_pct = df['atr_14'] / df['close'] * 100
                atr_score = atr_pct / 5  # Normalize to 0-1 assuming 5% ATR is high
                atr_score = atr_score.clip(0, 1) * 0.3  # 30% weight
                df['breakout_score'] += atr_score
            
            # Volume surge component
            if 'volume_ratio' in df.columns:
                vol_score = (df['volume_ratio'] - 1) / 2  # Normalize to 0-1 assuming 3x volume is high
                vol_score = vol_score.clip(0, 1) * 0.3  # 30% weight
                df['breakout_score'] += vol_score
            
            # Actual breakout component
            if 'breakout_up' in df.columns and 'breakout_down' in df.columns:
                breakout_score = df['breakout_up'].astype(float) * 0.4  # 40% weight for upward breakouts
                df['breakout_score'] += breakout_score
            
            # Distance from level component
            if 'dist_from_high' in df.columns:
                # Close to high = higher chance of breakout
                level_score = (1 - df['dist_from_high'] / 5)  # Normalize to 0-1 assuming 5% away is far
                level_score = level_score.clip(0, 1) * 0.2  # 20% weight
                df['breakout_score'] += level_score
            
            return df
            
        except Exception as e:
            logger.error(f"Error evaluating breakout strategy: {e}")
            df['breakout_score'] = 0.0
            return df


class VolatilityStrategy(TradingStrategy):
    """Volatility-based trading strategy."""
    
    def __init__(self):
        super().__init__("volatility")
    
    def evaluate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate volatility strategy on market data."""
        if data.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate volatility indicators
        try:
            # Historical volatility (20-day)
            if 'volatility_20d' not in df.columns and 'close' in df.columns and len(df) > 20:
                returns = df['close'].pct_change()
                df['volatility_20d'] = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
            
            # Average True Range (ATR)
            if 'atr_14' not in df.columns:
                if all(col in df.columns for col in ['high', 'low', 'close']) and len(df) > 14:
                    tr1 = df['high'] - df['low']
                    tr2 = abs(df['high'] - df['close'].shift())
                    tr3 = abs(df['low'] - df['close'].shift())
                    df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    df['atr_14'] = df['true_range'].rolling(window=14).mean()
                else:
                    df['atr_14'] = 0
            
            # ATR ratio (current ATR vs. historical)
            if 'atr_14' in df.columns:
                df['atr_sma60'] = df['atr_14'].rolling(window=60).mean()
                df['atr_ratio'] = df['atr_14'] / df['atr_sma60']
            
            # Bollinger Band width
            if 'close' in df.columns and len(df) > 20:
                if 'bb_width' not in df.columns:
                    df['sma_20'] = df['close'].rolling(window=20).mean()
                    df['bb_std'] = df['close'].rolling(window=20).std()
                    df['bb_upper'] = df['sma_20'] + 2 * df['bb_std']
                    df['bb_lower'] = df['sma_20'] - 2 * df['bb_std']
                    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
            
            # Calculate volatility score
            df['volatility_score'] = 0.0
            
            # Historical volatility component
            if 'volatility_20d' in df.columns:
                vol_score = df['volatility_20d'] / 0.5  # Normalize to 0-1 assuming 50% annual vol is high
                vol_score = vol_score.clip(0, 1) * 0.4  # 40% weight
                df['volatility_score'] += vol_score
            
            # ATR ratio component
            if 'atr_ratio' in df.columns:
                atr_score = df['atr_ratio'] / 2  # Normalize to 0-1 assuming 2x ATR ratio is high
                atr_score = atr_score.clip(0, 1) * 0.3  # 30% weight
                df['volatility_score'] += atr_score
            
            # Bollinger Band width component
            if 'bb_width' in df.columns:
                bb_score = df['bb_width'] / 0.1  # Normalize to 0-1 assuming 10% BB width is high
                bb_score = bb_score.clip(0, 1) * 0.3  # 30% weight
                df['volatility_score'] += bb_score
            
            return df
            
        except Exception as e:
            logger.error(f"Error evaluating volatility strategy: {e}")
            df['volatility_score'] = 0.0
            return df


class TrendFollowingStrategy(TradingStrategy):
    """Trend following trading strategy."""
    
    def __init__(self):
        super().__init__("trend_following")
    
    def evaluate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate trend following strategy on market data."""
        if data.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate trend following indicators
        try:
            # Moving average relationships
            if 'close' in df.columns:
                for period in [10, 20, 50]:
                    if len(df) >= period:
                        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                
                # Calculate price position relative to MAs
                if 'sma_10' in df.columns and 'sma_20' in df.columns:
                    df['sma_10_20_ratio'] = df['sma_10'] / df['sma_20']
                
                if 'sma_20' in df.columns and 'sma_50' in df.columns:
                    df['sma_20_50_ratio'] = df['sma_20'] / df['sma_50']
                
                if 'sma_10' in df.columns:
                    df['price_sma_10_ratio'] = df['close'] / df['sma_10']
            
            # ADX (Average Directional Index) for trend strength
            if all(col in df.columns for col in ['high', 'low', 'close']) and len(df) > 14:
                df['adx'] = self._calculate_adx(df)
            
            # Linear regression slope
            if 'close' in df.columns and len(df) > 20:
                df['slope_20d'] = self._calculate_slope(df['close'], 20)
            
            # Calculate trend following score
            df['trend_following_score'] = 0.0
            
            # Moving average alignment component
            if all(col in df.columns for col in ['sma_10_20_ratio', 'sma_20_50_ratio']):
                # If both ratios > 1, we have a positive trend alignment
                ma_alignment = (df['sma_10_20_ratio'] > 1).astype(float) * (df['sma_20_50_ratio'] > 1).astype(float)
                df['trend_following_score'] += ma_alignment * 0.3  # 30% weight
            
            # Price above short-term MA component
            if 'price_sma_10_ratio' in df.columns:
                price_ma_score = (df['price_sma_10_ratio'] > 1).astype(float) * 0.2  # 20% weight
                df['trend_following_score'] += price_ma_score
            
            # ADX component
            if 'adx' in df.columns:
                adx_score = df['adx'] / 50  # Normalize to 0-1 assuming ADX of 50 is strong trend
                adx_score = adx_score.clip(0, 1) * 0.3  # 30% weight
                df['trend_following_score'] += adx_score
            
            # Slope component
            if 'slope_20d' in df.columns:
                # Normalize slope to 0-1 (assuming slope of 0.1 is steep)
                slope_score = (df['slope_20d'] + 0.1) / 0.2
                slope_score = slope_score.clip(0, 1) * 0.2  # 20% weight
                df['trend_following_score'] += slope_score
            
            return df
            
        except Exception as e:
            logger.error(f"Error evaluating trend following strategy: {e}")
            df['trend_following_score'] = 0.0
            return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)."""
        # +DM, -DM, and TR
        high_diff = df['high'].diff()
        low_diff = df['low'].diff().abs()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smoothed averages of +DM, -DM, and TR
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum())
        
        # Directional movement index - avoid division by zero
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 0.000001))
        
        # Average directional index
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_slope(self, series: pd.Series, period: int = 20) -> pd.Series:
        """Calculate the slope of a linear regression over a rolling window."""
        slopes = pd.Series(index=series.index, dtype=float)
        
        for i in range(period - 1, len(series)):
            y = series.iloc[i - period + 1:i + 1].values
            x = np.arange(period)
            slope, _ = np.polyfit(x, y, 1)
            slopes.iloc[i] = slope
        
        return slopes


class MarketDataProcessor:
    """Processes raw market data from APIs and prepares it for ML models."""
    
    def __init__(self, session: aiohttp.ClientSession, redis_client: aioredis.Redis):
        self.session = session
        self.redis_client = redis_client
        self.scaler = StandardScaler()
        self.feature_columns = []
        self._load_scaler()
    
    def _load_scaler(self):
        """Load pre-trained scaler if available."""
        scaler_path = os.path.join(MODEL_PATH, "feature_scaler.joblib")
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded pre-trained scaler")
            except Exception as e:
                logger.error(f"Error loading scaler: {e}")
                self.scaler = StandardScaler()
    
    async def fetch_market_data(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch market data for a list of symbols from various APIs."""
        logger.info(f"Fetching market data for {len(symbols)} symbols...")
        
        # Get stock quotes
        try:
            quotes_data = await self._fetch_quotes(symbols)
            if not quotes_data.get("success", False):
                logger.error("Failed to fetch quotes")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching quotes: {str(e)}")
            return pd.DataFrame()
        
        # Collect data for each symbol
        all_data = []
        for symbol in symbols:
            try:
                # Get chart data
                chart_data = await self._fetch_chart(symbol)
                if not chart_data.get("success", False):
                    continue
                
                # Extract technical features
                symbol_data = self._extract_features(symbol, quotes_data, chart_data)
                if symbol_data:
                    all_data.append(symbol_data)
            except Exception as e:
                logger.warning(f"Error processing data for {symbol}: {str(e)}")
        
        # Convert to DataFrame
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        
        # Save feature columns list if not already done
        if not self.feature_columns:
            self.feature_columns = [col for col in df.columns if col != 'symbol']
            
        logger.info(f"Processed market data for {len(df)} symbols")
        return df
    
    async def _fetch_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch quotes for multiple symbols."""
        try:
            async with self.session.post(
                f"{DATA_API_BASE_URL}/api/quotes",
                json={"symbols": symbols},
                timeout=30  # Add timeout to prevent hanging
            ) as response:
                if response.status == 200:
                    return await response.json()
                logger.error(f"Failed to fetch quotes: {await response.text()}")
                return {"success": False}
        except asyncio.TimeoutError:
            logger.error("Timeout while fetching quotes")
            return {"success": False, "error": "Request timeout"}
        except Exception as e:
            logger.error(f"Error fetching quotes: {e}")
            return {"success": False, "error": str(e)}
    
    async def _fetch_chart(self, symbol: str) -> Dict[str, Any]:
        """Fetch chart data for a single symbol."""
        try:
            async with self.session.post(
                f"{DATA_API_BASE_URL}/api/chart",
                json={"symbol": symbol, "interval": "1d", "range": "1mo"},
                timeout=30  # Add timeout to prevent hanging
            ) as response:
                if response.status == 200:
                    return await response.json()
                logger.warning(f"Failed to fetch chart data for {symbol}: {await response.text()}")
                return {"success": False}
        except asyncio.TimeoutError:
            logger.warning(f"Timeout while fetching chart data for {symbol}")
            return {"success": False, "error": "Request timeout"}
        except Exception as e:
            logger.warning(f"Error fetching chart data for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_features(self, symbol: str, quotes_data: Dict, chart_data: Dict) -> Optional[Dict[str, Any]]:
        """Extract and calculate features for a single symbol."""
        try:
            quote = quotes_data.get("quotes", {}).get(symbol, {})
            if not quote or "error" in quote:
                return None
                
            chart = chart_data.get("chart_data", {})
            if not chart:
                return None
            
            # Convert chart data to DataFrame for calculations
            chart_df = self._process_chart_to_dataframe(chart)
            if chart_df.empty or len(chart_df) < 5:
                return None
            
            # Extract basic price and volume data
            features = {}
            features['symbol'] = symbol
            
            # Basic price data
            features['price'] = float(quote.get("Close", 0))
            features['open'] = float(quote.get("Open", 0))
            features['high'] = float(quote.get("High", 0))
            features['low'] = float(quote.get("Low", 0))
            features['volume'] = int(quote.get("Volume", 0))
            
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(chart_df)
            features.update(indicators)
            
            # Apply initial screening filters
            if (features['volume'] < MIN_VOLUME or
                features['relative_volume'] < MIN_RELATIVE_VOLUME or
                abs(features['price_change_pct']) < MIN_PRICE_CHANGE or
                features['atr_14'] < MIN_ATR):
                return None
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting features for {symbol}: {str(e)}")
            return None
    
    def _process_chart_to_dataframe(self, chart_data: Dict) -> pd.DataFrame:
        """Convert chart data to a pandas DataFrame."""
        if not chart_data:
            return pd.DataFrame()
            
        try:
            chart_df = pd.DataFrame()
            for date, values in chart_data.items():
                if isinstance(values, dict):
                    chart_df = pd.concat([chart_df, pd.DataFrame([values], index=[date])])
            
            if chart_df.empty:
                return pd.DataFrame()
                
            # Sort by date
            chart_df = chart_df.sort_index()
            
            # Ensure numeric columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in chart_df.columns:
                    chart_df[col] = pd.to_numeric(chart_df[col], errors='coerce')
            
            # Fill NaN values
            chart_df = chart_df.fillna(method='ffill').fillna(method='bfill')
            
            return chart_df
        except Exception as e:
            logger.error(f"Error processing chart data: {e}")
            return pd.DataFrame()
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators from price data."""
        indicators = {}
        
        # Price change
        indicators['price_change'] = df['Close'].iloc[-1] - df['Open'].iloc[-1]
        indicators['price_change_pct'] = indicators['price_change'] / df['Open'].iloc[-1] if df['Open'].iloc[-1] > 0 else 0
        
        # Volume metrics
        indicators['volume'] = df['Volume'].iloc[-1]
        avg_volume_10d = df['Volume'].rolling(10).mean().iloc[-1] if len(df) >= 10 else df['Volume'].mean()
        indicators['relative_volume'] = indicators['volume'] / avg_volume_10d if avg_volume_10d > 0 else 1.0
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['atr_14'] = true_range.rolling(14).mean().iloc[-1] if len(df) >= 14 else true_range.mean()
        
        # Moving Averages
        indicators['sma_10'] = df['Close'].rolling(10).mean().iloc[-1] if len(df) >= 10 else df['Close'].iloc[-1]
        indicators['sma_20'] = df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['Close'].iloc[-1]
        indicators['sma_50'] = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else df['Close'].iloc[-1]
        
        # Price relative to moving averages
        indicators['price_vs_sma10'] = df['Close'].iloc[-1] / indicators['sma_10'] - 1 if indicators['sma_10'] > 0 else 0
        indicators['price_vs_sma20'] = df['Close'].iloc[-1] / indicators['sma_20'] - 1 if indicators['sma_20'] > 0 else 0
        indicators['price_vs_sma50'] = df['Close'].iloc[-1] / indicators['sma_50'] - 1 if indicators['sma_50'] > 0 else 0
        
        # RSI (14-period)
        delta = df['Close'].diff().dropna()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        avg_gain = gain.rolling(14).mean().iloc[-1] if len(gain) >= 14 else gain.mean()
        avg_loss = abs(loss.rolling(14).mean().iloc[-1]) if len(loss) >= 14 else abs(loss.mean())
        rs = avg_gain / avg_loss if avg_loss > 0 else 1
        indicators['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean().iloc[-1] if len(df) >= 12 else df['Close'].iloc[-1]
        ema_26 = df['Close'].ewm(span=26).mean().iloc[-1] if len(df) >= 26 else df['Close'].iloc[-1]
        indicators['macd'] = ema_12 - ema_26
        
        # Bollinger Bands
        if len(df) >= 20:
            sma_20 = df['Close'].rolling(20).mean()
            stddev = df['Close'].rolling(20).std()
            indicators['bb_upper'] = sma_20.iloc[-1] + (stddev.iloc[-1] * 2)
            indicators['bb_lower'] = sma_20.iloc[-1] - (stddev.iloc[-1] * 2)
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / sma_20.iloc[-1]
            indicators['bb_position'] = (df['Close'].iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower']) if (indicators['bb_upper'] - indicators['bb_lower']) > 0 else 0.5
        else:
            indicators['bb_width'] = 0
            indicators['bb_position'] = 0.5
        
        # Volatility
        returns = df['Close'].pct_change().dropna()
        indicators['volatility_20d'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252) if len(returns) >= 20 else 0
        
        # Gap analysis
        if len(df) >= 2:
            indicators['gap'] = df['Open'].iloc[-1] - df['Close'].iloc[-2]
            indicators['gap_pct'] = indicators['gap'] / df['Close'].iloc[-2] if df['Close'].iloc[-2] > 0 else 0
        else:
            indicators['gap'] = 0
            indicators['gap_pct'] = 0
        
        # ADX (Average Directional Index) for trend strength
        if len(df) >= 14:
            # Simple ADX calculation
            plus_dm = df['High'].diff()
            minus_dm = df['Low'].shift().diff(-1)
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr = pd.DataFrame([
                df['High'] - df['Low'],
                abs(df['High'] - df['Close'].shift()),
                abs(df['Low'] - df['Close'].shift())
            ]).max()
            
            plus_di = 100 * plus_dm.rolling(14).mean() / tr.rolling(14).mean()
            minus_di = 100 * minus_dm.rolling(14).mean() / tr.rolling(14).mean()
            
            # Avoid division by zero with a small epsilon
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.001)
            indicators['adx'] = dx.rolling(14).mean().iloc[-1]
        else:
            indicators['adx'] = 0
        
        return indicators
    
    def preprocess_for_models(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare data for ML models."""
        if df.empty:
            return df, np.array([])
            
        # Make a copy to avoid modifying the original
        df_proc = df.copy()
        
        # Separate symbol for later reference
        symbols = df_proc['symbol'].values
        df_proc = df_proc.drop('symbol', axis=1)
        
        # Fill any missing values
        df_proc = df_proc.fillna(0)
        
        # Scale the features
        try:
            scaled_features = self.scaler.transform(df_proc)
        except:
            # If the scaler isn't fitted yet, fit it
            scaled_features = self.scaler.fit_transform(df_proc)
            
            # Save the scaler for future use
            os.makedirs(MODEL_PATH, exist_ok=True)
            joblib.dump(self.scaler, os.path.join(MODEL_PATH, "feature_scaler.joblib"))
        
        # Return preprocessed data with symbols
        return pd.DataFrame(scaled_features, columns=df_proc.columns), symbols


class ModelManager:
    """Manages ML model predictions for stock screening."""
    
    def __init__(self):
        # Model states
        self.xgb_model = None
        self.lgb_model = None
        self.rf_model = None
        self.feature_columns = []
        
        # Load pre-trained models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained ML models if available."""
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_PATH, exist_ok=True)
        
        # Load feature columns
        feature_cols_path = os.path.join(MODEL_PATH, "feature_columns.json")
        if os.path.exists(feature_cols_path):
            with open(feature_cols_path, 'r') as f:
                self.feature_columns = json.load(f)
                logger.info(f"Loaded {len(self.feature_columns)} feature columns")
        
        # Load XGBoost model
        xgb_path = os.path.join(MODEL_PATH, "xgboost_model.json")
        if os.path.exists(xgb_path):
            try:
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(xgb_path)
                logger.info("Loaded pre-trained XGBoost model")
            except Exception as e:
                logger.error(f"Error loading XGBoost model: {str(e)}")
        
        # Load LightGBM model
        lgb_path = os.path.join(MODEL_PATH, "lightgbm_model.txt")
        if os.path.exists(lgb_path):
            try:
                self.lgb_model = lgb.Booster(model_file=lgb_path)
                logger.info("Loaded pre-trained LightGBM model")
            except Exception as e:
                logger.error(f"Error loading LightGBM model: {str(e)}")
        
        # Load Random Forest model
        rf_path = os.path.join(MODEL_PATH, "random_forest_model.joblib")
        if os.path.exists(rf_path):
            try:
                self.rf_model = joblib.load(rf_path)
                logger.info("Loaded pre-trained Random Forest model")
            except Exception as e:
                logger.error(f"Error loading Random Forest model: {str(e)}")
    
    def predict_with_models(self, features: pd.DataFrame) -> np.ndarray:
        """Get predictions from all available models and ensemble them."""
        if features.empty:
            return np.array([])
        
        # Store predictions from each model
        predictions = []
        
        # XGBoost prediction
        if self.xgb_model:
            try:
                # Ensure there are no NaN or infinite values
                features_clean = features.fillna(0)
                dmatrix = xgb.DMatrix(features_clean.values)
                xgb_preds = self.xgb_model.predict(dmatrix)
                predictions.append(xgb_preds)
                logger.debug("Added XGBoost predictions")
            except Exception as e:
                logger.warning(f"Error making XGBoost predictions: {str(e)}")
        
        # LightGBM prediction
        if self.lgb_model:
            try:
                # Ensure there are no NaN or infinite values
                features_clean = features.fillna(0)
                lgb_preds = self.lgb_model.predict(features_clean.values)
                predictions.append(lgb_preds)
                logger.debug("Added LightGBM predictions")
            except Exception as e:
                logger.warning(f"Error making LightGBM predictions: {str(e)}")
        
        # Random Forest prediction
        if self.rf_model:
            try:
                # Ensure there are no NaN or infinite values
                features_clean = features.fillna(0)
                rf_preds = self.rf_model.predict_proba(features_clean.values)[:, 1]
                predictions.append(rf_preds)
                logger.debug("Added Random Forest predictions")
            except Exception as e:
                logger.warning(f"Error making Random Forest predictions: {str(e)}")
        
        # If we have no predictions, use a default metric
        if not predictions:
            logger.warning("No models available for prediction, using relative volume as ranking")
            # Find the column index for relative_volume
            if 'relative_volume' in features.columns:
                rel_vol_idx = features.columns.get_loc('relative_volume')
                return features.iloc[:, rel_vol_idx].values
            else:
                # Just return ones if we can't find relative volume
                return np.ones(len(features))
        
        # Ensemble the predictions (simple average)
        ensemble_preds = np.mean(predictions, axis=0)
        return ensemble_preds


class MarketAnalysisSystem:
    """Main system for market analysis and stock screening."""
    
    def __init__(self):
        self.session = None
        self.redis_client = None
        self.data_processor = None
        self.model_manager = None
        self.news_analyzer = None
        self.strategies = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize components and connections."""
        if self.initialized:
            return
        
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        # Connect to Redis
        try:
            self.redis_client = aioredis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.redis_client = None
        
        # Initialize components
        self.data_processor = MarketDataProcessor(self.session, self.redis_client)
        self.model_manager = ModelManager()
        self.news_analyzer = NewsSentimentAnalyzer()
        
        # Initialize trading strategies
        self.strategies = {
            "momentum": MomentumStrategy(),
            "mean_reversion": MeanReversionStrategy(),
            "breakout": BreakoutStrategy(),
            "volatility": VolatilityStrategy(),
            "trend_following": TrendFollowingStrategy()
        }
        
        self.initialized = True
        logger.info("Market Analysis System initialized")
    
    async def screen_stocks(self, universe: List[str]) -> List[Dict[str, Any]]:
        """Screen stocks from a universe to find the top trading candidates."""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        logger.info(f"Screening {len(universe)} stocks")
        
        # Initial screening - get market data for all symbols
        df = await self.data_processor.fetch_market_data(universe)
        if df.empty:
            logger.warning("No data available after initial screening")
            return []
        
        # Apply trading strategies
        df = await self._apply_strategies(df)
        
        # Preprocess data for ML models
        features_df, symbols = self.data_processor.preprocess_for_models(df)
        if len(symbols) == 0:
            logger.warning("No stocks passed preprocessing")
            return []
        
        # Get model predictions
        predictions = self.model_manager.predict_with_models(features_df)
        
        # Combine predictions with symbols and strategy scores
        results = []
        for i, symbol in enumerate(symbols):
            # Get the original unscaled features for this symbol
            original_row = df[df['symbol'] == symbol].iloc[0].to_dict() if len(df[df['symbol'] == symbol]) > 0 else {}
            
            # Add prediction score
            ml_score = float(predictions[i]) if i < len(predictions) else 0.0
            
            # Create result dictionary with technical features
            result = {
                'symbol': symbol,
                'ml_score': ml_score,
                'price': original_row.get('price', 0),
                'volume': original_row.get('volume', 0),
                'relative_volume': original_row.get('relative_volume', 0),
                'price_change_pct': original_row.get('price_change_pct', 0),
                'atr_14': original_row.get('atr_14', 0),
                'rsi_14': original_row.get('rsi_14', 0),
                'adx': original_row.get('adx', 0),
                'volatility_20d': original_row.get('volatility_20d', 0),
                
                # Strategy scores
                'momentum_score': original_row.get('momentum_score', 0),
                'mean_reversion_score': original_row.get('mean_reversion_score', 0),
                'breakout_score': original_row.get('breakout_score', 0),
                'volatility_score': original_row.get('volatility_score', 0),
                'trend_following_score': original_row.get('trend_following_score', 0),
                
                # Initialize news sentiment score (will be updated later)
                'news_sentiment_score': 0.5
            }
            
            # Calculate combined strategy score (average of all strategies)
            strategy_scores = [
                result['momentum_score'],
                result['mean_reversion_score'],
                result['breakout_score'],
                result['volatility_score'],
                result['trend_following_score']
            ]
            result['strategy_score'] = sum(strategy_scores) / len(strategy_scores)
            
            results.append(result)
        
        # Sort preliminary results by ML score to get the top candidates for further analysis
        results = sorted(results, key=lambda x: x['ml_score'], reverse=True)
        
        # Take top candidates for news analysis to avoid API overload
        top_candidates = results[:MAX_PRESCREEN_STOCKS]
        
        # Add news sentiment analysis for top candidates
        await self._add_news_sentiment(top_candidates)
        
        # Calculate final combined score
        for result in top_candidates:
            # Combine ML score, strategy score, and news sentiment score
            # Adjust weights as needed
            ml_weight = 0.4
            strategy_weight = 0.3
            news_weight = NEWS_WEIGHT
            
            result['final_score'] = (
                result['ml_score'] * ml_weight +
                result['strategy_score'] * strategy_weight +
                result['news_sentiment_score'] * news_weight
            )
        
        # Sort by final score
        top_candidates = sorted(top_candidates, key=lambda x: x['final_score'], reverse=True)
        
        # Store results in Redis for other components
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if self.redis_client:
            try:
                key = f"market_analysis:screened_stocks:{timestamp}"
                await self.redis_client.set(key, json.dumps(top_candidates))
                await self.redis_client.expire(key, 60 * 60 * 24)  # 24 hour expiry
                
                # Update the "latest" key for quick access
                await self.redis_client.set("market_analysis:screened_stocks:latest", json.dumps(top_candidates))
                
                logger.info(f"Stored {len(top_candidates)} screened stocks in Redis")
            except Exception as e:
                logger.error(f"Failed to store results in Redis: {str(e)}")
        
        # Return the top N stocks
        top_results = top_candidates[:TOP_N_STOCKS]
        logger.info(f"Screening completed in {time.time() - start_time:.2f} seconds, returning top {len(top_results)} stocks")
        
        return top_results
    
    async def _apply_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all trading strategies to the data."""
        if df.empty:
            return df
        
        # Apply each strategy
        for strategy_name, strategy in self.strategies.items():
            logger.debug(f"Applying {strategy_name} strategy")
            df = strategy.evaluate(df)
        
        return df
    
    async def _add_news_sentiment(self, results: List[Dict[str, Any]]) -> None:
        """Add news sentiment analysis to the results."""
        if not results:
            return
        
        # Analyze news sentiment for each stock
        for result in results:
            symbol = result['symbol']
            try:
                news_sentiment = await self.news_analyzer.analyze_stock_news(symbol, self.session)
                result['news_sentiment_score'] = news_sentiment['score']
                result['news_sentiment'] = news_sentiment['sentiment']
                result['news_count'] = news_sentiment['news_count']
                
                if 'details' in news_sentiment:
                    result['news_details'] = news_sentiment['details']
                
                logger.debug(f"Added news sentiment for {symbol}: {news_sentiment['sentiment']}")
            except Exception as e:
                logger.error(f"Error adding news sentiment for {symbol}: {e}")
                # Keep default neutral sentiment
                result['news_sentiment_score'] = 0.5
                result['news_sentiment'] = 'neutral'
                result['news_count'] = 0
    
    async def close(self):
        """Close connections and release resources."""
        if self.session:
            await self.session.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Market Analysis System resources released")


async def main():
    """Main function for testing."""
    # Sample universe of stocks (would be loaded from a database or API in production)
    
    try:
        # Get S&P 500 stocks
        sp500_stocks = si.tickers_sp500()
        sp500_stocks = [s.replace('.', '-') for s in sp500_stocks]  # Fix ticker format
        
        # Get additional liquid stocks
        nasdaq_stocks = si.tickers_nasdaq()[:200]  # Limit to top 200 for testing
        nasdaq_stocks = [s.replace('.', '-') for s in nasdaq_stocks]
        
        # Combine universes and remove duplicates
        universe = list(set(sp500_stocks + nasdaq_stocks))
        
        # Limit universe size for testing
        universe = universe[:300]
        
    except Exception as e:
        logger.error(f"Error loading stock universe: {e}")
        # Fallback to a small test universe
        universe = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", 
            "AMD", "INTC", "NFLX", "DIS", "BA", "GS", "JPM", "V", 
            "MA", "PG", "KO", "PEP", "MCD", "WMT", "HD", "NKE"
        ]
    
    # Initialize the system
    system = MarketAnalysisSystem()
    
    try:
        # Screen stocks
        results = await system.screen_stocks(universe)
        
        if not results:
            logger.warning("No stocks passed screening criteria")
            return
        
        # Print results
        print(f"\nTop {len(results)} stocks for trading:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['symbol']} - Final Score: {result['final_score']:.4f}")
            print(f"   ML Score: {result['ml_score']:.4f}, Strategy Score: {result['strategy_score']:.4f}, News Score: {result['news_sentiment_score']:.4f}")
            print(f"   Price: ${result['price']:.2f}, Change: {result['price_change_pct']*100:.2f}%, Vol Ratio: {result['relative_volume']:.2f}")
            print(f"   RSI: {result['rsi_14']:.2f}, ADX: {result['adx']:.2f}")
            print(f"   News Sentiment: {result['news_sentiment']} ({result['news_count']} articles)")
            print()
    finally:
        # Clean up
        await system.close()

if __name__ == "__main__":
    asyncio.run(main())