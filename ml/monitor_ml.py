"""
Position Monitor ML System

This module implements a comprehensive ML-based position monitoring system
that tracks active trading positions, analyzes their performance, and
generates alerts when predefined conditions are met.
"""

import os
import json
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import aiohttp
import redis.asyncio as aioredis

# ML Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import joblib

# Import centralized logging system
from monitor.logging_utils import get_logger
logger = get_logger("position_monitor_ml")

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
TRADE_LLM_API_URL = os.getenv("TRADE_LLM_API_URL", "http://localhost:8000/api/trade_signal")
MODEL_PATH = os.getenv("MODEL_PATH", "./models")

# Alert thresholds
PROFIT_TARGET_PCT = float(os.getenv("PROFIT_TARGET_PCT", "0.05"))  # 5% profit target
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.03"))  # 3% stop loss
TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "0.02"))  # 2% trailing stop
MAX_POSITION_HOLD_DAYS = int(os.getenv("MAX_POSITION_HOLD_DAYS", "10"))  # Max days to hold
VOLATILITY_ALERT_THRESHOLD = float(os.getenv("VOLATILITY_ALERT_THRESHOLD", "0.5"))  # 50% increase in volatility


class PositionMonitorSystem:
    """ML-based system for monitoring active trading positions."""
    
    def __init__(self):
        self.session = None
        self.redis_client = None
        self.model = None
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
        
        # Load ML model
        self._load_model()
        
        self.initialized = True
        logger.info("Position Monitor System initialized")
    
    def _load_model(self):
        """Load pre-trained position monitoring model."""
        model_path = os.path.join(MODEL_PATH, "position_monitor_model.pt")
        if os.path.exists(model_path):
            try:
                self.model = torch.load(model_path, map_location=device)
                self.model.eval()
                logger.info("Loaded position monitoring model")
            except Exception as e:
                logger.error(f"Error loading position monitoring model: {e}")
                self.model = None
    
    async def monitor_positions(self) -> List[Dict[str, Any]]:
        """Monitor all active positions."""
        if not self.initialized:
            await self.initialize()
        
        if not self.redis_client:
            logger.error("Redis client not available")
            return []
        
        # Get all active positions
        try:
            position_keys = await self.redis_client.keys("position:*:active")
            
            if not position_keys:
                logger.info("No active positions found")
                return []
            
            logger.info(f"Found {len(position_keys)} active positions")
            
            # Process each position
            results = []
            for key in position_keys:
                position_data = await self.redis_client.get(key)
                if not position_data:
                    continue
                
                try:
                    position = json.loads(position_data)
                    result = await self.monitor_position(position)
                    results.append(result)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in position data: {key}")
                except Exception as e:
                    logger.error(f"Error monitoring position {key}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching active positions: {e}")
            return []
    
    async def monitor_position(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor a single position and generate alerts if needed."""
        if not self.initialized:
            await self.initialize()
        
        position_id = position.get('position_id')
        symbol = position.get('symbol')
        entry_price = float(position.get('entry_price', 0))
        entry_time = position.get('entry_time')
        
        logger.info(f"Monitoring position {position_id} for {symbol}")
        
        # Get current market data
        market_data = await self._fetch_market_data(symbol)
        if not market_data:
            logger.warning(f"Could not fetch market data for {symbol}")
            return {
                'position_id': position_id,
                'symbol': symbol,
                'success': False,
                'error': 'Market data not available'
            }
        
        # Calculate position stats
        stats = self._calculate_position_stats(position, market_data)
        
        # Generate alerts based on position performance
        alerts = self._generate_alerts(position, stats)
        
        # Use ML model for additional insights if available
        ml_insights = self._get_ml_insights(position, market_data) if self.model else {}
        
        # Combine all monitoring data
        monitoring_result = {
            'position_id': position_id,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'stats': stats,
            'alerts': alerts,
            'ml_insights': ml_insights,
            'success': True
        }
        
        # Store monitoring result
        await self._store_monitoring_result(monitoring_result)
        
        # Send alerts to Trade LLM if any
        if alerts:
            await self._send_alerts_to_trade_llm(monitoring_result)
        
        return monitoring_result
    
    async def _fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch current market data for a symbol."""
        if not self.session:
            return {}
        
        try:
            # Fetch current quote
            async with self.session.post(
                f"{os.getenv('DATA_API_BASE_URL', 'http://localhost:8000')}/api/quotes",
                json={"symbols": [symbol]},
                timeout=30  # Add timeout to prevent hanging
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch quote for {symbol}: {await response.text()}")
                    return {}
                
                try:
                    data = await response.json()
                except Exception as e:
                    logger.error(f"Failed to parse quote response for {symbol}: {e}")
                    return {}
                
                if not data.get("success", False):
                    logger.error(f"API returned unsuccessful response: {data}")
                    return {}
                
                quote = data.get("quotes", {}).get(symbol, {})
                if not quote:
                    logger.error(f"No quote data for {symbol}")
                    return {}
                
                # Fetch recent chart data
                try:
                    async with self.session.post(
                        f"{os.getenv('DATA_API_BASE_URL', 'http://localhost:8000')}/api/chart",
                        json={"symbol": symbol, "interval": "1d", "range": "1mo"},
                        timeout=30  # Add timeout to prevent hanging
                    ) as chart_response:
                        chart_data = {}
                        if chart_response.status == 200:
                            try:
                                chart_data = await chart_response.json()
                            except Exception as e:
                                logger.error(f"Failed to parse chart response for {symbol}: {e}")
                        
                        return {
                            'quote': quote,
                            'chart': chart_data.get('chart_data', {})
                        }
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout while fetching chart data for {symbol}")
                    # Return with quote data only
                    return {'quote': quote, 'chart': {}}
                
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return {}
    
    def _calculate_position_stats(self, position: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate position statistics based on current market data."""
        entry_price = float(position.get('entry_price', 0))
        entry_time = position.get('entry_time')
        position_type = position.get('position_type', 'long')  # 'long' or 'short'
        
        # Get current price
        quote = market_data.get('quote', {})
        current_price = float(quote.get('Close', 0))
        
        if current_price == 0 or entry_price == 0:
            return {'price': 0, 'change': 0, 'change_pct': 0}
        
        # Calculate price change
        if position_type == 'long':
            price_change = current_price - entry_price
            price_change_pct = price_change / entry_price
        else:  # short position
            price_change = entry_price - current_price
            price_change_pct = price_change / entry_price
        
        # Calculate position age
        entry_datetime = datetime.fromisoformat(entry_time) if entry_time else datetime.now()
        position_age = (datetime.now() - entry_datetime).total_seconds() / (60 * 60 * 24)  # in days
        
        # Calculate volatility
        chart_data = market_data.get('chart', {})
        volatility = self._calculate_volatility(chart_data)
        
        # Calculate volume profile
        volume_profile = self._analyze_volume_profile(chart_data)
        
        return {
            'price': current_price,
            'change': price_change,
            'change_pct': price_change_pct,
            'age_days': position_age,
            'volatility': volatility,
            'volume_profile': volume_profile
        }
    
    def _convert_chart_to_dataframe(self, chart_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert chart data to a pandas DataFrame - utility method to avoid code duplication."""
        if not chart_data:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame()
            for date, values in chart_data.items():
                if isinstance(values, dict):
                    df = pd.concat([df, pd.DataFrame([values], index=[date])])
            
            if df.empty:
                return pd.DataFrame()
            
            # Sort by date
            df = df.sort_index()
            
            # Ensure numeric columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
        except Exception as e:
            logger.error(f"Error converting chart data to DataFrame: {e}")
            return pd.DataFrame()
    
    def _calculate_volatility(self, chart_data: Dict[str, Any]) -> float:
        """Calculate recent volatility from chart data."""
        if not chart_data:
            return 0.0
        
        try:
            # Convert chart data to DataFrame using the utility method
            df = self._convert_chart_to_dataframe(chart_data)
            
            if df.empty:
                return 0.0
            
            # Calculate daily returns
            if 'Close' in df.columns:
                returns = df['Close'].pct_change().dropna()
                
                # Calculate volatility (standard deviation of returns)
                if len(returns) > 1:
                    return float(returns.std() * np.sqrt(252))  # Annualized
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def _analyze_volume_profile(self, chart_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume profile from chart data."""
        if not chart_data:
            return {'increasing': False, 'avg_volume': 0}
        
        try:
            # Convert chart data to DataFrame using the utility method
            df = self._convert_chart_to_dataframe(chart_data)
            
            if df.empty:
                return {'increasing': False, 'avg_volume': 0}
            
            # Analyze volume
            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
                
                if len(df) >= 5:
                    recent_avg = df['Volume'].iloc[-5:].mean()
                    previous_avg = df['Volume'].iloc[-10:-5].mean() if len(df) >= 10 else df['Volume'].iloc[:-5].mean()
                    
                    return {
                        'increasing': recent_avg > previous_avg,
                        'avg_volume': float(recent_avg),
                        'volume_change': float((recent_avg / previous_avg) - 1) if previous_avg > 0 else 0
                    }
            
            return {'increasing': False, 'avg_volume': 0}
            
        except Exception as e:
            logger.error(f"Error analyzing volume profile: {e}")
            return {'increasing': False, 'avg_volume': 0}
    
    def _generate_alerts(self, position: Dict[str, Any], stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on position performance."""
        alerts = []
        position_type = position.get('position_type', 'long')
        
        # Profit target alert
        if stats.get('change_pct', 0) >= PROFIT_TARGET_PCT:
            alerts.append({
                'type': 'profit_target',
                'level': 'info',
                'message': f"Position reached profit target of {PROFIT_TARGET_PCT*100:.1f}%"
            })
        
        # Stop loss alert
        if stats.get('change_pct', 0) <= -STOP_LOSS_PCT:
            alerts.append({
                'type': 'stop_loss',
                'level': 'warning',
                'message': f"Position hit stop loss of {STOP_LOSS_PCT*100:.1f}%"
            })
        
        # Trailing stop alert
        max_profit = float(position.get('max_profit_pct', 0))
        if max_profit > 0 and (max_profit - stats.get('change_pct', 0)) >= TRAILING_STOP_PCT:
            alerts.append({
                'type': 'trailing_stop',
                'level': 'warning',
                'message': f"Position dropped {TRAILING_STOP_PCT*100:.1f}% from peak profit of {max_profit*100:.1f}%"
            })
        
        # Time-based alert
        if stats.get('age_days', 0) >= MAX_POSITION_HOLD_DAYS:
            alerts.append({
                'type': 'max_hold_time',
                'level': 'info',
                'message': f"Position held for {MAX_POSITION_HOLD_DAYS} days (maximum hold time)"
            })
        
        # Volatility alert
        entry_volatility = float(position.get('entry_volatility', 0))
        current_volatility = stats.get('volatility', 0)
        
        if entry_volatility > 0 and current_volatility > 0:
            volatility_change = (current_volatility / entry_volatility) - 1
            if volatility_change >= VOLATILITY_ALERT_THRESHOLD:
                alerts.append({
                    'type': 'volatility_increase',
                    'level': 'warning',
                    'message': f"Volatility increased by {volatility_change*100:.1f}% since entry"
                })
        
        # Volume alert
        volume_profile = stats.get('volume_profile', {})
        if volume_profile.get('increasing', False) and volume_profile.get('volume_change', 0) > 0.5:
            alerts.append({
                'type': 'volume_surge',
                'level': 'info',
                'message': f"Volume increased by {volume_profile.get('volume_change', 0)*100:.1f}%"
            })
        
        return alerts
    
    def _get_ml_insights(self, position: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get ML model insights for the position."""
        if not self.model:
            return {}
        
        try:
            # Extract features for the model
            features = self._extract_model_features(position, market_data)
            
            # Validate features
            if not features or len(features) == 0:
                logger.warning("No valid features extracted for ML model")
                return {}
            
            # Convert to tensor
            try:
                input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            except Exception as e:
                logger.error(f"Error converting features to tensor: {e}")
                return {}
            
            # Get model prediction
            with torch.no_grad():
                try:
                    outputs = self.model(input_tensor)
                    
                    # Interpret outputs
                    exit_probability = torch.sigmoid(outputs[0]).item()
                    
                    return {
                        'exit_probability': exit_probability,
                        'hold_probability': 1 - exit_probability,
                        'recommendation': 'exit' if exit_probability > 0.7 else 'hold'
                    }
                except Exception as e:
                    logger.error(f"Error during model inference: {e}")
                    return {}
                
        except Exception as e:
            logger.error(f"Error getting ML insights: {e}")
            return {}
    
    def _extract_model_features(self, position: Dict[str, Any], market_data: Dict[str, Any]) -> List[float]:
        """Extract features for the ML model."""
        # This would be customized based on the actual model's expected inputs
        features = []
        
        # Position features
        features.append(float(position.get('position_type', 'long') == 'long'))
        features.append(float(position.get('entry_price', 0)))
        
        # Market data features
        quote = market_data.get('quote', {})
        features.append(float(quote.get('Close', 0)))
        features.append(float(quote.get('Volume', 0)))
        
        # Technical indicators would be added here
        
        return features

    async def _send_alerts_to_trade_llm(self, monitoring_result: Dict[str, Any]) -> bool:
        """Send alerts directly to the Trade LLM for exit decisions."""
        alerts = monitoring_result.get('alerts', [])
        if not alerts:
            return False
        
        position_id = monitoring_result.get('position_id')
        symbol = monitoring_result.get('symbol')
        
        if not position_id or not symbol:
            logger.error("Missing position_id or symbol in monitoring result")
            return False
        
        try:
            # Prepare signal data
            signal_data = {
                'position_id': position_id,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'alerts': alerts,
                'current_price': monitoring_result.get('stats', {}).get('price', 0),
                'monitoring_data': monitoring_result,
                'source': 'position_monitor'
            }
            
            # Send to Trade LLM
            try:
                async with self.session.post(
                    TRADE_LLM_API_URL,
                    json=signal_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to send alerts to Trade LLM: {await response.text()}")
                        return False
                    
                    try:
                        response_data = await response.json()
                        logger.info(f"Sent alerts to Trade LLM: {response_data.get('status', 'unknown')}")
                        return True
                    except Exception as e:
                        logger.error(f"Error parsing Trade LLM response: {e}")
                        return False
            except asyncio.TimeoutError:
                logger.error(f"Timeout sending alerts to Trade LLM for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending alerts to Trade LLM: {e}")
            return False
    
    async def _store_monitoring_result(self, result: Dict[str, Any]) -> bool:
        """Store monitoring result in Redis."""
        if not self.redis_client:
            return False
            
        try:
            position_id = result.get('position_id')
            if not position_id:
                logger.error("Missing position_id in monitoring result")
                return False
            
            # Store current monitoring result
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            monitoring_key = f"position:{position_id}:monitoring:{timestamp}"
            
            try:
                result_json = json.dumps(result)
            except Exception as e:
                logger.error(f"Error serializing monitoring result to JSON: {e}")
                return False
            
            # Store in Redis with error handling
            try:
                await self.redis_client.set(monitoring_key, result_json)
                await self.redis_client.expire(monitoring_key, 60 * 60 * 24)  # 24 hour expiry
                
                # Update latest monitoring result
                latest_key = f"position:{position_id}:monitoring:latest"
                await self.redis_client.set(latest_key, result_json)
                await self.redis_client.expire(latest_key, 60 * 60 * 24)  # 24 hour expiry
                
                # Update position history
                history_key = f"position:{position_id}:history"
                current_stats = result.get('stats', {}).copy()  # Create a copy to avoid modifying the original
                current_stats.update({
                    'timestamp': datetime.now().isoformat(),
                    'monitoring_id': monitoring_key
                })
                
                # Add alerts to history
                alerts = result.get('alerts', [])
                if alerts:
                    current_stats['alerts'] = alerts
                
                # Get existing history
                history_data = await self.redis_client.get(history_key)
                if history_data:
                    try:
                        history = json.loads(history_data)
                        history.append(current_stats)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in history data for position {position_id}, creating new history")
                        history = [current_stats]
                else:
                    history = [current_stats]
                
                # Store updated history
                await self.redis_client.set(history_key, json.dumps(history))
                await self.redis_client.expire(history_key, 60 * 60 * 24 * 7)  # 7 day expiry
                
                logger.info(f"Stored monitoring result for position {position_id}")
                return True
            except Exception as e:
                logger.error(f"Redis operation failed: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Error storing monitoring result: {e}")
            return False
    
    async def close(self):
        """Close connections and release resources."""
        if self.session:
            await self.session.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Position Monitor System resources released")


async def run_continuous_monitoring(interval: int = 60):
    """Run continuous position monitoring in background."""
    system = PositionMonitorSystem()
    
    try:
        await system.initialize()
        
        while True:
            logger.info(f"Running position monitoring cycle")
            try:
                await system.monitor_positions()
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}")
            
            # Wait for next cycle
            logger.info(f"Waiting {interval} seconds for next monitoring cycle")
            await asyncio.sleep(interval)
            
    except asyncio.CancelledError:
        logger.info("Continuous monitoring task cancelled")
    except Exception as e:
        logger.error(f"Error in continuous monitoring: {e}")
    finally:
        await system.close()


async def monitor_single_position(position_id: str) -> Dict[str, Any]:
    """Monitor a specific position (API endpoint handler)."""
    if not position_id:
        logger.error("Invalid position_id provided")
        return {'success': False, 'error': 'Invalid position ID'}
        
    system = PositionMonitorSystem()
    
    try:
        await system.initialize()
        
        # Get position data from Redis
        position_key = f"position:{position_id}:active"
        position_data = await system.redis_client.get(position_key)
        
        if not position_data:
            logger.warning(f"Position {position_id} not found")
            return {'success': False, 'error': 'Position not found'}
        
        try:
            position = json.loads(position_data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in position data for {position_id}: {e}")
            return {'success': False, 'error': 'Invalid position data format'}
            
        result = await system.monitor_position(position)
        
        await system.close()
        return result
        
    except Exception as e:
        logger.error(f"Error monitoring position {position_id}: {e}")
        return {'success': False, 'error': str(e)}


async def main():
    """Main entry point for position monitoring system."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Position Monitor ML System')
    parser.add_argument('--mode', choices=['continuous', 'once'], default='continuous',
                       help='Monitoring mode: continuous or one-time')
    parser.add_argument('--interval', type=int, default=60,
                       help='Monitoring interval in seconds (for continuous mode)')
    parser.add_argument('--position', type=str, help='Monitor specific position ID (for one-time mode)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                       help='Logging level')
    parser.add_argument('--output-file', type=str, help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging_level = getattr(logging, args.log_level)
    logger.setLevel(logging_level)
    
    try:
        if args.mode == 'continuous':
            logger.info(f"Starting continuous monitoring with {args.interval}s interval")
            await run_continuous_monitoring(args.interval)
        elif args.mode == 'once':
            if args.position:
                logger.info(f"Monitoring single position: {args.position}")
                result = await monitor_single_position(args.position)
                
                # Output results
                if args.output_file:
                    try:
                        with open(args.output_file, 'w') as f:
                            json.dump(result, f, indent=2)
                        logger.info(f"Results saved to {args.output_file}")
                    except Exception as e:
                        logger.error(f"Error writing to output file: {e}")
                else:
                    print(json.dumps(result, indent=2))
            else:
                logger.info("Monitoring all active positions")
                system = PositionMonitorSystem()
                await system.initialize()
                results = await system.monitor_positions()
                await system.close()
                
                # Output results
                if args.output_file:
                    try:
                        with open(args.output_file, 'w') as f:
                            json.dump(results, f, indent=2)
                        logger.info(f"Results saved to {args.output_file}")
                    except Exception as e:
                        logger.error(f"Error writing to output file: {e}")
                else:
                    print(json.dumps(results, indent=2))
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))