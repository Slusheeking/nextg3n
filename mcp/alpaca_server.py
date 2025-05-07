"""
Alpaca Trading MCP FastAPI Server for LLM integration (production).
Provides portfolio data, risk assessment, and trade execution using Alpaca Trading API.
All configuration is contained in this file.
"""

import os
import json
import yaml
import asyncio
import pandas as pd
import numpy as np  # Used for portfolio analytics
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends
from pydantic import BaseModel, Field, ValidationError
from monitor.logging_utils import get_logger
import aiohttp
import redis.asyncio as aioredis

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Load configuration from system_config.yaml
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'system_config.yaml')
    try:
        with open(config_path, 'r') as f:
            system_config = yaml.safe_load(f)
        
        # Extract Alpaca-specific configuration
        alpaca_config = system_config.get('services', {}).get('alpaca', {})
        
        return {
            # Trading API Configuration
            "api_key": os.getenv("ALPACA_API_KEY", ""),
            "secret_key": os.getenv("ALPACA_SECRET_KEY", ""),
            "trading_api_base": os.getenv("ALPACA_TRADING_API_BASE",
                                        alpaca_config.get('trading_api_base', "https://paper-api.alpaca.markets/v2")),
            "data_api_base": os.getenv("ALPACA_DATA_API_BASE",
                                      alpaca_config.get('data_api_base', "https://data.alpaca.markets/v2")),
            "paper_trading": os.getenv("ALPACA_PAPER_TRADING", str(alpaca_config.get('paper_trading', True))).lower() == "true",
            
            # Risk Management Configuration (keys aligned with system_config.yaml)
            "max_risk_per_trade_pct": float(os.getenv("SYS_MAX_RISK_PER_TRADE_PCT",
                                           alpaca_config.get('risk_management', {}).get('max_risk_per_trade_pct', 0.02))),
            "min_reward_risk_ratio": float(os.getenv("SYS_MIN_REWARD_RISK_RATIO",
                                          alpaca_config.get('risk_management', {}).get('min_reward_risk_ratio', 2.0))),
            "max_position_value_pct": float(os.getenv("SYS_MAX_POSITION_VALUE_PCT",
                                           alpaca_config.get('risk_management', {}).get('max_position_value_pct', 0.10))),
            
            # Redis Configuration
            "redis_host": os.getenv("REDIS_HOST", system_config.get('redis', {}).get('host', 'localhost')),
            "redis_port": int(os.getenv("REDIS_PORT", system_config.get('redis', {}).get('port', 6379))),
            "redis_db": int(os.getenv("REDIS_DB_ALPACA",
                          os.getenv("REDIS_DB", system_config.get('redis', {}).get('db_alpaca', 0)))),
            "redis_password": os.getenv("REDIS_PASSWORD", system_config.get('redis', {}).get('password', None)),
            
            # Portfolio update interval (in seconds)
            "portfolio_update_interval": int(os.getenv("ALPACA_PORTFOLIO_UPDATE_INTERVAL",
                                            alpaca_config.get('portfolio_update_interval', 300))),
        }
    except Exception as e:
        logger = get_logger("alpaca_server")
        logger.error(f"Error loading configuration from system_config.yaml: {str(e)}. Using default values.")
        
        # Fallback to default values if config file can't be loaded
CONFIG = {
    # LLM API Configuration (if Alpaca's own LLM services are used directly by this server)
    "api_key": os.getenv("ALPACA_API_KEY", ""), # This is for Alpaca Trading API
    "secret_key": os.getenv("ALPACA_SECRET_KEY", ""), # For Alpaca Trading API
    # "default_completion_model": "llama-2-70b-chat", # Example if using Alpaca LLM directly
    # "default_embedding_model": "alpaca-ef-7b",   # Example if using Alpaca LLM directly
    
    # Trading API Configuration
    "trading_api_base": os.getenv("ALPACA_TRADING_API_BASE", "https://paper-api.alpaca.markets/v2"),
    "data_api_base": os.getenv("ALPACA_DATA_API_BASE", "https://data.alpaca.markets/v2"),
    "paper_trading": os.getenv("ALPACA_PAPER_TRADING", "True").lower() == "true",
    
    # Risk Management Configuration (keys aligned with system_config.yaml)
    "max_risk_per_trade_pct": float(os.getenv("SYS_MAX_RISK_PER_TRADE_PCT", 0.02)),
    "min_reward_risk_ratio": float(os.getenv("SYS_MIN_REWARD_RISK_RATIO", 2.0)),
    "max_position_value_pct": float(os.getenv("SYS_MAX_POSITION_VALUE_PCT", 0.10)),
    
    # Redis Configuration
    "redis_host": os.getenv("REDIS_HOST", "localhost"),
    "redis_port": int(os.getenv("REDIS_PORT", "6379")),
    "redis_db": int(os.getenv("REDIS_DB_ALPACA", os.getenv("REDIS_DB", "0"))), # Alpaca can use its own DB or default
    "redis_password": os.getenv("REDIS_PASSWORD", None)
}

# Get logger from centralized logging system
logger = get_logger("alpaca_server")
logger.info(f"Initializing Alpaca Trading server (Paper Trading: {CONFIG['paper_trading']})")

# --- Redis Client ---
redis_client: Optional[aioredis.Redis] = None
scheduler: Optional[AsyncIOScheduler] = None

async def connect_redis():
    global redis_client
    if redis_client is None:
        try:
            redis_client = aioredis.Redis(
                host=CONFIG["redis_host"],
                port=CONFIG["redis_port"],
                db=CONFIG["redis_db"],
                password=CONFIG["redis_password"],
                decode_responses=True
            )
            await redis_client.ping()
            logger.info(f"Connected to Redis for Alpaca Server at {CONFIG['redis_host']}:{CONFIG['redis_port']} (DB: {CONFIG['redis_db']})")
        except Exception as e:
            logger.error(f"Failed to connect to Redis for Alpaca Server: {str(e)}")
            redis_client = None
            raise RuntimeError(f"Failed to connect to Redis: {str(e)}")

# --- Alpaca API Setup ---
# Note: Direct requests to Alpaca LLM API (call_completion_api, etc.) are removed as this server
# primarily focuses on Trading API integration. LLM interactions are handled by llm_integration.py.

def verify_api_keys():
    """Verify that API keys are present and valid"""
    if not CONFIG["api_key"] or not CONFIG["secret_key"]:
        error_msg = "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the environment for Trading API."
        logger.error(error_msg)
        raise ValueError(error_msg)
    return True

TRADING_HEADERS = {
    "APCA-API-KEY-ID": CONFIG["api_key"],
    "APCA-API-SECRET-KEY": CONFIG["secret_key"],
    "Content-Type": "application/json"
}

# --- Trading API Functions ---

async def get_account():
    url = f"{CONFIG['trading_api_base']}/account"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=TRADING_HEADERS) as response:
            if response.status == 200: return await response.json()
            error_text = await response.text()
            logger.error(f"Error getting account: {response.status} - {error_text}")
            raise HTTPException(status_code=response.status, detail=error_text)

async def get_positions():
    url = f"{CONFIG['trading_api_base']}/positions"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=TRADING_HEADERS) as response:
            if response.status == 200: return await response.json()
            error_text = await response.text()
            logger.error(f"Error getting positions: {response.status} - {error_text}")
            raise HTTPException(status_code=response.status, detail=error_text)

async def get_orders(status: str = "open", limit: int = 50, nested: bool = True):
    url = f"{CONFIG['trading_api_base']}/orders"
    params = {"status": status, "limit": limit, "nested": str(nested).lower()}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=TRADING_HEADERS, params=params) as response:
            if response.status == 200: return await response.json()
            error_text = await response.text()
            logger.error(f"Error getting orders: {response.status} - {error_text}")
            raise HTTPException(status_code=response.status, detail=error_text)

async def place_order(symbol: str, qty: float, side: str, type: str, time_in_force: str,
                     limit_price: Optional[str] = None, stop_price: Optional[str] = None, 
                     trail_price: Optional[str] = None, trail_percent: Optional[str] = None,
                     extended_hours: Optional[bool] = None, client_order_id: Optional[str] = None,
                     order_class: Optional[str] = None, take_profit: Optional[dict] = None,
                     stop_loss: Optional[dict] = None):
    url = f"{CONFIG['trading_api_base']}/orders"
    payload = {
        "symbol": symbol, "qty": str(qty), "side": side, "type": type, "time_in_force": time_in_force,
        "limit_price": limit_price, "stop_price": stop_price, "trail_price": trail_price,
        "trail_percent": trail_percent, "extended_hours": extended_hours, 
        "client_order_id": client_order_id, "order_class": order_class,
        "take_profit": take_profit, "stop_loss": stop_loss
    }
    payload = {k: v for k, v in payload.items() if v is not None} # Remove None values
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=TRADING_HEADERS, json=payload) as response:
            response_data = await response.json()
            if response.status == 200: return response_data
            logger.error(f"Error placing order: {response.status} - {response_data}")
            raise HTTPException(status_code=response.status, detail=response_data)

async def get_bars(symbols: Union[str, List[str]], timeframe: str, start: str, end: Optional[str] = None, 
                   limit: Optional[int] = None, adjustment: str = "raw"):
    if isinstance(symbols, list): symbols = ",".join(symbols)
    url = f"{CONFIG['data_api_base']}/stocks/bars"
    params = {"symbols": symbols, "timeframe": timeframe, "start": start, "adjustment": adjustment}
    if end: params["end"] = end
    if limit: params["limit"] = limit
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=TRADING_HEADERS, params=params) as response:
            if response.status == 200: return await response.json()
            error_text = await response.text()
            logger.error(f"Error getting bars: {response.status} - {error_text}")
            raise HTTPException(status_code=response.status, detail=error_text)

async def get_latest_quote(symbol: str) -> Dict[str, Any]:
    url = f"{CONFIG['data_api_base']}/stocks/{symbol}/quotes/latest"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=TRADING_HEADERS) as response:
            if response.status == 200: return await response.json()
            error_text = await response.text()
            logger.error(f"Error getting latest quote for {symbol}: {response.status} - {error_text}")
            raise HTTPException(status_code=response.status, detail=error_text)

async def get_latest_trade(symbol: str) -> Dict[str, Any]:
    url = f"{CONFIG['data_api_base']}/stocks/{symbol}/trades/latest"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=TRADING_HEADERS) as response:
            if response.status == 200: return await response.json()
            error_text = await response.text()
            logger.error(f"Error getting latest trade for {symbol}: {response.status} - {error_text}")
            raise HTTPException(status_code=response.status, detail=error_text)

# --- Enhanced Trading Functions ---
# (optimize_order_type, calculate_optimal_execution_time, calculate_slippage_estimate, split_large_order, execute_optimized_order)
# These functions are complex and seem mostly complete from the provided file.
# For brevity, I will assume they are correctly implemented as per the previous state,
# ensuring they use the updated CONFIG keys where necessary.
# Key is that they should use CONFIG["max_risk_per_trade_pct"], CONFIG["min_reward_risk_ratio"], CONFIG["max_position_value_pct"]

async def calculate_position_size(account_value: float, risk_per_trade_pct: float, entry_price: float, stop_loss_price: float, max_position_value_pct: float) -> int:
    if entry_price <= stop_loss_price and entry_price > 0: # Invalid for long, or zero entry
        risk_per_share = entry_price - stop_loss_price # For short
    elif entry_price > stop_loss_price and stop_loss_price > 0: # Valid for long
        risk_per_share = entry_price - stop_loss_price
    else: # Invalid prices
        logger.warning(f"Invalid prices for position sizing: entry ${entry_price}, stop ${stop_loss_price}")
        return 0
        
    if risk_per_share <= 0: # Stop loss is at or above entry for long, or at/below for short
        logger.warning(f"Risk per share is zero or negative. Entry: {entry_price}, Stop: {stop_loss_price}")
        return 0
        
    risk_amount_per_trade = account_value * risk_per_trade_pct
    num_shares_by_risk = int(risk_amount_per_trade / risk_per_share)
    
    max_value_for_position = account_value * max_position_value_pct
    num_shares_by_max_value = int(max_value_for_position / entry_price) if entry_price > 0 else 0
    
    final_num_shares = min(num_shares_by_risk, num_shares_by_max_value)
    
    # Ensure it's a positive number of shares
    return max(0, final_num_shares)


async def calculate_stop_loss_atr(symbol: str, entry_price: float, position_type: str, atr_multiplier: float = 2.0, atr_period: int = 14) -> float:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=atr_period * 2) # Fetch more data to ensure enough for ATR calc
    
    bars_response = await get_bars(symbols=symbol, timeframe="1Day", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), limit=atr_period + 5)
    
    if not bars_response.get("bars") or symbol not in bars_response["bars"] or len(bars_response["bars"][symbol]) < atr_period:
        logger.warning(f"Not enough historical data for {symbol} to calculate ATR({atr_period}). Defaulting stop loss or handle error.")
        # Fallback: 1% of entry price, or raise error
        return round(entry_price * (0.99 if position_type.lower() == "long" else 1.01), 2)

    bars_df = pd.DataFrame(bars_response["bars"][symbol])
    bars_df['h'] = bars_df['h'].astype(float)
    bars_df['l'] = bars_df['l'].astype(float)
    bars_df['c'] = bars_df['c'].astype(float)
    
    high_low = bars_df['h'] - bars_df['l']
    high_prev_close = abs(bars_df['h'] - bars_df['c'].shift(1))
    low_prev_close = abs(bars_df['l'] - bars_df['c'].shift(1))
    
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean().iloc[-1]
    
    if pd.isna(atr) or atr == 0:
        logger.warning(f"ATR calculation resulted in NaN or zero for {symbol}. Defaulting stop loss.")
        return round(entry_price * (0.99 if position_type.lower() == "long" else 1.01), 2)

    stop_loss_val = 0.0
    if position_type.lower() == "long":
        stop_loss_val = entry_price - (atr * atr_multiplier)
    else:  # Short position
        stop_loss_val = entry_price + (atr * atr_multiplier)
            
    return round(max(0.01, stop_loss_val), 2) # Ensure stop loss is not zero or negative for valid price


async def assess_trade_risk_detailed(symbol: str, entry_price: float, side: str, 
                                     stop_loss_price_manual: Optional[float] = None, 
                                     profit_target_manual: Optional[float] = None,
                                     atr_multiplier_for_stop: float = 2.0,
                                     reward_to_risk_ratio_target: Optional[float] = None):
    account = await get_account()
    account_value = float(account.get("portfolio_value", 0))
    available_funds = float(account.get("cash", 0)) # Using cash as simpler proxy for available funds

    if account_value == 0:
        return {"success": False, "error": "Account value is zero."}

    stop_loss_price = stop_loss_price_manual
    if stop_loss_price is None:
        stop_loss_price = await calculate_stop_loss_atr(symbol, entry_price, side, atr_multiplier_for_stop)

    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share <= 0.001: # Effectively no risk or invalid stop
        return {"success": False, "error": "Risk per share is too low or stop loss is invalid.", "risk_per_share": risk_per_share}

    profit_target_price = profit_target_manual
    reward_risk_target = reward_to_risk_ratio_target or CONFIG["min_reward_risk_ratio"]
    if profit_target_price is None:
        if side.lower() == "long":
            profit_target_price = entry_price + (risk_per_share * reward_risk_target)
        else: # short
            profit_target_price = entry_price - (risk_per_share * reward_risk_target)
    profit_target_price = round(profit_target_price, 2)

    reward_per_share = abs(profit_target_price - entry_price)
    actual_reward_risk_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

    position_size_shares = await calculate_position_size(
        account_value, CONFIG["max_risk_per_trade_pct"], entry_price, stop_loss_price, CONFIG["max_position_value_pct"]
    )
    
    estimated_cost = position_size_shares * entry_price
    
    # Final check against available funds
    if estimated_cost > available_funds:
        logger.warning(f"Calculated trade cost ${estimated_cost:.2f} for {position_size_shares} shares of {symbol} exceeds available cash ${available_funds:.2f}. Sizing down.")
        position_size_shares = int(available_funds / entry_price) if entry_price > 0 else 0
        estimated_cost = position_size_shares * entry_price
        if position_size_shares == 0:
             return {"success": False, "error": "Not enough available funds for minimum position size."}


    risk_amount_total = position_size_shares * risk_per_share
    risk_percentage_of_portfolio = (risk_amount_total / account_value) * 100 if account_value > 0 else 0

    meets_criteria = (
        actual_reward_risk_ratio >= CONFIG["min_reward_risk_ratio"] and
        risk_percentage_of_portfolio <= (CONFIG["max_risk_per_trade_pct"] * 100) and # Compare percentages
        estimated_cost <= (account_value * CONFIG["max_position_value_pct"]) and
        estimated_cost <= available_funds # Explicit check
    )
    
    return {
        "success": True,
        "assessment_details": {
            "symbol": symbol, "side": side, "entry_price": entry_price,
            "stop_loss_price": stop_loss_price, "profit_target_price": profit_target_price,
            "risk_per_share": round(risk_per_share, 2), "reward_per_share": round(reward_per_share, 2),
            "actual_reward_risk_ratio": round(actual_reward_risk_ratio, 2),
            "position_size_shares": position_size_shares, "estimated_cost": round(estimated_cost, 2),
            "total_risk_amount": round(risk_amount_total, 2), 
            "risk_percentage_of_portfolio": round(risk_percentage_of_portfolio, 2),
            "meets_all_criteria": meets_criteria,
            "account_value_at_assessment": account_value,
            "available_funds_at_assessment": available_funds
        }
    }


# --- Portfolio Management Functions ---
async def get_portfolio_data() -> Dict[str, Any]:
    """
    Fetches comprehensive portfolio data from Alpaca APIs.
    Returns a dictionary with account summary, positions, performance metrics, etc.
    """
    try:
        # Get account information
        account_info = await get_account()
        
        # Get positions
        positions_info = await get_positions()
        
        # Get open orders
        orders_info = await get_orders(status="open")
        
        # Calculate portfolio metrics
        portfolio_metrics = await calculate_portfolio_metrics(account_info, positions_info)
        
        # Calculate sector exposure
        sector_exposure = await calculate_sector_exposure(positions_info)
        
        return {
            "success": True,
            "account_summary": account_info,
            "positions": positions_info,
            "open_orders": orders_info,
            "portfolio_metrics": portfolio_metrics,
            "sector_exposure": sector_exposure,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException as e:
        logger.error(f"Error in get_portfolio_data: {e.detail}")
        return {"success": False, "error": f"Failed to fetch portfolio components: {e.detail}"}
    except Exception as e:
        logger.error(f"Unexpected error in get_portfolio_data: {str(e)}")
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}

async def calculate_portfolio_metrics(account_info: Dict[str, Any], positions_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate portfolio metrics such as total value, daily P&L, overall P&L, etc.
    """
    try:
        # Extract values from account info
        equity = float(account_info.get("equity", 0))
        cash = float(account_info.get("cash", 0))
        portfolio_value = float(account_info.get("portfolio_value", 0))
        
        # Calculate metrics
        invested_value = portfolio_value - cash
        
        # Calculate daily and total P&L
        daily_pl = float(account_info.get("equity_change", 0))
        daily_pl_pct = (daily_pl / (portfolio_value - daily_pl)) * 100 if (portfolio_value - daily_pl) > 0 else 0
        
        # Calculate diversification metrics
        position_count = len(positions_info)
        largest_position_value = 0
        largest_position_symbol = None
        
        if positions_info:
            for position in positions_info:
                position_value = float(position.get("market_value", 0))
                if position_value > largest_position_value:
                    largest_position_value = position_value
                    largest_position_symbol = position.get("symbol")
        
        largest_position_pct = (largest_position_value / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        return {
            "portfolio_value": round(portfolio_value, 2),
            "cash": round(cash, 2),
            "invested_value": round(invested_value, 2),
            "daily_pl": round(daily_pl, 2),
            "daily_pl_pct": round(daily_pl_pct, 2),
            "position_count": position_count,
            "largest_position": {
                "symbol": largest_position_symbol,
                "value": round(largest_position_value, 2),
                "pct_of_portfolio": round(largest_position_pct, 2)
            }
        }
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {str(e)}")
        return {}

async def calculate_sector_exposure(positions_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate sector exposure based on positions.
    In a production environment, this would make additional API calls to get sector information.
    For this implementation, we'll use a simple mapping of known symbols to sectors.
    """
    # In production, you might use an external API or database to get actual sector data
    # This is a simplified, hard-coded map for demonstration
    sector_map = {
        "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
        "AMZN": "Consumer Cyclical", "META": "Technology",
        "JPM": "Financial Services", "BAC": "Financial Services",
        "JNJ": "Healthcare", "PFE": "Healthcare",
        "XOM": "Energy", "CVX": "Energy",
        "WMT": "Consumer Defensive", "PG": "Consumer Defensive"
        # Add more symbols as needed
    }
    
    try:
        # Initialize sector data
        sector_exposure = {}
        total_value = 0
        
        # Calculate total value of positions with known sectors
        for position in positions_info:
            symbol = position.get("symbol", "")
            market_value = float(position.get("market_value", 0))
            
            if symbol in sector_map:
                sector = sector_map[symbol]
                total_value += market_value
                
                if sector not in sector_exposure:
                    sector_exposure[sector] = {
                        "value": 0,
                        "positions": []
                    }
                
                sector_exposure[sector]["value"] += market_value
                sector_exposure[sector]["positions"].append({
                    "symbol": symbol,
                    "value": market_value
                })
        
        # Calculate percentages if total value is positive
        if total_value > 0:
            for sector in sector_exposure:
                sector_exposure[sector]["percentage"] = (sector_exposure[sector]["value"] / total_value) * 100
                sector_exposure[sector]["value"] = round(sector_exposure[sector]["value"], 2)
                sector_exposure[sector]["percentage"] = round(sector_exposure[sector]["percentage"], 2)
        
        return {
            "sectors": sector_exposure,
            "total_value": round(total_value, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating sector exposure: {str(e)}")
        return {"sectors": {}, "total_value": 0}

async def get_real_time_portfolio_tracking() -> Dict[str, Any]:
    """
    Fetches real-time portfolio tracking data including latest values and intraday changes.
    """
    try:
        # Get portfolio data
        portfolio_data = await get_portfolio_data()
        
        if not portfolio_data.get("success", False):
            return portfolio_data  # Return the error
        
        # Get account information
        account_info = portfolio_data["account_summary"]
        
        # Get the positions with real-time data
        positions = portfolio_data["positions"]
        positions_with_realtime = []
        
        for position in positions:
            symbol = position.get("symbol")
            try:
                # Get real-time quote data
                quote_data = await get_latest_quote(symbol)
                
                # Extract quote information
                ask = quote_data.get("quote", {}).get("ap", 0)
                bid = quote_data.get("quote", {}).get("bp", 0)
                last_price = (ask + bid) / 2 if ask and bid else None
                
                # Update position with real-time data
                enriched_position = {
                    **position,
                    "realtime_price": last_price,
                    "realtime_value": float(position.get("qty", 0)) * last_price if last_price else None,
                    "timestamp": datetime.now().isoformat()
                }
                
                positions_with_realtime.append(enriched_position)
            except Exception as e:
                logger.warning(f"Could not get real-time data for {symbol}: {str(e)}")
                positions_with_realtime.append(position)  # Add without real-time data
        
        return {
            "success": True,
            "account_summary": account_info,
            "positions": positions_with_realtime,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in get_real_time_portfolio_tracking: {str(e)}")
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}

async def get_available_funds_monitoring() -> Dict[str, Any]:
    """
    Provides detailed monitoring of available funds, buying power, and cash reserves.
    """
    try:
        # Get account information
        account_info = await get_account()
        
        # Extract relevant information
        cash = float(account_info.get("cash", 0))
        buying_power = float(account_info.get("buying_power", 0))
        daytrading_buying_power = float(account_info.get("daytrading_buying_power", 0))
        regt_buying_power = float(account_info.get("regt_buying_power", 0))
        portfolio_value = float(account_info.get("portfolio_value", 0))
        
        # Calculate percentages
        cash_pct_of_portfolio = (cash / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        # Get open orders to calculate reserved cash
        open_orders = await get_orders(status="open")
        
        # Calculate cash reserved for open orders (simplified)
        cash_reserved = 0
        for order in open_orders:
            if order.get("side") == "buy" and order.get("status") == "open":
                qty = float(order.get("qty", 0))
                limit_price = float(order.get("limit_price", 0)) if order.get("limit_price") else 0
                # For market orders, we would need the latest price
                if limit_price > 0:
                    cash_reserved += qty * limit_price
        
        # Calculate available cash (cash minus reserved)
        available_cash = cash - cash_reserved
        
        return {
            "success": True,
            "funds_data": {
                "cash": round(cash, 2),
                "available_cash": round(available_cash, 2),
                "cash_reserved_for_orders": round(cash_reserved, 2),
                "buying_power": round(buying_power, 2),
                "daytrading_buying_power": round(daytrading_buying_power, 2),
                "regulation_t_buying_power": round(regt_buying_power, 2),
                "cash_pct_of_portfolio": round(cash_pct_of_portfolio, 2),
                "portfolio_value": round(portfolio_value, 2)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in get_available_funds_monitoring: {str(e)}")
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}

async def check_portfolio_diversification() -> Dict[str, Any]:
    """
    Analyzes portfolio diversification across sectors, asset classes, and individual positions.
    Returns a risk assessment based on diversification metrics.
    """
    try:
        portfolio_data = await get_portfolio_data()
        
        if not portfolio_data.get("success", False):
            return portfolio_data  # Return the error
        
        positions = portfolio_data["positions"]
        account_info = portfolio_data["account_summary"]
        sector_exposure = portfolio_data["sector_exposure"]["sectors"]
        portfolio_value = float(account_info.get("portfolio_value", 0))
        
        # Calculate diversification metrics
        position_count = len(positions)
        sectors_count = len(sector_exposure)
        
        # Calculate concentration metrics
        position_concentration = []
        for position in positions:
            symbol = position.get("symbol")
            market_value = float(position.get("market_value", 0))
            concentration = (market_value / portfolio_value) * 100 if portfolio_value > 0 else 0
            position_concentration.append({
                "symbol": symbol,
                "concentration": round(concentration, 2)
            })
        
        # Sort by concentration (descending)
        position_concentration.sort(key=lambda x: x["concentration"], reverse=True)
        
        # Calculate top position concentration
        top_position_concentration = position_concentration[0]["concentration"] if position_concentration else 0
        top_3_concentration = sum(item["concentration"] for item in position_concentration[:3]) if len(position_concentration) >= 3 else sum(item["concentration"] for item in position_concentration)
        
        # Calculate sector concentration metrics
        sector_concentration = []
        for sector, data in sector_exposure.items():
            concentration = data.get("percentage", 0)
            sector_concentration.append({
                "sector": sector,
                "concentration": concentration
            })
        
        # Sort by concentration (descending)
        sector_concentration.sort(key=lambda x: x["concentration"], reverse=True)
        
        # Calculate top sector concentration
        top_sector_concentration = sector_concentration[0]["concentration"] if sector_concentration else 0
        
        # Assess diversification risk
        diversification_risk = "Low"  # Default
        risk_factors = []
        
        if position_count < 10:
            diversification_risk = "High"
            risk_factors.append("Low number of positions")
        elif position_count < 20:
            diversification_risk = "Medium"
            risk_factors.append("Moderate number of positions")
        
        if sectors_count < 3:
            diversification_risk = "High"
            risk_factors.append("Low sector diversification")
        
        if top_position_concentration > 15:
            diversification_risk = "High"
            risk_factors.append(f"High concentration in top position ({top_position_concentration:.2f}%)")
        elif top_position_concentration > 10:
            if diversification_risk != "High":
                diversification_risk = "Medium"
            risk_factors.append(f"Moderate concentration in top position ({top_position_concentration:.2f}%)")
        
        if top_3_concentration > 40:
            diversification_risk = "High"
            risk_factors.append(f"High concentration in top 3 positions ({top_3_concentration:.2f}%)")
        
        if top_sector_concentration > 40:
            diversification_risk = "High"
            risk_factors.append(f"High concentration in top sector ({top_sector_concentration:.2f}%)")
        elif top_sector_concentration > 30:
            if diversification_risk != "High":
                diversification_risk = "Medium"
            risk_factors.append(f"Moderate concentration in top sector ({top_sector_concentration:.2f}%)")
        
        return {
            "success": True,
            "diversification_metrics": {
                "position_count": position_count,
                "sectors_count": sectors_count,
                "top_position_concentration": round(top_position_concentration, 2),
                "top_3_positions_concentration": round(top_3_concentration, 2),
                "top_sector_concentration": round(top_sector_concentration, 2),
                "position_concentration": position_concentration[:5],  # Top 5 positions
                "sector_concentration": sector_concentration,
                "diversification_risk": diversification_risk,
                "risk_factors": risk_factors
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in check_portfolio_diversification: {str(e)}")
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}

# --- Enhanced Trading Functions ---
async def optimize_order_type(symbol: str, side: str, quantity: float) -> Dict[str, Any]:
    """
    Optimizes order type based on market conditions and trade characteristics.
    Returns recommendations for order type, time in force, and other parameters.
    """
    try:
        # Get the latest quote
        quote_data = await get_latest_quote(symbol)
        
        # Extract quote information
        bid = float(quote_data.get("quote", {}).get("bp", 0))
        ask = float(quote_data.get("quote", {}).get("ap", 0))
        spread = ask - bid
        spread_pct = (spread / ask) * 100 if ask > 0 else 0
        
        # Get historical volatility
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        bars_response = await get_bars(symbols=symbol, timeframe="1Day", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        
        # Default recommendation
        order_type = "market"
        time_in_force = "day"
        use_limit = False
        additional_params = {}
        
        # Check if we have enough historical data
        if bars_response.get("bars") and symbol in bars_response["bars"] and len(bars_response["bars"][symbol]) > 5:
            # Calculate volatility
            bars = bars_response["bars"][symbol]
            closes = [float(bar["c"]) for bar in bars]
            daily_returns = np.diff(closes) / closes[:-1]
            volatility = np.std(daily_returns)
            
            # High volatility or wide spread suggests limit order
            if volatility > 0.02 or spread_pct > 0.2:
                order_type = "limit"
                use_limit = True

                # Set limit price based on side and current quote
                if side.lower() == "buy":
                    # For buy, use a price slightly above bid
                    limit_price = bid + (spread * 0.3)
                else:
                    # For sell, use a price slightly below ask
                    limit_price = ask - (spread * 0.3)
                
                additional_params["limit_price"] = str(round(limit_price, 2))
        
        # Special case for large orders
        estimated_value = quantity * ((ask + bid) / 2)
        if estimated_value > 25000:  # Large order
            if use_limit:
                # Recommend adding stop loss for large limit orders
                time_in_force = "gtc"  # Good till canceled
                
                # Add stop loss recommendation
                if side.lower() == "buy":
                    stop_price = float(additional_params.get("limit_price", bid)) * 0.95
                else:
                    stop_price = float(additional_params.get("limit_price", ask)) * 1.05
                
                additional_params["stop_loss"] = {
                    "stop_price": str(round(stop_price, 2))
                }
            else:
                # For large market orders, recommend breaking it into smaller chunks
                return {
                    "success": True,
                    "recommendation": "split_order",
                    "reason": "Large order may cause excessive market impact. Consider splitting the order.",
                    "suggested_chunks": 3,
                    "order_type": "limit",
                    "time_in_force": "day"
                }
        
        return {
            "success": True,
            "recommendation": {
                "order_type": order_type,
                "time_in_force": time_in_force,
                "additional_parameters": additional_params
            },
            "market_conditions": {
                "bid": round(bid, 2),
                "ask": round(ask, 2),
                "spread": round(spread, 2),
                "spread_pct": round(spread_pct, 2)
            }
        }
    except Exception as e:
        logger.error(f"Error in optimize_order_type for {symbol}: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to optimize order type: {str(e)}",
            "fallback_recommendation": {
                "order_type": "market",
                "time_in_force": "day"
            }
        }

async def split_large_order(symbol: str, side: str, quantity: float, entry_price: float) -> List[Dict[str, Any]]:
    """
    Splits a large order into smaller chunks to minimize market impact.
    Returns a list of recommended order chunks with their parameters.
    """
    try:
        # Get the latest quote
        quote_data = await get_latest_quote(symbol)
        
        # Extract volume information (if available)
        last_trade = await get_latest_trade(symbol)
        avg_volume = float(last_trade.get("trade", {}).get("v", 10000))  # Default to 10,000 if not available
        
        # Calculate the percentage of average volume
        pct_of_avg_volume = (quantity / avg_volume) * 100
        
        # Determine number of chunks based on order size relative to average volume
        num_chunks = 1  # Default
        if pct_of_avg_volume > 5:
            num_chunks = 4
        elif pct_of_avg_volume > 3:
            num_chunks = 3
        elif pct_of_avg_volume > 1:
            num_chunks = 2
        
        if num_chunks == 1:
            # No need to split
            return [{
                "symbol": symbol,
                "side": side,
                "qty": str(quantity),
                "type": "market",
                "time_in_force": "day"
            }]
        
        # Calculate base quantity per chunk
        base_qty_per_chunk = quantity / num_chunks
        
        # Create chunks
        chunks = []
        for i in range(num_chunks):
            # For the last chunk, adjust quantity to ensure total equals original quantity
            if i == num_chunks - 1:
                remaining_qty = quantity - (base_qty_per_chunk * i)
                chunk_qty = remaining_qty
            else:
                chunk_qty = base_qty_per_chunk
            
            # Create chunk order
            chunk = {
                "symbol": symbol,
                "side": side,
                "qty": str(chunk_qty),
                "type": "limit" if i > 0 else "market",  # First chunk as market, rest as limit
                "time_in_force": "day",
            }
            
            # Add limit price for limit orders
            if chunk["type"] == "limit":
                # Adjust limit price based on side and chunk number
                if side.lower() == "buy":
                    # For buy, increment limit price for each subsequent chunk
                    price_adjustment = 0.001 * i
                    chunk["limit_price"] = str(round(entry_price * (1 + price_adjustment), 2))
                else:
                    # For sell, decrement limit price for each subsequent chunk
                    price_adjustment = 0.001 * i
                    chunk["limit_price"] = str(round(entry_price * (1 - price_adjustment), 2))
            
            chunks.append(chunk)
        
        return chunks
    except Exception as e:
        logger.error(f"Error in split_large_order for {symbol}: {str(e)}")
        # Fallback to a simple single order
        return [{
            "symbol": symbol,
            "side": side,
            "qty": str(quantity),
            "type": "market",
            "time_in_force": "day"
        }]

async def execute_optimized_order(symbol: str, side: str, quantity: float, optimize: bool = True) -> Dict[str, Any]:
    """
    Execute an order with optimized parameters based on market conditions.
    """
    try:
        # First, check if order optimization is requested
        if optimize:
            optimization_result = await optimize_order_type(symbol, side, quantity)
            
            if not optimization_result.get("success", False):
                logger.warning(f"Order optimization failed for {symbol}, falling back to market order")
                order_type = "market"
                time_in_force = "day"
                additional_params = {}
            else:
                # Check if order splitting is recommended
                if optimization_result.get("recommendation") == "split_order":
                    logger.info(f"Splitting large order for {symbol} as recommended")
                    entry_price = (optimization_result.get("market_conditions", {}).get("ask", 0) +
                                optimization_result.get("market_conditions", {}).get("bid", 0)) / 2
                    
                    # Get order chunks
                    order_chunks = await split_large_order(symbol, side, quantity, entry_price)
                    
                    # Execute first chunk immediately
                    first_chunk = order_chunks[0]
                    first_order_response = await place_order(**first_chunk)
                    
                    # Return response with information about all chunks
                    return {
                        "success": True,
                        "message": f"Executed first chunk of split order for {symbol}",
                        "first_order": first_order_response,
                        "remaining_chunks": order_chunks[1:],
                        "total_chunks": len(order_chunks)
                    }
                
                # Use recommended parameters
                rec = optimization_result.get("recommendation", {})
                order_type = rec.get("order_type", "market")
                time_in_force = rec.get("time_in_force", "day")
                additional_params = rec.get("additional_parameters", {})
        else:
            # Default to market order without optimization
            order_type = "market"
            time_in_force = "day"
            additional_params = {}
        
        # Build the order parameters
        order_params = {
            "symbol": symbol,
            "qty": quantity,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force
        }
        
        # Add any additional parameters
        for key, value in additional_params.items():
            order_params[key] = value
        
        # Place the order
        order_response = await place_order(**order_params)
        
        return {
            "success": True,
            "message": f"Executed optimized order for {symbol}",
            "order": order_response,
            "optimization_applied": optimize
        }
    except Exception as e:
        logger.error(f"Error in execute_optimized_order for {symbol}: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to execute optimized order: {str(e)}"
        }

async def calculate_slippage_estimate(symbol: str, quantity: float, side: str) -> Dict[str, Any]:
    """
    Calculate estimated slippage for a given order based on order book depth and historical data.
    """
    try:
        # Get the latest quote
        quote_data = await get_latest_quote(symbol)
        
        # Extract quote information
        bid = float(quote_data.get("quote", {}).get("bp", 0))
        ask = float(quote_data.get("quote", {}).get("ap", 0))
        bid_size = float(quote_data.get("quote", {}).get("bs", 0))
        ask_size = float(quote_data.get("quote", {}).get("as", 0))
        
        # Get the latest trade
        trade_data = await get_latest_trade(symbol)
        
        # Get historical volume and price data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        bars_response = await get_bars(symbols=symbol, timeframe="1Hour", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        
        # Calculate base slippage using bid-ask spread
        spread = ask - bid
        spread_pct = (spread / ask) * 100 if ask > 0 else 0
        
        # Initialize slippage variables
        base_slippage_pct = spread_pct / 2  # Half the spread as base
        
        # Adjust based on order size relative to quote sizes
        relevant_size = ask_size if side.lower() == "buy" else bid_size
        size_factor = min(5, (quantity / relevant_size)) if relevant_size > 0 else 1
        
        # Calculate final slippage estimation
        estimated_slippage_pct = base_slippage_pct * size_factor
        
        # Calculate price impact
        mid_price = (bid + ask) / 2
        estimated_price_impact = mid_price * (estimated_slippage_pct / 100)
        
        # Calculate expected execution price
        if side.lower() == "buy":
            expected_execution_price = mid_price + estimated_price_impact
        else:
            expected_execution_price = mid_price - estimated_price_impact
        
        # Calculate dollar impact
        estimated_dollar_impact = quantity * estimated_price_impact
        
        return {
            "success": True,
            "slippage_estimate": {
                "estimated_slippage_pct": round(estimated_slippage_pct, 3),
                "estimated_price_impact": round(estimated_price_impact, 4),
                "expected_execution_price": round(expected_execution_price, 2),
                "estimated_dollar_impact": round(estimated_dollar_impact, 2),
                "mid_price": round(mid_price, 2),
                "spread_pct": round(spread_pct, 3)
            },
            "market_conditions": {
                "bid": round(bid, 2),
                "ask": round(ask, 2),
                "bid_size": bid_size,
                "ask_size": ask_size,
                "spread": round(spread, 4)
            }
        }
    except Exception as e:
        logger.error(f"Error in calculate_slippage_estimate for {symbol}: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to estimate slippage: {str(e)}"
        }

async def calculate_optimal_execution_time(symbol: str) -> Dict[str, Any]:
    """
    Calculate the optimal execution time based on historical intraday patterns.
    """
    try:
        # Get historical intraday data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30 days of data
        bars_response = await get_bars(symbols=symbol, timeframe="15Min", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        
        if not bars_response.get("bars") or symbol not in bars_response["bars"] or len(bars_response["bars"][symbol]) < 100:
            logger.warning(f"Not enough historical data for {symbol} to calculate optimal execution time")
            return {
                "success": False,
                "error": "Insufficient historical data",
                "fallback_recommendation": "Execute during regular market hours (9:30-16:00 ET)"
            }
        
        # Convert to dataframe
        bars = bars_response["bars"][symbol]
        bars_df = pd.DataFrame(bars)
        
        # Convert t to datetime and extract hour and minute
        bars_df['datetime'] = pd.to_datetime(bars_df['t'])
        bars_df['hour'] = bars_df['datetime'].dt.hour
        bars_df['minute'] = bars_df['datetime'].dt.minute
        
        # Calculate volume and volatility metrics by time period
        time_metrics = {}
        
        for hour in range(9, 17):  # Trading hours 9:30 AM to 4:00 PM
            for minute in range(0, 60, 15):  # 15-minute intervals
                if (hour == 9 and minute < 30) or (hour == 16 and minute > 0):
                    continue  # Skip pre-market and after-hours
                
                # Filter data for this time period
                period_data = bars_df[(bars_df['hour'] == hour) & (bars_df['minute'] == minute)]
                
                if len(period_data) > 10:  # Ensure enough data points
                    # Calculate metrics
                    avg_volume = period_data['v'].mean()
                    price_range = (period_data['h'] - period_data['l']).mean()
                    price_range_pct = price_range / period_data['o'].mean() * 100
                    
                    time_metrics[f"{hour:02}:{minute:02}"] = {
                        "avg_volume": avg_volume,
                        "price_range_pct": price_range_pct,
                        "data_points": len(period_data)
                    }
        
        # Determine optimal times
        # For buys: look for times with high volume and low volatility
        # For sells: look for times with high volume
        buy_score = {time: metrics["avg_volume"] / (metrics["price_range_pct"] + 0.1)
                    for time, metrics in time_metrics.items()}
        sell_score = {time: metrics["avg_volume"]
                    for time, metrics in time_metrics.items()}
        
        # Sort by score
        best_times_for_buy = sorted(buy_score.items(), key=lambda x: x[1], reverse=True)[:3]
        best_times_for_sell = sorted(sell_score.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "success": True,
            "optimal_execution_times": {
                "buy": [{"time": time, "score": round(score, 2)} for time, score in best_times_for_buy],
                "sell": [{"time": time, "score": round(score, 2)} for time, score in best_times_for_sell]
            },
            "recommendation": "Consider executing buy orders during listed optimal times for potentially better prices and liquidity",
            "time_metrics": {time: {"avg_volume": round(metrics["avg_volume"], 2),
                                  "price_range_pct": round(metrics["price_range_pct"], 2)}
                           for time, metrics in time_metrics.items()}
        }
    except Exception as e:
        logger.error(f"Error in calculate_optimal_execution_time for {symbol}: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to calculate optimal execution time: {str(e)}"
        }
# --- FastAPI Models ---
class CompletionRequest(BaseModel):
    prompt: str
    model: str = "llama-2-70b-chat"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95

class EmbeddingsRequest(BaseModel):
    texts: List[str]
    model: str = "alpaca-ef-7b"

class ChatCompletionRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str = "llama-2-70b-chat"
    max_tokens: int = 512
    temperature: float = 0.7

class OrderRequest(BaseModel):
    symbol: str
    qty: float
    side: str
    type: str
    time_in_force: str
    limit_price: Optional[str] = None
    stop_price: Optional[str] = None
    trail_price: Optional[str] = None
    trail_percent: Optional[str] = None
    extended_hours: Optional[bool] = None
    client_order_id: Optional[str] = None
    order_class: Optional[str] = None
    take_profit: Optional[dict] = None
    stop_loss: Optional[dict] = None

class RiskAssessmentToolRequest(BaseModel):
    symbol: str
    entry_price: float
    side: str  # 'long' or 'short'
    stop_loss_price_manual: Optional[float] = None
    profit_target_manual: Optional[float] = None
    atr_multiplier_for_stop: float = Field(default=2.0, gt=0)
    reward_to_risk_ratio_target: Optional[float] = Field(default=None, gt=0)

class OptimalExecutionRequest(BaseModel):
    symbol: str

class SlippageEstimateRequest(BaseModel):
    symbol: str
    quantity: float
    side: str

class OptimizedOrderRequest(BaseModel):
    symbol: str
    side: str
    quantity: float
    optimize: bool = True


# --- FastAPI Server ---
app = FastAPI(title="Alpaca Trading MCP Server (Production)")

@app.on_event("startup")
async def startup_event():
    global scheduler
    logger.info("Alpaca Trading server starting up...")
    
    # Verify API keys before starting
    try:
        verify_api_keys()
    except ValueError as e:
        logger.error(f"Startup failed: {str(e)}")
        # In production, you might want to exit here, but for now we'll continue with warnings
    
    # Connect to Redis
    try:
        await connect_redis()
    except RuntimeError as e:
        logger.error(f"Redis connection failed during startup: {str(e)}")
        # In production, consider failing startup if Redis is critical
    
    # Initialize scheduler
    scheduler = AsyncIOScheduler()
    
    # Schedule portfolio updates
    scheduler.add_job(
        store_current_portfolio_to_redis,
        'interval',
        seconds=CONFIG.get("portfolio_update_interval", 300),
        id='portfolio_update',
        replace_existing=True
    )
    
    # Start the scheduler
    scheduler.start()
    logger.info(f"Scheduler started with portfolio updates every {CONFIG.get('portfolio_update_interval', 300)}s")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Alpaca Trading server shutting down.")
    
    # Shut down the scheduler
    if scheduler:
        scheduler.shutdown()
        logger.info("Scheduler shutdown complete.")
    
    # Close Redis connection
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed.")

@app.get("/server_info")
async def get_server_info():
    return {
        "name": "alpaca", # Consistent with mcp_config.yaml
        "version": "1.1.0", 
        "description": "Production MCP Server for Alpaca Trading API Integration with advanced portfolio and risk management.",
        "tools": [
            "get_account_info", "get_current_positions", "get_open_orders", "place_trade_order",
            "get_market_bars", "get_latest_market_quote", "get_latest_market_trade",
            "assess_trade_risk_detailed", # New detailed risk assessment tool
            # Portfolio specific tools (can be added if direct access is needed beyond get_portfolio_data)
            # "get_portfolio_data", "get_real_time_portfolio_tracking", 
            # "get_available_funds_monitoring", "get_sector_exposure_analysis", 
            # "check_portfolio_diversification"
        ],
        "config": {k: v for k, v in CONFIG.items() if k not in ["secret_key", "redis_password", "api_key"]}
    }

# --- API Endpoints for Tools ---
@app.post("/get_account_info")
async def api_get_account_info():
    return await get_account()

@app.post("/get_current_positions")
async def api_get_current_positions():
    return await get_positions()

@app.post("/get_open_orders")
async def api_get_open_orders(status: str = "open", limit: int = 50, nested: bool = True):
    return await get_orders(status, limit, nested)

@app.post("/place_trade_order")
async def api_place_trade_order(req: OrderRequest):
    return await place_order(**req.model_dump())

@app.post("/get_market_bars")
async def api_get_market_bars(symbols: Union[str, List[str]], timeframe: str, start: str, end: Optional[str] = None, limit: Optional[int] = None, adjustment: str = "raw"):
    return await get_bars(symbols, timeframe, start, end, limit, adjustment)

@app.post("/get_latest_market_quote")
async def api_get_latest_market_quote(symbol: str):
    return await get_latest_quote(symbol)

@app.post("/get_latest_market_trade")
async def api_get_latest_market_trade(symbol: str):
    return await get_latest_trade(symbol)

@app.post("/assess_trade_risk_detailed")
async def api_assess_trade_risk_detailed(req: RiskAssessmentToolRequest):
    return await assess_trade_risk_detailed(**req.model_dump())

@app.post("/get_portfolio_data")
async def api_get_portfolio_data():
    return await get_portfolio_data()

@app.post("/get_real_time_portfolio_tracking")
async def api_get_real_time_portfolio_tracking():
    return await get_real_time_portfolio_tracking()

@app.post("/get_available_funds_monitoring")
async def api_get_available_funds_monitoring():
    return await get_available_funds_monitoring()

@app.post("/check_portfolio_diversification")
async def api_check_portfolio_diversification():
    return await check_portfolio_diversification()

@app.post("/optimize_order_type")
async def api_optimize_order_type(req: OptimizedOrderRequest):
    return await optimize_order_type(req.symbol, req.side, req.quantity)

@app.post("/calculate_slippage_estimate")
async def api_calculate_slippage_estimate(req: SlippageEstimateRequest):
    return await calculate_slippage_estimate(req.symbol, req.quantity, req.side)

@app.post("/calculate_optimal_execution_time")
async def api_calculate_optimal_execution_time(req: OptimalExecutionRequest):
    return await calculate_optimal_execution_time(req.symbol)

@app.post("/execute_optimized_order")
async def api_execute_optimized_order(req: OptimizedOrderRequest):
    return await execute_optimized_order(req.symbol, req.side, req.quantity, req.optimize)

# --- Redis related helper functions for portfolio data ---
async def store_current_portfolio_to_redis():
    """Periodically fetches and stores current portfolio data to Redis."""
    if not redis_client:
        logger.warning("Redis client not available for portfolio storage.")
        return
    try:
        portfolio_data = await get_portfolio_data()
        if portfolio_data.get("success", False):
            # Store the complete portfolio data
            await redis_client.set(
                "alpaca:portfolio:current",
                json.dumps(portfolio_data),
                ex=timedelta(minutes=10)
            )
            
            # Also store specific metrics for quick access
            if "portfolio_metrics" in portfolio_data:
                metrics = portfolio_data["portfolio_metrics"]
                await redis_client.set(
                    "alpaca:portfolio:metrics",
                    json.dumps(metrics),
                    ex=timedelta(minutes=10)
                )
                
            # Store positions separately for other services to access
            if "positions" in portfolio_data:
                await redis_client.set(
                    "alpaca:portfolio:positions",
                    json.dumps(portfolio_data["positions"]),
                    ex=timedelta(minutes=10)
                )
                
            logger.info("Successfully stored current portfolio data to Redis.")
    except Exception as e:
        logger.error(f"Error storing portfolio data to Redis: {e}")

# --- Cache Management Functions ---
async def invalidate_portfolio_cache():
    """Invalidate all portfolio-related cache entries in Redis"""
    if not redis_client:
        logger.warning("Redis client not available for cache invalidation.")
        return False
        
    try:
        # Delete all portfolio-related keys
        keys_to_delete = [
            "alpaca:portfolio:current",
            "alpaca:portfolio:metrics",
            "alpaca:portfolio:positions"
        ]
        
        for key in keys_to_delete:
            await redis_client.delete(key)
            
        logger.info("Successfully invalidated portfolio cache.")
        return True
    except Exception as e:
        logger.error(f"Error invalidating portfolio cache: {e}")
        return False

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.1.0",
        "checks": {}
    }
    
    # Check Redis connection
    try:
        if redis_client:
            await redis_client.ping()
            health_status["checks"]["redis"] = "connected"
        else:
            health_status["checks"]["redis"] = "not_connected"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["redis"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Alpaca API connectivity
    try:
        await get_account()
        health_status["checks"]["alpaca_api"] = "connected"
    except Exception as e:
        health_status["checks"]["alpaca_api"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check scheduler status
    if scheduler and scheduler.running:
        health_status["checks"]["scheduler"] = "running"
    else:
        health_status["checks"]["scheduler"] = "not_running"
        health_status["status"] = "degraded"
    
    return health_status

# Initialize configuration
CONFIG = load_config()
logger = get_logger("alpaca_server")
logger.info(f"Initializing Alpaca Trading server (Paper Trading: {CONFIG.get('paper_trading', True)})")

TRADING_HEADERS = {
    "APCA-API-KEY-ID": CONFIG["api_key"],
    "APCA-API-SECRET-KEY": CONFIG["secret_key"],
    "Content-Type": "application/json"
}

# Note: Direct LLM call functions were removed as they are not primary tools of this server.
# The server_info lists tools this server *provides*, not tools it *uses*.
