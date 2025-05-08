"""
Main LLM Controller for Trading System

This module implements the Main LLM component that:
1. Receives processed market data from Redis
2. Makes decisions on potential trades
3. Triggers trade analysis and position sizing
4. Ultimately generates trade orders to Alpaca

It uses an LLM to analyze market conditions and make high-level trading decisions.
"""

import os
import json
import time
import asyncio
import logging
import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import aiohttp
import redis.asyncio as aioredis
import requests
from ratelimit import limits, sleep_and_retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main_llm")

# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/ubuntu/nextg3n/.env")

# Load configuration
CONFIG_PATH = "/home/ubuntu/nextg3n/config/llm_config.yaml"

def load_config():
    """Load configuration from YAML file."""
    try:
        with open(CONFIG_PATH, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {CONFIG_PATH}")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        # Return default configuration
        return {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "password": None,
                "channels": {
                    "market_data": "channel:market_data",
                    "market_analysis": "channel:ml_analysis",
                    "main_llm": "channel:main_llm",
                    "trade_analysis": "channel:trade_analysis",
                    "position_sizing": "channel:position_sizing",
                    "order_status": "channel:order_status"
                }
            },
            "llm": {
                "provider": "openrouter",
                "model": "google/gemini-2.0-flash-001",
                "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
                "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
                "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
                "rate_limit_per_minute": 20,
                "max_tokens": 8192,
                "temperature": 0.7,
                "site_url": "https://nextg3n.ai",
                "site_name": "NextG3N Trading System"
            },
            "main_llm": {
                "confidence_threshold": 0.75,
                "max_positions": 5,
                "max_position_size_pct": 10.0
            }
        }

# Load configuration
CONFIG = load_config()

# Extract configuration values
REDIS_CONFIG = CONFIG.get("redis", {})
LLM_CONFIG = CONFIG.get("llm", {})
MAIN_LLM_CONFIG = CONFIG.get("main_llm", {})

# Redis Configuration
REDIS_HOST = REDIS_CONFIG.get("host", "localhost")
REDIS_PORT = REDIS_CONFIG.get("port", 6379)
REDIS_DB = REDIS_CONFIG.get("db", 0)
REDIS_PASSWORD = REDIS_CONFIG.get("password")

# Redis channel names
REDIS_CHANNELS = REDIS_CONFIG.get("channels", {})
MARKET_DATA_CHANNEL = REDIS_CHANNELS.get("market_data", "channel:market_data")
ML_ANALYSIS_CHANNEL = REDIS_CHANNELS.get("market_analysis", "channel:ml_analysis")
MAIN_LLM_CHANNEL = REDIS_CHANNELS.get("main_llm", "channel:main_llm")
TRADE_ANALYSIS_CHANNEL = REDIS_CHANNELS.get("trade_analysis", "channel:trade_analysis")
POSITION_SIZING_CHANNEL = REDIS_CHANNELS.get("position_sizing", "channel:position_sizing")
ORDER_STATUS_CHANNEL = REDIS_CHANNELS.get("order_status", "channel:order_status")

# Trading parameters
MAX_POSITIONS = MAIN_LLM_CONFIG.get("max_positions", 5)
MAX_POSITION_SIZE_PCT = MAIN_LLM_CONFIG.get("max_position_size_pct", 10.0)
MIN_CONFIDENCE = MAIN_LLM_CONFIG.get("confidence_threshold", 0.75)

# Get LLM provider and credentials
LLM_PROVIDER = LLM_CONFIG.get("provider", "openrouter")
LLM_MODEL = LLM_CONFIG.get("model", "google/gemini-2.0-flash-001")
OPENAI_API_KEY = LLM_CONFIG.get("api_key") or os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = LLM_CONFIG.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = LLM_CONFIG.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY", "")
LLM_RATE_LIMIT = LLM_CONFIG.get("rate_limit_per_minute", 20)
LLM_MAX_TOKENS = LLM_CONFIG.get("max_tokens", 8192)
LLM_TEMPERATURE = LLM_CONFIG.get("temperature", 0.7)
LLM_SITE_URL = LLM_CONFIG.get("site_url", "https://nextg3n.ai")
LLM_SITE_NAME = LLM_CONFIG.get("site_name", "NextG3N Trading System")

class MainLLM:
    """Main LLM controller that analyzes market data and makes trade decisions."""
    
    def __init__(self):
        self.redis_client = None
        self.pubsub = None
        self.session = None
        self.initialized = False
        self.llm_system_prompt = self._load_system_prompt()
        self.tasks = []
        self.active_positions = {}
        self.last_ml_analysis = None
        self.portfolio_value = float(os.getenv("INITIAL_PORTFOLIO_VALUE", "100000"))
    
    def _load_system_prompt(self):
        """Load system prompt from file if available, otherwise use default."""
        prompt_file = MAIN_LLM_CONFIG.get("system_prompt_file")
        if prompt_file and os.path.exists(prompt_file):
            try:
                with open(prompt_file, 'r') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error loading system prompt file: {e}")
        
        # Default system prompt
        return """
        You are an expert trading advisor with deep knowledge of stock market analysis,
        technical indicators, and trading strategies. You will analyze market data and
        make trading recommendations based on a combination of technical analysis, 
        sentiment analysis, and ML model scores.
        
        Your job is to:
        1. Analyze the provided market data and trading signals
        2. Make a decision whether to recommend a trade or not
        3. Provide a confidence score for your recommendation (0.0-1.0)
        4. Explain your reasoning in a clear, concise manner
        
        Only recommend trades that have a clear rationale and strong signals.
        Be cautious and conservative in your recommendations.
        
        Your output should be structured as a JSON object with the following fields:
        - decision: "buy", "sell", or "no_action"
        - confidence: A number between 0.0 and 1.0
        - symbol: The ticker symbol for the recommendation
        - reasoning: A brief explanation of your rationale
        - time_horizon: "day_trade", "swing_trade", or "position_trade"
        - suggested_entry_price: Your suggested entry price
        - suggested_stop_loss: Your suggested stop loss price
        - suggested_take_profit: Your suggested take profit price
        
        Only make a buy or sell recommendation if your confidence is high (>0.75).
        """
    
    async def initialize(self):
        """Initialize connections to Redis and HTTP session."""
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
            
            # Test connection
            await self.redis_client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            
            # Subscribe to relevant Redis channels
            self.pubsub = self.redis_client.pubsub()
            await self.pubsub.subscribe(
                MARKET_DATA_CHANNEL,
                ML_ANALYSIS_CHANNEL,
                MAIN_LLM_CHANNEL,
                TRADE_ANALYSIS_CHANNEL,
                ORDER_STATUS_CHANNEL
            )
            
            # Start message listener
            listener_task = asyncio.create_task(self._listen_for_messages())
            self.tasks.append(listener_task)
            
            # Initialize active positions from Redis
            await self._load_active_positions()
            
            self.initialized = True
            logger.info("Main LLM initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Main LLM: {e}")
            await self.close()
    
    async def _load_active_positions(self):
        """Load active positions from Redis."""
        try:
            position_keys = await self.redis_client.keys("position:*:active")
            for key in position_keys:
                position_data = await self.redis_client.get(key)
                if position_data:
                    try:
                        position = json.loads(position_data)
                        position_id = position.get("position_id")
                        if position_id:
                            self.active_positions[position_id] = position
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse position data from {key}")
            
            logger.info(f"Loaded {len(self.active_positions)} active positions")
        except Exception as e:
            logger.error(f"Error loading active positions: {e}")
    
    async def _listen_for_messages(self):
        """Listen for messages on subscribed Redis channels."""
        try:
            while True:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message:
                    channel = message["channel"]
                    data = message["data"]
                    
                    # Try to parse JSON data
                    if isinstance(data, str):
                        try:
                            data = json.loads(data)
                        except json.JSONDecodeError:
                            logger.warning(f"Received non-JSON data on channel {channel}")
                            continue
                    
                    # Process message based on channel
                    if channel == MARKET_DATA_CHANNEL:
                        await self._handle_market_data(data)
                    elif channel == ML_ANALYSIS_CHANNEL:
                        await self._handle_ml_analysis(data)
                    elif channel == MAIN_LLM_CHANNEL:
                        await self._handle_main_llm_message(data)
                    elif channel == TRADE_ANALYSIS_CHANNEL:
                        await self._handle_trade_analysis(data)
                    elif channel == ORDER_STATUS_CHANNEL:
                        await self._handle_order_status(data)
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("Message listener task cancelled")
        except Exception as e:
            logger.error(f"Error in message listener: {e}")
    
    async def _handle_market_data(self, data):
        """Handle incoming market data."""
        logger.debug(f"Received market data: {data}")
        
        # If we have ML analysis and new market data, consider generating a trade
        if self.last_ml_analysis and len(self.active_positions) < MAX_POSITIONS:
            # Check if this is one of our top recommended stocks
            symbol = data.get("symbol")
            if symbol and symbol in [s.get("symbol") for s in self.last_ml_analysis.get("top_stocks", [])]:
                # This is a top stock with fresh market data - evaluate for trading
                await self._evaluate_trade_opportunity(symbol, data, self.last_ml_analysis)
    
    async def _handle_ml_analysis(self, data):
        """Handle incoming ML analysis."""
        logger.info(f"Received ML analysis update with {len(data.get('top_stocks', []))} top stocks")
        
        # Store the latest ML analysis
        self.last_ml_analysis = data
        
        # For each top stock, evaluate as a trade opportunity
        for stock in data.get("top_stocks", [])[:3]:  # Only consider top 3 to avoid overwhelming
            symbol = stock.get("symbol")
            if symbol:
                # Get latest market data for this symbol
                market_data = await self._get_latest_market_data(symbol)
                if market_data:
                    await self._evaluate_trade_opportunity(symbol, market_data, data)
    
    async def _handle_main_llm_message(self, data):
        """Handle messages sent directly to Main LLM."""
        message_type = data.get("type")
        logger.debug(f"Received Main LLM message of type: {message_type}")
        
        if message_type == "analysis_request":
            # Request to analyze a specific symbol
            symbol = data.get("symbol")
            if symbol:
                market_data = await self._get_latest_market_data(symbol)
                if market_data:
                    ml_data = await self._get_ml_data_for_symbol(symbol)
                    await self._evaluate_trade_opportunity(symbol, market_data, ml_data)
        
        elif message_type == "portfolio_update":
            # Update portfolio value
            new_value = data.get("portfolio_value")
            if new_value:
                self.portfolio_value = float(new_value)
                logger.info(f"Updated portfolio value to {self.portfolio_value}")
    
    async def _handle_trade_analysis(self, data):
        """Handle results from Trade Analysis component."""
        logger.debug(f"Received trade analysis: {data}")
        
        analysis_result = data.get("result")
        symbol = data.get("symbol")
        
        if analysis_result == "approved" and symbol:
            # Trade analysis approves the trade - proceed to position sizing
            await self._request_position_sizing(symbol, data)
    
    async def _handle_order_status(self, data):
        """Handle order status updates."""
        logger.debug(f"Received order status update: {data}")
        
        order_id = data.get("order_id")
        status = data.get("status")
        
        if status == "filled":
            # Order was filled - update our active positions
            position_id = data.get("position_id")
            if position_id:
                position_data = data.get("position_data", {})
                self.active_positions[position_id] = position_data
                logger.info(f"Added new position {position_id} for {position_data.get('symbol')}")
        
        elif status == "closed":
            # Position was closed - remove from active positions
            position_id = data.get("position_id")
            if position_id and position_id in self.active_positions:
                del self.active_positions[position_id]
                logger.info(f"Removed closed position {position_id}")
    
    async def _get_latest_market_data(self, symbol):
        """Get the latest market data for a symbol from Redis."""
        try:
            data = await self.redis_client.get(f"market_data:{symbol}:latest")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def _get_ml_data_for_symbol(self, symbol):
        """Get ML analysis data for a specific symbol."""
        if not self.last_ml_analysis:
            return None
            
        # Extract the data for this specific symbol from the last ML analysis
        for stock in self.last_ml_analysis.get("top_stocks", []):
            if stock.get("symbol") == symbol:
                return {
                    "symbol_data": stock,
                    "market_context": self.last_ml_analysis.get("market_context", {})
                }
        
        return None
    
    @sleep_and_retry
    @limits(calls=LLM_RATE_LIMIT, period=60)
    async def _call_llm_api(self, prompt, system_prompt=None):
        """Call the LLM API with rate limiting."""
        if not system_prompt:
            system_prompt = self.llm_system_prompt
            
        try:
            if LLM_PROVIDER == "openai":
                return await self._call_openai(prompt, system_prompt)
            elif LLM_PROVIDER == "anthropic":
                return await self._call_anthropic(prompt, system_prompt)
            elif LLM_PROVIDER == "openrouter":
                return await self._call_openrouter(prompt, system_prompt)
            else:
                logger.error(f"Unknown LLM provider: {LLM_PROVIDER}")
                return None
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return None
    
    async def _call_openai(self, prompt, system_prompt):
        """Call OpenAI API."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            data = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": LLM_MAX_TOKENS,
                "temperature": LLM_TEMPERATURE
            }
            
            async with self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    logger.error(f"OpenAI API error: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return None
    
    async def _call_anthropic(self, prompt, system_prompt):
        """Call Anthropic API."""
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": LLM_MODEL,
                "system": system_prompt,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": LLM_MAX_TOKENS,
                "temperature": LLM_TEMPERATURE
            }
            
            async with self.session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["content"][0]["text"]
                else:
                    error_text = await response.text()
                    logger.error(f"Anthropic API error: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return None
    
    async def _call_openrouter(self, prompt, system_prompt):
        """Call OpenRouter API."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": LLM_SITE_URL,
                "X-Title": LLM_SITE_NAME
            }
            
            data = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": LLM_MAX_TOKENS,
                "temperature": LLM_TEMPERATURE
            }
            
            async with self.session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            return None
    
    async def _evaluate_trade_opportunity(self, symbol, market_data, ml_data):
        """Evaluate a potential trade opportunity using the LLM."""
        logger.info(f"Evaluating trade opportunity for {symbol}")
        
        # Check if we already have a position in this symbol
        for position in self.active_positions.values():
            if position.get("symbol") == symbol:
                logger.info(f"Already have a position in {symbol}, skipping evaluation")
                return
        
        # Prepare LLM prompt
        prompt = self._prepare_trade_evaluation_prompt(symbol, market_data, ml_data)
        
        # Call LLM API
        llm_response = await self._call_llm_api(prompt)
        if not llm_response:
            logger.error(f"Failed to get LLM response for {symbol}")
            return
        
        # Process LLM response
        try:
            # Try to extract JSON from the response
            json_str = self._extract_json(llm_response)
            if not json_str:
                logger.warning(f"Failed to extract JSON from LLM response for {symbol}")
                return
                
            decision = json.loads(json_str)
            
            # Check if the decision is to take action
            if decision.get("decision") in ["buy", "sell"] and decision.get("confidence", 0) >= MIN_CONFIDENCE:
                # Forward to Trade Analysis component
                await self._request_trade_analysis(symbol, decision, market_data, ml_data)
            else:
                logger.info(f"No trade action for {symbol}: {decision.get('decision')} with confidence {decision.get('confidence')}")
                
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from LLM response for {symbol}")
        except Exception as e:
            logger.error(f"Error processing LLM response for {symbol}: {e}")
    
    def _prepare_trade_evaluation_prompt(self, symbol, market_data, ml_data):
        """Prepare the prompt for trade evaluation."""
        prompt = f"""
        I need you to evaluate a potential trade for {symbol}.
        
        Current Market Data:
        {json.dumps(market_data, indent=2)}
        
        ML Analysis:
        {json.dumps(ml_data, indent=2)}
        
        Current Portfolio:
        - Portfolio Value: ${self.portfolio_value}
        - Active Positions: {len(self.active_positions)}
        
        Based on this information, please evaluate whether we should take a trade on {symbol}.
        Consider the technical indicators, ML model predictions, market conditions, and our current portfolio.
        
        Respond with a JSON object containing your trade decision and reasoning.
        """
        return prompt
    
    def _extract_json(self, text):
        """Extract JSON from text that might contain other content."""
        try:
            # Try to find JSON content in the text
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                # Validate by parsing
                json.loads(json_str)
                return json_str
            
            return None
        except:
            return None
    
    async def _request_trade_analysis(self, symbol, decision, market_data, ml_data):
        """Request detailed trade analysis from the Trade Analysis component."""
        logger.info(f"Requesting trade analysis for {symbol}")
        
        try:
            # Combine all data for analysis
            analysis_request = {
                "type": "analysis_request",
                "symbol": symbol,
                "decision": decision,
                "market_data": market_data,
                "ml_data": ml_data,
                "portfolio": {
                    "value": self.portfolio_value,
                    "active_positions": len(self.active_positions)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Publish to Trade Analysis channel
            await self.redis_client.publish(
                TRADE_ANALYSIS_CHANNEL,
                json.dumps(analysis_request)
            )
            
            logger.info(f"Trade analysis request sent for {symbol}")
            
        except Exception as e:
            logger.error(f"Error requesting trade analysis for {symbol}: {e}")
    
    async def _request_position_sizing(self, symbol, analysis_data):
        """Request position sizing from the Position Sizing component."""
        logger.info(f"Requesting position sizing for {symbol}")
        
        try:
            # Extract trade details from analysis
            decision = analysis_data.get("decision", {})
            
            # Create position sizing request
            sizing_request = {
                "type": "sizing_request",
                "symbol": symbol,
                "trade_direction": decision.get("decision", "buy"),
                "suggested_entry": decision.get("suggested_entry_price"),
                "suggested_stop_loss": decision.get("suggested_stop_loss"),
                "suggested_take_profit": decision.get("suggested_take_profit"),
                "confidence": decision.get("confidence", 0.0),
                "portfolio_value": self.portfolio_value,
                "max_position_size_pct": MAX_POSITION_SIZE_PCT,
                "timestamp": datetime.now().isoformat()
            }
            
            # Publish to Position Sizing channel
            await self.redis_client.publish(
                POSITION_SIZING_CHANNEL,
                json.dumps(sizing_request)
            )
            
            logger.info(f"Position sizing request sent for {symbol}")
            
        except Exception as e:
            logger.error(f"Error requesting position sizing for {symbol}: {e}")
    
    async def close(self):
        """Close connections and cancel tasks."""
        logger.info("Shutting down Main LLM")
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
            
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.pubsub:
            await self.pubsub.unsubscribe()
            self.pubsub = None
            
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
        
        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None
        
        self.initialized = False
        logger.info("Main LLM shut down")

async def main():
    """Main function to run the Main LLM module."""
    main_llm = MainLLM()
    
    try:
        # Initialize and run
        await main_llm.initialize()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        # Clean shutdown
        await main_llm.close()

if __name__ == "__main__":
    asyncio.run(main())