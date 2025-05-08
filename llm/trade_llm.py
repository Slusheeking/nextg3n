"""
Trade LLM Controller for Trading System

This module implements the Trade LLM component that:
1. Receives position status and signals from ML Position Monitor
2. Makes decisions about when to exit trades
3. Sends exit orders to Alpaca Broker

It uses an LLM to analyze position data and make intelligent exit decisions.
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
logger = logging.getLogger("trade_llm")

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
                    "position_monitor": "channel:position_monitor",
                    "trade_llm": "channel:trade_llm",
                    "trade_order": "channel:trade_order",
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
            "trade_llm": {
                "confidence_threshold": 0.7,
                "position_monitoring_interval": 60,
                "min_signal_interval": 300
            }
        }

# Load configuration
CONFIG = load_config()

# Extract configuration values
REDIS_CONFIG = CONFIG.get("redis", {})
LLM_CONFIG = CONFIG.get("llm", {})
TRADE_LLM_CONFIG = CONFIG.get("trade_llm", {})

# Redis Configuration
REDIS_HOST = REDIS_CONFIG.get("host", "localhost")
REDIS_PORT = REDIS_CONFIG.get("port", 6379)
REDIS_DB = REDIS_CONFIG.get("db", 0)
REDIS_PASSWORD = REDIS_CONFIG.get("password")

# Redis channel names
REDIS_CHANNELS = REDIS_CONFIG.get("channels", {})
POSITION_MONITOR_CHANNEL = REDIS_CHANNELS.get("position_monitor", "channel:position_monitor")
TRADE_LLM_CHANNEL = REDIS_CHANNELS.get("trade_llm", "channel:trade_llm")
TRADE_ORDER_CHANNEL = REDIS_CHANNELS.get("trade_order", "channel:trade_order")
ORDER_STATUS_CHANNEL = REDIS_CHANNELS.get("order_status", "channel:order_status")

# Trade LLM parameters
CONFIDENCE_THRESHOLD = TRADE_LLM_CONFIG.get("confidence_threshold", 0.7)
POSITION_MONITORING_INTERVAL = TRADE_LLM_CONFIG.get("position_monitoring_interval", 60)
MIN_SIGNAL_INTERVAL = TRADE_LLM_CONFIG.get("min_signal_interval", 300)
HONOR_STOP_LOSS = TRADE_LLM_CONFIG.get("honor_stop_loss", True)
HONOR_TAKE_PROFIT = TRADE_LLM_CONFIG.get("honor_take_profit", True)
PARTIAL_EXIT_ENABLED = TRADE_LLM_CONFIG.get("partial_exit_enabled", True)

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

class TradeLLM:
    """Trade LLM controller that monitors positions and makes exit decisions."""
    
    def __init__(self):
        self.redis_client = None
        self.pubsub = None
        self.session = None
        self.initialized = False
        self.llm_system_prompt = self._load_system_prompt()
        self.tasks = []
        self.active_positions = {}
        self.last_signal_time = {}  # Track last signal time by position_id
    
    def _load_system_prompt(self):
        """Load system prompt from file if available, otherwise use default."""
        prompt_file = TRADE_LLM_CONFIG.get("system_prompt_file")
        if prompt_file and os.path.exists(prompt_file):
            try:
                with open(prompt_file, 'r') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error loading system prompt file: {e}")
        
        # Default system prompt
        return """
        You are an expert trading exit strategist with deep knowledge of stock market analysis,
        technical indicators, and risk management. You will analyze position data and
        make decisions about when to exit trades based on the current market conditions,
        position performance, and risk parameters.
        
        Your job is to:
        1. Analyze the provided position data and market signals
        2. Decide whether to hold the position or exit (fully or partially)
        3. Provide a confidence score for your recommendation (0.0-1.0)
        4. Explain your reasoning in a clear, concise manner
        
        Be disciplined about following stop losses and taking profits. Remember that 
        preserving capital is more important than maximizing gains. Don't hold losing
        positions hoping for a reversal without compelling evidence.
        
        Your output should be structured as a JSON object with the following fields:
        - decision: "hold", "exit", or "partial_exit"
        - exit_percentage: If partial_exit, what percentage to exit (0-100)
        - confidence: A number between 0.0 and 1.0
        - reasoning: A brief explanation of your rationale
        - suggested_exit_price: Your suggested exit price (limit order price)
        - exit_urgency: "low", "medium", or "high" (determines market vs limit order)
        
        Only recommend an exit if your confidence is high (>0.7).
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
                POSITION_MONITOR_CHANNEL,
                TRADE_LLM_CHANNEL,
                ORDER_STATUS_CHANNEL
            )
            
            # Start message listener
            listener_task = asyncio.create_task(self._listen_for_messages())
            self.tasks.append(listener_task)
            
            # Start position monitoring task
            monitor_task = asyncio.create_task(self._monitor_positions())
            self.tasks.append(monitor_task)
            
            # Initialize active positions from Redis
            await self._load_active_positions()
            
            self.initialized = True
            logger.info("Trade LLM initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Trade LLM: {e}")
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
                    if channel == POSITION_MONITOR_CHANNEL:
                        await self._handle_position_signal(data)
                    elif channel == TRADE_LLM_CHANNEL:
                        await self._handle_trade_llm_message(data)
                    elif channel == ORDER_STATUS_CHANNEL:
                        await self._handle_order_status(data)
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("Message listener task cancelled")
        except Exception as e:
            logger.error(f"Error in message listener: {e}")
    
    async def _monitor_positions(self):
        """Periodically check all active positions for potential exit."""
        try:
            while True:
                if self.active_positions:
                    logger.info(f"Monitoring {len(self.active_positions)} active positions")
                    
                    for position_id, position in list(self.active_positions.items()):
                        try:
                            # Get latest market data for this position
                            symbol = position.get("symbol")
                            if symbol:
                                # Check if we've evaluated this position recently
                                current_time = time.time()
                                last_time = self.last_signal_time.get(position_id, 0)
                                
                                if current_time - last_time >= MIN_SIGNAL_INTERVAL:
                                    market_data = await self._get_latest_market_data(symbol)
                                    if market_data:
                                        # Update position with latest price
                                        position["current_price"] = market_data.get("price", position.get("current_price", 0))
                                        position["updated_at"] = datetime.now().isoformat()
                                        
                                        # Check for stop loss/take profit triggers
                                        if await self._check_predefined_exits(position_id, position, market_data):
                                            # Exit was triggered by predefined rules
                                            continue
                                        
                                        # Evaluate position with LLM
                                        await self._evaluate_position_exit(position_id, position, market_data)
                        except Exception as e:
                            logger.error(f"Error monitoring position {position_id}: {e}")
                
                # Wait before next monitoring cycle
                await asyncio.sleep(POSITION_MONITORING_INTERVAL)
        except asyncio.CancelledError:
            logger.info("Position monitoring task cancelled")
        except Exception as e:
            logger.error(f"Error in position monitoring: {e}")
    
    async def _handle_position_signal(self, data):
        """Handle signals from the Position Monitor component."""
        logger.debug(f"Received position signal: {data}")
        
        signal_type = data.get("type")
        position_id = data.get("position_id")
        
        if position_id and position_id in self.active_positions:
            # Update last signal time
            self.last_signal_time[position_id] = time.time()
            
            if signal_type == "exit_signal":
                # High priority exit signal - evaluate immediately
                position = self.active_positions[position_id]
                symbol = position.get("symbol")
                
                if symbol:
                    market_data = await self._get_latest_market_data(symbol)
                    if market_data:
                        # Override normal interval check - this is a high priority signal
                        await self._evaluate_position_exit(position_id, position, market_data, data)
            
            elif signal_type == "position_update":
                # Update our position data
                position_data = data.get("position_data", {})
                self.active_positions[position_id].update(position_data)
    
    async def _handle_trade_llm_message(self, data):
        """Handle messages sent directly to Trade LLM."""
        message_type = data.get("type")
        logger.debug(f"Received Trade LLM message of type: {message_type}")
        
        if message_type == "evaluate_exit":
            # Request to evaluate a specific position for exit
            position_id = data.get("position_id")
            if position_id and position_id in self.active_positions:
                position = self.active_positions[position_id]
                symbol = position.get("symbol")
                
                if symbol:
                    market_data = await self._get_latest_market_data(symbol)
                    if market_data:
                        await self._evaluate_position_exit(position_id, position, market_data)
    
    async def _handle_order_status(self, data):
        """Handle order status updates."""
        logger.debug(f"Received order status update: {data}")
        
        order_id = data.get("order_id")
        status = data.get("status")
        
        if status == "filled" and data.get("order_type") == "exit":
            # Exit order was filled - remove from active positions
            position_id = data.get("position_id")
            if position_id and position_id in self.active_positions:
                del self.active_positions[position_id]
                logger.info(f"Removed closed position {position_id}")
                
                # Also remove from signal time tracking
                if position_id in self.last_signal_time:
                    del self.last_signal_time[position_id]
    
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
    
    async def _check_predefined_exits(self, position_id, position, market_data):
        """Check if any predefined exit rules are triggered (stop loss, take profit)."""
        if not (HONOR_STOP_LOSS or HONOR_TAKE_PROFIT):
            return False
            
        current_price = market_data.get("price", 0)
        if current_price <= 0:
            return False
            
        entry_price = position.get("entry_price", 0)
        stop_loss = position.get("stop_loss")
        take_profit = position.get("take_profit")
        
        # Check stop loss
        if HONOR_STOP_LOSS and stop_loss and current_price <= stop_loss:
            logger.info(f"Stop loss triggered for position {position_id} at {current_price}")
            
            # Create exit order
            await self._create_exit_order(
                position_id=position_id,
                symbol=position.get("symbol"),
                quantity=position.get("quantity", 0),
                price=current_price,
                order_type="market",  # Use market order for stop loss
                reason="stop_loss_triggered"
            )
            return True
            
        # Check take profit
        if HONOR_TAKE_PROFIT and take_profit and current_price >= take_profit:
            logger.info(f"Take profit triggered for position {position_id} at {current_price}")
            
            # Create exit order
            await self._create_exit_order(
                position_id=position_id,
                symbol=position.get("symbol"),
                quantity=position.get("quantity", 0),
                price=current_price,
                order_type="limit",  # Use limit order for take profit
                reason="take_profit_triggered"
            )
            return True
            
        return False
    
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
    
    async def _evaluate_position_exit(self, position_id, position, market_data, signal_data=None):
        """Evaluate whether to exit a position using the LLM."""
        logger.info(f"Evaluating exit for position {position_id}")
        
        # Prepare LLM prompt
        prompt = self._prepare_exit_evaluation_prompt(position, market_data, signal_data)
        
        # Call LLM API
        llm_response = await self._call_llm_api(prompt)
        if not llm_response:
            logger.error(f"Failed to get LLM response for position {position_id}")
            return
        
        # Process LLM response
        try:
            # Try to extract JSON from the response
            json_str = self._extract_json(llm_response)
            if not json_str:
                logger.warning(f"Failed to extract JSON from LLM response for position {position_id}")
                return
                
            decision = json.loads(json_str)
            
            # Check if the decision is to exit
            if decision.get("decision") in ["exit", "partial_exit"] and decision.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
                # Determine exit quantity
                quantity = position.get("quantity", 0)
                if PARTIAL_EXIT_ENABLED and decision.get("decision") == "partial_exit" and decision.get("exit_percentage"):
                    exit_pct = float(decision.get("exit_percentage")) / 100.0
                    exit_quantity = int(quantity * exit_pct)
                    if exit_quantity <= 0:
                        exit_quantity = 1  # Minimum exit quantity
                else:
                    exit_quantity = quantity
                
                # Determine order type based on urgency
                urgency = decision.get("exit_urgency", "medium")
                order_type = "market" if urgency == "high" else "limit"
                
                # Get exit price
                exit_price = decision.get("suggested_exit_price", market_data.get("price", 0))
                
                # Create exit order
                await self._create_exit_order(
                    position_id=position_id,
                    symbol=position.get("symbol"),
                    quantity=exit_quantity,
                    price=exit_price,
                    order_type=order_type,
                    reason=decision.get("reasoning", "llm_exit_decision")
                )
            else:
                logger.info(f"No exit action for position {position_id}: {decision.get('decision')} with confidence {decision.get('confidence')}")
                
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from LLM response for position {position_id}")
        except Exception as e:
            logger.error(f"Error processing LLM response for position {position_id}: {e}")
    
    def _prepare_exit_evaluation_prompt(self, position, market_data, signal_data=None):
        """Prepare the prompt for exit evaluation."""
        # Calculate current P&L
        entry_price = position.get("entry_price", 0)
        current_price = market_data.get("price", 0)
        quantity = position.get("quantity", 0)
        
        pnl = 0
        pnl_pct = 0
        if entry_price > 0 and current_price > 0:
            pnl = (current_price - entry_price) * quantity
            pnl_pct = (current_price / entry_price - 1) * 100
        
        # Prepare prompt
        prompt = f"""
        I need you to evaluate whether we should exit this position:
        
        Position Details:
        - Symbol: {position.get("symbol")}
        - Entry Price: ${entry_price}
        - Current Price: ${current_price}
        - Quantity: {quantity}
        - Current P&L: ${pnl:.2f} ({pnl_pct:.2f}%)
        - Entry Date: {position.get("entry_date")}
        - Stop Loss: ${position.get("stop_loss", "None")}
        - Take Profit: ${position.get("take_profit", "None")}
        - Position Notes: {position.get("notes", "None")}
        
        Current Market Data:
        {json.dumps(market_data, indent=2)}
        """
        
        # Add signal data if provided
        if signal_data:
            prompt += f"""
            
            Recent Signal:
            {json.dumps(signal_data, indent=2)}
            """
        
        # Add trailing instructions
        prompt += """
        
        Based on this information, please evaluate whether we should exit this position now,
        either partially or fully. Consider the technical indicators, current P&L, market conditions, 
        and any signals provided.
        
        Respond with a JSON object containing your exit decision and reasoning.
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
    
    async def _create_exit_order(self, position_id, symbol, quantity, price, order_type, reason):
        """Create an exit order and send it to the broker."""
        logger.info(f"Creating {order_type} exit order for {symbol} - {quantity} shares at ${price}")
        
        try:
            # Generate unique order ID
            order_id = f"exit_{position_id}_{int(time.time())}"
            
            # Create order data
            order = {
                "type": "exit_order",
                "order_id": order_id,
                "position_id": position_id,
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "order_type": order_type,
                "time_in_force": "day",
                "reason": reason,
                "created_at": datetime.now().isoformat()
            }
            
            # Store order in Redis
            await self.redis_client.set(
                f"exit_order:{order_id}",
                json.dumps(order)
            )
            
            # Publish to Trade Order channel
            await self.redis_client.publish(
                TRADE_ORDER_CHANNEL,
                json.dumps(order)
            )
            
            logger.info(f"Exit order {order_id} sent for position {position_id}")
            
            # Update last signal time to prevent rapid re-evaluation
            self.last_signal_time[position_id] = time.time()
            
        except Exception as e:
            logger.error(f"Error creating exit order for position {position_id}: {e}")
    
    async def close(self):
        """Close connections and cancel tasks."""
        logger.info("Shutting down Trade LLM")
        
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
        logger.info("Trade LLM shut down")

async def main():
    """Main function to run the Trade LLM module."""
    trade_llm = TradeLLM()
    
    try:
        # Initialize and run
        await trade_llm.initialize()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        # Clean shutdown
        await trade_llm.close()

if __name__ == "__main__":
    asyncio.run(main())