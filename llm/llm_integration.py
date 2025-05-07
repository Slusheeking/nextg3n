"""
LLM Integration for NextG3N Trading System.
Handles the integration of data sources with LLM for trading decisions.
"""

import asyncio
import json
import os
import re
from typing import Dict, Any, List, Optional, Union, Set
import datetime
from datetime import timedelta # Explicitly import timedelta
import pandas as pd
# import numpy as np # Not strictly needed with current full implementation
from dotenv import load_dotenv
from monitor.logging_utils import get_logger
from openai import OpenAI, AsyncOpenAI
import redis.asyncio as aioredis

from mcp.mcp_manager import mcp_manager


class LLMTradingIntegration:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("llm_integration")
        
        self.llm_config = config.get("llm", {})
        self.llm_provider = self.llm_config.get("provider", "openrouter")
        self.llm_model = self.llm_config.get("model", "openai/gpt-4o-mini") # Default to a capable model
        self.llm_temperature = self.llm_config.get("temperature", 0.5) # Slightly lower for more deterministic financial advice
        self.llm_max_tokens = self.llm_config.get("max_tokens", 2500) # Increased for potentially larger contexts
        
        self.trading_config = config.get("trading", {})
        self.min_confidence_threshold = self.trading_config.get("min_confidence_threshold", 0.7) # More descriptive name
        self.max_position_risk_pct = self.trading_config.get("max_position_risk_pct", 0.02) # Max % of capital to risk on one trade
        self.trading_capital = float(self.trading_config.get("capital", 10000.0))
        
        self._init_llm_client()
        
        self.mcp_servers = {}
        self.mcp_server_status = {}
        self.use_alpaca_llm = self.llm_config.get("use_alpaca_llm", False) # If true, Alpaca's LLM API will be used
        
        self.enable_decision_logging_to_redis = self.config.get("enable_decision_logging_to_redis", True)
        self.redis_client_decision_log: Optional[aioredis.Redis] = None # Specific client for decision logs
        
        # The initialization of MCP servers and Redis should be done in an async context.
        # Consider an `async def initialize(self):` method to be called after object creation.
        # For now, we assume these will be called before major operations.

    async def initialize_services(self):
        """Asynchronously initializes MCP servers and Redis client for decision logging."""
        await self._init_mcp_servers()
        if self.enable_decision_logging_to_redis:
            await self._init_decision_redis_client()

    async def _init_decision_redis_client(self):
        redis_config = self.config.get("decision_log_redis", {}) # Separate config section for this Redis
        host = redis_config.get("host", os.getenv("REDIS_HOST", "localhost"))
        port = int(redis_config.get("port", os.getenv("REDIS_PORT", 6379)))
        db = int(redis_config.get("db", 2)) # Use a different DB, e.g., DB 2 for decision logs
        password = redis_config.get("password", os.getenv("REDIS_PASSWORD", None))

        if self.redis_client_decision_log is None:
            try:
                self.redis_client_decision_log = aioredis.Redis(
                    host=host, port=port, db=db, password=password, decode_responses=True
                )
                await self.redis_client_decision_log.ping()
                self.logger.info(f"Connected to Redis for LLM Decision Logging at {host}:{port} (DB: {db})")
            except Exception as e:
                self.logger.error(f"Failed to connect to Redis for LLM Decision Logging: {e}")
                self.redis_client_decision_log = None

    def _init_llm_client(self):
        load_dotenv()
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or self.llm_config.get("api_key")
        if not self.openrouter_api_key and self.llm_provider == "openrouter":
            self.logger.error("OpenRouter API key not found")
            raise ValueError("OpenRouter API key not found")
            
        self.openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.openrouter_api_key)
        self.async_openai_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.openrouter_api_key)
        self.site_url = self.llm_config.get("site_url", "https://nextg3n.ai") # For OpenRouter identification
        self.site_name = self.llm_config.get("site_name", "NextG3N Trading System")
        self.logger.info(f"Initialized LLM client for provider '{self.llm_provider}' with model '{self.llm_model}'")
    
    async def _init_mcp_servers(self):
        available_servers = mcp_manager.get_available_servers()
        running_servers = mcp_manager.get_running_servers()
        self.logger.info(f"Initializing MCP servers. Available: {len(available_servers)}, Running: {len(running_servers)}")
        
        for server_config in available_servers:
            server_name = server_config.get("name")
            if not server_name: continue
            if not server_config.get("enabled", True):
                self.logger.info(f"MCP server {server_name} is disabled in config.")
                self.mcp_server_status[server_name] = "disabled"
                continue
            
            self.mcp_servers[server_name] = server_config
            if server_config.get("auto_start", False) and server_name not in running_servers:
                self.logger.info(f"Auto-starting {server_name} MCP server...")
                if not await mcp_manager.start_server(server_name):
                    self.logger.error(f"Failed to auto-start {server_name} MCP server.")
                    self.mcp_server_status[server_name] = "failed_to_start"
                    continue
            
            server_info = await mcp_manager.get_server_info(server_name)
            if "error" in server_info or not server_info.get("name"): # Check for error or empty info
                self.logger.error(f"Error getting {server_name} server info: {server_info.get('error', 'Unknown error or empty info')}")
                self.mcp_server_status[server_name] = "error_info"
                continue
            
            self.logger.info(f"Successfully connected to {server_name} MCP server (version {server_info.get('version', 'N/A')}).")
            self.mcp_server_status[server_name] = "connected"

    async def _execute_mcp_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any], error_context: str) -> Dict[str, Any]:
        if server_name not in self.mcp_servers or self.mcp_server_status.get(server_name) != "connected":
            self.logger.error(f"{error_context}: {server_name} MCP server is not connected or configured.")
            return {"success": False, "error": f"{server_name} MCP server not available."}
        
        self.logger.debug(f"Executing {tool_name} on {server_name} for {error_context} with args: {arguments}")
        result = await mcp_manager.execute_tool(server_name=server_name, tool_name=tool_name, arguments=arguments)
        
        if "error" in result or not result.get("success", True):
            error_msg = result.get("error", f"Unknown error from {server_name}.{tool_name}")
            self.logger.error(f"Error from {server_name}.{tool_name}: {error_msg}")
            return {"success": False, "error": error_msg, "details": result.get("details")}
        return result

    async def collect_trading_data(self, symbols: List[str]) -> Dict[str, Any]:
        self.logger.info(f"Collecting all trading data for symbols: {symbols}")
        tasks = {
            "real_time": self._collect_real_time_data(symbols),
            "historical": self._collect_historical_data(symbols),
            "options_flow": self._collect_options_flow(symbols),
            "social_sentiment": self._collect_social_sentiment(symbols),
            "news": self._collect_news_data(symbols),
            "portfolio": self._collect_portfolio_data()
        }
        
        results = await asyncio.gather(*(tasks.values()), return_exceptions=True)
        
        data = {}
        for i, key in enumerate(tasks.keys()):
            if isinstance(results[i], Exception):
                self.logger.error(f"Exception during data collection for {key}: {results[i]}")
                data[key] = {"success": False, "error": str(results[i])}
            else:
                data[key] = results[i]
        
        data["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        data["symbols_queried"] = symbols # Keep track of what was asked for
        return data

    async def _collect_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        return await self._execute_mcp_tool("polygon_websocket", "fetch_realtime_data", 
                                            {"symbols": symbols, "channels": ["T", "Q", "AM"], "duration_seconds": 15}, # Shorter duration for faster collection
                                            "real-time market data")

    async def _collect_historical_data(self, symbols: List[str]) -> Dict[str, Any]:
        end_date = datetime.datetime.now(datetime.timezone.utc)
        start_date = end_date - datetime.timedelta(days=60) # Extended history for better context
        return await self._execute_mcp_tool("polygon_rest", "fetch_aggregates", 
                                            {"symbols": symbols, "start_date": start_date.strftime("%Y-%m-%d"), 
                                             "end_date": end_date.strftime("%Y-%m-%d"), "timespan": "day", 
                                             "multiplier": 1, "adjusted": True, "limit": 500}, # Increased limit
                                            "historical market data")

    async def _collect_options_flow(self, symbols: List[str]) -> Dict[str, Any]:
        return await self._execute_mcp_tool("unusual_whales", "analyze_options_flow",  # Use the analysis endpoint
                                            {"symbols": symbols, "days": 2, "min_premium": 20000},  # Analyze last 2 days
                                            "options flow data and analysis")

    async def _collect_social_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        return await self._execute_mcp_tool("reddit_processor", "analyze_ticker_sentiment", 
                                            {"tickers": symbols, 
                                             "subreddits": self.config.get("reddit_subreddits", ["wallstreetbets", "stocks", "investing"]), 
                                             "time_filter": "day", "limit_per_ticker": 30}, # More recent, fewer posts
                                            "social sentiment data")

    async def _collect_news_data(self, symbols: List[str]) -> Dict[str, Any]:
        return await self._execute_mcp_tool("yahoo_finance", "analyze_news", 
                                            {"symbols": symbols, "count": 10}, # Fewer, more focused news items
                                            "news data and analysis")

    async def _collect_portfolio_data(self) -> Dict[str, Any]:
        return await self._execute_mcp_tool("alpaca", "get_portfolio_data", {}, "current portfolio data")

    async def _assess_trade_risk(self, symbol: str, entry_price: float, stop_price: float, profit_target: float) -> Dict[str, Any]:
        return await self._execute_mcp_tool("alpaca", "assess_trade_risk", 
                                            {"symbol": symbol, "entry_price": entry_price, 
                                             "stop_price": stop_price, "profit_target": profit_target}, 
                                            "trade risk assessment")
    
    def _format_data_for_prompt(self, data_value: Any, section_name: str, symbol: str) -> str:
        """Helper to format a section of data for the LLM prompt."""
        prompt_section = f"\n## {section_name} for {symbol}\n"
        if isinstance(data_value, dict) and data_value.get("success", False):
            # Extract relevant parts; this needs to be tailored to each data source's output structure
            # For example, for historical data:
            if section_name.startswith("Historical") and "results" in data_value:
                symbol_results = data_value["results"].get(symbol, [])
                if symbol_results:
                    prompt_section += f"Last {min(5, len(symbol_results))} daily bars:\n"
                    for bar in symbol_results[-5:]: # Last 5 bars
                        ts = datetime.datetime.fromtimestamp(bar['t']/1000).strftime('%Y-%m-%d') if 't' in bar else 'N/A'
                        prompt_section += f"- {ts}: O:{bar.get('o')} H:{bar.get('h')} L:{bar.get('l')} C:{bar.get('c')} V:{bar.get('v')}\n"
                else:
                    prompt_section += "No specific historical data found for this symbol in the response.\n"
            # For news:
            elif section_name.startswith("News") and "articles" in data_value:
                symbol_articles = [art for art in data_value["articles"] if symbol.lower() in [t.lower() for t in art.get("tickers",[])]]
                if symbol_articles:
                    prompt_section += f"Top {min(3, len(symbol_articles))} headlines:\n"
                    for art in symbol_articles[:3]:
                        prompt_section += f"- {art.get('title')} (Relevance: {art.get('relevance_score', 'N/A')}, Impact: {art.get('impact_score', 'N/A')})\n"
                else:
                    prompt_section += "No news articles specifically for this symbol in the response.\n"
            # For social sentiment:
            elif section_name.startswith("Social Sentiment") and "ticker_sentiment_analysis" in data_value:
                sentiment_data = data_value["ticker_sentiment_analysis"].get(symbol)
                if sentiment_data:
                    prompt_section += f"Mentions: {sentiment_data.get('mention_count')}\n"
                    prompt_section += f"Sentiment: {sentiment_data.get('sentiment',{}).get('dominant')}\n"
                    prompt_section += f"Scores: {json.dumps(sentiment_data.get('sentiment',{}).get('scores'))}\n"
                else:
                    prompt_section += "No social sentiment data for this symbol.\n"
            # For options flow:
            elif section_name.startswith("Options Flow") and "data" in data_value: # Assuming 'data' contains 'unusual_activity'
                unusual = data_value["data"].get("unusual_activity", [])
                symbol_unusual = [flow for flow in unusual if flow.get("underlying_symbol", "").upper() == symbol.upper()]
                if symbol_unusual:
                    prompt_section += f"Unusual Activity Count: {len(symbol_unusual)}\n"
                    for flow in symbol_unusual[:3]:
                        prompt_section += f"- Type: {flow.get('type')}, Strike: {flow.get('strike_price')}, Exp: {flow.get('expiration_date')}, Premium: {flow.get('total_premium')}\n"
                else:
                    prompt_section += "No specific unusual options activity for this symbol.\n"
            else:
                # Generic dump for other successful data sections, limited length
                content_str = json.dumps(data_value, indent=2)
                prompt_section += content_str[:1000] + ("..." if len(content_str) > 1000 else "") + "\n"
        elif isinstance(data_value, dict) and "error" in data_value:
            prompt_section += f"Error collecting data: {data_value['error']}\n"
        else:
            prompt_section += "Data not available or in unexpected format.\n"
        return prompt_section

    def _prepare_llm_prompt(self, data: Dict[str, Any], symbol: str) -> str:
        prompt = f"""
You are an expert day trading advisor analyzing data for {symbol}. Your goal is to identify high-probability, short-term (intraday to 1-2 days) trading opportunities.

**Key Day Trading Criteria to Emphasize:**
1. **Liquidity & Volume:** High trading volume (e.g., >1M shares daily average), significant relative volume (e.g., >1.5x average).
2. **Volatility:** Sufficient price movement for profit (e.g., ATR > $0.50 or >2% of stock price).
3. **Clear Trend/Catalyst:** Identifiable intraday trend, news catalyst, or strong pre-market movement.
4. **Technical Confirmation:** Support/resistance levels, chart patterns, key moving average behavior.
5. **Options Activity:** Unusual or significant options flow indicating directional bias.
6. **Sentiment:** Strong social or news sentiment aligning with technicals.

**Current Portfolio Context:**
"""
        portfolio_data = data.get("portfolio", {})
        if portfolio_data.get("success"):
            prompt += f"- Account Value: ${portfolio_data.get('account_value', 0):,.2f}\n"
            prompt += f"- Available Funds for Trading: ${portfolio_data.get('available_funds', 0):,.2f}\n"
            prompt += f"- Current Positions: {portfolio_data.get('positions_count', 0)}\n"
            current_pos_for_symbol = next((p for p in portfolio_data.get("positions",[]) if p.get("symbol") == symbol), None)
            if current_pos_for_symbol:
                prompt += f"- Existing Position in {symbol}: Qty {current_pos_for_symbol.get('qty')}, Avg Entry ${current_pos_for_symbol.get('avg_entry_price')}\n"
            else:
                prompt += f"- No existing position in {symbol}.\n"
        else:
            prompt += "- Portfolio data not available.\n"

        prompt += self._format_data_for_prompt(data.get("real_time"), "Real-Time Market Data", symbol)
        prompt += self._format_data_for_prompt(data.get("historical"), "Historical Market Data (Daily - Last 60d)", symbol)
        prompt += self._format_data_for_prompt(data.get("options_flow"), "Options Flow Analysis (Last 2 days)", symbol)
        prompt += self._format_data_for_prompt(data.get("social_sentiment"), "Social Sentiment (Reddit - Last Day)", symbol)
        prompt += self._format_data_for_prompt(data.get("news"), "News Analysis (Recent)", symbol)
        
        prompt += f"""
**Trading Decision Request for {symbol}:**

Based on all the above data and prioritizing the Key Day Trading Criteria, provide a concise trading decision:

1. **Decision:** (BUY, SELL, HOLD)
2. **Confidence:** (0-100%) - Your confidence in this specific setup.
3. **Strategy Name:** (e.g., "News Catalyst Momentum Play", "Options Flow Breakout", "Sentiment Reversal")
4. **Reasoning (max 3 bullet points):** Key factors supporting the decision, referencing specific data points and criteria.
5. **Entry Price:** Suggested price or range.
6. **Stop Loss:** Price level.
7. **Take Profit Target(s):** Price level(s).
8. **Position Size (% of Trading Capital):** Recommended percentage (e.g., 1%, 2%, max 5%) considering risk and available funds.
9. **Holding Period:** (e.g., "Intraday", "1-2 hours", "End of Day")

If HOLD, briefly state why and what would change your mind.
If no clear opportunity meets the criteria, state HOLD and explain.
Focus on actionable, high-conviction setups.
"""
        return prompt
    
    async def get_llm_decision(self, prompt: str, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        try:
            llm_response_content = ""
            provider_used = ""

            if self.use_alpaca_llm and self.mcp_server_status.get("alpaca_llm") == "connected":
                provider_used = "alpaca_llm"
                llm_api_result = await self._get_alpaca_llm_decision(prompt)
                if llm_api_result.get("success"):
                    llm_response_content = llm_api_result.get("raw_response", "")
                else:
                    self.logger.warning("Alpaca LLM failed, falling back to OpenRouter if configured.")
                    # Fallback logic can be more sophisticated, e.g., retry or use OpenRouter by default
                    if self.llm_provider == "openrouter": # Check if OpenRouter is the primary or fallback
                         provider_used = "openrouter" # Update provider if falling back
                         llm_api_result = await self._get_openrouter_llm_decision(prompt)
                         if llm_api_result.get("success"):
                             llm_response_content = llm_api_result.get("raw_response", "")
            elif self.llm_provider == "openrouter":
                 provider_used = "openrouter"
                 llm_api_result = await self._get_openrouter_llm_decision(prompt)
                 if llm_api_result.get("success"):
                     llm_response_content = llm_api_result.get("raw_response", "")
            else:
                self.logger.error("No suitable LLM provider configured or available.")
                return {"success": False, "error": "No LLM provider available."}

            if not llm_response_content:
                self.logger.error(f"LLM ({provider_used}) did not return content.")
                return {"success": False, "error": f"LLM ({provider_used}) did not return content.", "provider": provider_used}

            extracted_decision = self._extract_decision(llm_response_content)
            extracted_decision["symbol"] = symbol # Ensure symbol is part of the decision object

            llm_full_response = {
                "success": True,
                "raw_response": llm_response_content,
                "decision": extracted_decision,
                "provider": provider_used
            }
            
            # Perform risk assessment and funds check if it's a trade decision
            if extracted_decision.get("action") in ["BUY", "SELL"] and \
               all(extracted_decision.get(k) is not None for k in ["entry_price", "stop_loss", "take_profit"]):
                
                risk_assessment_result = await self._assess_trade_risk(
                    symbol=symbol,
                    entry_price=extracted_decision["entry_price"],
                    stop_price=extracted_decision["stop_loss"],
                    profit_target=extracted_decision["take_profit"]
                )
                
                if risk_assessment_result.get("success"):
                    extracted_decision["risk_assessment"] = risk_assessment_result.get("assessment_details", risk_assessment_result) # Use nested if exists
                    
                    # Use position size from risk assessment (shares)
                    calculated_shares = risk_assessment_result.get("position_size", 0)
                    entry_price = extracted_decision["entry_price"]
                    estimated_cost = calculated_shares * entry_price
                    
                    portfolio_info = data.get("portfolio", {})
                    available_funds = float(portfolio_info.get("available_funds", 0))

                    # Final check on affordability and adjust if necessary
                    if available_funds > 0 and entry_price > 0 and estimated_cost > available_funds:
                        self.logger.warning(f"Cost {estimated_cost} for {calculated_shares} shares of {symbol} exceeds available funds {available_funds}. Adjusting.")
                        adjusted_shares = int(available_funds / entry_price)
                        calculated_shares = max(0, adjusted_shares)
                        estimated_cost = calculated_shares * entry_price
                        extracted_decision["notes_on_sizing"] = f"Sized down to {calculated_shares} shares due to available funds. Original risk-based size: {risk_assessment_result.get('position_size')}."
                    
                    extracted_decision["position_size_shares"] = calculated_shares
                    extracted_decision["estimated_cost"] = estimated_cost
                    # Update meets_criteria based on final cost and funds
                    extracted_decision["meets_criteria"] = risk_assessment_result.get("meets_criteria", False) and (estimated_cost <= available_funds if available_funds > 0 else True)

                else:
                    extracted_decision["notes_on_sizing"] = "Risk assessment failed. Position size not finalized."
                    extracted_decision["position_size_shares"] = 0
            
            await self.log_llm_decision_and_context(symbol, data, llm_full_response)
            return llm_full_response
            
        except Exception as e:
            self.logger.error(f"Critical error in get_llm_decision for {symbol}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def _get_openrouter_llm_decision(self, prompt: str) -> Dict[str, Any]:
        try:
            completion = await self.async_openai_client.chat.completions.create(
                extra_headers={"HTTP-Referer": self.site_url, "X-Title": self.site_name},
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert day trading advisor making precise trading decisions based on market data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_temperature, max_tokens=self.llm_max_tokens
            )
            decision_text = completion.choices[0].message.content
            return {"success": True, "raw_response": decision_text}
        except Exception as e:
            self.logger.error(f"Error calling OpenRouter: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_alpaca_llm_decision(self, prompt: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are an expert day trading advisor making precise trading decisions based on market data."},
            {"role": "user", "content": prompt}
        ]
        result = await self._execute_mcp_tool("alpaca_llm", "chat_completion", 
                                              {"messages": messages, "model": self.llm_config.get("alpaca_model", "llama-2-70b-chat"), 
                                               "max_tokens": self.llm_max_tokens, "temperature": self.llm_temperature},
                                              "Alpaca LLM decision")
        if result.get("success") and "message" in result.get("result", {}):
            return {"success": True, "raw_response": result["result"]["message"]["content"]}
        else:
            return {"success": False, "error": result.get("error", "Unexpected response from Alpaca LLM")}

    def _extract_decision(self, decision_text: str) -> Dict[str, Any]:
        decision = {"action": "HOLD", "confidence": 0.0, "reasoning": "Default due to parsing difficulty."}
        try:
            # Action
            action_match = re.search(r"Decision:\s*(BUY|SELL|HOLD)", decision_text, re.IGNORECASE)
            if action_match: decision["action"] = action_match.group(1).upper()

            # Confidence
            conf_match = re.search(r"Confidence:\s*(\d+)%", decision_text, re.IGNORECASE)
            if conf_match: decision["confidence"] = float(conf_match.group(1)) / 100.0
            
            # Strategy Name
            strat_match = re.search(r"Strategy Name:\s*(.+)", decision_text, re.IGNORECASE)
            if strat_match: decision["strategy_name"] = strat_match.group(1).strip()

            # Reasoning
            reason_match = re.search(r"Reasoning:\s*([\s\S]*?)(?=\n\d+\.\s|\nEntry Price:|$)", decision_text, re.IGNORECASE)
            if reason_match: decision["reasoning"] = reason_match.group(1).strip()

            # Prices
            for field, regex_pattern in [
                ("entry_price", r"Entry Price:\s*\$?([\d\.]+)(?:-([\d\.]+))?"), # Handles range like $100-$101
                ("stop_loss", r"Stop Loss:\s*\$?([\d\.]+)"),
                ("take_profit", r"Take Profit Target\(s\)?:\s*\$?([\d\.]+)") # Simpler, takes first target
            ]:
                match = re.search(regex_pattern, decision_text, re.IGNORECASE)
                if match:
                    try:
                        decision[field] = float(match.group(1))
                        if field == "entry_price" and match.group(2): # Handle entry range, take midpoint or lower
                             decision[field] = (float(match.group(1)) + float(match.group(2))) / 2.0
                    except ValueError: self.logger.warning(f"Could not parse {field} from '{match.group(1)}'")


            # Position Size
            pos_size_match = re.search(r"Position Size(?: \(% of Trading Capital\))?:\s*([\d\.]+)%", decision_text, re.IGNORECASE)
            if pos_size_match: decision["position_size_pct_capital"] = float(pos_size_match.group(1)) / 100.0

            # Holding Period
            hold_period_match = re.search(r"Holding Period:\s*(.+)", decision_text, re.IGNORECASE)
            if hold_period_match: decision["holding_period"] = hold_period_match.group(1).strip()

        except Exception as e:
            self.logger.error(f"Error parsing LLM decision text: {e}\nText was: {decision_text[:500]}")
        return decision

    async def log_llm_decision_and_context(self, symbol: str, data_context: Dict[str, Any], llm_response: Dict[str, Any]):
        log_entry = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "symbol": symbol,
            "llm_provider": llm_response.get("provider", "unknown"),
            "llm_model_used": self.llm_model if llm_response.get("provider") == "openrouter" else self.llm_config.get("alpaca_model", "unknown"),
            "decision_details": llm_response.get("decision", {}),
            # Optionally include a summary of data_context if needed, be mindful of log size
            # "data_context_summary": {key: val.get("success", "N/A") for key, val in data_context.items() if isinstance(val, dict)}
        }
        
        self.logger.info(f"LLM_TRADING_DECISION: {json.dumps(log_entry)}")

        if self.enable_decision_logging_to_redis and self.redis_client_decision_log:
            try:
                redis_key = f"llm_decision_log:{symbol}:{log_entry['timestamp_utc']}"
                # Store the detailed log entry, potentially with more context if required for feedback.
                # For feedback loop, raw prompt and full LLM response might be useful.
                full_log_for_redis = {**log_entry, "raw_llm_response": llm_response.get("raw_response")}
                # Consider adding a summary of key input data points from `data_context` as well.
                await self.redis_client_decision_log.set(redis_key, json.dumps(full_log_for_redis), ex=timedelta(days=90))
                self.logger.info(f"LLM decision for {symbol} logged to Redis for feedback: {redis_key}")
            except Exception as e:
                self.logger.error(f"Error logging LLM decision to Redis: {e}")

    async def get_trading_decision_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Orchestrates data collection, prompt preparation, and LLM decision making for a single symbol."""
        self.logger.info(f"Starting trading decision process for symbol: {symbol}")
        
        # Ensure services are initialized (especially if __init__ can't be async)
        await self.initialize_services()

        # 1. Collect all necessary data
        # For a single symbol, pass it as a list
        trading_data = await self.collect_trading_data(symbols=[symbol])

        # 2. Prepare the prompt for the LLM
        prompt = self._prepare_llm_prompt(trading_data, symbol)
        self.logger.debug(f"Generated LLM prompt for {symbol}:\n{prompt[:500]}...") # Log snippet

        # 3. Get decision from LLM (passing trading_data for context like portfolio)
        llm_decision_response = await self.get_llm_decision(prompt, trading_data, symbol)
        
        self.logger.info(f"LLM ({llm_decision_response.get('provider', 'N/A')}) decision for {symbol}: {llm_decision_response.get('decision', {}).get('action', 'Error')}")
        return llm_decision_response
