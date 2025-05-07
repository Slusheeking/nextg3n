"""
Trade Flow Orchestrator for NextG3N Trading System

Coordinates AutoGen 0.9.0 agents and MCP tools for the trading workflow.
Manages stock screening, analysis, decision-making, execution, and monitoring.
Publishes to Kafka topic nextg3n-orchestration-events.
"""

import logging
import asyncio
import json
import os
import time
import datetime
import pipeline
from typing import Dict, Any, List
from autogen import ConversableAgent, GroupChat, GroupChatManager
from kafka import KafkaProducer
from redis import Redis
from monitoring.metrics_logger import MetricsLogger
from services.mcp_client import MCPClient
from models.decision.decision_model import DecisionModel


class TradeFlowOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.logger = MetricsLogger(
            component_name="trade_flow_orchestrator")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.config = config
        self.kafka_config = config.get("kafka", {})
        self.redis_config = config.get("storage", {}).get("redis", {})
        self.llm_config = config.get("llm", {})
        self.mcp_client = MCPClient(config)
        self.decision_model = DecisionModel(config)
        self.strategy_refinement_enabled = self.config.get("orchestration", {}).get("strategy_refinement", {}).get("enabled", False)
        self.strategy_refinement_model_name = self.config.get("orchestration", {}).get("strategy_refinement", {}).get("model_name", "google/flan-t5-base")
        self.llm_pipeline = pipeline("text2text-generation", model=self.strategy_refinement_model_name)
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get(
                "bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.redis = Redis(
            host=self.redis_config.get("host", "localhost"),
            port=self.redis_config.get("port", 6379),
            db=self.redis_config.get("db", 0)
        )

        # AutoGen agents - REMOVED
        # self.stock_picker = ConversableAgent(
        #     name="StockPicker",
        #     llm_config={"config_list": [{"model": self.llm_config.get(
        #         "model", "gpt-4"), "api_key": os.getenv(
        #             "OPENROUTER_API_KEY")}]},
        #     system_message="Select promising stocks based on volatility,
        #     liquidity, and sentiment."
        # )
        # self.sentiment_analyzer = ConversableAgent(
        #     name="SentimentAnalyzer",
        #     llm_config={"config_list": [{"model": self.llm_config.get(
        #         "model", "gpt-4"), "api_key": os.getenv(
        #             "OPENROUTER_API_KEY")}]},
        #     system_message="Analyze sentiment data for trading
        #     decisions."
        # )
        # self.predictor = ConversableAgent(
        #     name="Predictor",
        #     llm_config={"config_list": [{"model": self.llm_config.get(
        #         "model", "gpt-4"), "api_key": os.getenv(
        #             "OPENROUTER_API_KEY")}]},
        #     system_message="Predict price movements based on historical
        #     and real-time data."
        # )
        # self.context_analyzer = ConversableAgent(
        #     name="ContextAnalyzer",
        #     llm_config={"config_list": [{"model": self.llm_config.get(
        #         "model", "gpt-4"), "api_key": os.getenv(
        #             "OPENROUTER_API_KEY")}]},
        #     system_message="Provide market context for trading
        #     decisions."
        # )
        # self.trade_decider = ConversableAgent(
        #     name="TradeDecider",
        #     llm_config={"config_list": [{"model": self.llm_config.get(
        #         "model", "gpt-4"), "api_key": os.getenv(
        #             "OPENROUTER_API_KEY")}]},
        #     system_message="Make final trading decisions based on agent
        #     inputs."
        # )
        # self.group_chat = GroupChat(
        #     agents=[self.stock_picker, self.sentiment_analyzer,
        #             self.predictor, self.context_analyzer,
        #             self.trade_decider],
        #     messages=[],
        #     max_round=3
        # )
        # self.group_chat_manager = GroupChatManager(
        #     groupchat=self.group_chat,
        #     llm_config={"config_list": [{"model": self.llm_config.get(
        #         "model", "gpt-4"), "api_key": os.getenv(
        #             "OPENROUTER_API_KEY")}]}
        # )
        self.logger.info("TradeFlowOrchestrator initialized")

    async def run_trading_workflow(self) -> Dict[str, Any]:
        operation_id = f"workflow_{int(time.time())}"
        self.logger.info(f"Running trading workflow - Operation: {operation_id}")

        try:
            # Step 1: Stock screening
            watchlist_key = "watchlist"
            watchlist = self.redis.get(watchlist_key)
            if watchlist:
                symbols = json.loads(watchlist.decode('utf-8'))
                self.logger.info(f"Fetched watchlist from Redis: {symbols}")
            else:
                self.logger.warning("Watchlist not found in Redis. Using default symbols.")
                symbols = ["AAPL", "GOOGL", "TSLA"]  # Default symbols if watchlist is not found
            ranking = await self.mcp_client.call_tool("stock_ranker", "rank_stocks", {"symbols": symbols})
            if not ranking["success"]:
                return {"success": False, "error": "Stock ranking failed", "operation_id": operation_id}

            top_stocks = [s["symbol"] for s in ranking["stocks"]]
            self.logger.info(f"Top stocks: {top_stocks}")

            # Step 2: Analysis and decision
            decisions = []
            for symbol in top_stocks:
                # sentiment = await self.mcp_client.call_tool("sentiment", "analyze_sentiment", {"symbol": symbol})
                # forecast = await self.mcp_client.call_tool("forecast", "predict_price", {"symbol": symbol, "timeframe": "1m"})
                # context = await self.mcp_client.call_tool("context", "retrieve_context", {"symbol": symbol})

                # message = f"Evaluate {symbol}: Sentiment={sentiment.get('sentiment_score', 0)}, Forecast={forecast.get('prediction', 0)}, Context={context.get('context_summary', '')}"
                # loop = asyncio.get_event_loop()
                # chat_result = await loop.run_in_executor(
                #     None,
                #     lambda: self.group_chat_manager.initiate_chat(self.stock_picker, message=message)
                # )

                # action = "buy" if "buy" in chat_result.lower() else "sell" if "sell" in chat_result.lower() else "hold"
                # if action != "hold":
                #     trade = await self.mcp_client.call_tool("alpaca", "place_order", {
                #         "symbol": symbol,
                #         "action": action,
                #         "quantity": 10,  # Placeholder; calculate based on $1,000 max
                #         "order_type": "market"
                #     })
                #     if trade["success"]:
                #         decisions.append({"symbol": symbol, "action": action, "order_id": trade["order_id"]})
                decision = await self.decision_model.make_decision(symbol)
                if decision["success"]:
                    action = decision["action"]
                    if action != "hold":
                        # Calculate quantity based on max position
                        # size and current price
                        max_position_size = self.config.get(
                            "trade", {}).get("max_position_size", 0.2)
                        capital = self.config.get("trade", {}).get(
                            "capital", 5000.0)
                        max_trade_amount = capital * max_position_size

                        # Fetch current price from market data service
                        current_price_data = await self.mcp_client.call_tool(
                            "market_data", "get_latest_price",
                            {"symbol": symbol})
                        if current_price_data["success"]:
                            current_price = current_price_data["price"]
                            quantity = int(
                                max_trade_amount / current_price)
                            self.logger.info(
                                f"Calculated quantity for {symbol}: "
                                f"{quantity}")
                        else:
                            self.logger.warning(
                                f"Failed to fetch current price for "
                                f"{symbol}. Using default quantity of 1.")
                            quantity = 1

                        trade = await self.mcp_client.call_tool(
                            "alpaca", "place_order", {
                                "symbol": symbol,
                                "action": action,
                                "quantity": quantity,
                                "order_type": "market"
                            })
                        if trade["success"]:
                            decisions.append({
                                "symbol": symbol,
                                "action": action,
                                "order_id": trade["order_id"]
                            })
            # Step 3: Monitoring
            for decision in decisions:
                monitor = await self.mcp_client.call_tool("trade_executor", "monitor_trade", {
                    "symbol": decision["symbol"],
                    "order_id": decision["order_id"]
                })
                if monitor["should_exit"]:
                    await self.mcp_client.call_tool("alpaca", "place_order", {
                        "symbol": decision["symbol"],
                        "action": "sell" if decision["action"] == "buy" else "buy",
                        "quantity": 10,
                        "order_type": "market"
                    })

            result = {
                "success": True,
                "decisions": decisions,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.redis.setex(f"workflow:{operation_id}", 300, json.dumps(result))
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n-')}orchestration-events",
                {"event": "workflow_completed", "data": result}
            )
            return result

        except Exception as e:
            self.logger.error(f"Error running trading workflow: {e}")
            return {"success": False, "error": str(e), "operation_id": operation_id}

        finally:
            if self.strategy_refinement_enabled:
                await self.refine_strategy_with_llm(result)

    async def shutdown(self):
        self.producer.close()
        self.redis.close()
        await self.mcp_client.shutdown()
        await self.decision_model.shutdown()
        self.logger.info("TradeFlowOrchestrator shutdown")

    async def refine_strategy_with_llm(self, workflow_result: Dict[str, Any]) -> None:
        """
        Refines the trading strategy using a large language model.
        """
        try:
            # Fetch historical trade data from Redis or a database
            # For simplicity, let's assume the workflow_result contains the necessary data
            trade_data = workflow_result.get("decisions", [])

            # Prepare a prompt for the LLM
            prompt = f"Analyze the following trading decisions and suggest improvements to the strategy: {trade_data}. Consider factors like profitability, risk management, and market conditions."

            # Call the LLM for analysis
            llm_response = self.llm_pipeline(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]

            # Log the LLM response
            self.logger.info(f"LLM strategy refinement suggestions: {llm_response}")

            # Implement the suggested improvements (e.g., update DecisionModel, adjust weights)
            # This part requires careful consideration and implementation based on the LLM's suggestions
            # For now, let's just log a message indicating that the improvements need to be implemented
            self.logger.info("LLM strategy refinement suggestions need to be implemented.")

        except Exception as e:
            self.logger.error(f"LLM strategy refinement failed: {e}")
