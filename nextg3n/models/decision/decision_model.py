"""
Decision Model for NextG3N Trading System

This module implements the DecisionModel class, using a Decision Transformer and
OpenRouter LLM for trading decisions and explanations. It supports the TradeAgent in
TradeFlowOrchestrator.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import torch
import aiohttp
import numpy as np
from torch import nn

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

# Kafka imports
from kafka import KafkaProducer

class DecisionTransformer(nn.Module):
    """
    Decision Transformer model for trading decisions.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        state_dim = config.get("state_dim", 64)
        action_dim = config.get("action_dim", 3)
        hidden_size = config.get("hidden_size", 256)
        n_layer = config.get("n_layer", 6)
        n_head = config.get("n_head", 8)
        
        # Simplified transformer architecture
        self.embedding = nn.Linear(state_dim, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=n_head,
            num_encoder_layers=n_layer,
            dim_feedforward=hidden_size * 4
        )
        self.action_head = nn.Linear(hidden_size, action_dim)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.to('cuda')
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for decision prediction.

        Args:
            states: Input states tensor (batch, sequence, state_dim)

        Returns:
            Predicted action probabilities
        """
        states = self.embedding(states)
        states = self.transformer(states, states)
        actions = self.action_head(states)
        return actions

class DecisionModel:
    """
    Model for making trading decisions using a Decision Transformer and OpenRouter LLM.
    Supports the TradeAgent in TradeFlowOrchestrator.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DecisionModel with configuration and model settings.

        Args:
            config: Configuration dictionary with model and Kafka settings
        """
        init_start_time = time.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="decision_model")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.model_config = config.get("models", {}).get("decision", {})
        self.kafka_config = config.get("kafka", {})
        self.llm_config = config.get("llm", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize Decision Transformer model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialize_model()
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("decision_model.initialization_time_ms", init_duration)
        self.logger.info("DecisionModel initialized")

    def _initialize_model(self):
        """
        Initialize the Decision Transformer model.
        """
        try:
            self.model = DecisionTransformer(self.model_config)
            
            # Load pre-trained checkpoint if provided
            checkpoint_path = self.model_config.get("checkpoint_path")
            if checkpoint_path and os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.logger.info(f"Loaded pre-trained checkpoint from {checkpoint_path}")
            
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Decision Transformer model: {e}")
            self.model = None
            raise

    async def make_decision(
        self,
        symbol: str,
        state: Dict[str, Any],
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Make a trading decision for a stock, optionally generating an LLM explanation.

        Args:
            symbol: Stock symbol
            state: Input state dictionary (price predictions, sentiment, etc.)
            explain: Generate an LLM explanation for the decision (default: False)

        Returns:
            Dictionary containing decision and optional explanation
        """
        start_time = time.time()
        operation_id = f"make_decision_{int(start_time)}"
        self.logger.info(f"Making decision for {symbol} (explain={explain}) - Operation: {operation_id}")

        if not self.model:
            self.logger.error("Decision Transformer model not initialized")
            return {
                "success": False,
                "error": "Decision Transformer model not initialized",
                "symbol": symbol,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                
                # Process state
                state_vector = await loop.run_in_executor(
                    self.executor,
                    lambda: self._process_state(state)
                )
                
                # Convert to tensor
                state_tensor = torch.tensor(state_vector, dtype=torch.float32).to(self.device)
                state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, state_dim)
                
                # Generate decision
                with torch.no_grad():
                    action_logits = await loop.run_in_executor(
                        self.executor,
                        lambda: self.model(state_tensor)
                    )
                    action_probs = torch.softmax(action_logits, dim=-1).cpu().numpy()[0, 0]
                    action_idx = np.argmax(action_probs)
                    action = ["buy", "sell", "hold"][action_idx]
                    confidence = float(action_probs[action_idx])
                
                result = {
                    "success": True,
                    "symbol": symbol,
                    "action": action,
                    "confidence": confidence,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Generate LLM explanation if requested
                if explain:
                    result["explanation"] = await self._generate_explanation(symbol, state, action, confidence)
                    self.logger.track_llm_usage(tokens=200, model=self.llm_config.get("model"))
                
                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}decision_events",
                    {"event": "decision_made", "data": result}
                )
                
                duration = (time.time() - start_time) * 1000
                self.logger.timing("decision_model.make_decision_time_ms", duration)
                self.logger.info(f"Decision made for {symbol}: {action} (confidence: {confidence:.2f})")
                self.logger.counter("decision_model.decisions_made", 1)
                return result
                
        except Exception as e:
            self.logger.error(f"Error making decision for {symbol}: {e}")
            self.logger.counter("decision_model.decision_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def _process_state(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Process input state into a fixed-size vector.

        Args:
            state: Input state dictionary

        Returns:
            State vector as numpy array
        """
        try:
            state_dim = self.model_config.get("state_dim", 64)
            state_vector = np.zeros(state_dim, dtype=np.float32)
            
            # Example: Extract features (customize based on actual state structure)
            price_pred = state.get("price_prediction", {}).get("predicted_price", 0.0)
            sentiment = state.get("sentiment", {}).get("sentiment_score", 0.0)
            rsi = state.get("technical_indicators", {}).get("rsi", 50.0)
            macd = state.get("technical_indicators", {}).get("macd", 0.0)
            context_score = state.get("context", {}).get("context_score", 0.0)
            
            # Fill state vector (simplified example)
            state_vector[0] = price_pred
            state_vector[1] = sentiment
            state_vector[2] = rsi / 100.0
            state_vector[3] = macd
            state_vector[4] = context_score
            
            return state_vector
        
        except Exception as e:
            self.logger.error(f"Error processing state: {e}")
            return np.zeros(self.model_config.get("state_dim", 64), dtype=np.float32)

    async def _generate_explanation(self, symbol: str, state: Dict[str, Any], action: str, confidence: float) -> str:
        """
        Generate an explanation for the trading decision using OpenRouter LLM.

        Args:
            symbol: Stock symbol
            state: Input state dictionary
            action: Trading action (buy, sell, hold)
            confidence: Confidence score

        Returns:
            Explanation text
        """
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }
        state_summary = (
            f"Price prediction: {state.get('price_prediction', {}).get('predicted_price', 0.0):.2f}, "
            f"Sentiment: {state.get('sentiment', {}).get('sentiment_score', 0.0):.2f}, "
            f"RSI: {state.get('technical_indicators', {}).get('rsi', 50.0):.2f}, "
            f"MACD: {state.get('technical_indicators', {}).get('macd', 0.0):.2f}"
        )
        payload = {
            "model": self.llm_config.get("model", "openai/gpt-4"),
            "messages": [
                {"role": "user", "content": f"Explain why a trading decision to {action} {symbol} with confidence {confidence:.2f} was made based on: {state_summary}"}
            ],
            "max_tokens": self.llm_config.get("max_tokens", 512),
            "temperature": self.llm_config.get("temperature", 0.7)
        }
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.llm_config.get("retry_attempts", 3)):
                try:
                    async with session.post(
                        self.llm_config.get("base_url", "https://openrouter.ai/api/v1") + "/chat/completions",
                        json=payload,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result["choices"][0]["message"]["content"]
                        else:
                            await asyncio.sleep(self.llm_config.get("retry_delay", 1000) / 1000)
                except Exception as e:
                    if attempt == self.llm_config.get("retry_attempts", 3) - 1:
                        self.logger.error(f"Failed to generate explanation: {e}")
                        return "Failed to generate explanation"
                    await asyncio.sleep(self.llm_config.get("retry_delay", 1000) / 1000)
        return "Failed to generate explanation"

    def shutdown(self):
        """
        Shutdown the DecisionModel and close resources.
        """
        self.logger.info("Shutting down DecisionModel")
        self.executor.shutdown(wait=True)
        self.producer.close()
        self.logger.info("Kafka producer closed")