"""
Sentiment Model for NextG3N Trading System

This module implements the SentimentModel class, using a fine-tuned RoBERTa model and
OpenRouter LLM for sentiment analysis of financial texts. It supports the SentimentAgent
in TradeFlowOrchestrator.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import torch
import pandas as pd
import aiohttp
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

# Kafka imports
from kafka import KafkaProducer

class SentimentDataset(Dataset):
    """
    Custom dataset for sentiment analysis.
    """
    def __init__(self, texts: List[str], tokenizer: RobertaTokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class SentimentModel:
    """
    Model for analyzing sentiment in financial texts using a fine-tuned RoBERTa model
    and OpenRouter LLM. Supports the SentimentAgent in TradeFlowOrchestrator.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SentimentModel with configuration and model settings.

        Args:
            config: Configuration dictionary with model and Kafka settings
        """
        init_start_time = time.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="sentiment_model")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.model_config = config.get("models", {}).get("sentiment", {})
        self.kafka_config = config.get("kafka", {})
        self.llm_config = config.get("llm", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize RoBERTa model and tokenizer
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialize_model()
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("sentiment_model.initialization_time_ms", init_duration)
        self.logger.info("SentimentModel initialized")

    def _initialize_model(self):
        """
        Initialize the RoBERTa model and tokenizer.
        """
        try:
            model_name = self.model_config.get("model_name", "roberta-base")
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3  # Positive, neutral, negative
            )
            
            # Load pre-trained checkpoint if provided
            checkpoint_path = self.model_config.get("checkpoint_path")
            if checkpoint_path and os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.logger.info(f"Loaded pre-trained checkpoint from {checkpoint_path}")
            
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RoBERTa model: {e}")
            self.model = None
            self.tokenizer = None
            raise

    async def analyze_sentiment(
        self,
        texts: List[str],
        batch_size: int = 16,
        use_llm: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze sentiment for a list of texts using RoBERTa or OpenRouter LLM.

        Args:
            texts: List of texts to analyze
            batch_size: Batch size for RoBERTa inference (default: 16)
            use_llm: Use OpenRouter LLM instead of RoBERTa (default: False)

        Returns:
            Dictionary containing sentiment scores
        """
        start_time = time.time()
        operation_id = f"analyze_sentiment_{int(start_time)}"
        self.logger.info(f"Analyzing sentiment for {len(texts)} texts (use_llm={use_llm}) - Operation: {operation_id}")

        if not texts:
            self.logger.error("No texts provided for sentiment analysis")
            return {
                "success": False,
                "error": "No texts provided",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            if use_llm:
                # Use OpenRouter LLM
                result = await self._analyze_sentiment_with_llm(texts)
            else:
                # Use RoBERTa
                result = await self._analyze_sentiment_with_roberta(texts, batch_size)
            
            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}sentiment_events",
                {"event": "sentiment_analyzed", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("sentiment_model.analyze_sentiment_time_ms", duration)
            self.logger.info(f"Sentiment analyzed for {len(texts)} texts: Avg Score={result['average_sentiment']['sentiment_score']:.2f}")
            self.logger.counter("sentiment_model.analyses_completed", len(texts))
            self.logger.track_llm_usage(tokens=len(texts) * 100 if use_llm else 0, model=self.llm_config.get("model", "roberta-base"))
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            self.logger.counter("sentiment_model.analysis_errors", len(texts))
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_sentiment_with_roberta(self, texts: List[str], batch_size: int) -> Dict[str, Any]:
        """
        Analyze sentiment using RoBERTa model.

        Args:
            texts: List of texts to analyze
            batch_size: Batch size for inference

        Returns:
            Dictionary containing sentiment scores
        """
        if not self.model or not self.tokenizer:
            self.logger.error("RoBERTa model or tokenizer not initialized")
            return {
                "success": False,
                "error": "RoBERTa model or tokenizer not initialized",
                "text_count": 0,
                "timestamp": datetime.utcnow().isoformat()
            }

        async with aiohttp.ClientSession() as session:
            loop = asyncio.get_event_loop()
            
            # Create dataset and dataloader
            dataset = SentimentDataset(
                texts,
                self.tokenizer,
                max_length=self.model_config.get("max_length", 512)
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            # Perform inference
            sentiment_scores = []
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    outputs = await loop.run_in_executor(
                        self.executor,
                        lambda: self.model(input_ids, attention_mask=attention_mask)
                    )
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    sentiment_scores.extend(probs.tolist())
            
            # Aggregate sentiment
            results = [
                {
                    "text": text,
                    "positive": score[2],  # Adjusted for RoBERTa label order
                    "neutral": score[1],
                    "negative": score[0],
                    "sentiment_score": score[2] - score[0]  # Positive - negative
                }
                for text, score in zip(texts, sentiment_scores)
            ]
            
            # Calculate average sentiment
            avg_sentiment = {
                "positive": sum(r["positive"] for r in results) / len(results) if results else 0.0,
                "neutral": sum(r["neutral"] for r in results) / len(results) if results else 0.0,
                "negative": sum(r["negative"] for r in results) / len(results) if results else 0.0,
                "sentiment_score": sum(r["sentiment_score"] for r in results) / len(results) if results else 0.0
            }

            return {
                "success": True,
                "results": results,
                "average_sentiment": avg_sentiment,
                "text_count": len(texts),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_sentiment_with_llm(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment using OpenRouter LLM.

        Args:
            texts: List of texts to analyze

        Returns:
            Dictionary containing sentiment scores
        """
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }
        results = []
        
        for text in texts:
            payload = {
                "model": self.llm_config.get("model", "openai/gpt-4"),
                "messages": [
                    {"role": "user", "content": f"Analyze the sentiment of this financial text (positive, neutral, negative) and provide a score (-1 to 1): {text}"}
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
                                sentiment_text = result["choices"][0]["message"]["content"]
                                # Parse sentiment score (assuming LLM returns a number or text like "positive: 0.7")
                                try:
                                    score = float(sentiment_text.split(":")[-1].strip())
                                    sentiment = "positive" if score > 0 else "negative" if score < 0 else "neutral"
                                except:
                                    score = 0.0
                                    sentiment = "neutral"
                                results.append({
                                    "text": text,
                                    "positive": max(0, score),
                                    "neutral": 1 - abs(score),
                                    "negative": max(0, -score),
                                    "sentiment_score": score
                                })
                                break
                            else:
                                await asyncio.sleep(self.llm_config.get("retry_delay", 1000) / 1000)
                    except Exception as e:
                        if attempt == self.llm_config.get("retry_attempts", 3) - 1:
                            self.logger.error(f"Failed to analyze sentiment for text: {e}")
                            results.append({
                                "text": text,
                                "positive": 0.0,
                                "neutral": 1.0,
                                "negative": 0.0,
                                "sentiment_score": 0.0
                            })
                        await asyncio.sleep(self.llm_config.get("retry_delay", 1000) / 1000)
        
        # Calculate average sentiment
        avg_sentiment = {
            "positive": sum(r["positive"] for r in results) / len(results) if results else 0.0,
            "neutral": sum(r["neutral"] for r in results) / len(results) if results else 0.0,
            "negative": sum(r["negative"] for r in results) / len(results) if results else 0.0,
            "sentiment_score": sum(r["sentiment_score"] for r in results) / len(results) if results else 0.0
        }

        return {
            "success": True,
            "results": results,
            "average_sentiment": avg_sentiment,
            "text_count": len(texts),
            "timestamp": datetime.utcnow().isoformat()
        }

    async def train_model(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train or fine-tune the RoBERTa model.

        Args:
            training_data: Training data DataFrame with 'text' and 'label' columns

        Returns:
            Dictionary indicating training result
        """
        start_time = time.time()
        operation_id = f"train_model_{int(start_time)}"
        self.logger.info(f"Training RoBERTa model - Operation: {operation_id}")

        if not self.model or not self.tokenizer:
            self.logger.error("RoBERTa model or tokenizer not initialized")
            return {
                "success": False,
                "error": "RoBERTa model or tokenizer not initialized",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                
                # Create dataset
                texts = training_data["text"].tolist()
                labels = training_data["label"].tolist()  # Expected: 0 (negative), 1 (neutral), 2 (positive)
                dataset = SentimentDataset(
                    texts,
                    self.tokenizer,
                    max_length=self.model_config.get("max_length", 512)
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.model_config.get("batch_size", 16),
                    shuffle=True
                )
                
                # Train model
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.model_config.get("learning_rate", 2e-5)
                )
                self.model.train()
                
                for epoch in range(self.model_config.get("max_epochs", 3)):
                    for batch_idx, batch in enumerate(dataloader):
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels_batch = torch.tensor(labels[batch_idx * dataloader.batch_size:(batch_idx + 1) * dataloader.batch_size]).to(self.device)
                        
                        outputs = await loop.run_in_executor(
                            self.executor,
                            lambda: self.model(input_ids, attention_mask=attention_mask, labels=labels_batch)
                        )
                        loss = outputs.loss
                        
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        self.logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
                
                self.model.eval()

            result = {
                "success": True,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}sentiment_events",
                {"event": "model_trained", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("sentiment_model.train_model_time_ms", duration)
            self.logger.info("RoBERTa model trained successfully")
            self.logger.counter("sentiment_model.trainings_completed", 1)
            return result

        except Exception as e:
            self.logger.error(f"Error training RoBERTa model: {e}")
            self.logger.counter("sentiment_model.training_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def shutdown(self):
        """
        Shutdown the SentimentModel and close resources.
        """
        self.logger.info("Shutting down SentimentModel")
        self.executor.shutdown(wait=True)
        self.producer.close()
        self.logger.info("Kafka producer closed")