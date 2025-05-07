"""
Context Retriever for NextG3N Trading System

This module implements the ContextRetriever class, using RAG to retrieve relevant context
and OpenRouter LLM for generating summaries. It supports the ContextAgent in TradeFlowOrchestrator.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
import torch
from sentence_transformers import SentenceTransformer

# Storage imports
from nextg3n.storage.vector_db import VectorDB

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

# Kafka imports
from kafka import KafkaProducer

class ContextRetriever:
    """
    Class for retrieving and summarizing context using RAG and OpenRouter LLM.
    Supports the ContextAgent in TradeFlowOrchestrator.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ContextRetriever with configuration and model settings.

        Args:
            config: Configuration dictionary with model and Kafka settings
        """
        init_start_time = time.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="context_retriever")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.model_config = config.get("models", {}).get("context", {})
        self.kafka_config = config.get("kafka", {})
        self.llm_config = config.get("llm", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize embedding model
        self.embedder = None
        self.vector_db = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_models()
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("context_retriever.initialization_time_ms", init_duration)
        self.logger.info("ContextRetriever initialized")

    def _initialize_models(self):
        """
        Initialize the embedding model and vector database.
        """
        try:
            model_name = self.model_config.get("embedder_model", "all-MiniLM-L6-v2")
            self.embedder = SentenceTransformer(model_name, device=self.device)
            self.vector_db = VectorDB(self.config)
            self.logger.info(f"Initialized embedder model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            self.embedder = None
            self.vector_db = None
            raise

    async def store_context(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Store context texts in the vector database.

        Args:
            texts: List of texts to store
            metadatas: Optional list of metadata dictionaries

        Returns:
            Dictionary indicating storage result
        """
        start_time = time.time()
        operation_id = f"store_context_{int(start_time)}"
        self.logger.info(f"Storing {len(texts)} context texts - Operation: {operation_id}")

        if not texts:
            self.logger.error("No texts provided for storage")
            return {
                "success": False,
                "error": "No texts provided",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        if not self.embedder or not self.vector_db:
            self.logger.error("Embedder or vector database not initialized")
            return {
                "success": False,
                "error": "Embedder or vector database not initialized",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                
                # Generate embeddings
                embeddings = await loop.run_in_executor(
                    self.executor,
                    lambda: self.embedder.encode(
                        texts,
                        batch_size=self.model_config.get("batch_size", 32),
                        convert_to_numpy=True
                    )
                )
                
                # Store in vector database
                result = await self.vector_db.store_texts(texts, embeddings, metadatas)
                
                if result["success"]:
                    result.update({
                        "operation_id": operation_id,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Publish to Kafka
                    self.producer.send(
                        f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}context_events",
                        {"event": "context_stored", "data": result}
                    )
                    
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("context_retriever.store_context_time_ms", duration)
                    self.logger.info(f"Stored {len(texts)} context texts")
                    self.logger.counter("context_retriever.contexts_stored", len(texts))
                    return result
                
                else:
                    self.logger.error(f"Failed to store context: {result.get('error')}")
                    return result
                
        except Exception as e:
            self.logger.error(f"Error storing context: {e}")
            self.logger.counter("context_retriever.store_errors", len(texts))
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def retrieve_context(
        self,
        query: str,
        k: int = 5,
        generate_summary: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context texts from the vector database, optionally generating
        an LLM summary.

        Args:
            query: Query text for retrieval
            k: Number of top results to retrieve (default: 5)
            generate_summary: Generate an LLM summary of retrieved texts (default: False)

        Returns:
            Dictionary containing retrieved texts and optional summary
        """
        start_time = time.time()
        operation_id = f"retrieve_context_{int(start_time)}"
        self.logger.info(f"Retrieving context for query (k={k}, generate_summary={generate_summary}) - Operation: {operation_id}")

        if not query:
            self.logger.error("No query provided for retrieval")
            return {
                "success": False,
                "error": "No query provided",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        if not self.embedder or not self.vector_db:
            self.logger.error("Embedder or vector database not initialized")
            return {
                "success": False,
                "error": "Embedder or vector database not initialized",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                
                # Generate query embedding
                query_embedding = await loop.run_in_executor(
                    self.executor,
                    lambda: self.embedder.encode([query], convert_to_numpy=True)[0]
                )
                
                # Query vector database
                result = await self.vector_db.query_texts(query_embedding, k=k)
                
                if not result["success"]:
                    self.logger.error(f"Failed to retrieve context: {result.get('error')}")
                    return result
                
                # Optionally generate LLM summary
                summary = None
                if generate_summary and result["texts"]:
                    summary = await self._generate_summary(result["texts"])
                    self.logger.track_llm_usage(tokens=len(" ".join(result["texts"]).split()) * 2, model=self.llm_config.get("model"))
                
                result.update({
                    "summary": summary,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}context_events",
                    {"event": "context_retrieved", "data": result}
                )
                
                duration = (time.time() - start_time) * 1000
                self.logger.timing("context_retriever.retrieve_context_time_ms", duration)
                self.logger.info(f"Retrieved {result['count']} context texts")
                self.logger.counter("context_retriever.contexts_retrieved", result["count"])
                return result
                
        except Exception as e:
            self.logger.error(f"Error retrieving context: {e}")
            self.logger.counter("context_retriever.retrieve_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _generate_summary(self, texts: List[str]) -> str:
        """
        Generate a summary of texts using OpenRouter LLM.

        Args:
            texts: List of texts to summarize

        Returns:
            Summary text
        """
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }
        combined_text = "\n".join(texts)
        payload = {
            "model": self.llm_config.get("model", "openai/gpt-4"),
            "messages": [
                {"role": "user", "content": f"Summarize the following financial texts in 100 words or less:\n{combined_text}"}
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
                        self.logger.error(f"Failed to generate summary: {e}")
                        return "Failed to generate summary"
                    await asyncio.sleep(self.llm_config.get("retry_delay", 1000) / 1000)
        return "Failed to generate summary"

    def shutdown(self):
        """
        Shutdown the ContextRetriever and close resources.
        """
        self.logger.info("Shutting down ContextRetriever")
        self.executor.shutdown(wait=True)
        self.producer.close()
        self.logger.info("Kafka producer closed")