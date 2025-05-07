"""
Vector Database for NextG3N Trading System

This module implements the VectorDB class, managing a vector database using ChromaDB and FAISS
for the RAG system. It provides methods for storing and retrieving text embeddings, supporting
the ContextRetriever model in the NextG3N system.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import chromadb
from chromadb.config import Settings
import faiss

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

# Kafka imports
from kafka import KafkaProducer

class VectorDB:
    """
    Class for managing a vector database in the NextG3N system using ChromaDB and FAISS.
    Provides methods for storing and retrieving text embeddings for RAG.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the VectorDB with configuration and vector database settings.

        Args:
            config: Configuration dictionary with ChromaDB, FAISS, and Kafka settings
        """
        init_start_time = time.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="vector_db")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.vector_db_config = config.get("storage", {}).get("vector_db", {})
        self.kafka_config = config.get("kafka", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize ChromaDB client
        self.chroma_client = None
        self.collection = None
        self.faiss_index = None
        self.embedding_dim = self.vector_db_config.get("embedding_dim", 384)  # Default for all-MiniLM-L6-v2
        self._initialize_vector_db()
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        init_duration = (datetime.time() - init_start_time) * 1000
        self.logger.timing("vector_db.initialization_time_ms", init_duration)
        self.logger.info("VectorDB initialized")

    def _initialize_vector_db(self):
        """
        Initialize the ChromaDB client and FAISS index.
        """
        try:
            # Initialize ChromaDB
            persist_dir = self.vector_db_config.get("persist_dir", "./vector_db_data")
            self.chroma_client = chromadb.Client(Settings(
                persist_directory=persist_dir,
                is_persistent=True
            ))
            
            # Create or get collection
            collection_name = self.vector_db_config.get("collection_name", "financial_texts")
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity for FAISS
            )
            
            # Initialize FAISS index
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            if faiss.get_num_gpus() > 0:
                self.faiss_index = faiss.index_cpu_to_all_gpus(self.faiss_index)
                self.logger.info("FAISS index initialized with GPU support")
            else:
                self.logger.warning("No GPUs detected for FAISS; using CPU")
            
            # Sync FAISS with ChromaDB
            existing_docs = self.collection.get(include=["embeddings"])
            if existing_docs["embeddings"]:
                embeddings = np.array(existing_docs["embeddings"], dtype=np.float32)
                self.faiss_index.add(embeddings)
                self.logger.info(f"Loaded {len(embeddings)} existing embeddings into FAISS")
            
            self.logger.info(f"Connected to ChromaDB collection: {collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize VectorDB: {e}")
            self.chroma_client = None
            self.collection = None
            self.faiss_index = None
            raise

    async def store_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Store text documents with their embeddings in the vector database.

        Args:
            texts: List of text documents
            embeddings: List of embedding vectors
            ids: Optional list of document IDs
            metadatas: Optional list of metadata dictionaries

        Returns:
            Boolean indicating success
        """
        start_time = datetime.time()
        operation_id = f"store_texts_{int(start_time)}"
        self.logger.info(f"Storing {len(texts)} texts in VectorDB - Operation: {operation_id}")

        if not self.collection or not self.faiss_index:
            self.logger.error("VectorDB not initialized")
            return False

        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}_{int(start_time)}" for i in range(len(texts))]
            
            # Validate inputs
            if len(texts) != len(embeddings) or (metadatas and len(metadatas) != len(texts)):
                self.logger.error("Mismatched input lengths")
                return False
            
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    lambda: self.collection.add(
                        documents=texts,
                        embeddings=embeddings,
                        ids=ids,
                        metadatas=metadatas
                    )
                )
                
                # Add to FAISS
                embeddings_np = np.array(embeddings, dtype=np.float32)
                await loop.run_in_executor(
                    self.executor,
                    lambda: self.faiss_index.add(embeddings_np)
                )

            result = {
                "success": True,
                "text_count": len(texts),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}vector_db_events",
                {"event": "texts_stored", "data": result}
            )

            duration = (datetime.time() - start_time) * 1000
            self.logger.timing("vector_db.store_texts_time_ms", duration)
            self.logger.info(f"Stored {len(texts)} texts in VectorDB")
            self.logger.counter("vector_db.texts_stored", len(texts))
            return True

        except Exception as e:
            self.logger.error(f"Error storing texts in VectorDB: {e}")
            self.logger.counter("vector_db.store_errors", len(texts))
            return False

    async def get_texts(
        self,
        ids: List[str]
    ) -> Dict[str, Any]:
        """
        Retrieve texts by their IDs from the vector database.

        Args:
            ids: List of document IDs

        Returns:
            Dictionary containing retrieved texts and metadata
        """
        start_time = datetime.time()
        operation_id = f"get_texts_{int(start_time)}"
        self.logger.info(f"Retrieving {len(ids)} texts from VectorDB - Operation: {operation_id}")

        if not self.collection:
            self.logger.error("VectorDB not initialized")
            return {
                "success": False,
                "error": "VectorDB not initialized",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    lambda: self.collection.get(
                        ids=ids,
                        include=["documents", "metadatas"]
                    )
                )

            retrieved = {
                "success": True,
                "texts": result.get("documents", []),
                "metadatas": result.get("metadatas", []),
                "ids": result.get("ids", []),
                "count": len(result.get("ids", [])),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}vector_db_events",
                {"event": "texts_retrieved", "data": retrieved}
            )

            duration = (datetime.time() - start_time) * 1000
            self.logger.timing("vector_db.get_texts_time_ms", duration)
            self.logger.info(f"Retrieved {retrieved['count']} texts from VectorDB")
            self.logger.counter("vector_db.texts_retrieved", retrieved['count'])
            return retrieved

        except Exception as e:
            self.logger.error(f"Error retrieving texts from VectorDB: {e}")
            self.logger.counter("vector_db.get_errors", len(ids))
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def query_texts(
        self,
        query_embedding: List[float],
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Query texts by similarity to a given embedding.

        Args:
            query_embedding: Query embedding vector
            k: Number of nearest neighbors to retrieve

        Returns:
            Dictionary containing similar texts, metadata, and distances
        """
        start_time = datetime.time()
        operation_id = f"query_texts_{int(start_time)}"
        self.logger.info(f"Querying {k} similar texts from VectorDB - Operation: {operation_id}")

        if not self.collection or not self.faiss_index:
            self.logger.error("VectorDB not initialized")
            return {
                "success": False,
                "error": "VectorDB not initialized",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            # Convert query embedding to numpy array
            query_np = np.array([query_embedding], dtype=np.float32)
            
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                # Perform FAISS search
                distances, indices = await loop.run_in_executor(
                    self.executor,
                    lambda: self.faiss_index.search(query_np, k)
                )
                
                # Retrieve texts from ChromaDB
                ids = [f"doc_{i}" for i in indices[0] if i >= 0]
                if not ids:
                    return {
                        "success": True,
                        "texts": [],
                        "metadatas": [],
                        "ids": [],
                        "distances": [],
                        "count": 0,
                        "operation_id": operation_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                result = await loop.run_in_executor(
                    self.executor,
                    lambda: self.collection.get(
                        ids=ids,
                        include=["documents", "metadatas"]
                    )
                )

            retrieved = {
                "success": True,
                "texts": result.get("documents", []),
                "metadatas": result.get("metadatas", []),
                "ids": result.get("ids", []),
                "distances": distances[0].tolist(),
                "count": len(result.get("ids", [])),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}vector_db_events",
                {"event": "texts_queried", "data": retrieved}
            )

            duration = (datetime.time() - start_time) * 1000
            self.logger.timing("vector_db.query_texts_time_ms", duration)
            self.logger.info(f"Queried {retrieved['count']} similar texts from VectorDB")
            self.logger.counter("vector_db.queries", retrieved['count'])
            return retrieved

        except Exception as e:
            self.logger.error(f"Error querying texts from VectorDB: {e}")
            self.logger.counter("vector_db.query_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def shutdown(self):
        """
        Shutdown the VectorDB client and close resources.
        """
        self.logger.info("Shutting down VectorDB")
        self.executor.shutdown(wait=True)
        self.producer.close()
        if self.chroma_client:
            self.chroma_client.persist()
        self.logger.info("Kafka producer closed and ChromaDB persisted")