"""
Cache Service for NextG3N Trading System

This module implements the CacheService, providing Redis-based caching for low-latency data
access. It offers tools for storing, retrieving, and deleting cached data, integrated with
the MonitorAgent, StockPickerAgent, and other agents in TradeFlowOrchestrator.
"""

import os
import json
import logging
import asyncio
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
from redis.cluster import RedisCluster
from kafka import KafkaProducer

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

class CacheService:
    """
    Service for caching data in Redis for the NextG3N system.
    Provides tools for storing, retrieving, and deleting cached data.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CacheService with configuration and Redis settings.

        Args:
            config: Configuration dictionary with Redis and Kafka settings
        """
        init_start_time = datetime.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="cache_service")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.redis_config = config.get("services", {}).get("cache", {}).get("redis", {})
        self.kafka_config = config.get("kafka", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize Redis cluster
        self.redis = None
        self._initialize_redis()
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        init_duration = (datetime.time() - init_start_time) * 1000
        self.logger.timing("cache_service.initialization_time_ms", init_duration)
        self.logger.info("CacheService initialized")

    def _initialize_redis(self):
        """
        Initialize the Redis cluster client.
        """
        try:
            redis_host = self.redis_config.get("host", "localhost")
            redis_port = self.redis_config.get("port", 6379)
            redis_db = self.redis_config.get("db", 0)
            
            self.redis = RedisCluster(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                max_connections=100
            )
            
            # Test connection
            self.redis.ping()
            self.logger.info(f"Connected to Redis cluster at {redis_host}:{redis_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis client: {e}")
            self.redis = None
            raise

    async def set_cache(
        self,
        key: str,
        value: Any,
        expiry: Optional[int] = None
    ) -> bool:
        """
        Store data in Redis cache with an optional expiry time.

        Args:
            key: Cache key
            value: Data to cache (JSON-serializable)
            expiry: Time-to-live in seconds (optional)

        Returns:
            Boolean indicating success
        """
        start_time = datetime.time()
        operation_id = f"set_cache_{int(start_time)}"
        self.logger.info(f"Setting cache for key: {key} - Operation: {operation_id}")

        if not self.redis:
            self.logger.error("Redis client not initialized")
            return False

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                serialized_value = json.dumps(value)
                success = await loop.run_in_executor(
                    self.executor,
                    lambda: self.redis.set(key, serialized_value, ex=expiry)
                )

            result = {
                "success": success,
                "key": key,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}cache_events",
                {"event": "cache_set", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("cache_service.set_cache_time_ms", duration)
            if success:
                self.logger.info(f"Cache set for key: {key}")
                self.logger.counter("cache_service.cache_sets", 1)
            else:
                self.logger.error(f"Failed to set cache for key: {key}")
                self.logger.counter("cache_service.cache_set_errors", 1)
            
            return success

        except Exception as e:
            self.logger.error(f"Error setting cache for key {key}: {e}")
            self.logger.counter("cache_service.cache_set_errors", 1)
            return False

    async def get_cache(self, key: str) -> Any:
        """
        Retrieve data from Redis cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found
        """
        start_time = time.time()
        operation_id = f"get_cache_{int(start_time)}"
        self.logger.info(f"Getting cache for key: {key} - Operation: {operation_id}")

        if not self.redis:
            self.logger.error("Redis client not initialized")
            return None

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                value = await loop.run_in_executor(
                    self.executor,
                    lambda: self.redis.get(key)
                )

            result = {
                "success": value is not None,
                "key": key,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}cache_events",
                {"event": "cache_get", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("cache_service.get_cache_time_ms", duration)
            
            if value is not None:
                deserialized_value = json.loads(value)
                self.logger.info(f"Cache retrieved for key: {key}")
                self.logger.counter("cache_service.cache_hits", 1)
                return deserialized_value
            else:
                self.logger.info(f"Cache miss for key: {key}")
                self.logger.counter("cache_service.cache_misses", 1)
                return None

        except Exception as e:
            self.logger.error(f"Error getting cache for key {key}: {e}")
            self.logger.counter("cache_service.cache_get_errors", 1)
            return None

    async def delete_cache(self, key: str) -> int:
        """
        Delete a key from Redis cache.

        Args:
            key: Cache key

        Returns:
            Number of keys deleted (0 or 1)
        """
        start_time = time.time()
        operation_id = f"delete_cache_{int(start_time)}"
        self.logger.info(f"Deleting cache for key: {key} - Operation: {operation_id}")

        if not self.redis:
            self.logger.error("Redis client not initialized")
            return 0

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                deleted_count = await loop.run_in_executor(
                    self.executor,
                    lambda: self.redis.delete(key)
                )

            result = {
                "success": deleted_count > 0,
                "key": key,
                "deleted_count": deleted_count,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}cache_events",
                {"event": "cache_delete", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("cache_service.delete_cache_time_ms", duration)
            self.logger.info(f"Deleted {deleted_count} keys from cache for key: {key}")
            self.logger.counter("cache_service.cache_deletes", deleted_count)
            return deleted_count

        except Exception as e:
            self.logger.error(f"Error deleting cache for key {key}: {e}")
            self.logger.counter("cache_service.cache_delete_errors", 1)
            return 0

    def shutdown(self):
        """
        Shutdown the service and close resources.
        """
        self.logger.info("Shutting down CacheService")
        self.executor.shutdown(wait=True)
        self.producer.close()
        if self.redis:
            self.redis.close()
        self.logger.info("Kafka producer and Redis client closed")