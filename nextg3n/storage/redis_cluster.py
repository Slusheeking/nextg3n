"""
Redis Cluster for NextG3N Trading System

This module implements the RedisCluster class, managing a Redis cluster for distributed
caching. It provides methods for storing, retrieving, and deleting data, supporting
the CacheService and other components in the NextG3N system.
"""

import os
import json
import logging
import asyncio
import time
from typing import Any, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
from redis.cluster import RedisCluster
from redis.exceptions import ConnectionError, TimeoutError

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

# Kafka imports
from kafka import KafkaProducer

class RedisCluster:
    """
    Class for managing a Redis cluster in the NextG3N system.
    Provides methods for distributed caching operations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RedisCluster with configuration and Redis settings.

        Args:
            config: Configuration dictionary with Redis and Kafka settings
        """
        init_start_time = time.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="redis_cluster")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.redis_config = config.get("storage", {}).get("redis", {})
        self.kafka_config = config.get("kafka", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize Redis cluster
        self.redis = None
        self.max_retries = self.redis_config.get("max_retries", 3)
        self.retry_delay = self.redis_config.get("retry_delay", 1.0)  # Seconds
        self._initialize_redis()
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("redis_cluster.initialization_time_ms", init_duration)
        self.logger.info("RedisCluster initialized")

    def _initialize_redis(self):
        """
        Initialize the Redis cluster client with retry logic.
        """
        for attempt in range(self.max_retries):
            try:
                redis_host = self.redis_config.get("host", "localhost")
                redis_port = self.redis_config.get("port", 6379)
                redis_db = self.redis_config.get("db", 0)
                redis_password = self.redis_config.get("password") or os.environ.get("REDIS_PASSWORD")
                
                self.redis = RedisCluster(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password,
                    decode_responses=True,
                    max_connections=100,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0
                )
                
                # Test connection
                self.redis.ping()
                self.logger.info(f"Connected to Redis cluster at {redis_host}:{redis_port}")
                return
            
            except (ConnectionError, TimeoutError) as e:
                self.logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
            
            except Exception as e:
                self.logger.error(f"Failed to initialize Redis client: {e}")
                raise
        
        self.logger.error("Failed to connect to Redis after all retries")
        self.redis = None
        raise ConnectionError("Unable to connect to Redis cluster")

    async def set(
        self,
        key: str,
        value: Any,
        expiry: Optional[int] = None,
        retry: int = 0
    ) -> bool:
        """
        Store data in Redis with an optional expiry time.

        Args:
            key: Cache key
            value: Data to cache (JSON-serializable)
            expiry: Time-to-live in seconds (optional)
            retry: Current retry attempt (internal)

        Returns:
            Boolean indicating success
        """
        start_time = time.time()
        operation_id = f"set_{int(start_time)}"
        self.logger.info(f"Setting Redis key: {key} - Operation: {operation_id}")

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
                {"event": "redis_set", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("redis_cluster.set_time_ms", duration)
            if success:
                self.logger.info(f"Redis set for key: {key}")
                self.logger.counter("redis_cluster.sets", 1)
            else:
                self.logger.error(f"Failed to set Redis key: {key}")
                self.logger.counter("redis_cluster.set_errors", 1)
            
            return success

        except (ConnectionError, TimeoutError) as e:
            if retry < self.max_retries:
                self.logger.warning(f"Redis set retry {retry + 1} for key {key}: {e}")
                await asyncio.sleep(self.retry_delay)
                return await self.set(key, value, expiry, retry + 1)
            self.logger.error(f"Error setting Redis key {key} after retries: {e}")
            self.logger.counter("redis_cluster.set_errors", 1)
            return False
        
        except Exception as e:
            self.logger.error(f"Error setting Redis key {key}: {e}")
            self.logger.counter("redis_cluster.set_errors", 1)
            return False

    async def get(self, key: str, retry: int = 0) -> Any:
        """
        Retrieve data from Redis.

        Args:
            key: Cache key
            retry: Current retry attempt (internal)

        Returns:
            Cached data or None if not found
        """
        start_time = time.time()
        operation_id = f"get_{int(start_time)}"
        self.logger.info(f"Getting Redis key: {key} - Operation: {operation_id}")

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
                {"event": "redis_get", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("redis_cluster.get_time_ms", duration)
            
            if value is not None:
                deserialized_value = json.loads(value)
                self.logger.info(f"Redis retrieved for key: {key}")
                self.logger.counter("redis_cluster.hits", 1)
                return deserialized_value
            else:
                self.logger.info(f"Redis miss for key: {key}")
                self.logger.counter("redis_cluster.misses", 1)
                return None

        except (ConnectionError, TimeoutError) as e:
            if retry < self.max_retries:
                self.logger.warning(f"Redis get retry {retry + 1} for key {key}: {e}")
                await asyncio.sleep(self.retry_delay)
                return await self.get(key, retry + 1)
            self.logger.error(f"Error getting Redis key {key} after retries: {e}")
            self.logger.counter("redis_cluster.get_errors", 1)
            return None
        
        except Exception as e:
            self.logger.error(f"Error getting Redis key {key}: {e}")
            self.logger.counter("redis_cluster.get_errors", 1)
            return None

    async def delete(self, key: str, retry: int = 0) -> int:
        """
        Delete a key from Redis.

        Args:
            key: Cache key
            retry: Current retry attempt (internal)

        Returns:
            Number of keys deleted (0 or 1)
        """
        start_time = time.time()
        operation_id = f"delete_{int(start_time)}"
        self.logger.info(f"Deleting Redis key: {key} - Operation: {operation_id}")

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
                {"event": "redis_delete", "data": result}
            )

            duration = (time.time() - start_time) * 1000
            self.logger.timing("redis_cluster.delete_time_ms", duration)
            self.logger.info(f"Deleted {deleted_count} keys from Redis for key: {key}")
            self.logger.counter("redis_cluster.deletes", deleted_count)
            return deleted_count

        except (ConnectionError, TimeoutError) as e:
            if retry < self.max_retries:
                self.logger.warning(f"Redis delete retry {retry + 1} for key {key}: {e}")
                await asyncio.sleep(self.retry_delay)
                return await self.delete(key, retry + 1)
            self.logger.error(f"Error deleting Redis key {key} after retries: {e}")
            self.logger.counter("redis_cluster.delete_errors", 1)
            return 0
        
        except Exception as e:
            self.logger.error(f"Error deleting Redis key {key}: {e}")
            self.logger.counter("redis_cluster.delete_errors", 1)
            return 0

    async def batch_set(
        self,
        items: List[Dict[str, Any]],
        expiry: Optional[int] = None
    ) -> int:
        """
        Store multiple keys in Redis in a batch operation.

        Args:
            items: List of dictionaries with 'key' and 'value' fields
            expiry: Time-to-live in seconds (optional)

        Returns:
            Number of keys successfully set
        """
        start_time = time.time()
        operation_id = f"batch_set_{int(start_time)}"
        self.logger.info(f"Batch setting {len(items)} Redis keys - Operation: {operation_id}")

        if not self.redis:
            self.logger.error("Redis client not initialized")
            return 0

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                pipeline = self.redis.pipeline()
                for item in items:
                    key = item.get("key")
                    value = json.dumps(item.get("value"))
                    pipeline.set(key, value, ex=expiry)
                results = await loop.run_in_executor(self.executor, pipeline.execute)
                
                success_count = sum(1 for result in results if result)
                
                result = {
                    "success": success_count > 0,
                    "keys_set": success_count,
                    "total_keys": len(items),
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}cache_events",
                    {"event": "redis_batch_set", "data": result}
                )

                duration = (time.time() - start_time) * 1000
                self.logger.timing("redis_cluster.batch_set_time_ms", duration)
                self.logger.info(f"Batch set {success_count}/{len(items)} Redis keys")
                self.logger.counter("redis_cluster.batch_sets", success_count)
                if success_count < len(items):
                    self.logger.warning(f"Failed to set {len(items) - success_count} keys in batch")
                    self.logger.counter("redis_cluster.batch_set_errors", len(items) - success_count)
                
                return success_count

        except Exception as e:
            self.logger.error(f"Error batch setting Redis keys: {e}")
            self.logger.counter("redis_cluster.batch_set_errors", len(items))
            return 0

    async def batch_get(self, keys: List[str]) -> List[Any]:
        """
        Retrieve multiple keys from Redis in a batch operation.

        Args:
            keys: List of cache keys

        Returns:
            List of cached data (None for misses)
        """
        start_time = time.time()
        operation_id = f"batch_get_{int(start_time)}"
        self.logger.info(f"Batch getting {len(keys)} Redis keys - Operation: {operation_id}")

        if not self.redis:
            self.logger.error("Redis client not initialized")
            return [None] * len(keys)

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                values = await loop.run_in_executor(self.executor, lambda: self.redis.mget(keys))
                
                results = [json.loads(v) if v else None for v in values]
                hit_count = sum(1 for v in values if v is not None)
                
                result = {
                    "success": hit_count > 0,
                    "keys_hit": hit_count,
                    "total_keys": len(keys),
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Publish to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}cache_events",
                    {"event": "redis_batch_get", "data": result}
                )

                duration = (time.time() - start_time) * 1000
                self.logger.timing("redis_cluster.batch_get_time_ms", duration)
                self.logger.info(f"Batch retrieved {hit_count}/{len(keys)} Redis keys")
                self.logger.counter("redis_cluster.batch_hits", hit_count)
                self.logger.counter("redis_cluster.batch_misses", len(keys) - hit_count)
                
                return results

        except Exception as e:
            self.logger.error(f"Error batch getting Redis keys: {e}")
            self.logger.counter("redis_cluster.batch_get_errors", len(keys))
            return [None] * len(keys)

    def shutdown(self):
        """
        Shutdown the Redis cluster client and close resources.
        """
        self.logger.info("Shutting down RedisCluster")
        self.executor.shutdown(wait=True)
        self.producer.close()
        if self.redis:
            self.redis.close()
        self.logger.info("Kafka producer and Redis client closed")