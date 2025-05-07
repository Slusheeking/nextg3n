"""
Redis Cluster for NextG3N Trading System

This module implements the RedisCluster class, managing a Redis cluster for distributed
caching. It provides methods for storing, retrieving, and deleting data, supporting
the CacheService and other components in the NextG3N system.
"""

import os
import json
import asyncio
import time
from typing import Any, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
from monitor.logging_utils import get_logger
import aiohttp
import redis
from redis.cluster import RedisCluster
from redis.exceptions import ConnectionError, TimeoutError, ResponseError
from redisearch import Client, IndexDefinition, TextField, NumericField, TagField
from redisearch import aggregations as aggregations, reducers as reducers
from redisearch.query import Query

from datetime import datetime, date  # Add date import

class RedisClusterManager:
    """
    Class for managing a Redis cluster in the NextG3N system.
    Provides methods for distributed caching operations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RedisCluster with configuration and Redis settings.

        Args:
            config: Configuration dictionary with Redis settings
        """
        init_start_time = time.time()
        
        # Initialize logging
        self.logger = get_logger("redis_cluster")

        # Load configuration
        self.config = config
        self.redis_config = config.get("storage", {}).get("redis", {})

        # Initialize Redis cluster
        self.redis = None
        self.max_retries = self.redis_config.get("max_retries", 3)
        self.retry_delay = self.redis_config.get("retry_delay", 1.0)  # Seconds
        self.health_check_interval = self.redis_config.get("health_check_interval", 30)  # Seconds
        self.pool_size = self.redis_config.get("pool_size", 10)
        
        # Initialize Redis client
        self._initialize_redis()
        
        # Set up health check task
        self._health_check_task = None
        
        # Initialize thread pool for parallel processing
        max_workers = self.redis_config.get("thread_pool_size", 5)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Track connection state
        self.is_connected = self.redis is not None
        
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.info(f"RedisCluster initialized in {init_duration:.2f}ms")

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
                connection_timeout = self.redis_config.get("connection_timeout", 5.0)
                retry_on_timeout = self.redis_config.get("retry_on_timeout", True)
                
                # Use standalone Redis client if not in cluster mode
                use_cluster = self.redis_config.get("use_cluster", False)
                
                if use_cluster:
                    self.logger.info(f"Initializing Redis cluster client at {redis_host}:{redis_port}")
                    self.redis = RedisCluster(
                        host=redis_host,
                        port=redis_port,
                        db=redis_db,
                        password=redis_password,
                        decode_responses=True,
                        max_connections=self.pool_size,
                        socket_timeout=connection_timeout,
                        socket_connect_timeout=connection_timeout,
                        retry_on_timeout=retry_on_timeout,
                        health_check_interval=self.health_check_interval
                    )
                else:
                    self.logger.info(f"Initializing standalone Redis client at {redis_host}:{redis_port}")
                    self.redis = redis.Redis(
                        host=redis_host,
                        port=redis_port,
                        db=redis_db,
                        password=redis_password,
                        decode_responses=True,
                        socket_timeout=connection_timeout,
                        socket_connect_timeout=connection_timeout,
                        retry_on_timeout=retry_on_timeout,
                        health_check_interval=self.health_check_interval
                    )
                
                # Test connection
                self.redis.ping()
                self.is_connected = True
                self.logger.info(f"Connected to Redis at {redis_host}:{redis_port} (DB: {redis_db})")
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
        self.is_connected = False
        raise ConnectionError("Unable to connect to Redis")
    
    async def start_health_check(self):
        """Start periodic health check for Redis connection."""
        if self._health_check_task is not None:
            self._health_check_task.cancel()
            
        loop = asyncio.get_running_loop()
        self._health_check_task = loop.create_task(self._periodic_health_check())
        self.logger.debug("Redis health check task started")
    
    async def _periodic_health_check(self):
        """Periodically check the Redis connection health."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                if self.redis is None:
                    self.logger.warning("Redis client is not initialized, attempting reconnection")
                    self._initialize_redis()
                    continue
                
                loop = asyncio.get_running_loop()
                ping_result = await loop.run_in_executor(self.executor, lambda: self.redis.ping())
                
                if ping_result:
                    self.logger.debug("Redis health check: connection is healthy")
                    self.is_connected = True
                else:
                    self.logger.warning("Redis health check failed: invalid response")
                    self.is_connected = False
                    self._initialize_redis()
                    
            except asyncio.CancelledError:
                self.logger.debug("Redis health check task cancelled")
                break
                
            except Exception as e:
                self.logger.warning(f"Redis health check error: {e}")
                self.is_connected = False
                try:
                    self._initialize_redis()
                except Exception as re:
                    self.logger.error(f"Failed to reconnect to Redis: {re}")

    def create_index(self, index_name: str, redis_key: str, schema_fields: List, stopwords: List = None) -> bool:
        """Create a Redisearch index with the specified schema.

        Args:
            index_name: The name of the index to create.
            redis_key: The Redis key pattern to index.
            schema_fields: The fields to include in the schema (list of TextField, NumericField, etc.)
            stopwords: Optional list of stopwords to ignore in text fields

        Returns:
            bool: True if index was created or already exists, False otherwise
        """
        try:
            # Define the index schema
            definition = IndexDefinition(prefix=[redis_key])
            
            # Create the index
            client = Client(index_name, redis_conn=self.redis)
            try:
                client.info()
                self.logger.info(f"Index {index_name} already exists.")
                return True
            except ResponseError:
                # Index doesn't exist, create it
                client.create_index(fields=schema_fields, definition=definition, stopwords=stopwords)
                self.logger.info(f"Index {index_name} created successfully with {len(schema_fields)} fields.")
                return True
            except Exception as e:
                self.logger.error(f"Error checking if index {index_name} exists: {e}")
                return False

        except Exception as e:
            self.logger.error(f"Error creating index {index_name}: {e}")
            return False
            
    def create_index_risk_assessed(self, index_name: str, redis_key: str) -> bool:
        """Create a Redisearch index for risk-assessed stock candidates.

        Args:
            index_name: The name of the index to create.
            redis_key: The Redis key pattern to index.
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Define the schema fields
        schema = [
            TextField("ticker", weight=5.0),
            TextField("source"),
            NumericField("volume"),
            NumericField("price"),
            NumericField("atr"),
            NumericField("risk_score", sortable=True),
            NumericField("reward_risk_ratio", sortable=True)
        ]
        
        return self.create_index(index_name, redis_key, schema)

    def create_trade_position_key(self, position_id: str) -> str:
        """Create the Redis key structure for active trade positions.
        
        Args:
            position_id: The ID of the trade position.

        Returns:
            The Redis key string.
        """
        return f"trade_positions:active:{position_id}"

    def create_trade_position_history_key(self, date: str, position_id: str) -> str:
        """Create the Redis key structure for historical trade positions.

        Args:
            date: The date for which to create the key.
            position_id: The ID of the trade position.

        Returns:
            The Redis key string.
        """
        return f"trade_positions:history:{date}:{position_id}"

    def create_index_trade_positions(self, index_name: str, redis_key: str) -> bool:
        """Create a Redisearch index for trade positions.

        Args:
            index_name: The name of the index to create.
            redis_key: The Redis key pattern to index.
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info(f"Creating trade positions index {index_name} with prefix {redis_key}")
        
        # Define the schema fields for trade positions
        schema = [
            TextField("symbol", weight=10.0),
            NumericField("entry_price", sortable=True),
            NumericField("current_price", sortable=True),
            NumericField("quantity", sortable=True),
            NumericField("stop_price"),
            NumericField("profit_target"),
            TextField("entry_time"),
            NumericField("unrealized_pnl", sortable=True),
            NumericField("unrealized_pnl_pct", sortable=True),
            TagField("status")  # Can be 'active', 'closed', etc.
        ]
        
        return self.create_index(index_name, redis_key, schema)

    def create_risk_assessed_key(self, date: str) -> str:
        """Create the Redis key structure for risk-assessed stock candidates.
        
        Args:
            date: The date for which to create the key.

        Returns:
            The Redis key string.
        """
        return f"stock_pool:{date}:risk_assessed"

    async def index_data(self, index_name: str, key: str, value: Dict[str, Any]) -> bool:
        """Index the given data in Redisearch.

        Args:
            index_name: The name of the index.
            key: The Redis key for the data.
            value: The data to index.
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            client = Client(index_name, redis_conn=self.redis)
            
            # Convert value dict to kwargs for add_document
            # Filter out None values
            document_fields = {k: v for k, v in value.items() if v is not None}
            
            # Use the key as the document ID
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: client.add_document(key, **document_fields)
            )
            
            self.logger.debug(f"Indexed document {key} in {index_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error indexing document {key} in {index_name}: {e}")
            return False
            
    async def search(self, index_name: str, query_string: str, offset: int = 0, limit: int = 10) -> Dict:
        """Search for documents in a RediSearch index.
        
        Args:
            index_name: The name of the index to search.
            query_string: The query string to execute.
            offset: Pagination offset.
            limit: Max number of results to return.
            
        Returns:
            Dict containing search results and metadata.
        """
        try:
            client = Client(index_name, redis_conn=self.redis)
            query = Query(query_string).paging(offset, limit)
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: client.search(query)
            )
            
            # Format results for easier consumption
            docs = []
            for doc in result.docs:
                doc_dict = {"id": doc.id}
                for field, value in doc.__dict__.items():
                    if field != "id" and not field.startswith("_"):
                        doc_dict[field] = value
                docs.append(doc_dict)
                
            self.logger.debug(f"Search in {index_name} for '{query_string}' returned {len(docs)} results")
            return {
                "total": result.total,
                "docs": docs,
                "duration_ms": result.duration
            }
        except Exception as e:
            self.logger.error(f"Error searching index {index_name}: {e}")
            return {"total": 0, "docs": [], "error": str(e)}

    async def set(
        self,
        key: str,
        value: Any,
        expiry: Optional[int] = None,
        retry: int = 0,
        index_name: Optional[str] = None
    ) -> bool:
        """Store data in Redis with an optional expiry time.

        Args:
            key: Cache key
            value: Data to cache (JSON-serializable)
            expiry: Time-to-live in seconds (optional)
            retry: Current retry attempt (internal)
            index_name: Name of the index to add this document to (optional)

        Returns:
            Boolean indicating success
        """
        start_time = time.time()
        operation_id = f"set_{int(start_time)}"
        self.logger.info(f"Setting Redis key: {key} - Operation: {operation_id}")

        if not self.redis or not self.is_connected:
            self.logger.error("Redis client not initialized or not connected")
            # Try to reconnect
            try:
                self._initialize_redis()
                if not self.is_connected:
                    return False
            except Exception:
                return False

        try:
            loop = asyncio.get_running_loop()
            serialized_value = json.dumps(value)
            success = await loop.run_in_executor(
                self.executor,
                lambda: self.redis.set(key, serialized_value, ex=expiry)
            )

            # Log the operation
            self.logger.debug(f"Redis operation: {operation_id}, Set key: {key}")

            duration = (time.time() - start_time) * 1000
            if success:
                self.logger.info(f"Redis set for key: {key} (took {duration:.2f}ms)")
                
                # Index the data if requested
                if index_name:
                    await self.index_data(index_name, key, value)
            else:
                self.logger.error(f"Failed to set Redis key: {key} (took {duration:.2f}ms)")
            
            return bool(success)

        except (ConnectionError, TimeoutError) as e:
            if retry < self.max_retries:
                self.logger.warning(f"Redis set retry {retry + 1} for key {key}: {e}")
                await asyncio.sleep(self.retry_delay)
                return await self.set(key, value, expiry, retry + 1, index_name)
            self.logger.error(f"Error setting Redis key {key} after retries: {e}")
            return False
        
        except Exception as e:
            self.logger.error(f"Error setting Redis key {key}: {e}")
            return False
            
    async def get(self, key: str, parse_json: bool = True, retry: int = 0) -> Any:
        """Get data from Redis by key.
        
        Args:
            key: Cache key
            parse_json: Whether to parse the value as JSON
            retry: Current retry attempt (internal)
            
        Returns:
            The data if found, or None if not found
        """
        start_time = time.time()
        self.logger.debug(f"Getting Redis key: {key}")
        
        if not self.redis or not self.is_connected:
            self.logger.error("Redis client not initialized or not connected")
            try:
                self._initialize_redis()
                if not self.is_connected:
                    return None
            except Exception:
                return None
        
        try:
            loop = asyncio.get_running_loop()
            value = await loop.run_in_executor(
                self.executor,
                lambda: self.redis.get(key)
            )
            
            if value is None:
                self.logger.debug(f"Key not found: {key}")
                return None
                
            if parse_json and value:
                try:
                    value = json.loads(value)
                except Exception as e:
                    self.logger.debug(f"Failed to parse JSON for key {key}: {e}")
            
            duration = (time.time() - start_time) * 1000
            self.logger.debug(f"Redis get for key: {key} (took {duration:.2f}ms)")
            return value
            
        except (ConnectionError, TimeoutError) as e:
            if retry < self.max_retries:
                self.logger.warning(f"Redis get retry {retry + 1} for key {key}: {e}")
                await asyncio.sleep(self.retry_delay)
                return await self.get(key, parse_json, retry + 1)
            self.logger.error(f"Error getting Redis key {key} after retries: {e}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting Redis key {key}: {e}")
            return None
            
    async def delete(self, *keys, retry: int = 0) -> int:
        """Delete one or more keys from Redis.
        
        Args:
            *keys: Keys to delete
            retry: Current retry attempt (internal)
            
        Returns:
            Number of keys deleted
        """
        if not keys:
            return 0
            
        start_time = time.time()
        self.logger.debug(f"Deleting Redis keys: {keys}")
        
        if not self.redis or not self.is_connected:
            self.logger.error("Redis client not initialized or not connected")
            try:
                self._initialize_redis()
                if not self.is_connected:
                    return 0
            except Exception:
                return 0
                
        try:
            loop = asyncio.get_running_loop()
            count = await loop.run_in_executor(
                self.executor,
                lambda: self.redis.delete(*keys)
            )
            
            duration = (time.time() - start_time) * 1000
            self.logger.info(f"Deleted {count} Redis keys (took {duration:.2f}ms)")
            return count
            
        except (ConnectionError, TimeoutError) as e:
            if retry < self.max_retries:
                self.logger.warning(f"Redis delete retry {retry + 1} for keys {keys}: {e}")
                await asyncio.sleep(self.retry_delay)
                return await self.delete(*keys, retry=retry + 1)
            self.logger.error(f"Error deleting Redis keys {keys} after retries: {e}")
            return 0
            
        except Exception as e:
            self.logger.error(f"Error deleting Redis keys {keys}: {e}")
            return 0
            
    async def exists(self, *keys, retry: int = 0) -> int:
        """Check if keys exist in Redis.
        
        Args:
            *keys: Keys to check
            retry: Current retry attempt (internal)
            
        Returns:
            Number of keys that exist
        """
        if not keys:
            return 0
            
        if not self.redis or not self.is_connected:
            self.logger.error("Redis client not initialized or not connected")
            try:
                self._initialize_redis()
                if not self.is_connected:
                    return 0
            except Exception:
                return 0
                
        try:
            loop = asyncio.get_running_loop()
            count = await loop.run_in_executor(
                self.executor,
                lambda: self.redis.exists(*keys)
            )
            
            return count
            
        except (ConnectionError, TimeoutError) as e:
            if retry < self.max_retries:
                self.logger.warning(f"Redis exists retry {retry + 1} for keys {keys}: {e}")
                await asyncio.sleep(self.retry_delay)
                return await self.exists(*keys, retry=retry + 1)
            self.logger.error(f"Error checking existence of Redis keys {keys} after retries: {e}")
            return 0
            
        except Exception as e:
            self.logger.error(f"Error checking existence of Redis keys {keys}: {e}")
            return 0
            
    async def expire(self, key: str, seconds: int, retry: int = 0) -> bool:
        """Set an expiration time on a key.
        
        Args:
            key: The key to set expiration on
            seconds: Time to live in seconds
            retry: Current retry attempt (internal)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis or not self.is_connected:
            self.logger.error("Redis client not initialized or not connected")
            try:
                self._initialize_redis()
                if not self.is_connected:
                    return False
            except Exception:
                return False
                
        try:
            loop = asyncio.get_running_loop()
            success = await loop.run_in_executor(
                self.executor,
                lambda: self.redis.expire(key, seconds)
            )
            
            return bool(success)
            
        except (ConnectionError, TimeoutError) as e:
            if retry < self.max_retries:
                self.logger.warning(f"Redis expire retry {retry + 1} for key {key}: {e}")
                await asyncio.sleep(self.retry_delay)
                return await self.expire(key, seconds, retry + 1)
            self.logger.error(f"Error setting expiry on Redis key {key} after retries: {e}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error setting expiry on Redis key {key}: {e}")
            return False

    async def initialize_index(self):
        """Initialize all required Redisearch indices.
        """
        # Initialize risk_assessed index
        risk_assessed_index = "risk_assessed_idx"
        risk_assessed_key = "stock_pool:*:risk_assessed"
        result1 = self.create_index_risk_assessed(risk_assessed_index, risk_assessed_key)
        
        # Initialize trade positions index
        positions_index = "trade_positions_idx"
        positions_key = "trade_positions:*"
        result2 = self.create_index_trade_positions(positions_index, positions_key)
        
        self.logger.info(f"Index initialization complete - Risk assessed: {result1}, Trade positions: {result2}")
        
        # Start health check
        await self.start_health_check()

    async def get_health(self) -> Dict[str, Any]:
        """Get health information about the Redis connection.
        
        Returns:
            Dictionary with health status and metrics
        """
        health_info = {
            "is_connected": self.is_connected,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            if self.redis and self.is_connected:
                loop = asyncio.get_running_loop()
                info = await loop.run_in_executor(
                    self.executor,
                    lambda: self.redis.info()
                )
                
                # Extract useful metrics
                health_info.update({
                    "redis_version": info.get("redis_version", "unknown"),
                    "uptime_seconds": info.get("uptime_in_seconds", 0),
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "total_connections_received": info.get("total_connections_received", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0)
                })
        except Exception as e:
            self.logger.error(f"Error getting Redis health info: {e}")
            health_info["error"] = str(e)
            
        return health_info
    
    def shutdown(self):
        """Shutdown the Redis cluster client and close resources.
        """
        self.logger.info("Shutting down RedisCluster")
        
        # Cancel health check task if running
        if self._health_check_task is not None:
            self._health_check_task.cancel()
            self._health_check_task = None
        
        # Shutdown thread executor
        self.executor.shutdown(wait=True)
        
        # Close Redis connection
        if self.redis:
            try:
                self.redis.close()
                self.is_connected = False
                self.logger.info("Redis connection closed")
            except Exception as e:
                self.logger.error(f"Error closing Redis connection: {e}")
