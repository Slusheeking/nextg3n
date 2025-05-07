"""
LLM Integration for NextG3N Trading System.
Handles the integration of data sources with large language models for trading decisions.
"""


import aiohttp
import asyncio
import json
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import hashlib
from dotenv import load_dotenv
import redis.asyncio as aioredis
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security, Request
from fastapi.security import APIKeyHeader
import tiktoken
import yaml
from pathlib import Path
from copy import deepcopy

# Fallback logger
try:
    from monitor.logging_utils import get_logger
except ImportError:
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

# Load environment variables
load_dotenv()

# --- Configuration ---
def validate_config():
    config_path = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'llm_config.yaml'))
    default_config = {
        "llm": {
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "model": os.getenv("LLM_MODEL", "gpt-4o"),
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
            "rate_limit_per_minute": 20,
            "max_tokens": 8192,
            "temperature": 0.7,
            "context_window": 32000
        },
        "redis": {
            "host": os.getenv("LLM_REDIS_HOST", os.getenv("REDIS_HOST", "localhost")),
            "port": int(os.getenv("LLM_REDIS_PORT", os.getenv("REDIS_PORT", "6379"))),
            "db": int(os.getenv("LLM_REDIS_DB", os.getenv("REDIS_DB", "2"))),
            "password": os.getenv("LLM_REDIS_PASSWORD", os.getenv("REDIS_PASSWORD", None)),
            "max_retries": 5,
            "reconnect_cooldown": 30
        },
        "caching": {
            "enabled": True,
            "ttl": 3600
        },
        "retries": {
            "max_attempts": 3,
            "backoff_factor": 1.5
        },
        "security": {
            "enable_auth": True
        },
        "mcp_endpoints": {
            "alpaca": "http://localhost:8000",
            "polygon_rest": "http://localhost:8001",
            "polygon_websocket": "http://localhost:8002",
            "reddit_processor": "http://localhost:8003",
            "redis": "http://localhost:8004",
            "unusual_whales": "http://localhost:8005",
            "yahoo_finance": "http://localhost:8006"
        },
        "shutdown": {
            "timeout_seconds": 30
        }
    }
    try:
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f) or {}
            def deep_merge(default: dict, update: dict) -> dict:
                merged = deepcopy(default)
                for key, value in update.items():
                    if isinstance(value, dict) and key in merged:
                        merged[key] = deep_merge(merged[key], value)
                    else:
                        merged[key] = value
                return merged
            config = deep_merge(default_config, file_config)
        else:
            logger.warning(f"Config file not found at {config_path}. Using default configuration.")
            config = default_config

        # Override with environment variables
        config["llm"]["provider"] = os.getenv("LLM_PROVIDER", config["llm"]["provider"])
        config["llm"]["model"] = os.getenv("LLM_MODEL", config["llm"]["model"])
        config["llm"]["api_key"] = os.getenv("OPENAI_API_KEY", config["llm"]["api_key"])
        config["llm"]["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY", config["llm"]["anthropic_api_key"])
        config["llm"]["rate_limit_per_minute"] = int(os.getenv("LLM_RATE_LIMIT", config["llm"]["rate_limit_per_minute"]))
        config["llm"]["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS", config["llm"]["max_tokens"]))
        config["llm"]["temperature"] = float(os.getenv("LLM_TEMPERATURE", config["llm"]["temperature"]))
        config["llm"]["context_window"] = int(os.getenv("LLM_CONTEXT_WINDOW", config["llm"]["context_window"]))
        config["redis"]["host"] = os.getenv("LLM_REDIS_HOST", os.getenv("REDIS_HOST", config["redis"]["host"]))
        config["redis"]["port"] = int(os.getenv("LLM_REDIS_PORT", os.getenv("REDIS_PORT", config["redis"]["port"])))
        config["redis"]["db"] = int(os.getenv("LLM_REDIS_DB", os.getenv("REDIS_DB", config["redis"]["db"])))
        config["redis"]["password"] = os.getenv("LLM_REDIS_PASSWORD", os.getenv("REDIS_PASSWORD", config["redis"]["password"]))
        config["redis"]["max_retries"] = int(os.getenv("REDIS_MAX_RETRIES", config["redis"]["max_retries"]))
        config["redis"]["reconnect_cooldown"] = int(os.getenv("REDIS_RECONNECT_COOLDOWN", config["redis"]["reconnect_cooldown"]))
        config["caching"]["enabled"] = os.getenv("LLM_CACHING_ENABLED", str(config["caching"]["enabled"])).lower() == "true"
        config["caching"]["ttl"] = int(os.getenv("LLM_CACHE_TTL", config["caching"]["ttl"]))
        config["retries"]["max_attempts"] = int(os.getenv("LLM_MAX_RETRY_ATTEMPTS", config["retries"]["max_attempts"]))
        config["retries"]["backoff_factor"] = float(os.getenv("LLM_RETRY_BACKOFF_FACTOR", config["retries"]["backoff_factor"]))
        config["security"]["enable_auth"] = os.getenv("LLM_ENABLE_AUTH", str(config["security"]["enable_auth"])).lower() == "true"
        config["shutdown"]["timeout_seconds"] = int(os.getenv("SHUTDOWN_TIMEOUT_SECONDS", config["shutdown"]["timeout_seconds"]))

        # Validate configuration
        if config["llm"]["rate_limit_per_minute"] <= 0:
            logger.warning(f"Invalid rate_limit_per_minute: {config['llm']['rate_limit_per_minute']}. Using default: 20")
            config["llm"]["rate_limit_per_minute"] = 20
        if config["llm"]["max_tokens"] <= 0 or config["llm"]["max_tokens"] > 32000:
            logger.warning(f"Invalid max_tokens: {config['llm']['max_tokens']}. Using default: 8192")
            config["llm"]["max_tokens"] = 8192
        if config["llm"]["temperature"] < 0.0 or config["llm"]["temperature"] > 1.0:
            logger.warning(f"Invalid temperature: {config['llm']['temperature']}. Using default: 0.7")
            config["llm"]["temperature"] = 0.7
        if config["redis"]["port"] <= 0:
            logger.warning(f"Invalid redis_port: {config['redis']['port']}. Using default: 6379")
            config["redis"]["port"] = 6379
        if config["redis"]["db"] < 0:
            logger.warning(f"Invalid redis_db: {config['redis']['db']}. Using default: 2")
            config["redis"]["db"] = 2
        if config["redis"]["max_retries"] < 0:
            logger.warning(f"Invalid max_retries: {config['redis']['max_retries']}. Using default: 5")
            config["redis"]["max_retries"] = 5
        if config["redis"]["reconnect_cooldown"] <= 0:
            logger.warning(f"Invalid reconnect_cooldown: {config['redis']['reconnect_cooldown']}. Using default: 30")
            config["redis"]["reconnect_cooldown"] = 30
        if config["caching"]["ttl"] <= 0:
            logger.warning(f"Invalid cache_ttl: {config['caching']['ttl']}. Using default: 3600")
            config["caching"]["ttl"] = 3600
        if config["retries"]["max_attempts"] <= 0:
            logger.warning(f"Invalid max_attempts: {config['retries']['max_attempts']}. Using default: 3")
            config["retries"]["max_attempts"] = 3
        if config["retries"]["backoff_factor"] <= 0:
            logger.warning(f"Invalid backoff_factor: {config['retries']['backoff_factor']}. Using default: 1.5")
            config["retries"]["backoff_factor"] = 1.5
        if config["shutdown"]["timeout_seconds"] <= 0:
            logger.warning(f"Invalid timeout_seconds: {config['shutdown']['timeout_seconds']}. Using default: 30")
            config["shutdown"]["timeout_seconds"] = 30
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}. Using default configuration.")
        return default_config

CONFIG = validate_config()

logger = get_logger("llm_integration")
logger.info("Initializing LLM Integration Module")

# --- Redis Client ---
redis_client = None
http_session = None
active_requests = 0
shutdown_event = asyncio.Event()

async def connect_redis(force_reconnect=False):
    global redis_client
    current_time = time.time()
    if not force_reconnect and redis_client is not None:
        return True
    if (current_time - redis_last_error_time < CONFIG["redis"]["reconnect_cooldown"] and
        redis_last_error_time > 0 and redis_connection_attempts > CONFIG["redis"]["max_retries"]):
        logger.debug("Skipping Redis reconnect - in cooldown period")
        return False
    try:
        if redis_client:
            try:
                await redis_client.close()
            except Exception:
                pass
        redis_client = aioredis.Redis(
            host=CONFIG["redis"]["host"],
            port=CONFIG["redis"]["port"],
            db=CONFIG["redis"]["db"],
            password=CONFIG["redis"]["password"],
            decode_responses=True,
            socket_timeout=5.0,
            socket_keepalive=True,
            health_check_interval=30,
            retry_on_timeout=True,
            max_connections=20,
            retry=aioredis.retry.Retry(retries=2, backoff=1.5)
        )
        await asyncio.wait_for(redis_client.ping(), timeout=2.0)
        redis_connection_attempts = 0
        redis_last_error_time = 0
        logger.info(f"Connected to Redis at {CONFIG['redis']['host']}:{CONFIG['redis']['port']} (DB: {CONFIG['redis']['db']})")
        return True
    except (aioredis.RedisError, asyncio.TimeoutError, ConnectionError) as e:
        redis_connection_attempts += 1
        redis_last_error_time = current_time
        if redis_connection_attempts <= 3:
            logger.exception(f"Redis connection attempt {redis_connection_attempts}/{CONFIG['redis']['max_retries']} failed: {str(e)}")
        else:
            logger.warning(f"Redis connection attempt {redis_connection_attempts}/{CONFIG['redis']['max_retries']} failed: {str(e)}")
        redis_client = None
        return False
    except Exception as e:
        logger.exception("Unexpected error connecting to Redis")
        redis_client = None
        redis_last_error_time = current_time
        return False

async def get_http_session():
    global http_session
    if http_session is None or http_session.closed:
        http_session = aiohttp.ClientSession()
    return http_session

# --- Rate Limiter ---
class RateLimiter:
    def __init__(self, calls_limit: int, time_window: int = 60):
        self.calls_limit = calls_limit
        self.time_window = time_window
        self.redis_key_prefix = "llm_rate_limit"
    
    async def check_rate_limit(self, request: Request):
        if not redis_client:
            logger.warning("Redis unavailable, skipping rate limiting")
            return True
        ip = request.headers.get("X-Forwarded-For", request.client.host)
        if ip and "," in ip:
            ip = ip.split(",")[0].strip()
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        key = f"{self.redis_key_prefix}:{ip}"
        if api_key:
            hashed_key = hashlib.md5(api_key.encode()).hexdigest()
            key = f"{self.redis_key_prefix}:{hashed_key}"
        now = int(datetime.now().timestamp())
        try:
            async with redis_client.pipeline() as pipe:
                pipe.zremrangebyscore(key, 0, now - self.time_window)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, self.time_window)
                _, _, count, _ = await pipe.execute()
            if count > self.calls_limit:
                logger.warning(f"Rate limit exceeded for {key}: {count} requests in last {self.time_window}s")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking rate limit in Redis: {str(e)}")
            return True

rate_limiter = RateLimiter(calls_limit=CONFIG["llm"]["rate_limit_per_minute"])

# --- Authentication ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not CONFIG["security"]["enable_auth"]:
        return
    valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")
    if not api_key or api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# --- MCP Tool Integration ---
async def fetch_mcp_data(endpoint: str, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    session = await get_http_session()
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    try:
        async with session.post(endpoint, json=payload, headers=headers) as response:
            if response.status == 401:
                raise HTTPException(status_code=401, detail="Invalid API key for MCP endpoint")
            if response.status != 200:
                error_detail = await response.text()
                raise HTTPException(status_code=response.status, detail=error_detail)
            return await response.json()
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching data from MCP endpoint {endpoint}: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

async def fetch_alpaca_data(symbol: str, api_key: str) -> Dict[str, Any]:
    endpoint = f"{CONFIG['mcp_endpoints']['alpaca']}/get_quotes"
    payload = {"symbol": symbol}
    return await fetch_mcp_data(endpoint, payload, api_key)

async def fetch_polygon_data(symbol: str, api_key: str) -> Dict[str, Any]:
    endpoint = f"{CONFIG['mcp_endpoints']['polygon_rest']}/fetch_quotes"
    payload = {"symbols": [symbol]}
    return await fetch_mcp_data(endpoint, payload, api_key)

async def fetch_reddit_data(symbol: str, api_key: str) -> Dict[str, Any]:
    endpoint = f"{CONFIG['mcp_endpoints']['reddit_processor']}/sentiment_analysis"
    payload = {"symbol": symbol}
    return await fetch_mcp_data(endpoint, payload, api_key)

async def fetch_unusual_whales_data(symbol: str, api_key: str) -> Dict[str, Any]:
    endpoint = f"{CONFIG['mcp_endpoints']['unusual_whales']}/fetch_options_flow"
    payload = {"symbols": [symbol], "min_premium": 10000, "min_volume": 100}
    return await fetch_mcp_data(endpoint, payload, api_key)

async def fetch_yahoo_finance_data(symbol: str, api_key: str) -> Dict[str, Any]:
    endpoint = f"{CONFIG['mcp_endpoints']['yahoo_finance']}/fetch_news"
    payload = {"symbols": [symbol], "count": 10}
    return await fetch_mcp_data(endpoint, payload, api_key)

async def fetch_redis_data(key_pattern: str) -> List[Dict[str, Any]]:
    if not redis_client:
        return []
    try:
        cursor = 0
        results = []
        while True:
            cursor, keys = await redis_client.scan(cursor, match=key_pattern, count=100)
            for key in keys:
                data = await redis_client.get(key)
                if data:
                    results.append(json.loads(data))
            if cursor == 0:
                break
        return results
    except Exception as e:
        logger.error(f"Error fetching Redis data for {key_pattern}: {str(e)}")
        return []

# --- LLM Utilities ---
class TokenUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }

class LLMResponse:
    def __init__(self, text: str, usage: TokenUsage, model_used: str):
        self.text = text
        self.usage = usage
        self.model_used = model_used
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "usage": self.usage.to_dict(),
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat()
        }

def get_tokenizer(model_name: str):
    try:
        if "gpt" in model_name.lower():
            return tiktoken.encoding_for_model(model_name)
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.warning(f"Error getting tokenizer for {model_name}: {str(e)}. Using cl100k_base encoding.")
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model_name: str) -> int:
    try:
        tokenizer = get_tokenizer(model_name)
        return len(tokenizer.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {str(e)}. Using approximation.")
        return len(text) // 4

def truncate_to_token_limit(text: str, max_tokens: int, model_name: str) -> str:
    try:
        tokenizer = get_tokenizer(model_name)
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens)
    except Exception as e:
        logger.warning(f"Error truncating text: {str(e)}. Using character-based approximation.")
        approx_chars = max_tokens * 4
        return text[:approx_chars]

async def call_openai_api(prompt: str, system_message: str = None, model: str = None) -> LLMResponse:
    session = await get_http_session()
    api_model = model or CONFIG["llm"]["model"]
    api_key = CONFIG["llm"]["api_key"]
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    system_tokens = count_tokens(system_message or "", api_model) if system_message else 0
    prompt_tokens = count_tokens(prompt, api_model)
    total_prompt_tokens = system_tokens + prompt_tokens
    max_context_tokens = CONFIG["llm"]["context_window"]
    max_completion_tokens = CONFIG["llm"]["max_tokens"]
    if total_prompt_tokens >= max_context_tokens - max_completion_tokens:
        logger.warning(f"Prompt too long ({total_prompt_tokens} tokens). Truncating...")
        available_tokens = max_context_tokens - max_completion_tokens - (system_tokens + 100)
        prompt = truncate_to_token_limit(prompt, available_tokens, api_model)
        prompt_tokens = count_tokens(prompt, api_model)
        total_prompt_tokens = system_tokens + prompt_tokens
    logger.debug(f"Request to OpenAI API with model {api_model}. Prompt tokens: {total_prompt_tokens}")
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    url = "https://api.openai.com/v1/chat/completions"
    request_body = {
        "model": api_model,
        "messages": messages,
        "temperature": CONFIG["llm"]["temperature"],
        "max_tokens": max_completion_tokens
    }
    cache_key = None
    if CONFIG["caching"]["enabled"] and redis_client:
        cache_dict = {
            "model": api_model,
            "messages": messages,
            "temperature": CONFIG["llm"]["temperature"],
            "timestamp": datetime.now().isoformat()
        }
        cache_str = json.dumps(cache_dict, sort_keys=True)
        cache_key = f"llm:cache:{hashlib.sha256(cache_str.encode()).hexdigest()}"
        try:
            cached_response = await redis_client.get(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for LLM request")
                response_data = json.loads(cached_response)
                return LLMResponse(
                    text=response_data["text"],
                    usage=TokenUsage(**response_data["usage"]),
                    model_used=response_data["model_used"]
                )
        except Exception as e:
            logger.warning(f"Error reading from cache: {str(e)}")
    try:
        async with session.post(url, headers=headers, json=request_body) as response:
            if response.status == 401:
                raise HTTPException(status_code=401, detail="Invalid OpenAI API key")
            if response.status == 429:
                retry_after = int(response.headers.get("Retry-After", "60"))
                logger.warning(f"OpenAI API rate limit hit. Retrying after {retry_after}s")
                await asyncio.sleep(retry_after)
                return await call_openai_api(prompt, system_message, model)
            if response.status != 200:
                error_detail = await response.text()
                logger.error(f"OpenAI API error: {response.status} - {error_detail}")
                raise HTTPException(status_code=response.status, detail=error_detail)
            response_json = await response.json()
            completion_text = response_json["choices"][0]["message"]["content"]
            usage = TokenUsage(
                prompt_tokens=response_json["usage"]["prompt_tokens"],
                completion_tokens=response_json["usage"]["completion_tokens"],
                total_tokens=response_json["usage"]["total_tokens"]
            )
            result = LLMResponse(
                text=completion_text,
                usage=usage,
                model_used=api_model
            )
            if CONFIG["caching"]["enabled"] and redis_client and cache_key:
                try:
                    await redis_client.setex(cache_key, CONFIG["caching"]["ttl"], json.dumps(result.to_dict()))
                except Exception as e:
                    logger.warning(f"Error writing to cache: {str(e)}")
            return result
    except aiohttp.ClientError as e:
        logger.error(f"AIOHTTP ClientError for OpenAI API: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI API: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def call_anthropic_api(prompt: str, system_message: str = None, model: str = None) -> LLMResponse:
    session = await get_http_session()
    api_model = model or CONFIG["llm"]["model"]
    if "claude" not in api_model.lower():
        api_model = "claude-3-opus-20240229"
    api_key = CONFIG["llm"]["anthropic_api_key"]
    if not api_key:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured")
    system_tokens = count_tokens(system_message or "", api_model) if system_message else 0
    prompt_tokens = count_tokens(prompt, api_model)
    total_prompt_tokens = system_tokens + prompt_tokens
    max_context_tokens = CONFIG["llm"]["context_window"]
    max_completion_tokens = CONFIG["llm"]["max_tokens"]
    if total_prompt_tokens >= max_context_tokens - max_completion_tokens:
        logger.warning(f"Prompt too long ({total_prompt_tokens} tokens). Truncating...")
        available_tokens = max_context_tokens - max_completion_tokens - (system_tokens + 100)
        prompt = truncate_to_token_limit(prompt, available_tokens, api_model)
        prompt_tokens = count_tokens(prompt, api_model)
        total_prompt_tokens = system_tokens + prompt_tokens
    logger.debug(f"Request to Anthropic API with model {api_model}. Prompt tokens: {total_prompt_tokens}")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    url = "https://api.anthropic.com/v1/messages"
    request_body = {
        "model": api_model,
        "max_tokens": max_completion_tokens,
        "temperature": CONFIG["llm"]["temperature"],
        "messages": [{"role": "user", "content": prompt}]
    }
    if system_message:
        request_body["system"] = system_message
    cache_key = None
    if CONFIG["caching"]["enabled"] and redis_client:
        cache_dict = {
            "model": api_model,
            "prompt": prompt,
            "system": system_message,
            "temperature": CONFIG["llm"]["temperature"],
            "timestamp": datetime.now().isoformat()
        }
        cache_str = json.dumps(cache_dict, sort_keys=True)
        cache_key = f"llm:cache:{hashlib.sha256(cache_str.encode()).hexdigest()}"
        try:
            cached_response = await redis_client.get(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for LLM request")
                response_data = json.loads(cached_response)
                return LLMResponse(
                    text=response_data["text"],
                    usage=TokenUsage(**response_data["usage"]),
                    model_used=response_data["model_used"]
                )
        except Exception as e:
            logger.warning(f"Error reading from cache: {str(e)}")
    try:
        async with session.post(url, headers=headers, json=request_body) as response:
            if response.status == 401:
                raise HTTPException(status_code=401, detail="Invalid Anthropic API key")
            if response.status == 429:
                retry_after = int(response.headers.get("Retry-After", "60"))
                logger.warning(f"Anthropic API rate limit hit. Retrying after {retry_after}s")
                await asyncio.sleep(retry_after)
                return await call_anthropic_api(prompt, system_message, model)
            if response.status != 200:
                error_detail = await response.text()
                logger.error(f"Anthropic API error: {response.status} - {error_detail}")
                raise HTTPException(status_code=response.status, detail=error_detail)
            response_json = await response.json()
            completion_text = response_json["content"][0]["text"]
            usage = TokenUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=count_tokens(completion_text, api_model),
                total_tokens=total_prompt_tokens + count_tokens(completion_text, api_model)
            )
            result = LLMResponse(
                text=completion_text,
                usage=usage,
                model_used=api_model
            )
            if CONFIG["caching"]["enabled"] and redis_client and cache_key:
                try:
                    await redis_client.setex(cache_key, CONFIG["caching"]["ttl"], json.dumps(result.to_dict()))
                except Exception as e:
                    logger.warning(f"Error writing to cache: {str(e)}")
            return result
    except aiohttp.ClientError as e:
        logger.error(f"AIOHTTP ClientError for Anthropic API: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error calling Anthropic API: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def query_llm(prompt: str, system_message: str = None, model: str = None) -> LLMResponse:
    provider = CONFIG["llm"]["provider"].lower()
    selected_model = model or CONFIG["llm"]["model"]
    if "gpt" in selected_model.lower():
        provider = "openai"
    elif "claude" in selected_model.lower():
        provider = "anthropic"
    max_attempts = CONFIG["retries"]["max_attempts"]
    backoff_factor = CONFIG["retries"]["backoff_factor"]
    for attempt in range(max_attempts):
        try:
            if provider == "openai":
                return await call_openai_api(prompt, system_message, selected_model)
            elif provider == "anthropic":
                return await call_anthropic_api(prompt, system_message, selected_model)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {provider}")
        except HTTPException as e:
            if e.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts - 1:
                wait_time = backoff_factor ** attempt
                logger.warning(f"LLM API error (attempt {attempt+1}/{max_attempts}): {e.detail}. Retrying in {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt < max_attempts - 1:
                wait_time = backoff_factor ** attempt
                logger.warning(f"Unexpected error (attempt {attempt+1}/{max_attempts}): {str(e)}. Retrying in {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
            else:
                raise HTTPException(status_code=500, detail=str(e))

# --- Trading-specific LLM Prompt Templates ---
class TradingAnalysisRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    market_data: Optional[Dict[str, Any]] = None
    news_data: Optional[List[Dict[str, Any]]] = None
    sentiment_data: Optional[Dict[str, Any]] = None
    technical_indicators: Optional[Dict[str, Any]] = None
    historical_context: Optional[str] = None
    specific_query: Optional[str] = None
    output_format: str = "text"
    fetch_data: bool = True

async def analyze_trade_opportunity(request: TradingAnalysisRequest, api_key: str = Depends(verify_api_key)) -> Dict[str, Any]:
    market_data = request.market_data or {}
    news_data = request.news_data or []
    sentiment_data = request.sentiment_data or {}
    technical_indicators = request.technical_indicators or {}
    
    if request.fetch_data:
        try:
            alpaca_data = await fetch_alpaca_data(request.symbol, api_key)
            market_data.update(alpaca_data.get("quotes", {}))
        except Exception as e:
            logger.warning(f"Failed to fetch Alpaca data: {str(e)}")
        
        try:
            polygon_data = await fetch_polygon_data(request.symbol, api_key)
            market_data.update(polygon_data.get("quotes", {}))
        except Exception as e:
            logger.warning(f"Failed to fetch Polygon data: {str(e)}")
        
        try:
            reddit_data = await fetch_reddit_data(request.symbol, api_key)
            sentiment_data.update(reddit_data.get("sentiment", {}))
        except Exception as e:
            logger.warning(f"Failed to fetch Reddit data: {str(e)}")
        
        try:
            unusual_whales_data = await fetch_unusual_whales_data(request.symbol, api_key)
            market_data["options_flow"] = unusual_whales_data.get("data", {})
        except Exception as e:
            logger.warning(f"Failed to fetch Unusual Whales data: {str(e)}")
        
        try:
            yahoo_finance_data = await fetch_yahoo_finance_data(request.symbol, api_key)
            news_data.extend(yahoo_finance_data.get("articles", []))
        except Exception as e:
            logger.warning(f"Failed to fetch Yahoo Finance data: {str(e)}")
        
        try:
            redis_news = await fetch_redis_data(f"yahoo_news_analysis:*{request.symbol}*")
            news_data.extend([item for sublist in redis_news for item in sublist.get("articles", [])])
        except Exception as e:
            logger.warning(f"Failed to fetch Redis news data: {str(e)}")

    prompt = f"""
    Analyze the trading opportunity for {request.symbol}:

    === Market Data ===
    {json.dumps(market_data, indent=2)}

    === Technical Indicators ===
    {json.dumps(technical_indicators, indent=2) if technical_indicators else "No technical indicators provided."}

    === Recent News ===
    {''.join(f"{idx+1}. {news.get('title', 'No Title')} [{news.get('published_at', 'No Date')}]\n   {news.get('summary', news.get('content', 'No content'))[:200]}...\n\n" for idx, news in enumerate(news_data[:5])) if news_data else "No recent news provided."}

    === Market Sentiment ===
    {json.dumps(sentiment_data, indent=2) if sentiment_data else "No sentiment data provided."}

    === Historical Context ===
    {request.historical_context if request.historical_context else "No historical context provided."}

    === Instructions ===
    {request.specific_query or "Provide a trading analysis and recommendation."}

    Provide:
    1. Summary of market conditions
    2. Significant technical indicators and patterns
    3. News impact assessment
    4. Options flow analysis (if available)
    5. Social media sentiment impact (if available)
    6. Trading recommendation (buy, sell, hold)
    7. Risk assessment and price targets
    8. Confidence level (low, medium, high)

    Output Format: {"JSON" if request.output_format.lower() == "json" else "Detailed text analysis"}
    """

    system_message = """
    You are an expert trading analyst for the NextG3N trading system. Analyze market data, technical indicators, news, sentiment, and options flow to provide accurate trading recommendations. Base your analysis strictly on the provided data, avoiding speculation. Include precise numerical values for entry points, targets, and stops. Provide a clear risk assessment and confidence level.
    """

    try:
        response = await query_llm(prompt, system_message)
        result = {
            "symbol": request.symbol,
            "analysis": response.text,
            "model_used": response.model_used,
            "tokens_used": response.usage.total_tokens,
            "timestamp": datetime.now().isoformat()
        }
        if redis_client:
            try:
                key = f"llm:analysis:{request.symbol}:{datetime.now().strftime('%Y%m%d%H%M%S')}"
                await redis_client.setex(key, 86400 * 7, json.dumps(result))
                await redis_client.zadd("llm:analysis:index", {request.symbol: datetime.now().timestamp()})
            except Exception as e:
                logger.warning(f"Failed to store analysis in Redis: {str(e)}")
        return result
    except Exception as e:
        logger.error(f"Error in analyze_trade_opportunity: {str(e)}")
        return {
            "symbol": request.symbol,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

class NewsAnalysisRequest(BaseModel):
    news_items: List[Dict[str, Any]] = Field(..., min_items=1)
    symbols: Optional[List[str]] = None
    context: Optional[str] = None
    output_format: str = "text"
    fetch_data: bool = True

async def analyze_news_impact(request: NewsAnalysisRequest, api_key: str = Depends(verify_api_key)) -> Dict[str, Any]:
    news_items = request.news_items
    symbols = request.symbols or []
    
    if request.fetch_data and symbols:
        try:
            yahoo_finance_data = await fetch_yahoo_finance_data(symbols[0], api_key)
            news_items.extend(yahoo_finance_data.get("articles", []))
        except Exception as e:
            logger.warning(f"Failed to fetch Yahoo Finance news: {str(e)}")
        try:
            redis_news = await fetch_redis_data(f"yahoo_news_analysis:*{symbols[0]}*")
            news_items.extend([item for sublist in redis_news for item in sublist.get("articles", [])])
        except Exception as e:
            logger.warning(f"Failed to fetch Redis news data: {str(e)}")

    prompt = f"""
    Analyze the market impact of the following news items:

    === News Items ===
    {''.join(f"{idx+1}. {news.get('title', 'No Title')} [{news.get('published_at', 'No Date')}]\n   Source: {news.get('source', 'Unknown Source')}\n   {news.get('content', news.get('summary', 'No content'))[:200]}...\n\n" for idx, news in enumerate(news_items))}

    === Related Symbols ===
    {', '.join(symbols) if symbols else 'None'}

    === Additional Context ===
    {request.context if request.context else 'None'}

    === Instructions ===
    Provide:
    1. Summary of each news item
    2. Market impact assessment (positive, negative, neutral)
    3. Affected sectors or stocks
    4. Severity of impact (minimal, moderate, significant)
    5. Timeframe of impact (immediate, short-term, long-term)

    Output Format: {"JSON" if request.output_format.lower() == "json" else "Detailed text analysis"}
    """

    system_message = """
    You are an expert financial news analyst for the NextG3N trading system. Extract insights from news to determine market impact. Focus on facts, differentiate between established information and forward-looking statements, and maintain objectivity.
    """

    try:
        response = await query_llm(prompt, system_message)
        result = {
            "analysis": response.text,
            "model_used": response.model_used,
            "tokens_used": response.usage.total_tokens,
            "timestamp": datetime.now().isoformat()
        }
        if redis_client and symbols:
            try:
                news_hash = hashlib.md5("".join([news.get("title", "")[:50] for news in news_items]).encode()).hexdigest()
                key = f"llm:news_analysis:{news_hash}"
                await redis_client.setex(key, 86400 * 3, json.dumps(result))
                for symbol in symbols:
                    await redis_client.zadd(f"llm:news_analysis:symbol:{symbol}", {key: datetime.now().timestamp()})
            except Exception as e:
                logger.warning(f"Failed to store news analysis in Redis: {str(e)}")
        return result
    except Exception as e:
        logger.error(f"Error in analyze_news_impact: {str(e)}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

class TradingReportRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    timeframe: str = Field(..., min_length=1)
    market_data: Optional[Dict[str, Any]] = None
    technical_analysis: Optional[Dict[str, Any]] = None
    recent_trades: Optional[List[Dict[str, Any]]] = None
    news_mentions: Optional[List[Dict[str, Any]]] = None
    fetch_data: bool = True

async def generate_trading_report(
    request: TradingReportRequest,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    market_data = request.market_data or {}
    technical_analysis = request.technical_analysis or {}
    recent_trades = request.recent_trades or []
    news_mentions = request.news_mentions or []

    if request.fetch_data:
        try:
            alpaca_data = await fetch_alpaca_data(request.symbol, api_key)
            market_data.update(alpaca_data.get("quotes", {}))
        except Exception as e:
            logger.warning(f"Failed to fetch Alpaca data: {str(e)}")
        
        try:
            polygon_data = await fetch_polygon_data(request.symbol, api_key)
            market_data.update(polygon_data.get("quotes", {}))
        except Exception as e:
            logger.warning(f"Failed to fetch Polygon data: {str(e)}")
        
        try:
            unusual_whales_data = await fetch_unusual_whales_data(request.symbol, api_key)
            market_data["options_flow"] = unusual_whales_data.get("data", {})
        except Exception as e:
            logger.warning(f"Failed to fetch Unusual Whales data: {str(e)}")
        
        try:
            yahoo_finance_data = await fetch_yahoo_finance_data(request.symbol, api_key)
            news_mentions.extend(yahoo_finance_data.get("articles", []))
        except Exception as e:
            logger.warning(f"Failed to fetch Yahoo Finance data: {str(e)}")

    prompt = f"""
    Generate a trading report for {request.symbol} on the {request.timeframe} timeframe:

    === Market Data ===
    {json.dumps(market_data, indent=2)}

    === Technical Analysis ===
    {json.dumps(technical_analysis, indent=2) if technical_analysis else "No technical analysis provided."}

    === Recent Trades ===
    {json.dumps(recent_trades, indent=2) if recent_trades else "No recent trades provided."}

    === Recent News Mentions ===
    {''.join(f"{idx+1}. {news.get('title', 'No Title')} [{news.get('published_at', 'No Date')}]\n   {news.get('summary', '')[:150]}...\n" for idx, news in enumerate(news_mentions[:5])) if news_mentions else "No news mentions provided."}

    === Instructions ===
    Create a professional trading report including:

    1. Executive Summary
       - Market position and trend
       - Key takeaways
    2. Technical Analysis Details
       - Support and resistance levels
       - Key indicators (moving averages, RSI, MACD)
       - Chart patterns
    3. Performance Analysis
       - Price action
       - Volume analysis
       - Volatility
    4. News & Sentiment Impact
       - Significant news events
       - Market sentiment
    5. Options Flow Analysis
       - Unusual options activity (if available)
    6. Trade Recommendations
       - Entry points
       - Exit targets
       - Stop loss
       - Position sizing
    7. Risk Assessment
       - Potential downsides
       - Conditions to monitor
       - Confidence level
    """

    system_message = """
    You are a professional trading analyst for the NextG3N trading system. Generate detailed, data-driven trading reports. Ensure conclusions are backed by data, with precise numerical values for entry points, targets, and stops. Present information in a structured format with clear sections and bullet points.
    """

    try:
        response = await query_llm(prompt, system_message)
        result = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "report": response.text,
            "model_used": response.model_used,
            "tokens_used": response.usage.total_tokens,
            "timestamp": datetime.now().isoformat()
        }
        if redis_client:
            try:
                key = f"llm:report:{request.symbol}:{request.timeframe}:{datetime.now().strftime('%Y%m%d')}"
                await redis_client.setex(key, 86400 * 14, json.dumps(result))
                await redis_client.zadd(f"llm:reports:{request.symbol}", {key: datetime.now().timestamp()})
            except Exception as e:
                logger.warning(f"Failed to store report in Redis: {str(e)}")
        return result
    except Exception as e:
        logger.error(f"Error in generate_trading_report: {str(e)}")
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# --- FastAPI Server ---
app = FastAPI(
    title="LLM Integration MCP Server",
    description="Server for integrating LLM capabilities with the NextG3N trading system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "General", "description": "Server information and health checks"},
        {"name": "LLM", "description": "Direct LLM query endpoints"},
        {"name": "Trading", "description": "Trading analysis endpoints"},
        {"name": "Analysis", "description": "News and sentiment analysis endpoints"},
        {"name": "Reports", "description": "Comprehensive trading report endpoints"}
    ]
)

app.add_middleware(
    lambda request, call_next: rate_limiter.check_rate_limit(request) and call_next(request)
)

@app.middleware("http")
async def track_requests(request: Request, call_next):
    global active_requests
    active_requests += 1
    try:
        return await call_next(request)
    finally:
        active_requests -= 1

@app.on_event("startup")
async def startup_event():
    logger.info("LLM Integration server starting up")
    provider = CONFIG["llm"]["provider"].lower()
    if provider == "openai" and not CONFIG["llm"]["api_key"]:
        logger.warning("OpenAI API key not configured")
    elif provider == "anthropic" and not CONFIG["llm"]["anthropic_api_key"]:
        logger.warning("Anthropic API key not configured")
    await connect_redis()
    await get_http_session()
    logger.info("LLM Integration server startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("LLM Integration server shutting down")
    shutdown_event.set()
    shutdown_start = time.time()
    while active_requests > 0:
        if time.time() - shutdown_start > CONFIG["shutdown"]["timeout_seconds"]:
            logger.warning(f"Shutdown timeout exceeded with {active_requests} pending requests")
            break
        logger.info(f"Waiting for {active_requests} active requests")
        await asyncio.sleep(1)
    if http_session and not http_session.closed:
        await http_session.close()
    if redis_client:
        try:
            await redis_client.close()
            redis_client = None
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {str(e)}")
    logger.info("LLM Integration server shutdown complete")

@app.get("/server_info", tags=["General"], dependencies=[Depends(verify_api_key)])
async def get_server_info():
    safe_config = {
        "provider": CONFIG["llm"]["provider"],
        "model": CONFIG["llm"]["model"],
        "rate_limit_per_minute": CONFIG["llm"]["rate_limit_per_minute"],
        "max_tokens": CONFIG["llm"]["max_tokens"],
        "temperature": CONFIG["llm"]["temperature"],
        "caching_enabled": CONFIG["caching"]["enabled"]
    }
    return {
        "name": "llm_integration",
        "version": "1.0.0",
        "description": "LLM Integration MCP Server for NextG3N trading system",
        "tools": [
            "query_llm",
            "analyze_trade_opportunity",
            "analyze_news_impact",
            "generate_trading_report"
        ],
        "config": safe_config
    }

@app.post("/query_llm", tags=["LLM"], dependencies=[Depends(verify_api_key)])
async def api_query_llm(
    prompt: str,
    system_message: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    try:
        response = await query_llm(prompt, system_message, model)
        return {
            "success": True,
            "response": response.text,
            "usage": response.usage.to_dict(),
            "model_used": response.model_used
        }
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error in query_llm API: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_trade_opportunity", tags=["Trading"], dependencies=[Depends(verify_api_key)])
async def api_analyze_trade_opportunity(request: TradingAnalysisRequest) -> Dict[str, Any]:
    try:
        result = await analyze_trade_opportunity(request)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"success": True, **result}
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_trade_opportunity API: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_news_impact", tags=["Analysis"], dependencies=[Depends(verify_api_key)])
async def api_analyze_news_impact(request: NewsAnalysisRequest) -> Dict[str, Any]:
    try:
        result = await analyze_news_impact(request)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"success": True, **result}
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_news_impact API: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_trading_report", tags=["Reports"], dependencies=[Depends(verify_api_key)])
async def api_generate_trading_report(request: TradingReportRequest) -> Dict[str, Any]:
    try:
        result = await generate_trading_report(request)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"success": True, **result}
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error in generate_trading_report API: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting LLM Integration server in standalone mode")
    uvicorn.run(app, host="0.0.0.0", port=8007)