"""
Yahoo Finance MCP FastAPI Server for LLM integration (production).
Provides financial news, quotes, chart data, summary, and sentiment analysis from Yahoo Finance.
All configuration is contained in this file.
"""


import os
import json
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Response, status, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
import redis.asyncio as aioredis
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import yaml
from pathlib import Path
from copy import deepcopy
import logging
import hashlib
import gc
import pandas as pd
import numpy as np

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

# Fallback logger for use in validate_config
logger = logging.getLogger("yahoo_finance_server")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Configuration ---
def validate_config():
    config_path = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'yahoo_finance_config.yaml'))
    default_config = {
        "yahoo_finance": {
            "rate_limit_per_minute": 60,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "default_news_count": 20,
            "default_chart_interval": "1d",
            "default_chart_range": "1mo",
            "request_delay": 0.5  # Seconds between yfinance calls
        },
        "redis": {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": int(os.getenv("REDIS_DB", "0")),
            "password": os.getenv("REDIS_PASSWORD", None),
            "max_retries": 5,
            "reconnect_cooldown": 30
        },
        "cache": {
            "news_ttl": 300,  # 5 minutes
            "quotes_ttl": 300,  # 5 minutes
            "chart_ttl_intraday": 300,  # 5 minutes
            "chart_ttl_daily": 3600,  # 1 hour
            "summary_ttl": 86400,  # 24 hours
            "analysis_ttl_low": 7 * 24 * 60 * 60,  # 7 days
            "analysis_ttl_high": 14 * 24 * 60 * 60,  # 14 days
            "alert_ttl": 3 * 24 * 60 * 60  # 3 days
        },
        "models": {
            "pegasus": {
                "model_name": "google/pegasus-xsum",
                "max_length": 150,
                "min_length": 30,
                "num_beams": 4
            },
            "rebel": {
                "model_name": "Babelscape/rebel-large",
                "max_length": 256,
                "num_beams": 4
            },
            "financial_bert": {
                "model_name": "yiyanghkust/finbert-tone",
                "max_length": 512
            }
        },
        "security": {
            "enable_auth": True
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
        config["yahoo_finance"]["rate_limit_per_minute"] = int(os.getenv("RATE_LIMIT_PER_MINUTE", config["yahoo_finance"]["rate_limit_per_minute"]))
        config["yahoo_finance"]["user_agent"] = os.getenv("USER_AGENT", config["yahoo_finance"]["user_agent"])
        config["yahoo_finance"]["default_news_count"] = int(os.getenv("DEFAULT_NEWS_COUNT", config["yahoo_finance"]["default_news_count"]))
        config["yahoo_finance"]["default_chart_interval"] = os.getenv("DEFAULT_CHART_INTERVAL", config["yahoo_finance"]["default_chart_interval"])
        config["yahoo_finance"]["default_chart_range"] = os.getenv("DEFAULT_CHART_RANGE", config["yahoo_finance"]["default_chart_range"])
        config["yahoo_finance"]["request_delay"] = float(os.getenv("REQUEST_DELAY", config["yahoo_finance"]["request_delay"]))
        config["redis"]["host"] = os.getenv("REDIS_HOST", config["redis"]["host"])
        config["redis"]["port"] = int(os.getenv("REDIS_PORT", config["redis"]["port"]))
        config["redis"]["db"] = int(os.getenv("REDIS_DB", config["redis"]["db"]))
        config["redis"]["password"] = os.getenv("REDIS_PASSWORD", config["redis"]["password"])
        config["redis"]["max_retries"] = int(os.getenv("REDIS_MAX_RETRIES", config["redis"]["max_retries"]))
        config["redis"]["reconnect_cooldown"] = int(os.getenv("REDIS_RECONNECT_COOLDOWN", config["redis"]["reconnect_cooldown"]))
        config["security"]["enable_auth"] = os.getenv("YAHOO_FINANCE_ENABLE_AUTH", str(config["security"]["enable_auth"])).lower() == "true"
        config["shutdown"]["timeout_seconds"] = int(os.getenv("SHUTDOWN_TIMEOUT_SECONDS", config["shutdown"]["timeout_seconds"]))

        # Validate configuration
        if config["yahoo_finance"]["rate_limit_per_minute"] <= 0:
            logger.warning(f"Invalid rate_limit_per_minute: {config['yahoo_finance']['rate_limit_per_minute']}. Using default: 60")
            config["yahoo_finance"]["rate_limit_per_minute"] = 60
        if config["yahoo_finance"]["default_news_count"] <= 0:
            logger.warning(f"Invalid default_news_count: {config['yahoo_finance']['default_news_count']}. Using default: 20")
            config["yahoo_finance"]["default_news_count"] = 20
        if config["yahoo_finance"]["request_delay"] < 0:
            logger.warning(f"Invalid request_delay: {config['yahoo_finance']['request_delay']}. Using default: 0.5")
            config["yahoo_finance"]["request_delay"] = 0.5
        if config["redis"]["port"] <= 0:
            logger.warning(f"Invalid redis_port: {config['redis']['port']}. Using default: 6379")
            config["redis"]["port"] = 6379
        if config["redis"]["db"] < 0:
            logger.warning(f"Invalid redis_db: {config['redis']['db']}. Using default: 0")
            config["redis"]["db"] = 0
        if config["redis"]["max_retries"] < 0:
            logger.warning(f"Invalid max_retries: {config['redis']['max_retries']}. Using default: 5")
            config["redis"]["max_retries"] = 5
        if config["redis"]["reconnect_cooldown"] <= 0:
            logger.warning(f"Invalid reconnect_cooldown: {config['redis']['reconnect_cooldown']}. Using default: 30")
            config["redis"]["reconnect_cooldown"] = 30
        if config["cache"]["news_ttl"] <= 0:
            logger.warning(f"Invalid news_ttl: {config['cache']['news_ttl']}. Using default: 300")
            config["cache"]["news_ttl"] = 300
        if config["cache"]["quotes_ttl"] <= 0:
            logger.warning(f"Invalid quotes_ttl: {config['cache']['quotes_ttl']}. Using default: 300")
            config["cache"]["quotes_ttl"] = 300
        if config["cache"]["chart_ttl_intraday"] <= 0:
            logger.warning(f"Invalid chart_ttl_intraday: {config['cache']['chart_ttl_intraday']}. Using default: 300")
            config["cache"]["chart_ttl_intraday"] = 300
        if config["cache"]["chart_ttl_daily"] <= 0:
            logger.warning(f"Invalid chart_ttl_daily: {config['cache']['chart_ttl_daily']}. Using default: 3600")
            config["cache"]["chart_ttl_daily"] = 3600
        if config["cache"]["summary_ttl"] <= 0:
            logger.warning(f"Invalid summary_ttl: {config['cache']['summary_ttl']}. Using default: 86400")
            config["cache"]["summary_ttl"] = 86400
        if config["cache"]["analysis_ttl_low"] <= 0:
            logger.warning(f"Invalid analysis_ttl_low: {config['cache']['analysis_ttl_low']}. Using default: 7 days")
            config["cache"]["analysis_ttl_low"] = 7 * 24 * 60 * 60
        if config["cache"]["analysis_ttl_high"] <= 0:
            logger.warning(f"Invalid analysis_ttl_high: {config['cache']['analysis_ttl_high']}. Using default: 14 days")
            config["cache"]["analysis_ttl_high"] = 14 * 24 * 60 * 60
        if config["cache"]["alert_ttl"] <= 0:
            logger.warning(f"Invalid alert_ttl: {config['cache']['alert_ttl']}. Using default: 3 days")
            config["cache"]["alert_ttl"] = 3 * 24 * 60 * 60
        if config["shutdown"]["timeout_seconds"] <= 0:
            logger.warning(f"Invalid timeout_seconds: {config['shutdown']['timeout_seconds']}. Using default: 30")
            config["shutdown"]["timeout_seconds"] = 30
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}. Using default configuration.")
        return default_config

CONFIG = validate_config()

# Replace the fallback logger with the proper one
logger = get_logger("yahoo_finance_server")
logger.info("Initializing Yahoo Finance server")

# --- Redis Client ---
redis_client = None
redis_last_error_time = 0
redis_connection_attempts = 0

async def connect_redis(force_reconnect=False):
    global redis_client, redis_last_error_time, redis_connection_attempts
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

# --- Rate Limiter ---
class RateLimiter:
    def __init__(self, calls_limit: int, time_window: int = 60):
        self.calls_limit = calls_limit
        self.time_window = time_window
        self.redis_key_prefix = "yahoo_finance_rate_limit"
    
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
        now = int(time.time())
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

rate_limiter = RateLimiter(calls_limit=CONFIG["yahoo_finance"]["rate_limit_per_minute"])

# --- Authentication ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not CONFIG["security"]["enable_auth"]:
        return
    valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")
    if not api_key or api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# --- Utility Functions ---
def serialize_pandas_data(data: Any) -> Any:
    if isinstance(data, pd.Series):
        return {k: serialize_pandas_data(v) for k, v in data.to_dict().items()}
    elif isinstance(data, pd.DataFrame):
        return {k: serialize_pandas_data(v) for k, v in data.to_dict(orient="index").items()}
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif isinstance(data, (list, dict)):
        return json.loads(json.dumps(data, default=str))
    elif data is None or isinstance(data, (int, float, str, bool)):
        return data
    return str(data)

async def cuda_safe_predict(model, tokenizer, text: str, device: str, max_length: int, **kwargs) -> Any:
    if not text or len(text.strip()) < 10:
        logger.warning("Text too short for model prediction")
        return None
    if len(text) > 10000:
        logger.warning(f"Truncating long text ({len(text)} chars) for model prediction")
        text = text[:10000] + "..."
    try:
        inputs = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, **kwargs)
        return outputs
    except RuntimeError as cuda_err:
        if "CUDA out of memory" in str(cuda_err) and device == "cuda":
            logger.warning("CUDA out of memory, falling back to CPU")
            prev_device = device
            model.to("cpu")
            inputs = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, **kwargs)
            model.to(prev_device)
            return outputs
        raise
    except Exception as e:
        logger.error(f"Error in model prediction: {str(e)}")
        return None

# --- AI/ML Models ---
# Initialize model instances
pegasus_model = None
rebel_model = None
financial_bert_model = None

class PEGASUSModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["model_name"]
        self.max_length = config["max_length"]
        self.min_length = config["min_length"]
        self.num_beams = config["num_beams"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        logger.info(f"PEGASUSModel ({self.model_name}) will run on {self.device}")

    def load_model(self):
        if self.tokenizer is None or self.model is None:
            logger.info(f"Loading PEGASUS model: {self.model_name}")
            try:
                self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
                self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
                logger.info(f"PEGASUS model {self.model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Error loading PEGASUS model {self.model_name}: {str(e)}")
                raise

    def predict(self, text: str) -> str:
        if self.tokenizer is None or self.model is None:
            try:
                self.load_model()
            except Exception:
                logger.error("Failed to load PEGASUS model")
                return text[:197] + "..." if len(text) > 200 else text
        outputs = asyncio.run_coroutine_threadsafe(
            cuda_safe_predict(
                self.model, self.tokenizer, text, self.device, 1024,
                max_length=self.max_length, min_length=self.min_length, num_beams=self.num_beams,
                length_penalty=2.0, early_stopping=True, no_repeat_ngram_size=3
            ),
            asyncio.get_event_loop()
        ).result()
        if outputs is None:
            return text[:197] + "..." if len(text) > 200 else text
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if not summary or len(summary) < 10:
            logger.warning("Generated summary too short or empty")
            return text[:197] + "..." if len(text) > 200 else text
        return summary

class REBELModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["model_name"]
        self.max_length = config["max_length"]
        self.num_beams = config["num_beams"]
        self.gen_max_length = config.get("gen_max_length", 64)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        logger.info(f"REBELModel ({self.model_name}) will run on {self.device}")

    def load_model(self):
        if self.tokenizer is None or self.model is None:
            logger.info(f"Loading REBEL model: {self.model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
                logger.info(f"REBEL model {self.model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Error loading REBEL model {self.model_name}: {str(e)}")
                raise

    def predict(self, text: str) -> List[Dict[str, str]]:
        if self.tokenizer is None or self.model is None:
            try:
                self.load_model()
            except Exception:
                logger.error("Failed to load REBEL model")
                return []
        outputs = asyncio.run_coroutine_threadsafe(
            cuda_safe_predict(
                self.model, self.tokenizer, text, self.device, self.max_length,
                max_length=self.gen_max_length, num_beams=self.num_beams, num_return_sequences=1
            ),
            asyncio.get_event_loop()
        ).result()
        if outputs is None:
            return []
        decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        extracted_events = []
        current_triplet = {"subject": "", "relation": "", "object": ""}
        state = None
        for token in decoded_preds[0].split():
            if token == "<triplet>":
                if any(current_triplet.values()):
                    extracted_events.append({
                        "type": current_triplet["relation"].strip() or "unknown",
                        "subject": current_triplet["subject"].strip(),
                        "object": current_triplet["object"].strip(),
                        "description": f"{current_triplet['subject'].strip()} {current_triplet['relation'].strip()} {current_triplet['object'].strip()}"
                    })
                current_triplet = {"subject": "", "relation": "", "object": ""}
                state = "relation"
            elif token == "<subj>":
                state = "subject"
            elif token == "<obj>":
                state = "object"
            elif state == "relation":
                current_triplet["relation"] += (token + " ")
            elif state == "subject":
                current_triplet["subject"] += (token + " ")
            elif state == "object":
                current_triplet["object"] += (token + " ")
        if any(current_triplet.values()):
            extracted_events.append({
                "type": current_triplet["relation"].strip() or "unknown",
                "subject": current_triplet["subject"].strip(),
                "object": current_triplet["object"].strip(),
                "description": f"{current_triplet['subject'].strip()} {current_triplet['relation'].strip()} {current_triplet['object'].strip()}"
            })
        return [e for e in extracted_events if e["subject"] and e["object"] and e["type"] != "unknown"]

    def extract_events(self, text: str) -> List[Dict[str, str]]:
        return self.predict(text)

class FinancialBERTModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["model_name"]
        self.max_length = config["max_length"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        logger.info(f"FinancialBERTModel ({self.model_name}) will run on {self.device}")

    def load_model(self):
        if self.tokenizer is None or self.model is None:
            logger.info(f"Loading FinancialBERT model: {self.model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
                logger.info(f"FinancialBERT model {self.model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Error loading FinancialBERT model {self.model_name}: {str(e)}")
                raise

    def predict(self, text: str) -> Dict[str, float]:
        if self.tokenizer is None or self.model is None:
            try:
                self.load_model()
            except Exception:
                logger.error("Failed to load FinancialBERT model")
                return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
        outputs = asyncio.run_coroutine_threadsafe(
            cuda_safe_predict(self.model, self.tokenizer, text, self.device, self.max_length, padding=True),
            asyncio.get_event_loop()
        ).result()
        if outputs is None:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        try:
            scores = probs[0].tolist()
            sentiment = {}
            if hasattr(self.model.config, "id2label"):
                labels = [self.model.config.id2label[i].lower() for i in range(len(scores))]
                for label, score in zip(labels, scores):
                    sentiment[label] = float(score)
                for key in ["positive", "negative", "neutral"]:
                    sentiment.setdefault(key, 0.0)
            else:
                label2id = self.model.config.label2id if hasattr(self.model.config, "label2id") else {}
                positive_idx = label2id.get("Positive", label2id.get("positive", 0))
                negative_idx = label2id.get("Negative", label2id.get("negative", 1))
                neutral_idx = label2id.get("Neutral", label2id.get("neutral", 2))
                if positive_idx < len(scores) and negative_idx < len(scores) and neutral_idx < len(scores):
                    sentiment = {
                        "positive": scores[positive_idx],
                        "negative": scores[negative_idx],
                        "neutral": scores[neutral_idx]
                    }
                else:
                    logger.warning("Invalid label indices for FinBERT")
                    sentiment = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
            total = sum(sentiment.values())
            if abs(total - 1.0) > 0.1:
                logger.warning(f"Sentiment scores don't sum to 1.0 (sum={total})")
                if total > 0:
                    sentiment = {k: v/total for k, v in sentiment.items()}
            return sentiment
        except Exception as e:
            logger.warning(f"Error in sentiment mapping: {str(e)}")
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

# --- News Analysis Component ---
# Initialize model instances
pegasus_model = PEGASUSModel(CONFIG["models"]["pegasus"])
rebel_model = REBELModel(CONFIG["models"]["rebel"])
financial_bert_model = FinancialBERTModel(CONFIG["models"]["financial_bert"])

class NewsAnalysisComponent:
    def __init__(self, pegasus_model: PEGASUSModel, rebel_model: REBELModel, financial_bert_model: FinancialBERTModel):
        self.pegasus_model = pegasus_model
        self.rebel_model = rebel_model
        self.financial_bert_model = financial_bert_model
        self.logger = get_logger("news_analysis_component")

    async def analyze_news(self, articles: List[Dict[str, Any]], symbols: List[str]) -> Dict[str, Any]:
        self.logger.info(f"Analyzing {len(articles)} news articles for symbols: {symbols}")
        try:
            relevant_articles_data = []
            for article in articles:
                article_tickers = article.get("tickers", []) or []
                article_tickers_lower = [t.lower() for t in article_tickers if isinstance(t, str)]
                target_symbols_lower = [s.lower() for s in symbols]
                if any(s in article_tickers_lower for s in target_symbols_lower):
                    analysis = await self.analyze_article(article, symbols)
                    relevant_articles_data.append(analysis)
            self.logger.info(f"Found and analyzed {len(relevant_articles_data)} relevant articles")
            aggregated = self.aggregate_analysis(relevant_articles_data, symbols)
            return {"success": True, "articles": relevant_articles_data, "aggregated": aggregated}
        except Exception as e:
            self.logger.error(f"Error in analyze_news: {str(e)}")
            return {"success": False, "error": str(e)}

    async def analyze_article(self, article: Dict[str, Any], target_symbols: List[str]) -> Dict[str, Any]:
        try:
            title = article.get("title", "") or ""
            summary = article.get("summary", "") or ""
            text_content = f"{title}. {summary}" if summary else title
            generated_summary = self.pegasus_model.predict(text_content)
            events = self.rebel_model.extract_events(text_content)
            sentiment = self.financial_bert_model.predict(text_content)
            relevance_score = self.calculate_relevance_score(article, target_symbols)
            impact_score = self.calculate_impact_score(events, sentiment)
            return {
                "article_id": str(article.get("id", article.get("uuid", "unknown"))),
                "title": title,
                "summary": summary,
                "generated_summary": generated_summary,
                "events": events,
                "sentiment": sentiment,
                "relevance_score": relevance_score,
                "impact_score": impact_score,
                "tickers": article.get("tickers", []),
                "published_at": article.get("published_at", article.get("providerPublishTime", ""))
            }
        except Exception as e:
            self.logger.error(f"Error analyzing article (ID: {article.get('id', 'N/A')}): {str(e)}")
            return {"article_id": article.get("id", "N/A"), "title": article.get("title", ""), "error": str(e)}

    def calculate_relevance_score(self, article: Dict[str, Any], target_symbols: List[str]) -> float:
        score = 0.0
        title = article.get("title", "").lower() or ""
        summary = article.get("summary", "").lower() or ""
        content = title + " " + summary
        article_tickers = article.get("tickers", []) or []
        article_tickers_lower = [t.lower() for t in article_tickers if isinstance(t, str)]
        target_symbols_lower = [s.lower() for s in target_symbols]
        symbol_match_score = 0.0
        for target_sym_l in target_symbols_lower:
            if target_sym_l in title:
                symbol_match_score += 0.35
            if target_sym_l in summary:
                symbol_match_score += 0.25
            if target_sym_l in article_tickers_lower:
                symbol_match_score += 0.4
        score += min(symbol_match_score, 0.7)
        financial_keywords = ["earnings", "profit", "loss", "revenue", "guidance", "fda", "approval", "partnership", "acquisition", "rating", "upgrade", "downgrade", "dividend", "buyback", "ipo", "sec filing", "investigation", "lawsuit", "debt", "equity", "analyst", "target price"]
        keyword_hits = sum(1 for keyword in financial_keywords if keyword in content)
        score += min(keyword_hits * 0.03, 0.2)
        published_ts = article.get("published_at", article.get("providerPublishTime", ""))
        if published_ts:
            try:
                if isinstance(published_ts, (int, float)):
                    published_dt = datetime.fromtimestamp(published_ts, tz=datetime.timezone.utc)
                elif isinstance(published_ts, str):
                    try:
                        published_dt = datetime.fromisoformat(published_ts.replace("Z", "+00:00"))
                    except ValueError:
                        from dateutil.parser import parse
                        published_dt = parse(published_ts)
                        if published_dt.tzinfo is None:
                            published_dt = published_dt.replace(tzinfo=datetime.timezone.utc)
                else:
                    published_dt = published_ts
                    if published_dt.tzinfo is None:
                        published_dt = published_dt.replace(tzinfo=datetime.timezone.utc)
                age_days = (datetime.now(datetime.timezone.utc) - published_dt).total_seconds() / (24 * 60 * 60)
                if age_days <= 1:
                    score += 0.15
                elif age_days <= 3:
                    score += 0.10
                elif age_days <= 7:
                    score += 0.05
            except Exception as e:
                self.logger.warning(f"Could not parse date for relevance scoring ('{published_ts}'): {str(e)}")
        if len(summary) > 50:
            score += 0.05
        if len(summary) == 0 and len(title) > 30:
            score -= 0.05
        return round(min(score, 1.0), 3)

    def calculate_impact_score(self, events: List[Dict[str, str]], sentiment: Dict[str, float]) -> float:
        score = 0.0
        event_impact_weights = {
            "acquisition": 0.3, "earnings": 0.25, "management_change": 0.2, "product_launch": 0.15,
            "fda": 0.3, "approval": 0.3, "lawsuit": 0.25, "investigation": 0.25
        }
        max_event_score = 0.5
        current_event_score = 0.0
        for event in events:
            event_type = event.get("type", "").lower()
            current_event_score += event_impact_weights.get(event_type, 0.05)
        score += min(current_event_score, max_event_score)
        positive_score = sentiment.get("positive", 0.0)
        negative_score = sentiment.get("negative", 0.0)
        if positive_score > 0.7:
            score += 0.3 * positive_score
        elif negative_score > 0.7:
            score += 0.3 * negative_score
        elif positive_score > 0.5:
            score += 0.15 * positive_score
        elif negative_score > 0.5:
            score += 0.15 * negative_score
        return round(min(score, 1.0), 3)

    def aggregate_analysis(self, articles: List[Dict[str, Any]], symbols: List[str]) -> Dict[str, Any]:
        try:
            aggregated = {
                "sentiment": {"overall": "neutral", "positive": 0.0, "negative": 0.0, "neutral": 0.0, "avg_score": 0.0},
                "events": [],
                "impact": {"overall": "low", "avg_score": 0.0},
                "relevance": {"avg_score": 0.0},
                "by_symbol": {
                    s: {
                        "sentiment": {"overall": "neutral", "positive": 0.0, "negative": 0.0, "neutral": 0.0, "avg_score": 0.0},
                        "events": [],
                        "impact": {"overall": "low", "avg_score": 0.0},
                        "relevance": {"avg_score": 0.0},
                        "article_count": 0
                    } for s in symbols
                }
            }
            all_events_map = {}
            total_sentiment_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            total_impact_score = 0.0
            total_relevance_score = 0.0
            article_count_total = 0
            for article in articles:
                if article.get("error"):
                    continue
                article_count_total += 1
                sentiment = article.get("sentiment", {})
                total_sentiment_scores["positive"] += sentiment.get("positive", 0)
                total_sentiment_scores["negative"] += sentiment.get("negative", 0)
                total_sentiment_scores["neutral"] += sentiment.get("neutral", 0)
                total_impact_score += article.get("impact_score", 0)
                total_relevance_score += article.get("relevance_score", 0)
                for event in article.get("events", []):
                    event_desc = f"{event.get('type')}:{event.get('subject','')}-{event.get('object','')}"
                    all_events_map[event_desc] = all_events_map.get(event_desc, 0) + 1
                for sym in article.get("tickers", []):
                    if sym in aggregated["by_symbol"]:
                        s_data = aggregated["by_symbol"][sym]
                        s_data["article_count"] += 1
                        s_data["sentiment"]["positive"] += sentiment.get("positive", 0)
                        s_data["sentiment"]["negative"] += sentiment.get("negative", 0)
                        s_data["sentiment"]["neutral"] += sentiment.get("neutral", 0)
                        s_data["impact"]["avg_score"] += article.get("impact_score", 0)
                        s_data["relevance"]["avg_score"] += article.get("relevance_score", 0)
                        for event in article.get("events", []):
                            s_data["events"].append(event)
            if article_count_total > 0:
                aggregated["sentiment"]["positive"] = total_sentiment_scores["positive"] / article_count_total
                aggregated["sentiment"]["negative"] = total_sentiment_scores["negative"] / article_count_total
                aggregated["sentiment"]["neutral"] = total_sentiment_scores["neutral"] / article_count_total
                aggregated["sentiment"]["avg_score"] = (
                    total_sentiment_scores["positive"] + total_sentiment_scores["negative"] +
                    total_sentiment_scores["neutral"]) / 3
                if aggregated["sentiment"]["positive"] > 0.4:
                    aggregated["sentiment"]["overall"] = "positive"
                elif aggregated["sentiment"]["negative"] > 0.4:
                    aggregated["sentiment"]["overall"] = "negative"
                aggregated["impact"]["avg_score"] = total_impact_score / article_count_total
                aggregated["relevance"]["avg_score"] = total_relevance_score / article_count_total
                if aggregated["impact"]["avg_score"] > 0.6:
                    aggregated["impact"]["overall"] = "high"
                elif aggregated["impact"]["avg_score"] > 0.3:
                    aggregated["impact"]["overall"] = "medium"
            aggregated["events"] = [{"event": k, "count": v} for k, v in sorted(all_events_map.items(), key=lambda item: item[1], reverse=True)][:10]
            for sym, data in aggregated["by_symbol"].items():
                if data["article_count"] > 0:
                    data["sentiment"]["positive"] /= data["article_count"]
                    data["sentiment"]["negative"] /= data["article_count"]
                    data["sentiment"]["neutral"] /= data["article_count"]
                    data["impact"]["avg_score"] /= data["article_count"]
                    data["relevance"]["avg_score"] /= data["article_count"]
                    if data["sentiment"]["positive"] > 0.4:
                        data["sentiment"]["overall"] = "positive"
                    elif data["sentiment"]["negative"] > 0.4:
                        data["sentiment"]["overall"] = "negative"
                    if data["impact"]["avg_score"] > 0.6:
                        data["impact"]["overall"] = "high"
                    elif data["impact"]["avg_score"] > 0.3:
                        data["impact"]["overall"] = "medium"
                    unique_events = []
                    seen_event_descs = set()
                    for ev in data["events"]:
                        ev_desc = f"{ev.get('type')}:{ev.get('subject','')}-{ev.get('object','')}"
                        if ev_desc not in seen_event_descs:
                            unique_events.append(ev)
                            seen_event_descs.add(ev_desc)
                    data["events"] = unique_events[:5]
            return aggregated
        except Exception as e:
            self.logger.error(f"Error in aggregate_analysis: {str(e)}")
            return {"success": False, "error": str(e), "aggregated": {}}

# --- Redis Integration ---
async def store_news_analysis(analysis: Dict[str, Any], symbols: List[str]) -> bool:
    if not redis_client and not await connect_redis():
        logger.warning("Redis client not available, skipping news analysis storage")
        return False
    try:
        date_str = datetime.now().strftime("%Y-%m-%d")
        key_symbols = hashlib.md5("_".join(sorted(symbols)).encode()).hexdigest()
        key = f"yahoo_news_analysis:{date_str}:{key_symbols}"
        avg_relevance = analysis.get("aggregated", {}).get("relevance", {}).get("avg_score", 0.0)
        ttl = CONFIG["cache"]["analysis_ttl_high"] if avg_relevance > 0.7 else CONFIG["cache"]["analysis_ttl_low"]
        json_data = json.dumps(analysis)
        result = await redis_client.setex(key, ttl, json_data)
        if result:
            logger.info(f"Stored news analysis in Redis with key {key}, TTL: {ttl}s")
        else:
            logger.warning(f"Failed to store news analysis in Redis with key {key}")
        agg_impact_score = analysis.get("aggregated", {}).get("impact", {}).get("avg_score", 0.0)
        if agg_impact_score > 0.6:
            alert_key = f"yahoo_news_alert:{date_str}:{key_symbols}"
            alert_data = {
                "timestamp": datetime.now().isoformat(),
                "symbols": symbols,
                "aggregated_impact_score": agg_impact_score,
                "top_articles_sample": [
                    {"title": art.get("title"), "impact": art.get("impact_score"), "relevance": art.get("relevance_score")}
                    for art in analysis.get("articles", [])[:3]
                ]
            }
            alert_result = await redis_client.setex(alert_key, CONFIG["cache"]["alert_ttl"], json.dumps(alert_data))
            if alert_result:
                logger.info(f"Stored market-moving news alert in Redis with key {alert_key}")
            else:
                logger.warning(f"Failed to store market-moving news alert in Redis")
        return result is not None
    except Exception as e:
        logger.error(f"Error storing news analysis/alerts in Redis: {str(e)}")
        return False

async def get_stored_news_analysis_from_redis(symbols: List[str], date: Optional[str] = None) -> Dict[str, Any]:
    if not redis_client and not await connect_redis():
        raise HTTPException(status_code=503, detail="Redis cache service unavailable")
    try:
        if date:
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        date_str = date if date else datetime.now().strftime("%Y-%m-%d")
        key_symbols = hashlib.md5("_".join(sorted(symbols)).encode()).hexdigest()
        key = f"yahoo_news_analysis:{date_str}:{key_symbols}"
        analysis_json = await redis_client.get(key)
        if analysis_json:
            logger.info(f"Retrieved stored news analysis from Redis for key {key}")
            return {"success": True, "analysis": json.loads(analysis_json)}
        logger.info(f"No stored news analysis found in Redis for key {key}")
        return {"success": False, "message": "No analysis found for the given symbols and date"}
    except Exception as e:
        logger.error(f"Error retrieving stored news analysis from Redis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving stored analysis: {str(e)}")

# --- FastAPI Models ---
class NewsRequest(BaseModel):
    symbols: List[str] = []
    count: int = Field(CONFIG["yahoo_finance"]["default_news_count"], ge=1, le=100)

class NewsAnalysisRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=1)
    count: int = Field(CONFIG["yahoo_finance"]["default_news_count"], ge=1, le=100)

class EventExtractionRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=1)
    count: int = Field(CONFIG["yahoo_finance"]["default_news_count"], ge=1, le=100)

class QuotesRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=1, max_items=50)

class ChartRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    interval: str = Field(CONFIG["yahoo_finance"]["default_chart_interval"])
    range: str = Field(CONFIG["yahoo_finance"]["default_chart_range"])

class SummaryRequest(BaseModel):
    symbol: str = Field(..., min_length=1)

class StoredNewsAnalysisRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=1)
    date: Optional[str] = None

# --- FastAPI Server ---
news_analysis_component = NewsAnalysisComponent(pegasus_model, rebel_model, financial_bert_model)
app = FastAPI(
    title="Yahoo Finance MCP Server for LLM",
    description="Production-ready server providing financial news, quotes, chart data and sentiment analysis",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "General", "description": "Server information and health checks"},
        {"name": "Data", "description": "Endpoints for fetching financial data"},
        {"name": "Analysis", "description": "Endpoints for news analysis and event extraction"}
    ]
)

app.add_middleware(
    lambda request, call_next: rate_limiter.check_rate_limit(request) and call_next(request)
)

active_requests = 0
shutdown_event = asyncio.Event()

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
    logger.info("Yahoo Finance server starting up")
    await connect_redis()
    try:
        server_health = await check_server_health()
        logger.info(f"Server health check on startup: {server_health}")
    except Exception as e:
        logger.error(f"Error performing initial health check: {str(e)}")
    asyncio.create_task(load_models_background())
    shutdown_event.clear()
    logger.info("Yahoo Finance server startup complete")

async def load_models_background():
    logger.info("Loading ML models in background")
    try:
        pegasus_model.load_model()
        rebel_model.load_model()
        financial_bert_model.load_model()
        logger.info("All ML models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading ML models in background: {str(e)}")

async def check_server_health() -> Dict[str, Any]:
    health = {
        "status": "healthy",
        "components": {
            "redis": False,
            "yfinance": False
        },
        "timestamp": datetime.now().isoformat()
    }
    try:
        redis_result = await asyncio.wait_for(redis_client.ping(), timeout=2.0)
        health["components"]["redis"] = bool(redis_result)
    except Exception as e:
        logger.warning(f"Redis health check failed: {str(e)}")
        health["status"] = "degraded"
    try:
        ticker = yfinance.Ticker("AAPL")
        info = ticker.info
        health["components"]["yfinance"] = bool(info and "symbol" in info)
    except Exception as e:
        logger.warning(f"YFinance health check failed: {str(e)}")
        health["status"] = "degraded"
    return health

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Yahoo Finance server shutting down")
    shutdown_event.set()
    shutdown_start = time.time()
    while active_requests > 0:
        if time.time() - shutdown_start > CONFIG["shutdown"]["timeout_seconds"]:
            logger.warning(f"Shutdown timeout exceeded with {active_requests} pending requests")
            break
        logger.info(f"Waiting for {active_requests} active requests")
        await asyncio.sleep(1)
    if redis_client:
        try:
            await redis_client.close()
            logger.info("Redis connection closed")
            redis_client = None
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {str(e)}")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA memory cache cleared")
        pegasus_model.model = None
        pegasus_model.tokenizer = None
        rebel_model.model = None
        rebel_model.tokenizer = None
        financial_bert_model.model = None
        financial_bert_model.tokenizer = None
        gc.collect()
        logger.info("ML model references cleared and garbage collected")
    except Exception as e:
        logger.warning(f"Error clearing model resources: {str(e)}")
    logger.info("Yahoo Finance server shutdown complete")

@app.get("/server_info", tags=["General"], dependencies=[Depends(verify_api_key)])
async def get_server_info():
    return {
        "name": "yahoo_finance",
        "version": "1.1.0",
        "description": "Production MCP Server for Yahoo Finance News and Market Data Integration",
        "tools": [
            "fetch_news", "fetch_quotes", "fetch_chart_data", "fetch_summary",
            "analyze_news", "extract_events", "get_stored_news_analysis"
        ],
        "models": [
            CONFIG["models"]["pegasus"]["model_name"],
            CONFIG["models"]["rebel"]["model_name"],
            CONFIG["models"]["financial_bert"]["model_name"]
        ],
        "config": {k: v for k, v in CONFIG["yahoo_finance"].items() if k != "redis_password"}
    }

@app.get("/health", tags=["General"], dependencies=[Depends(verify_api_key)])
async def health_check(response: Response):
    health_status = await check_server_health()
    if health_status["status"] != "healthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return health_status

@app.post("/fetch_news", tags=["Data"], dependencies=[Depends(verify_api_key)])
async def fetch_news(req: NewsRequest):
    try:
        result = await api_fetch_news(req)
        if not result.get("success", False):
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=result.get("error", "Failed to fetch news"))
        return result
    except Exception as e:
        logger.error(f"Error in /fetch_news endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def api_fetch_news(req: NewsRequest) -> Dict[str, Any]:
    try:
        symbols_list = req.symbols if req.symbols else []
        count = req.count if req.count else CONFIG["yahoo_finance"]["default_news_count"]
        cache_key = f"yahoo_news:{datetime.now().strftime('%Y-%m-%d')}:{','.join(sorted(symbols_list))}:{count}"
        if redis_client:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Returning cached news for {symbols_list}")
                return json.loads(cached_data)
        all_articles = []
        tickers_to_fetch = symbols_list if symbols_list else ["^GSPC"]
        for ticker_str in tickers_to_fetch:
            try:
                ticker_obj = yfinance.Ticker(ticker_str)
                news_items = ticker_obj.news
                if not news_items:
                    logger.warning(f"No news found for {ticker_str}")
                    continue
                for article in news_items[:count if len(tickers_to_fetch) == 1 else count // len(tickers_to_fetch) or 1]:
                    processed_article = {
                        "id": str(article.get("uuid", "")),
                        "title": article.get("title", ""),
                        "publisher": article.get("publisher", ""),
                        "link": article.get("link", ""),
                        "published_at": article.get("providerPublishTime", ""),
                        "summary": article.get("summary", ""),
                        "tickers": article.get("relatedTickers", [ticker_str]),
                        "type": article.get("type", "")
                    }
                    all_articles.append(processed_article)
                await asyncio.sleep(CONFIG["yahoo_finance"]["request_delay"])
            except Exception as e:
                logger.warning(f"Error fetching news for {ticker_str}: {str(e)}")
        unique_articles = []
        seen_links = set()
        for art in all_articles:
            if art["link"] not in seen_links:
                unique_articles.append(art)
                seen_links.add(art["link"])
        result = {"success": True, "articles": unique_articles[:count]}
        if redis_client:
            await redis_client.setex(cache_key, CONFIG["cache"]["news_ttl"], json.dumps(result))
            logger.debug(f"Cached news data for {symbols_list}")
        logger.info(f"Fetched {len(unique_articles)} unique news articles")
        return result
    except Exception as e:
        logger.error(f"Error in api_fetch_news: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/analyze_news", tags=["Analysis"], dependencies=[Depends(verify_api_key)])
async def api_analyze_news(req: NewsAnalysisRequest, background_tasks: BackgroundTasks):
    try:
        cached_analysis = await get_stored_news_analysis_from_redis(req.symbols)
        if cached_analysis.get("success"):
            logger.info(f"Using cached news analysis for {req.symbols}")
            return cached_analysis["analysis"]
        news_response = await api_fetch_news(NewsRequest(symbols=req.symbols, count=req.count))
        if not news_response.get("success"):
            raise HTTPException(status_code=503, detail=news_response.get("error", "Failed to fetch news"))
        articles = news_response.get("articles", [])
        if not articles:
            return {"success": True, "message": "No articles found to analyze", "articles": [], "aggregated": {}}
        analysis_result = await news_analysis_component.analyze_news(articles, req.symbols)
        if analysis_result.get("success"):
            background_tasks.add_task(store_news_analysis, analysis_result, req.symbols)
        return analysis_result
    except Exception as e:
        logger.error(f"Error in /analyze_news endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_events", tags=["Analysis"], dependencies=[Depends(verify_api_key)])
async def api_extract_events(req: EventExtractionRequest):
    try:
        news_response = await api_fetch_news(NewsRequest(symbols=req.symbols, count=req.count))
        if not news_response.get("success"):
            raise HTTPException(status_code=503, detail=news_response.get("error", "Failed to fetch news"))
        articles = news_response.get("articles", [])
        if not articles:
            return {"success": True, "message": "No articles found for event extraction", "events": []}
        extracted_events_list = []
        for i, article in enumerate(articles):
            article_id = article.get("id", f"unknown-{i}")
            text_content = f"{article.get('title', '')}. {article.get('summary', '')}"
            try:
                events = rebel_model.extract_events(text_content)
                if events:
                    extracted_events_list.append({
                        "article_id": article_id,
                        "title": article.get("title"),
                        "link": article.get("link"),
                        "events": events
                    })
            except Exception as e:
                logger.warning(f"Error extracting events from article {article_id}: {str(e)}")
        logger.info(f"Extracted events from {len(extracted_events_list)} articles")
        return {"success": True, "extracted_data": extracted_events_list}
    except Exception as e:
        logger.error(f"Error in /extract_events endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_stored_news_analysis", tags=["Analysis"], dependencies=[Depends(verify_api_key)])
async def api_get_stored_news_analysis(req: StoredNewsAnalysisRequest):
    try:
        return await get_stored_news_analysis_from_redis(req.symbols, req.date)
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error retrieving stored news analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fetch_quotes", tags=["Data"], dependencies=[Depends(verify_api_key)])
async def api_fetch_quotes(req: QuotesRequest):
    try:
        cache_key = f"yahoo_quotes:{datetime.now().strftime('%Y-%m-%d')}:{','.join(sorted(req.symbols))}"
        if redis_client:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Returning cached quotes for {len(req.symbols)} symbols")
                return json.loads(cached_data)
        data = {}
        for symbol in req.symbols:
            try:
                ticker = yfinance.Ticker(symbol)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    data[symbol] = serialize_pandas_data(hist.iloc[-1])
                else:
                    data[symbol] = {"error": "No data found"}
                await asyncio.sleep(CONFIG["yahoo_finance"]["request_delay"])
            except Exception as e:
                logger.warning(f"Error fetching quote for {symbol}: {str(e)}")
                data[symbol] = {"error": str(e)}
        result = {"success": True, "quotes": data, "timestamp": datetime.now().isoformat()}
        if redis_client:
            await redis_client.setex(cache_key, CONFIG["cache"]["quotes_ttl"], json.dumps(result))
            logger.debug(f"Cached quotes data for {len(req.symbols)} symbols")
        return result
    except Exception as e:
        logger.error(f"Error fetching quotes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fetch_chart_data", tags=["Data"], dependencies=[Depends(verify_api_key)])
async def api_fetch_chart_data(req: ChartRequest):
    logger.info(f"Fetching chart data for {req.symbol}")
    try:
        cache_key = f"yahoo_chart:{req.symbol}:{req.interval}:{req.range}"
        if redis_client:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Returning cached chart data for {req.symbol}")
                return json.loads(cached_data)
        ticker = yfinance.Ticker(req.symbol)
        hist = ticker.history(period=req.range, interval=req.interval)
        if hist.empty:
            logger.warning(f"No chart data found for {req.symbol}")
            return {"success": False, "error": "No chart data found"}
        hist.index = hist.index.strftime('%Y-%m-%d %H:%M:%S')
        result = {"success": True, "chart_data": serialize_pandas_data(hist)}
        ttl = CONFIG["cache"]["chart_ttl_intraday"] if req.interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m'] else CONFIG["cache"]["chart_ttl_daily"]
        if redis_client:
            await redis_client.setex(cache_key, ttl, json.dumps(result))
            logger.debug(f"Cached chart data for {req.symbol}")
        return result
    except Exception as e:
        logger.error(f"Error fetching chart data for {req.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fetch_summary", tags=["Data"], dependencies=[Depends(verify_api_key)])
async def api_fetch_summary(req: SummaryRequest):
    if shutdown_event.is_set():
        raise HTTPException(status_code=503, detail="Server is shutting down")
    logger.info(f"Fetching summary for {req.symbol}")
    try:
        cache_key = f"yahoo_summary:{req.symbol}"
        if redis_client:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Using cached summary data for {req.symbol}")
                return json.loads(cached_data)
        ticker = yfinance.Ticker(req.symbol)
        info = ticker.info
        summary_data = {
            "longName": info.get("longName"),
            "symbol": info.get("symbol"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "website": info.get("website"),
            "marketCap": info.get("marketCap"),
            "previousClose": info.get("previousClose"),
            "open": info.get("open"),
            "dayLow": info.get("dayLow"),
            "dayHigh": info.get("dayHigh"),
            "volume": info.get("volume"),
            "averageVolume": info.get("averageVolume"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "beta": info.get("beta"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "dividendYield": info.get("dividendYield"),
            "description": info.get("longBusinessSummary", ""),
            "employees": info.get("fullTimeEmployees"),
            "exchange": info.get("exchange"),
        }
        result = {"success": True, "summary": summary_data}
        if redis_client:
            await redis_client.setex(cache_key, CONFIG["cache"]["summary_ttl"], json.dumps(result))
            logger.debug(f"Cached summary data for {req.symbol}")
        return result
    except Exception as e:
        logger.error(f"Error fetching summary for {req.symbol}: {str(e)}")
        if "No fundamentals found" in str(e) or "No data found for symbol" in str(e):
            return {"success": False, "error": f"No summary data found for symbol {req.symbol}"}
        raise HTTPException(status_code=500, detail=str(e))
