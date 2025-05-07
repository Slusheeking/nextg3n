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
from monitor.logging_utils import get_logger

import yfinance as yf
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Response, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import redis.asyncio as aioredis
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
CONFIG = {
    "rate_limit_per_minute": 60,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "default_news_count": 20,
    "default_chart_interval": "1d",
    "default_chart_range": "1mo",
    "redis_host": os.getenv("REDIS_HOST", "localhost"),
    "redis_port": int(os.getenv("REDIS_PORT", "6379")),
    "redis_db": int(os.getenv("REDIS_DB", "0")), # Dedicated Redis DB for Yahoo Finance data
    "redis_password": os.getenv("REDIS_PASSWORD", None),
    "models": {
        "pegasus": {
            "model_name": "google/pegasus-xsum",
            "max_length": 150, # Max length of summary
            "min_length": 30,  # Min length of summary
            "num_beams": 4     # Beam search width
        },
        "rebel": {
            "model_name": "Babelscape/rebel-large", # Model for relation and event extraction
            "max_length": 256, # Max input sequence length for REBEL
            "num_beams": 4     # Beam search for generation
        },
        "financial_bert": {
            "model_name": "yiyanghkust/finbert-tone", # FinBERT for sentiment/tone
            "max_length": 512 # Max input sequence length
        }
    }
}

# Get logger from centralized logging system
logger = get_logger("yahoo_finance_server")
logger.info("Initializing Yahoo Finance server")

# --- Redis Client ---
redis_client: Optional[aioredis.Redis] = None

# Redis connection status tracking
redis_last_error_time = 0
redis_connection_attempts = 0
REDIS_RECONNECT_COOLDOWN = 30  # seconds between reconnection attempts

async def connect_redis(force_reconnect=False):
    """
    Establish connection to Redis cache server.
    Uses connection parameters from CONFIG dictionary.
    Sets global redis_client variable when successful.
    
    Args:
        force_reconnect: If True, forces reconnection even if client exists
    
    Returns:
        bool: True if connected successfully, False otherwise
    """
    global redis_client, redis_last_error_time, redis_connection_attempts
    
    # Check if we should attempt reconnection
    current_time = time.time()
    if not force_reconnect and redis_client is not None:
        return True  # Already connected
        
    # Rate-limit reconnection attempts if they're failing
    if (current_time - redis_last_error_time < REDIS_RECONNECT_COOLDOWN and
        redis_last_error_time > 0 and redis_connection_attempts > 3):
        logger.debug("Skipping Redis reconnect - in cooldown period")
        return False
    
    try:
        # Create new client if needed
        if redis_client is None or force_reconnect:
            if redis_client:
                try:
                    await redis_client.close()
                except Exception:
                    pass  # Ignore errors from closing
                    
            redis_client = aioredis.Redis(
                host=CONFIG["redis_host"],
                port=CONFIG["redis_port"],
                db=CONFIG["redis_db"],
                password=CONFIG["redis_password"],
                decode_responses=True,
                socket_timeout=5.0,  # Add timeout for production reliability
                socket_keepalive=True,  # Keep connection alive
                health_check_interval=30,  # Periodic health checks
                retry_on_timeout=True,
                max_connections=20,  # Limit connection pool size for stability
                retry=aioredis.retry.Retry(retries=2, backoff=1.5)  # Auto-retry with backoff
            )
        
        # Test connection with timeout
        await asyncio.wait_for(redis_client.ping(), timeout=2.0)
        
        # Reset error tracking on successful connection
        redis_connection_attempts = 0
        redis_last_error_time = 0
        logger.info(f"Connected to Redis for Yahoo Finance at {CONFIG['redis_host']}:{CONFIG['redis_port']} (DB: {CONFIG['redis_db']})")
        return True
        
    except (aioredis.RedisError, asyncio.TimeoutError, ConnectionError) as e:
        redis_connection_attempts += 1
        redis_last_error_time = time.time()
        # Only log full exception for first few attempts to avoid log spam
        if redis_connection_attempts <= 3:
            logger.exception("Failed to connect to Redis for Yahoo Finance")
        else:
            logger.warning(f"Redis connection attempt {redis_connection_attempts} failed: {str(e)}")
        redis_client = None  # Ensure it's None if connection fails
        return False
        
    except Exception as e:
        # For unexpected errors
        logger.exception("Unexpected error connecting to Redis")
        redis_client = None
        redis_last_error_time = time.time()
        return False

# --- AI/ML Models ---

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
        """
        Load the PEGASUS tokenizer and model from pretrained sources.
        Places model on CUDA if available, otherwise CPU.
        
        Raises:
            Exception: If model loading fails
        """
        if self.tokenizer is None or self.model is None:
            logger.info(f"Loading PEGASUS model: {self.model_name}")
            try:
                self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
                self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
                self.model.eval()  # Set to evaluation mode for inference
                logger.info(f"PEGASUS model {self.model_name} loaded successfully.")
            except Exception as e:
                logger.exception(f"Error loading PEGASUS model {self.model_name}")
                raise  # Re-raise to indicate failure

    def predict(self, text: str) -> str:
        """
        Generate a concise summary of the given text using the PEGASUS model.
        
        Args:
            text: Source text to summarize
            
        Returns:
            String containing the generated summary or empty string on error
        """
        if self.tokenizer is None or self.model is None:
            try:
                self.load_model()
            except Exception as e:
                logger.error(f"Failed to load PEGASUS model: {str(e)}")
                return ""  # Return empty string if model can't be loaded
        
        if not text:
            logger.warning("Empty text provided for summarization")
            return ""
            
        if len(text.strip()) < 20:
            logger.warning("Text too short for meaningful summarization")
            return text  # Return original if too short
            
        try:
            # Truncate extremely long texts to prevent memory issues
            if len(text) > 10000:  # Arbitrary limit to prevent memory issues
                logger.warning(f"Truncating overly long text ({len(text)} chars) for summarization")
                text = text[:10000] + "..."
                
            with torch.no_grad():  # Disable gradient calculation for inference
                # Handle CUDA out-of-memory by falling back to CPU if needed
                try:
                    inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(self.device)
                    summary_ids = self.model.generate(
                        inputs["input_ids"],
                        max_length=self.max_length,
                        min_length=self.min_length,
                        num_beams=self.num_beams,
                        length_penalty=2.0,
                        early_stopping=True,
                        no_repeat_ngram_size=3  # Avoid repetition in generated text
                    )
                except RuntimeError as cuda_err:
                    if "CUDA out of memory" in str(cuda_err):
                        logger.warning("CUDA out of memory, falling back to CPU for this prediction")
                        # Move model to CPU temporarily for this prediction
                        prev_device = self.device
                        self.device = "cpu"
                        self.model = self.model.to("cpu")
                        
                        # Retry on CPU
                        inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
                        summary_ids = self.model.generate(
                            inputs["input_ids"],
                            max_length=self.max_length,
                            min_length=self.min_length,
                            num_beams=2,  # Reduce beam size on CPU
                            length_penalty=2.0,
                            early_stopping=True,
                            no_repeat_ngram_size=3
                        )
                        
                        # Restore device setting
                        self.device = prev_device
                        if prev_device == "cuda":
                            self.model = self.model.to("cuda")
                    else:
                        raise
                        
                summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                # Validate summary quality
                if not summary or len(summary) < 10:
                    logger.warning("Generated summary too short or empty")
                    if len(text) < 200:  # For very short inputs
                        return text  # Return original for very short inputs
                    else:
                        return text[:197] + "..."  # Simple truncation as fallback
                        
                return summary
        except Exception as e:
            logger.exception(f"Error generating PEGASUS summary: {str(e)}")
            # Provide a degraded experience rather than empty response
            if len(text) > 200:
                return text[:197] + "..."  # Simple truncation as fallback
            return text  # Return original text as fallback

class REBELModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["model_name"]
        self.max_length = config["max_length"] # Max length for tokenized input
        self.gen_max_length = config.get("gen_max_length", 64) # Max length for generated output (triplets)
        self.num_beams = config["num_beams"]
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
                logger.info(f"REBEL model {self.model_name} loaded successfully.")
            except Exception as e:
                logger.exception(f"Error loading REBEL model {self.model_name}")
                raise

    def predict(self, text: str) -> List[Dict[str, str]]:
        if self.tokenizer is None or self.model is None:
            try:
                self.load_model()
            except Exception as e:
                logger.error(f"Failed to load REBEL model: {str(e)}")
                return []  # Return empty list if model can't be loaded
        
        # Early validation of input
        if not text or len(text.strip()) < 20:
            logger.warning("Text too short for meaningful event extraction")
            return []  # Nothing meaningful to extract
        
        # Limit very large inputs to prevent memory issues
        if len(text) > 10000:
            logger.warning(f"Truncating long text ({len(text)} chars) for event extraction")
            text = text[:10000] + "..."
        
        extracted_events = []
        try:
            # Handle CUDA out-of-memory by falling back to CPU if needed
            try:
                inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, return_tensors="pt").to(self.device)
                
                generated_tokens = self.model.generate(
                    **inputs,
                    max_length=self.gen_max_length, # Max length of the generated sequence of triplets
                    num_beams=self.num_beams,
                    num_return_sequences=1 # We want the single best sequence
                )
            except RuntimeError as cuda_err:
                if "CUDA out of memory" in str(cuda_err):
                    logger.warning("CUDA out of memory in REBEL, falling back to CPU for this prediction")
                    # Move model to CPU temporarily for this prediction
                    prev_device = self.device
                    self.device = "cpu"
                    self.model = self.model.to("cpu")
                    
                    # Retry on CPU with reduced beam search
                    inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, return_tensors="pt")
                    generated_tokens = self.model.generate(
                        **inputs,
                        max_length=self.gen_max_length,
                        num_beams=2, # Reduced beam search for CPU
                        num_return_sequences=1
                    )
                    
                    # Restore device setting
                    self.device = prev_device
                    if prev_device == "cuda":
                        self.model = self.model.to("cuda")
                else:
                    raise
                    
            decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False) # Keep special tokens for parsing

            # Robust REBEL output parser: handles all special tokens and extracts triplets accurately.
            tokens = decoded_preds[0].split()
            extracted_events = []
            current_triplet = {"subject": "", "relation": "", "object": ""}
            state = None  # None, "relation", "subject", "object"
            for token in tokens:
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
                else:
                    # If state is None, ignore token (should not happen in well-formed output)
                    continue
            # Add last triplet if present
            if any(current_triplet.values()):
                extracted_events.append({
                    "type": current_triplet["relation"].strip() or "unknown",
                    "subject": current_triplet["subject"].strip(),
                    "object": current_triplet["object"].strip(),
                    "description": f"{current_triplet['subject'].strip()} {current_triplet['relation'].strip()} {current_triplet['object'].strip()}"
                })
            # Filter out incomplete or generic events
            extracted_events = [e for e in extracted_events if e["subject"] and e["object"] and e["type"] != "unknown"]
            return extracted_events
        except Exception as e:
            logger.exception(f"Error extracting REBEL events for text '{text[:100]}...'")
            return []  # Return empty list on error

    def extract_events(self, text: str) -> List[Dict[str, str]]: # Kept for consistency if called elsewhere
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
                logger.info(f"FinancialBERT model {self.model_name} loaded successfully.")
            except Exception as e:
                logger.exception(f"Error loading FinancialBERT model {self.model_name}")
                raise

    def predict(self, text: str) -> Dict[str, float]:
        if self.tokenizer is None or self.model is None:
            try:
                self.load_model()
            except Exception as e:
                logger.error(f"Failed to load FinancialBERT model: {str(e)}")
                return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}  # Balanced fallback
        
        # Skip empty or very short texts
        if not text or len(text.strip()) < 10:
            logger.warning("Text too short for meaningful sentiment analysis")
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}  # Balanced fallback
            
        # Truncate extremely long texts to prevent memory issues
        if len(text) > 10000:
            logger.warning(f"Truncating overly long text ({len(text)} chars) for sentiment analysis")
            text = text[:10000] + "..."
            
        try:
            # Handle CUDA out-of-memory by falling back to CPU if needed
            try:
                inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                     outputs = self.model(**inputs)
            except RuntimeError as cuda_err:
                if "CUDA out of memory" in str(cuda_err):
                    logger.warning("CUDA out of memory in FinancialBERT, falling back to CPU for this prediction")
                    # Move model to CPU temporarily for this prediction
                    prev_device = self.device
                    self.device = "cpu"
                    self.model = self.model.to("cpu")
                    
                    # Retry on CPU
                    inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding=True, truncation=True)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # Restore device setting
                    self.device = prev_device
                    if prev_device == "cuda":
                        self.model = self.model.to("cuda")
                else:
                    raise
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Robust FinBERT-tone label mapping with error handling
            try:
                scores = probs[0].tolist()
                sentiment = {}
                if hasattr(self.model.config, "id2label"):
                    # Use id2label for mapping, lowercased for consistency
                    labels = [self.model.config.id2label[i].lower() for i in range(len(scores))]
                    for label, score in zip(labels, scores):
                        sentiment[label] = float(score)
                    # Ensure all keys present
                    for key in ["positive", "negative", "neutral"]:
                        sentiment.setdefault(key, 0.0)
                else:
                    # Manual mapping as fallback
                    label2id = self.model.config.label2id if hasattr(self.model.config, "label2id") else {}
                    positive_idx = label2id.get("Positive", label2id.get("positive", 0))
                    negative_idx = label2id.get("Negative", label2id.get("negative", 1))
                    neutral_idx = label2id.get("Neutral", label2id.get("neutral", 2))
                    
                    # Add safe bounds checking
                    if positive_idx < len(scores) and negative_idx < len(scores) and neutral_idx < len(scores):
                        sentiment = {
                            "positive": scores[positive_idx],
                            "negative": scores[negative_idx],
                            "neutral": scores[neutral_idx]
                        }
                    else:
                        logger.warning("Invalid label indices for FinBERT, using balanced fallback")
                        sentiment = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
                        
                # Validate sentiment scores
                total = sum(sentiment.values())
                if abs(total - 1.0) > 0.1:  # If values don't sum close to 1.0
                    logger.warning(f"Sentiment scores don't sum to 1.0 (sum={total}), normalizing")
                    if total > 0:
                        sentiment = {k: v/total for k, v in sentiment.items()}  # Normalize
                
                return sentiment
            except Exception as mapping_err:
                logger.warning(f"Error in sentiment mapping: {str(mapping_err)}")
                return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}  # Fallback
        except Exception as e:
            logger.exception(f"Error classifying FinancialBERT rating for '{text[:100]}...'")
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}  # Fallback


# Initialize models
pegasus_model = PEGASUSModel(CONFIG["models"]["pegasus"])
rebel_model = REBELModel(CONFIG["models"]["rebel"])
financial_bert_model = FinancialBERTModel(CONFIG["models"]["financial_bert"])

# --- News Analysis Component ---

class NewsAnalysisComponent:
    def __init__(self, pegasus_model: PEGASUSModel, rebel_model: REBELModel, financial_bert_model: FinancialBERTModel):
        self.pegasus_model = pegasus_model
        self.rebel_model = rebel_model
        self.financial_bert_model = financial_bert_model
        self.logger = get_logger("news_analysis_component") # Changed logger name

    async def analyze_news(self, articles: List[Dict[str, Any]], symbols: List[str]) -> Dict[str, Any]:
        self.logger.info(f"Analyzing {len(articles)} news articles for symbols: {symbols}")
        try:
            relevant_articles_data = []
            for article in articles:
                # Check if any of the target symbols are mentioned in the article's tickers
                article_tickers_lower = [t.lower() for t in article.get("tickers", [])]
                target_symbols_lower = [s.lower() for s in symbols]
                if any(s in article_tickers_lower for s in target_symbols_lower):
                    analysis = await self.analyze_article(article, symbols) # Pass symbols parameter
                    relevant_articles_data.append(analysis)
            
            self.logger.info(f"Found and analyzed {len(relevant_articles_data)} relevant articles.")
            aggregated = self.aggregate_analysis(relevant_articles_data, symbols)
            return {"success": True, "articles": relevant_articles_data, "aggregated": aggregated}
        except Exception as e:
            self.logger.exception("Error in NewsAnalysisComponent.analyze_news")
            return {"success": False, "error": str(e)}

    async def analyze_article(self, article: Dict[str, Any], target_symbols: List[str] = None) -> Dict[str, Any]:
        try:
            if target_symbols is None: target_symbols = []
            title = article.get("title", "")
            summary = article.get("summary", "")
            text_content = f"{title}. {summary}" if summary else title # Use title if summary is empty

            # Ensure models are loaded (they load on first predict if not already)
            generated_summary = self.pegasus_model.predict(text_content)
            events = self.rebel_model.extract_events(text_content)
            sentiment = self.financial_bert_model.predict(text_content)
            relevance_score = self.calculate_relevance_score(article, target_symbols) # Pass target_symbols
            impact_score = self.calculate_impact_score(events, sentiment)
            
            return {
                "article_id": article.get("id", article.get("uuid")), # Use uuid as fallback for id
                "title": title,
                "summary": summary, # Original summary
                "generated_summary": generated_summary, # Model-generated summary
                "events": events,
                "sentiment": sentiment,
                "relevance_score": relevance_score,
                "impact_score": impact_score,
                "tickers": article.get("tickers", []),
                "published_at": article.get("published_at", article.get("providerPublishTime")) # Use providerPublishTime as fallback
            }
        except Exception as e:
            self.logger.exception(f"Error analyzing article (ID: {article.get('id', 'N/A')})")
            return {"article_id": article.get("id", "N/A"), "title": article.get("title", ""), "error": str(e)}

    def calculate_relevance_score(self, article: Dict[str, Any], target_symbols: List[str] = None) -> float:
        score = 0.0
        if target_symbols is None: target_symbols = []

        title = article.get("title", "").lower()
        summary = article.get("summary", "").lower()
        content = title + " " + summary
        article_tickers_lower = [t.lower() for t in article.get("tickers", [])]
        target_symbols_lower = [s.lower() for s in target_symbols]

        symbol_match_score = 0.0
        for target_sym_l in target_symbols_lower:
            if target_sym_l in title: symbol_match_score += 0.35 # Higher weight for title
            if target_sym_l in summary: symbol_match_score += 0.25
            if target_sym_l in article_tickers_lower: symbol_match_score += 0.4 # Highest if explicitly tagged
        score += min(symbol_match_score, 0.7) 

        financial_keywords = ["earnings", "profit", "loss", "revenue", "guidance", "fda", "approval", "partnership", "acquisition", "rating", "upgrade", "downgrade", "dividend", "buyback", "ipo", "sec filing", "investigation", "lawsuit", "debt", "equity", "analyst", "target price"]
        keyword_hits = sum(1 for keyword in financial_keywords if keyword in content)
        score += min(keyword_hits * 0.03, 0.2) # Reduced weight per keyword, max 0.2

        published_ts = article.get("published_at", article.get("providerPublishTime"))
        if published_ts:
            try:
                # Handle both string ISO dates and Unix timestamps
                if isinstance(published_ts, (int, float)):
                    published_dt = datetime.fromtimestamp(published_ts)
                elif isinstance(published_ts, str):
                    published_dt = datetime.fromisoformat(published_ts.replace("Z", "+00:00"))
                else: # If it's already a datetime object (less likely from raw JSON)
                    published_dt = published_ts
                
                # Ensure published_dt is offset-aware for comparison with offset-aware datetime.now()
                if published_dt.tzinfo is None:
                    published_dt = published_dt.replace(tzinfo=datetime.timezone.utc) # Assume UTC if naive

                age_days = (datetime.now(datetime.timezone.utc) - published_dt).total_seconds() / (24 * 60 * 60)

                if age_days <= 1: score += 0.15
                elif age_days <= 3: score += 0.10
                elif age_days <= 7: score += 0.05
            except Exception as e:
                self.logger.warning(f"Could not parse date for relevance scoring ('{published_ts}'): {e}", exc_info=True)
        
        if len(summary) > 50 : score += 0.05
        if len(summary) == 0 and len(title) > 30: score -= 0.05 # Penalize if only title and it's short
            
        return round(min(score, 1.0), 3)

    def calculate_impact_score(self, events: List[Dict[str, str]], sentiment: Dict[str, float]) -> float:
        score = 0.0
        # Event impact based on type and count
        event_impact_weights = {"acquisition": 0.3, "earnings": 0.25, "management_change": 0.2, "product_launch": 0.15, "fda": 0.3, "approval":0.3, "lawsuit":0.25, "investigation":0.25}
        max_event_score = 0.5
        current_event_score = 0.0
        if events:
            for event in events:
                current_event_score += event_impact_weights.get(event.get("type","").lower(), 0.05) # Default small score for unknown events
        score += min(current_event_score, max_event_score)

        # Sentiment impact (strong sentiment is more impactful)
        # Consider the magnitude of positive/negative sentiment
        positive_score = sentiment.get("positive", 0.0)
        negative_score = sentiment.get("negative", 0.0)
        
        if positive_score > 0.7: score += 0.3 * positive_score # Weighted by score
        elif negative_score > 0.7: score += 0.3 * negative_score # Weighted by score
        elif positive_score > 0.5: score += 0.15 * positive_score
        elif negative_score > 0.5: score += 0.15 * negative_score
        
        return round(min(score, 1.0), 3)

    def aggregate_analysis(self, articles: List[Dict[str, Any]], symbols: List[str]) -> Dict[str, Any]:
        """
        Aggregate analysis results from multiple articles.
        
        Args:
            articles: List of analyzed article data
            symbols: List of stock symbols to track
            
        Returns:
            Dict with aggregated sentiment, events, impact and relevance metrics
        """
        try:
            aggregated = {
                "sentiment": {"overall": "neutral", "positive": 0.0, "negative": 0.0, "neutral": 0.0, "avg_score": 0.0},
                "events": [],
                "impact": {"overall": "low", "avg_score": 0.0},
                "relevance": {"avg_score": 0.0},
                "by_symbol": {s: {"sentiment": {"overall": "neutral", "positive": 0.0, "negative": 0.0, "neutral": 0.0, "avg_score": 0.0}, 
                                   "events": [], "impact": {"overall": "low", "avg_score": 0.0}, "relevance": {"avg_score": 0.0}, "article_count": 0} for s in symbols}
            }
            
            # To count event occurrences
            all_events_map = {}  # type: Dict[str, int]
            total_sentiment_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            total_impact_score = 0.0
            total_relevance_score = 0.0
            article_count_total = 0

            for article in articles:
                if article.get("error"): continue # Skip errored articles
                article_count_total +=1
                
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
                        s_data["impact"]["avg_score"] += article.get("impact_score", 0) # Sum first, then average
                        s_data["relevance"]["avg_score"] += article.get("relevance_score", 0)
                        for event in article.get("events", []):
                             s_data["events"].append(event) # Can deduplicate later if needed

            if article_count_total > 0:
                aggregated["sentiment"]["positive"] = total_sentiment_scores["positive"] / article_count_total
                aggregated["sentiment"]["negative"] = total_sentiment_scores["negative"] / article_count_total
                aggregated["sentiment"]["neutral"] = total_sentiment_scores["neutral"] / article_count_total
                aggregated["sentiment"]["avg_score"] = (total_sentiment_scores["positive"] + total_sentiment_scores["negative"] + total_sentiment_scores["neutral"]) / 3 if article_count_total > 0 else 0.0

                # Determine overall sentiment label
                if aggregated["sentiment"]["positive"] > 0.4:
                    aggregated["sentiment"]["overall"] = "positive"
                elif aggregated["sentiment"]["negative"] > 0.4:
                    aggregated["sentiment"]["overall"] = "negative"
                else:
                    aggregated["sentiment"]["overall"] = "neutral"

                aggregated["impact"]["avg_score"] = total_impact_score / article_count_total if article_count_total > 0 else 0.0
                aggregated["relevance"]["avg_score"] = total_relevance_score / article_count_total if article_count_total > 0 else 0.0

                # Determine overall impact label
                if aggregated["impact"]["avg_score"] > 0.6:
                    aggregated["impact"]["overall"] = "high"
                elif aggregated["impact"]["avg_score"] > 0.3:
                    aggregated["impact"]["overall"] = "medium"
                else:
                    aggregated["impact"]["overall"] = "low"


            aggregated["events"] = [{"event": k, "count": v} for k,v in sorted(all_events_map.items(), key=lambda item: item[1], reverse=True)][:10] # Top 10 events

            for sym, data in aggregated["by_symbol"].items():
                if data["article_count"] > 0:
                    data["sentiment"]["positive"] /= data["article_count"]
                    data["sentiment"]["negative"] /= data["article_count"]
                    data["sentiment"]["neutral"] /= data["article_count"]
                    data["impact"]["avg_score"] /= data["article_count"]
                    data["relevance"]["avg_score"] /= data["article_count"]
                    # Determine per-symbol sentiment and impact labels
                    if data["sentiment"]["positive"] > 0.4:
                        data["sentiment"]["overall"] = "positive"
                    elif data["sentiment"]["negative"] > 0.4:
                        data["sentiment"]["overall"] = "negative"
                    else:
                        data["sentiment"]["overall"] = "neutral"

                    if data["impact"]["avg_score"] > 0.6:
                        data["impact"]["overall"] = "high"
                    elif data["impact"]["avg_score"] > 0.3:
                        data["impact"]["overall"] = "medium"
                    else:
                        data["impact"]["overall"] = "low"
                    
                    # Deduplicate per-symbol events if necessary
                    unique_sym_events = []
                    seen_sym_event_descs = set()
                    for ev in data["events"]:
                        ev_desc = f"{ev.get('type')}:{ev.get('subject','')}-{ev.get('object','')}"
                        if ev_desc not in seen_sym_event_descs:
                            unique_sym_events.append(ev)
                            seen_sym_event_descs.add(ev_desc)
                    data["events"] = unique_sym_events[:5]  # Top 5 for symbol

            return aggregated
        except Exception as e:
            self.logger.exception("Error in NewsAnalysisComponent.aggregate_analysis") # Changed to logger.exception for full traceback
            # Return a default structure on error
            return {"success": False, "error": str(e), "aggregated": {}}


# Initialize news analysis component
news_analysis_component = NewsAnalysisComponent(pegasus_model, rebel_model, financial_bert_model) # Renamed instance

# --- Redis Integration for News Data ---

async def store_news_analysis(analysis: Dict[str, Any], symbols: List[str]) -> bool:
    """
    Store news analysis results in Redis cache with appropriate TTL.
    
    Args:
        analysis: The complete news analysis results
        symbols: List of stock symbols the analysis relates to
        
    Returns:
        Boolean indicating if storage was successful
    """
    if not redis_client:
        logger.warning("Redis client not available, skipping news analysis storage")
        return False
    try:
        date_str = datetime.now().strftime("%Y-%m-%d")
        # Sanitize symbols for key, e.g. AAPL_MSFT, max 2-3 symbols for key length
        key_symbols = "_".join(sorted(list(set(symbols))))[:50] # Ensure reasonable key length
        key = f"yahoo_news_analysis:{date_str}:{key_symbols}"
        
        # Store with different TTL based on relevance - more relevant data kept longer
        avg_relevance = analysis.get("aggregated", {}).get("relevance", {}).get("avg_score", 0.0)
        ttl = timedelta(days=7)  # Default TTL
        if avg_relevance > 0.7:
            ttl = timedelta(days=14)  # Keep highly relevant analysis longer
        
        # Compress large analysis objects for efficient storage
        json_data = json.dumps(analysis)
        if len(json_data) > 10000:  # If data is large
            logger.info(f"Large analysis data ({len(json_data)} bytes) for {key_symbols}, consider optimizing")
            
        await redis_client.set(key, json_data, ex=ttl)
        logger.info(f"Stored news analysis in Redis with key {key}, TTL: {ttl}")

        # Store alerts for market-moving events based on aggregated analysis
        agg_impact_score = analysis.get("aggregated", {}).get("impact", {}).get("avg_score", 0.0)
        if agg_impact_score > 0.6: # Threshold for market-moving
            alert_key = f"yahoo_news_alert:{date_str}:{key_symbols}"
            alert_data = {
                "timestamp": datetime.now().isoformat(),
                "symbols": symbols,
                "aggregated_impact_score": agg_impact_score,
                "top_articles_sample": [
                    {"title": art.get("title"), "impact": art.get("impact_score"), "relevance": art.get("relevance_score")}
                    for art in analysis.get("articles", [])[:3] # Sample of top 3 articles
                ]
            }
            await redis_client.set(alert_key, json.dumps(alert_data), ex=timedelta(days=3))
            logger.info(f"Stored market-moving news alert in Redis with key {alert_key}")
        return True
    except Exception as e:
        logger.exception("Error storing news analysis/alerts in Redis")
        return False

async def get_stored_news_analysis_from_redis(symbols: List[str], date: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve previously stored news analysis from Redis cache.
    
    Args:
        symbols: List of stock symbols to retrieve analysis for
        date: Optional specific date (YYYY-MM-DD format), defaults to today
        
    Returns:
        Dictionary with analysis results or error message
        
    Raises:
        HTTPException: If Redis is unavailable or retrieval fails
    """
    if not redis_client:
        logger.warning("Redis client not available for get_stored_news_analysis.")
        raise HTTPException(status_code=503, detail="Redis cache service unavailable.")
        
    try:
        # Validate date format if provided
        if date:
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
                
        date_str = date if date else datetime.now().strftime("%Y-%m-%d")
        key_symbols = "_".join(sorted(list(set(symbols))))[:50]
        key = f"yahoo_news_analysis:{date_str}:{key_symbols}"
        
        # Try to get data with timeout for production reliability
        analysis_json = await asyncio.wait_for(redis_client.get(key), timeout=2.0)
        
        if analysis_json:
            logger.info(f"Retrieved stored news analysis from Redis for key {key}")
            try:
                return {"success": True, "analysis": json.loads(analysis_json)}
            except json.JSONDecodeError:
                logger.error(f"Corrupted JSON data in Redis for key {key}")
                return {"success": False, "message": "Corrupted data found"}
        else:
            logger.info(f"No stored news analysis found in Redis for key {key}")
            return {"success": False, "message": "No analysis found for the given symbols and date."}
    except asyncio.TimeoutError:
        logger.error("Redis operation timed out when retrieving news analysis")
        raise HTTPException(status_code=504, detail="Cache service timed out")
    except Exception as e:
        logger.exception("Error retrieving stored news analysis from Redis")
        raise HTTPException(status_code=500, detail=str(e))


# --- FastAPI Models --- (NewsRequest, etc. are fine)
class NewsRequest(BaseModel):
    symbols: List[str] = []
    count: int = CONFIG["default_news_count"]

class NewsAnalysisRequest(BaseModel):
    symbols: List[str]
    count: int = CONFIG["default_news_count"]

class EventExtractionRequest(BaseModel):
    symbols: List[str]
    count: int = CONFIG["default_news_count"]

class QuotesRequest(BaseModel):
    symbols: List[str]
class StoredNewsAnalysisRequest(BaseModel):
    symbols: List[str]
    date: Optional[str] = None # YYYY-MM-DD format

# --- FastAPI Server ---
# --- FastAPI Setup ---
app = FastAPI(
    title="Yahoo Finance MCP Server for LLM",
    description="Production-ready server providing financial news, quotes, chart data and sentiment analysis",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for production use
app.add_middleware(
    CORSMiddleware,
    # IMPORTANT: Set allowed origins for production!
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"],  # Replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Server active requests tracker for graceful shutdown
active_requests = 0
shutdown_event = asyncio.Event()

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track active requests for graceful shutdown"""
    global active_requests
    active_requests += 1
    try:
        return await call_next(request)
    finally:
        active_requests -= 1

@app.on_event("startup")
async def startup_event():
    """Initialize resources when FastAPI server starts"""
    global redis_client
    retry_count = 0
    max_retries = 5
    
    logger.info("Yahoo Finance server starting up")
    
    # Connect to Redis with retries for production reliability
    while retry_count < max_retries:
        try:
            await connect_redis()
            if redis_client:
                logger.info("Redis connection established successfully")
                break
            retry_count += 1
            wait_time = min(2 ** retry_count, 30)  # Exponential backoff with max 30 seconds
            logger.warning(f"Redis connection failed. Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})")
            await asyncio.sleep(wait_time)
        except Exception as e:
            logger.exception(f"Error during Redis connection attempt {retry_count}/{max_retries}")
            retry_count += 1
            if retry_count >= max_retries:
                logger.error("Maximum Redis connection attempts reached. Server will start in degraded mode.")
            else:
                wait_time = min(2 ** retry_count, 30)
                await asyncio.sleep(wait_time)
    
    # Perform health check
    try:
        server_health = await check_server_health()
        logger.info(f"Server health check on startup: {server_health}")
    except Exception as e:
        logger.exception("Error performing initial health check")
    
    # Load ML models in background to avoid blocking startup
    asyncio.create_task(load_models_background())
    
    # Reset shutdown event if server is restarting
    shutdown_event.clear()
    
    logger.info("Yahoo Finance server startup complete.")

async def load_models_background():
    """Load ML models in background to avoid blocking server startup"""
    logger.info("Loading ML models in background")
    try:
        # Load models in sequence to manage memory
        pegasus_model.load_model()
        rebel_model.load_model()
        financial_bert_model.load_model()
        logger.info("All ML models loaded successfully")
    except Exception as e:
        logger.exception("Error loading ML models in background")

async def check_server_health() -> Dict[str, Any]:
    """Check all system components are functioning properly"""
    health = {
        "status": "healthy",
        "components": {
            "redis": False,
            "yfinance": False
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Check Redis
    try:
        if redis_client:
            await asyncio.wait_for(redis_client.ping(), timeout=2.0)
            health["components"]["redis"] = True
        else:
            logger.warning("Redis client not available during health check")
            health["status"] = "degraded"
    except (asyncio.TimeoutError, Exception) as e:
        health["status"] = "degraded"
        logger.warning(f"Redis health check failed: {str(e)}")
    
    # Check YFinance with a simple query (with timeout safeguard)
    try:
        async def check_yfinance():
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            return info and "symbol" in info
            
        yfinance_result = await asyncio.wait_for(
            asyncio.to_thread(check_yfinance),
            timeout=5.0
        )
        if yfinance_result:
            health["components"]["yfinance"] = True
        else:
            health["status"] = "degraded"
            logger.warning("YFinance health check returned incomplete data")
    except asyncio.TimeoutError:
        health["status"] = "degraded"
        logger.warning("YFinance health check timed out after 5 seconds")
    except Exception as e:
        health["status"] = "degraded"
        logger.warning(f"YFinance health check failed: {str(e)}")
    
    return health

@app.on_event("shutdown")
async def shutdown_event():
    """
    Graceful shutdown with resource cleanup
    
    Allows pending requests to complete (with timeout), then closes resources
    and performs cleanup operations.
    """
    logger.info("Yahoo Finance server shutting down - waiting for pending requests")
    
    # Signal shutdown to prevent new long-running operations
    shutdown_event.set()
    
    # Wait for active requests to complete with timeout
    shutdown_timeout = 30  # seconds
    shutdown_start = time.time()
    
    while active_requests > 0:
        if time.time() - shutdown_start > shutdown_timeout:
            logger.warning(f"Shutdown timeout exceeded. Forcing shutdown with {active_requests} pending requests.")
            break
        logger.info(f"Waiting for {active_requests} active requests to complete...")
        await asyncio.sleep(1)
    
    # Clean up Redis connections
    global redis_client
    if redis_client:
        try:
            await redis_client.close()
            logger.info("Redis connection closed successfully")
            redis_client = None
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {str(e)}")
    
    # Release ML model resources
    try:
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA memory cache cleared")
        
        # Set models to None to help garbage collection
        pegasus_model.model = None
        pegasus_model.tokenizer = None
        rebel_model.model = None
        rebel_model.tokenizer = None
        financial_bert_model.model = None
        financial_bert_model.tokenizer = None
        logger.info("ML model references cleared")
    except Exception as e:
        logger.warning(f"Error clearing model resources: {str(e)}")
    
    logger.info("Yahoo Finance server shutdown complete")

@app.get("/server_info")
async def get_server_info():
    """
    Return information about this server and its capabilities.
    """
    return {
        "name": "yahoo_finance",
        "version": "1.1.0",
        "description": "Production MCP Server for Yahoo Finance News and Market Data Integration with Advanced Analysis",
        "tools": [
            "fetch_news", "fetch_quotes", "fetch_chart_data", "fetch_summary",
            "analyze_news", "extract_events", "get_stored_news_analysis"
        ],
        "models": [CONFIG["models"]["pegasus"]["model_name"], CONFIG["models"]["rebel"]["model_name"], CONFIG["models"]["financial_bert"]["model_name"]],
        "config": {k: v for k, v in CONFIG.items() if k not in ["redis_password"]} # Exclude sensitive
    }

@app.get("/health")
async def health_check(response: Response):
    """
    Health check endpoint for monitoring system components.
    Returns status of system components for health monitoring.
    """
    health_status = await check_server_health()
    
    if health_status["status"] != "healthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
    return health_status

# Endpoints: /fetch_news, /fetch_quotes, /fetch_chart_data, /fetch_summary are defined below.

@app.post("/fetch_news")
async def fetch_news(req: NewsRequest):
    """
    Fetch news articles for the requested symbols.
    
    Args:
        req: NewsRequest with symbols and count parameters
        
    Returns:
        List of news articles with metadata
        
    Raises:
        HTTPException: On fetch failure or timeout
    """
    try:
        result = await api_fetch_news(req)
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.get("error", "Failed to fetch news")
            )
        return result
    except Exception as e:
        logger.exception("Error in /fetch_news endpoint")
        raise HTTPException(status_code=500, detail=str(e))

async def api_fetch_news(req: NewsRequest) -> Dict[str, Any]:
    """
    Fetch news articles for the given symbols using yfinance.
    """
    try:
        symbols_list = req.symbols if req.symbols else []
        count = req.count if req.count else CONFIG["default_news_count"]
        
        all_articles = []
        # If multiple symbols, fetch for each. yfinance news() is per-ticker.
        # If no symbols, fetch general market news (e.g., S&P 500)
        tickers_to_fetch = symbols_list if symbols_list else ["^GSPC"]

        for ticker_str in tickers_to_fetch:
            logger.info(f"Fetching news for ticker: {ticker_str}")
            try:
                ticker_obj = yf.Ticker(ticker_str)
                news_items = ticker_obj.news
                
                if not news_items:
                    logger.warning(f"No news found for {ticker_str}")
                    continue

                for article in news_items[:count if len(tickers_to_fetch) == 1 else CONFIG["default_news_count"] // len(tickers_to_fetch) or 1]: # Distribute count for multiple tickers
                    processed_article = {
                        "id": str(article.get("uuid", "")),
                        "title": article.get("title", ""),
                        "publisher": article.get("publisher", ""),
                        "link": article.get("link", ""),
                        "published_at": article.get("providerPublishTime"), # Keep as timestamp or convert to ISO
                        "summary": article.get("summary", ""), # yfinance often lacks summary
                        "tickers": article.get("relatedTickers", [ticker_str]), # Ensure current ticker is included
                        "type": article.get("type", "")
                    }
                    all_articles.append(processed_article)
            except Exception as e_yf:
                logger.error(f"Error fetching yfinance news for {ticker_str}: {str(e_yf)}")
        
        # Deduplicate articles by link or title if fetching from multiple sources/tickers
        unique_articles = []
        seen_links = set()
        for art in all_articles:
            if art["link"] not in seen_links:
                unique_articles.append(art)
                seen_links.add(art["link"])
        
        logger.info(f"Successfully fetched {len(unique_articles)} unique news articles.")
        return {"success": True, "articles": unique_articles[:count]} # Apply overall count limit
            
    except Exception as e:
        logger.exception("Error in api_fetch_news")
        return {"success": False, "error": str(e)}


@app.post("/analyze_news")
async def api_analyze_news(req: NewsAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze financial news articles for the requested symbols.
    
    Fetches news, performs sentiment analysis, extracts events, and returns
    both individual article analysis and aggregated insights.
    
    Args:
        req: NewsAnalysisRequest with symbols and count parameters
        background_tasks: FastAPI BackgroundTasks for async operations after response
        
    Returns:
        Analysis results including sentiment scores and extracted events
        
    Raises:
        HTTPException: On news fetch or processing failure
    """
    try:
        # First check if we have cached results
        cached_analysis = await get_stored_news_analysis_from_redis(req.symbols)
        if cached_analysis.get("success") == True:
            logger.info(f"Using cached news analysis for {req.symbols}")
            return cached_analysis["analysis"]
        
        # Otherwise fetch and analyze new data
        news_response = await api_fetch_news(NewsRequest(symbols=req.symbols, count=req.count))
        if not news_response.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=news_response.get("error", "Failed to fetch news for analysis")
            )
        
        articles = news_response.get("articles", [])
        if not articles:
            return {"success": True, "message": "No articles found to analyze.", "articles": [], "aggregated": {}}

        analysis_result = await news_analysis_component.analyze_news(articles, req.symbols)
        
        # Store in cache for future requests using background task
        if analysis_result.get("success"):
            background_tasks.add_task(store_news_analysis, analysis_result, req.symbols)
        
        return analysis_result
    except Exception as e:
        logger.exception("Error in /analyze_news endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_events")
async def api_extract_events(req: EventExtractionRequest):
    """
    Extract financial events and relationships from news articles.
    
    Uses REBEL NLP model to identify relationships and events in financial news
    such as acquisitions, product launches, management changes, etc.
    
    Args:
        req: EventExtractionRequest with symbols and count parameters
        
    Returns:
        Dictionary with extracted events grouped by article
        
    Raises:
        HTTPException: On news fetch or extraction failure
    """
    try:
        logger.info(f"Processing event extraction request for symbols: {req.symbols}")
        news_response = await api_fetch_news(NewsRequest(symbols=req.symbols, count=req.count))
        if not news_response.get("success", False):
            logger.error(f"Failed to fetch news for event extraction: {news_response.get('error')}")
            raise HTTPException(status_code=503, detail=news_response.get("error", "Failed to fetch news for event extraction"))

        articles = news_response.get("articles", [])
        if not articles:
            logger.info(f"No articles found for event extraction for symbols: {req.symbols}")
            return {"success": True, "message": "No articles found for event extraction.", "events": []}

        logger.info(f"Extracting events from {len(articles)} articles")
        extracted_events_list = []
        for i, article in enumerate(articles):
            article_id = article.get("id", f"unknown-{i}")
            text_content = f"{article.get('title', '')}. {article.get('summary', '')}"
            try:
                events = rebel_model.extract_events(text_content)
                if events: # Only add if events were found
                    extracted_events_list.append({
                        "article_id": article_id,
                        "title": article.get("title"),
                        "link": article.get("link"),
                        "events": events
                    })
                    logger.debug(f"Extracted {len(events)} events from article {article_id}")
                else:
                    logger.debug(f"No events found in article {article_id}")
            except Exception as event_error:
                logger.warning(f"Error extracting events from article {article_id}: {str(event_error)}")
                # Continue processing other articles
        
        logger.info(f"Successfully extracted events from {len(extracted_events_list)} articles out of {len(articles)} total")
        return {"success": True, "extracted_data": extracted_events_list}
    except Exception as e:
        logger.exception(f"Error in /extract_events endpoint for symbols {req.symbols}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_stored_news_analysis")
async def api_get_stored_news_analysis(req: StoredNewsAnalysisRequest):
    """
    Retrieve previously stored news analysis from Redis cache.
    
    This endpoint allows retrieving pre-computed news analysis results rather
    than generating them on demand, improving response time and reducing load.
    
    Args:
        req: StoredNewsAnalysisRequest with symbols and optional date parameters
        
    Returns:
        Dictionary with stored analysis results if available
        
    Raises:
        HTTPException: If Redis is unavailable or retrieval fails
    """
    logger.info(f"Retrieving stored news analysis for symbols {req.symbols}, date: {req.date or 'today'}")
    try:
        return await get_stored_news_analysis_from_redis(req.symbols, req.date)
    except HTTPException as http_ex:
        # Re-raise HTTP exceptions with the same status code
        logger.warning(f"HTTP error in stored news retrieval: {http_ex.detail}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error retrieving stored news analysis for {req.symbols}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve stored analysis: {str(e)}"
        )

# Endpoint implementations for financial data access
@app.post("/fetch_quotes")
async def api_fetch_quotes(req: QuotesRequest):
    """
    Fetch current stock quotes for requested symbols.
    
    Args:
        req: QuotesRequest with list of stock symbols
        
    Returns:
        Dictionary with quote data for each requested symbol
        
    Raises:
        HTTPException: On fetch failure or timeout
    """
    try:
        # Validate symbols
        if not req.symbols:
            raise HTTPException(status_code=400, detail="No symbols provided")
        if len(req.symbols) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 symbols allowed per request")
        
        # Create cache key for batch lookup
        cache_key = f"yahoo_quotes:{datetime.now().strftime('%Y-%m-%d')}:{','.join(sorted(req.symbols))}"
        
        # Try cache first
        if redis_client:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Returning cached quotes for {len(req.symbols)} symbols")
                return json.loads(cached_data)
        
        # Fetch from API if not in cache
        data = {}
        for symbol in req.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")  # Current day's quote
                if not hist.empty:
                    quote_data = hist.iloc[-1].to_dict()
                    # Robust conversion of numpy/pandas values to Python native types
                    processed_data = {}
                    for k, v in quote_data.items():
                        try:
                            if hasattr(v, 'item'):  # Convert numpy values
                                processed_data[k] = v.item()
                            elif hasattr(v, 'timestamp'):  # Convert pandas timestamps
                                processed_data[k] = v.isoformat()
                            elif v is None or isinstance(v, (int, float, str, bool)):
                                processed_data[k] = v
                            else:
                                processed_data[k] = str(v)  # Fallback to string for unknown types
                        except Exception as conv_error:
                            logger.warning(f"Error converting value for {k} in {symbol}: {str(conv_error)}")
                            processed_data[k] = str(v)  # Use string as fallback
                    data[symbol] = processed_data
                else:
                    logger.info(f"No quote data found for {symbol}")
                    data[symbol] = {"error": "No data found"}
            except Exception as symbol_error:
                logger.warning(f"Error fetching quote for {symbol}: {str(symbol_error)}")
                data[symbol] = {"error": str(symbol_error)}
        
        result = {"success": True, "quotes": data, "timestamp": datetime.now().isoformat()}
        
        # Store in cache with short TTL since quotes change frequently
        if redis_client and data:
            await redis_client.set(cache_key, json.dumps(result), ex=300)  # 5 minute cache
            
        return result
    except Exception as e:
        logger.exception("Error fetching quotes")
        raise HTTPException(status_code=500, detail=str(e))

class ChartRequest(BaseModel): # Ensure defined if not already
    symbol: str
    interval: str = CONFIG["default_chart_interval"]
    range: str = CONFIG["default_chart_range"]

@app.post("/fetch_chart_data")
async def api_fetch_chart_data(req: ChartRequest):
    """
    Fetch historical chart data for the requested symbol.
    
    Retrieves OHLCV (Open, High, Low, Close, Volume) data for a stock
    at the specified interval and range.
    
    Args:
        req: ChartRequest with symbol, interval and range parameters
        
    Returns:
        Dictionary with time-series chart data
        
    Raises:
        HTTPException: On fetch failure or timeout
    """
    logger.info(f"Fetching chart data for {req.symbol} with interval={req.interval}, range={req.range}")
    try:
        # Try to get from cache first with a short TTL
        cache_key = f"yahoo_chart:{req.symbol}:{req.interval}:{req.range}"
        
        if redis_client:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Returning cached chart data for {req.symbol}")
                return json.loads(cached_data)
        
        # Otherwise fetch from API
        ticker = yf.Ticker(req.symbol)
        hist = ticker.history(period=req.range, interval=req.interval)
        
        if hist.empty:
            logger.warning(f"No chart data found for {req.symbol}")
            return {"success": False, "error": "No chart data found"}
            
        # Convert timestamp to string for JSON compatibility
        hist.index = hist.index.strftime('%Y-%m-%d %H:%M:%S')
        result = {"success": True, "chart_data": hist.to_dict(orient="index")}
        
        # Cache the result (shorter TTL for more frequently changing data)
        if redis_client:
            # Calculate appropriate TTL based on interval
            if req.interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
                ttl = 300  # 5 minutes for intraday data
            else:
                ttl = 3600  # 1 hour for daily/weekly data
            await redis_client.set(cache_key, json.dumps(result), ex=ttl)
            
        return result
    except Exception as e:
        logger.exception(f"Error fetching chart data for {req.symbol}")
        raise HTTPException(status_code=500, detail=str(e))

class SummaryRequest(BaseModel): # Ensure defined
    symbol: str

@app.post("/fetch_summary")
async def api_fetch_summary(req: SummaryRequest):
    """
    Fetch company summary information for a stock symbol.
    
    Retrieves key company data including name, sector, financial metrics,
    and market statistics.
    
    Args:
        req: SummaryRequest with symbol parameter
        
    Returns:
        Dictionary with company summary information
        
    Raises:
        HTTPException: On fetch failure or invalid symbol
    """
    if shutdown_event.is_set():
        logger.warning(f"Rejecting summary request for {req.symbol} during shutdown")
        raise HTTPException(status_code=503, detail="Server is shutting down")
        
    logger.info(f"Fetching company summary for {req.symbol}")
    
    try:
        # Check cache first with longer TTL (company info doesn't change often)
        cache_key = f"yahoo_summary:{req.symbol}"
        
        if redis_client:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Using cached summary data for {req.symbol}")
                return json.loads(cached_data)
        
        # Fetch from API if not in cache
        ticker = yf.Ticker(req.symbol)
        # yfinance info can be very large, select key fields
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
            # Add some additional useful fields
            "description": info.get("longBusinessSummary", ""),
            "employees": info.get("fullTimeEmployees"),
            "exchange": info.get("exchange"),
        }
        
        result = {"success": True, "summary": summary_data}
        
        # Cache the result for 24 hours - company info doesn't change frequently
        if redis_client:
            await redis_client.set(cache_key, json.dumps(result), ex=86400)  # 24 hours TTL
            
        return result
    except Exception as e:
        logger.exception(f"Error fetching summary for {req.symbol}")
        # Check if it's a yfinance specific error for "No summary data found"
        if "No fundamentals found" in str(e) or "No data found for symbol" in str(e):  # Common yf errors
            return {"success": False, "error": f"No summary data found for symbol {req.symbol}"}
        raise HTTPException(status_code=500, detail=str(e))


# --- Production-ready Redis-based rate limiter ---
from fastapi import Request
import time

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Production-ready Redis-based rate limiter middleware.
    Implements a sliding window rate limit using Redis for distributed environments.
    
    Args:
        request: FastAPI request object
        call_next: Next middleware/endpoint in chain
        
    Returns:
        Response: Either the next middleware's response or a 429 status code
    """
    if not redis_client:
        # Fallback to allow requests if Redis is unavailable
        logger.warning("Redis unavailable for rate limiting, allowing request")
        return await call_next(request)
    
    try:
        # Get client IP - properly handle proxy forwarding in production
        ip = request.headers.get("X-Forwarded-For", request.client.host)
        if ip and "," in ip:  # X-Forwarded-For can contain multiple IPs
            ip = ip.split(",")[0].strip()  # Use the original client IP
            
        # Extract API key if present for per-API key rate limiting
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        
        # Create rate limit key based on IP and optionally API key
        path = request.url.path
        rate_key = f"ratelimit:{ip}:{path}"
        if api_key:
            # Hash the API key for privacy
            import hashlib
            hashed_key = hashlib.md5(api_key.encode()).hexdigest()
            rate_key = f"ratelimit:{hashed_key}:{path}"
        
        # Configuration
        window = 60  # seconds
        max_requests = CONFIG.get("rate_limit_per_minute", 60)
        now = time.time()
        window_start = now - window
        
        # Atomic Redis pipeline for rate limiting
        pipe = redis_client.pipeline()
        # Add current timestamp to sorted set
        pipe.zadd(rate_key, {str(now): now})
        # Remove timestamps outside the window
        pipe.zremrangebyscore(rate_key, 0, window_start)
        # Count requests in current window
        pipe.zcard(rate_key)
        # Set expiry on the key
        pipe.expire(rate_key, window)
        results = await pipe.execute()
        
        request_count = results[2]
        
        # If too many requests, return 429
        if request_count > max_requests:
            # Add retry-after header
            return Response(
                content=json.dumps({"detail": "Rate limit exceeded. Try again later."}),
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(window)},
                media_type="application/json"
            )
            
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-Rate-Limit-Limit"] = str(max_requests)
        response.headers["X-Rate-Limit-Remaining"] = str(max(0, max_requests - request_count))
        response.headers["X-Rate-Limit-Reset"] = str(int(now + window))
        
        return response
    except Exception as e:
        logger.exception(f"Rate limiting error: {str(e)}")
        # Fallback to allow request in case of error
        return await call_next(request)

# --- End of file ---
