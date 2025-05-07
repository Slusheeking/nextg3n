"""
Tests for the Reddit Processor MCP FastAPI Server.
Tests core functionality by mocking external dependencies.
"""

import os
import sys
import json
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

# Add parent directory to path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path setup
from mcp.reddit_processor_server import app, validate_config, connect_redis
from mcp.reddit_processor_server import api_fetch_subreddit_posts, api_fetch_post_comments, api_search_reddit
from mcp.reddit_processor_server import get_trending_sentiment, get_trending_tickers, get_trending_topics
from mcp.reddit_processor_server import finbert_tone_model, roberta_sec_model, financial_bert_ner_model


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_env_vars():
    """Mock environment variables required by the application"""
    with patch.dict(os.environ, {
        "REDDIT_CLIENT_ID": "test_client_id",
        "REDDIT_CLIENT_SECRET": "test_client_secret",
        "REDDIT_USER_AGENT": "TestUserAgent",
        "REDDIT_USERNAME": "test_username",
        "REDDIT_PASSWORD": "test_password",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_DB": "1",
        "VALID_API_KEYS": "test-key-123"
    }):
        yield


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    with patch('redis.asyncio.Redis', return_value=AsyncMock()) as mock:
        # Configure the mock
        mock.return_value.ping = AsyncMock(return_value=True)
        mock.return_value.get = AsyncMock(return_value=None)
        mock.return_value.set = AsyncMock(return_value=True)
        mock.return_value.delete = AsyncMock(return_value=True)
        mock.return_value.close = AsyncMock(return_value=None)
        mock.return_value.exists = AsyncMock(return_value=False)
        mock.return_value.keys = AsyncMock(return_value=[])
        mock.return_value.expire = AsyncMock(return_value=True)
        mock.return_value.zrange = AsyncMock(return_value=[])
        mock.return_value.zrangebyscore = AsyncMock(return_value=[])
        mock.return_value.zadd = AsyncMock(return_value=True)
        
        # Pipeline mocking
        pipeline_instance = AsyncMock()
        pipeline_instance.__aenter__ = AsyncMock(return_value=pipeline_instance)
        pipeline_instance.__aexit__ = AsyncMock(return_value=None)
        pipeline_instance.zremrangebyscore = AsyncMock(return_value=None)
        pipeline_instance.zadd = AsyncMock(return_value=None)
        pipeline_instance.zcard = AsyncMock(return_value=None)
        pipeline_instance.expire = AsyncMock(return_value=None)
        pipeline_instance.execute = AsyncMock(return_value=[None, None, 0, None])
        mock.return_value.pipeline = AsyncMock(return_value=pipeline_instance)
        
        # Patch the global variable
        with patch('mcp.reddit_processor_server.redis_client', mock.return_value):
            yield mock


@pytest.fixture
def mock_reddit():
    """Mock Reddit API client for testing"""
    with patch('asyncpraw.Reddit') as mock:
        # Create a mock Reddit client
        mock_client = MagicMock()
        
        # Mock subreddit
        mock_subreddit = AsyncMock()
        mock_subreddit.hot = AsyncMock()
        mock_subreddit.new = AsyncMock()
        mock_subreddit.top = AsyncMock()
        mock_subreddit.rising = AsyncMock()
        mock_subreddit.search = AsyncMock()
        
        # Mock post
        mock_post = MagicMock()
        mock_post.id = "test_post_id"
        mock_post.title = "Test Post"
        mock_post.score = 100
        mock_post.created_utc = datetime.utcnow().timestamp()
        mock_post.num_comments = 25
        mock_post.permalink = "/r/test/comments/test_post_id/test_post/"
        mock_post.author = "test_author"
        mock_post.url = "https://reddit.com/test"
        mock_post.selftext = "Test post content"
        
        # Mock submission
        mock_submission = AsyncMock()
        mock_submission.load = AsyncMock()
        mock_submission.comments = MagicMock()
        mock_comment = MagicMock()
        mock_comment.id = "test_comment_id"
        mock_comment.body = "Test comment"
        mock_comment.score = 10
        mock_comment.created_utc = datetime.utcnow().timestamp()
        mock_comment.author = "test_commenter"
        mock_comment.parent_id = "t3_test_post_id"
        mock_submission.comments.list = MagicMock(return_value=[mock_comment])
        mock_submission.comments.replace_more = AsyncMock()
        
        # Functions to generate async iterators for different endpoints
        async def mock_async_iter_factory(items):
            for item in items:
                yield item
        
        # Mock the subreddit hot/new/top iterators
        mock_post_list = [mock_post]
        mock_subreddit.hot.return_value = mock_async_iter_factory(mock_post_list)
        mock_subreddit.new.return_value = mock_async_iter_factory(mock_post_list)
        mock_subreddit.top.return_value = mock_async_iter_factory(mock_post_list)
        mock_subreddit.rising.return_value = mock_async_iter_factory(mock_post_list)
        mock_subreddit.search.return_value = mock_async_iter_factory(mock_post_list)
        
        # Connect the mock objects
        mock_client.subreddit = AsyncMock(return_value=mock_subreddit)
        mock_client.submission = AsyncMock(return_value=mock_submission)
        
        # Set the mock return value
        mock.return_value = mock_client
        
        # Patch the global variable
        with patch('mcp.reddit_processor_server.reddit_client', mock_client):
            yield mock


@pytest.fixture
def mock_ml_models():
    """Mock ML models for testing sentiment analysis"""
    # Mock FinBERT tone model
    with patch.object(finbert_tone_model, 'load_model', MagicMock()), \
         patch.object(finbert_tone_model, 'predict', MagicMock(return_value=[{"label": "POSITIVE", "score": 0.8}])), \
         patch.object(roberta_sec_model, 'load_model', MagicMock()), \
         patch.object(roberta_sec_model, 'predict', MagicMock(return_value=[{"label": "positive", "score": 0.7}])), \
         patch.object(financial_bert_ner_model, 'load_model', MagicMock()), \
         patch.object(financial_bert_ner_model, 'predict', MagicMock(return_value=[
             {"type": "ORG", "text": "Apple", "score": 0.95, "start": 10, "end": 15},
             {"type": "MISC", "text": "iPhone", "score": 0.92, "start": 20, "end": 26}
         ])):
            # Set model attributes to indicate they are loaded
            finbert_tone_model.model = MagicMock()
            finbert_tone_model.tokenizer = MagicMock()
            roberta_sec_model.model = MagicMock()
            roberta_sec_model.tokenizer = MagicMock()
            financial_bert_ner_model.model = MagicMock()
            financial_bert_ner_model.tokenizer = MagicMock()
            
            yield


@pytest.mark.asyncio
async def test_validate_config(mock_env_vars):
    """Test that configuration loads correctly"""
    with patch('pathlib.Path.exists', return_value=False):
        config = validate_config()
        
        assert config["reddit"]["client_id"] == "test_client_id"
        assert config["reddit"]["client_secret"] == "test_client_secret"
        assert config["reddit"]["user_agent"] == "TestUserAgent"
        assert config["reddit"]["username"] == "test_username"
        assert config["reddit"]["password"] == "test_password"
        assert config["redis"]["host"] == "localhost"
        assert config["redis"]["port"] == 6379
        assert config["redis"]["db"] == 1


@pytest.mark.asyncio
async def test_connect_redis(mock_redis, mock_env_vars):
    """Test Redis connection function"""
    await connect_redis()
    mock_redis.assert_called_once()
    mock_redis.return_value.ping.assert_called_once()


@pytest.mark.asyncio
async def test_api_fetch_subreddit_posts(mock_reddit, mock_redis, mock_env_vars, mock_ml_models):
    """Test fetching subreddit posts API endpoint"""
    from mcp.reddit_processor_server import SubredditPostsRequest
    
    request = SubredditPostsRequest(
        subreddits=["wallstreetbets", "stocks"],
        sort="hot",
        time_filter="day",
        limit=10
    )
    
    result = await api_fetch_subreddit_posts(request)
    
    assert result["success"] is True
    assert "posts" in result
    assert len(result["posts"]) > 0
    assert result["posts"][0]["subreddit"] == "wallstreetbets"
    
    # Verify that Reddit API was called correctly
    mock_reddit.return_value.subreddit.assert_called()
    
    # Verify caching attempt
    mock_redis.return_value.get.assert_called()
    mock_redis.return_value.set.assert_called()


@pytest.mark.asyncio
async def test_api_fetch_post_comments(mock_reddit, mock_redis, mock_env_vars, mock_ml_models):
    """Test fetching post comments API endpoint"""
    from mcp.reddit_processor_server import PostCommentsRequest
    
    request = PostCommentsRequest(
        subreddit="wallstreetbets",
        post_id="test_post_id",
        sort="confidence",
        limit=10
    )
    
    result = await api_fetch_post_comments(request)
    
    assert result["success"] is True
    assert "comments" in result
    assert len(result["comments"]) > 0
    assert result["comments"][0]["id"] == "test_comment_id"
    
    # Verify Reddit API was called correctly
    mock_reddit.return_value.submission.assert_called_with(id="test_post_id")
    
    # Verify caching attempt
    mock_redis.return_value.get.assert_called()
    mock_redis.return_value.set.assert_called()


@pytest.mark.asyncio
async def test_api_search_reddit(mock_reddit, mock_redis, mock_env_vars, mock_ml_models):
    """Test searching Reddit API endpoint"""
    from mcp.reddit_processor_server import SearchRedditRequest
    
    request = SearchRedditRequest(
        query="AAPL earnings",
        subreddits=["wallstreetbets", "stocks"],
        sort="relevance",
        time_filter="week",
        limit=10
    )
    
    result = await api_search_reddit(request)
    
    assert result["success"] is True
    assert "posts" in result
    assert len(result["posts"]) > 0
    
    # Verify Reddit API was called correctly
    mock_reddit.return_value.subreddit.assert_called()
    
    # Verify caching attempt
    mock_redis.return_value.get.assert_called()
    mock_redis.return_value.set.assert_called()


@pytest.mark.asyncio
async def test_api_analyze_ticker_sentiment(mock_reddit, mock_redis, mock_env_vars, mock_ml_models):
    """Test analyzing ticker sentiment API endpoint"""
    from mcp.reddit_processor_server import AnalyzeTickerSentimentRequest
    
    # Mock the reddit API search to return data
    mock_subreddit = mock_reddit.return_value.subreddit.return_value
    
    request = AnalyzeTickerSentimentRequest(
        tickers=["AAPL", "MSFT"],
        subreddits=["wallstreetbets", "stocks"],
        time_filter="week",
        limit_per_ticker=10
    )
    
    with patch('mcp.reddit_processor_server.reddit_api_with_retry', AsyncMock()) as mock_retry:
        mock_retry.return_value = [{"content": "Apple is doing well", "score": 10}]
        
        result = await api_analyze_ticker_sentiment(request)
        
        assert result["success"] is True
        assert "tickers" in result
        assert "AAPL" in result["tickers"]
        assert "sentiment" in result["tickers"]["AAPL"]
        assert "summary" in result
        assert result["tickers"]["AAPL"]["mention_count"] >= 1
        
        # Verify that the ML models were called
        finbert_tone_model.predict.assert_called()
        financial_bert_ner_model.predict.assert_called()


@pytest.mark.asyncio
async def test_get_trending_sentiment(mock_redis, mock_env_vars):
    """Test getting trending sentiment from Redis"""
    # Mock Redis to return sentiment data
    mock_redis.return_value.zrange.return_value = ["AAPL", "MSFT"]
    mock_redis.return_value.zrangebyscore.return_value = [
        json.dumps({
            "date": datetime.utcnow().strftime('%Y-%m-%d'),
            "sentiment": {
                "scores": {"positive": 0.7, "negative": 0.1, "neutral": 0.2},
                "positive_mentions": 10,
                "negative_mentions": 3,
                "neutral_mentions": 5
            }
        })
    ]
    
    result = await get_trending_sentiment(["wallstreetbets"], 7, 5)
    
    assert len(result) > 0
    assert "ticker" in result[0]
    assert "avg_sentiment" in result[0]
    assert "total_mentions" in result[0]
    assert "dominant_sentiment" in result[0]


@pytest.mark.asyncio
async def test_get_trending_tickers(mock_redis, mock_env_vars):
    """Test getting trending tickers from Redis"""
    # Mock Redis to return ticker data
    mock_redis.return_value.zrange.return_value = [
        ("AAPL", 100.0),
        ("MSFT", 75.0),
        ("TSLA", 50.0)
    ]
    
    result = await get_trending_tickers(1, 10)
    
    assert len(result) > 0
    assert "ticker" in result[0]
    assert "mentions" in result[0]


def test_server_info_endpoint(test_client, mock_env_vars, mock_ml_models):
    """Test the server info endpoint"""
    response = test_client.get("/server_info")
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "reddit_processor"
    assert "version" in data
    assert "tools" in data
    assert "models" in data
    assert len(data["tools"]) > 0
    assert len(data["models"]) > 0


def test_models_status_endpoint(test_client, mock_env_vars, mock_ml_models):
    """Test the models status endpoint"""
    with patch('mcp.reddit_processor_server.check_models_loaded') as mock_check:
        mock_check.return_value = {
            "finbert_tone": True,
            "roberta_sec": True,
            "financial_bert_ner": True
        }
        
        response = test_client.get("/models/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert "models" in data
        assert data["models"]["finbert_tone"] is True
        assert data["models"]["roberta_sec"] is True
        assert data["models"]["financial_bert_ner"] is True
        assert "device" in data
        assert "timestamp" in data


def test_authentication(test_client, mock_env_vars):
    """Test API endpoints require authentication"""
    # Try to access an authenticated endpoint without providing API key
    response = test_client.post("/fetch_subreddit_posts", json={
        "subreddits": ["wallstreetbets"],
        "sort": "hot",
        "time_filter": "day",
        "limit": 10
    })
    assert response.status_code == 401
    
    # With API key but invalid request format
    response = test_client.post(
        "/fetch_subreddit_posts",
        json={
            "subreddits": [],  # Empty list, will fail validation
            "sort": "invalid",
            "time_filter": "day",
            "limit": 10
        },
        headers={"X-API-Key": "test-key-123"}
    )
    assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main(["-v", "test_reddit_processor_server.py"])