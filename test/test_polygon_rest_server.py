"""
Tests for the Polygon REST API FastAPI Server.
Tests core functionality by mocking external dependencies.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

# Add parent directory to path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path setup
from mcp.polygon_rest_server import app, validate_config, connect_redis
from mcp.polygon_rest_server import polygon_request, fetch_aggregates
from mcp.polygon_rest_server import extract_features_from_aggregates, identify_simple_support_resistance
from mcp.polygon_rest_server import PatternClassifier, analyze_historical_patterns_with_ml


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_env_vars():
    """Mock environment variables required by the application"""
    with patch.dict(os.environ, {
        "POLYGON_API_KEY": "test_api_key",
        "POLYGON_RATE_LIMIT": "10",
        "POLYGON_USE_CACHE": "True",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_DB": "3",
        "VALID_API_KEYS": "test-key-123"
    }):
        yield


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    with patch('mcp.polygon_rest_server.aioredis.Redis', return_value=AsyncMock()) as mock:
        # Configure the mock
        mock.return_value.ping = AsyncMock(return_value=True)
        mock.return_value.get = AsyncMock(return_value=None)
        mock.return_value.set = AsyncMock(return_value=True)
        mock.return_value.delete = AsyncMock(return_value=True)
        mock.return_value.close = AsyncMock(return_value=None)
        mock.return_value.pipeline = AsyncMock()
        pipeline_instance = AsyncMock()
        pipeline_instance.__aenter__ = AsyncMock(return_value=pipeline_instance)
        pipeline_instance.__aexit__ = AsyncMock(return_value=None)
        pipeline_instance.zremrangebyscore = AsyncMock(return_value=None)
        pipeline_instance.zadd = AsyncMock(return_value=None)
        pipeline_instance.zcard = AsyncMock(return_value=None)
        pipeline_instance.expire = AsyncMock(return_value=None)
        pipeline_instance.execute = AsyncMock(return_value=[None, None, 0, None])
        mock.return_value.pipeline.return_value = pipeline_instance
        mock.return_value.zrange = AsyncMock(return_value=[])
        yield mock


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for testing API calls"""
    with patch('aiohttp.ClientSession') as mock:
        # Configure AsyncMock responses
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"success": True})
        mock_resp.text = AsyncMock(return_value=json.dumps({"success": True}))
        mock_resp.headers = {}
        
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_resp
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_context)
        mock_session.post = AsyncMock(return_value=mock_context)
        
        mock.return_value = mock_session
        mock.return_value.__aenter__.return_value = mock_session
        
        yield mock


@pytest.fixture
def mock_joblib():
    """Mock joblib for testing model loading/saving"""
    with patch('mcp.polygon_rest_server.joblib') as mock:
        mock.load = MagicMock(return_value=MagicMock())
        mock.dump = MagicMock()
        yield mock


@pytest.mark.asyncio
async def test_validate_config(mock_env_vars):
    """Test that configuration validates correctly"""
    config = validate_config()
    
    assert config["polygon"]["api_key"] == "test_api_key"
    assert config["polygon"]["rate_limit_per_minute"] == 10
    assert config["polygon"]["use_cache"] == True
    assert config["redis"]["host"] == "localhost"
    assert config["redis"]["port"] == 6379
    assert config["redis"]["db"] == 3
    assert "feature_engineering" in config
    assert "sma_periods" in config["feature_engineering"]


@pytest.mark.asyncio
async def test_connect_redis(mock_redis, mock_env_vars):
    """Test Redis connection function"""
    await connect_redis()
    mock_redis.return_value.ping.assert_called_once()


@pytest.mark.asyncio
async def test_polygon_request(mock_aiohttp_session, mock_redis, mock_env_vars):
    """Test polygon_request function handles correct API responses"""
    # Configure mock to return specific data
    sample_response = {
        "status": "OK",
        "ticker": "AAPL",
        "results": [
            {
                "c": 150.0,
                "h": 152.0,
                "l": 148.0,
                "o": 149.0,
                "t": 1625097600000,
                "v": 30000000
            }
        ]
    }
    
    session_instance = mock_aiohttp_session.return_value
    context_instance = session_instance.get.return_value.__aenter__.return_value
    context_instance.json.return_value = sample_response
    
    with patch('mcp.polygon_rest_server.redis_client', mock_redis.return_value):
        result = await polygon_request("/v2/aggs/ticker/AAPL/range/1/day/2021-07-01/2021-07-01")
        assert result == sample_response
        session_instance.get.assert_called_once()


@pytest.mark.asyncio
async def test_polygon_request_with_rate_limit_handling(mock_aiohttp_session, mock_env_vars):
    """Test polygon_request handles rate limiting correctly"""
    # First response is rate limited (429), second response succeeds
    rate_limited_resp = AsyncMock()
    rate_limited_resp.status = 429
    rate_limited_resp.headers = {"Retry-After": "1"}
    rate_limited_context = AsyncMock()
    rate_limited_context.__aenter__.return_value = rate_limited_resp
    
    success_resp = AsyncMock()
    success_resp.status = 200
    success_resp.json = AsyncMock(return_value={"status": "OK", "results": []})
    success_context = AsyncMock()
    success_context.__aenter__.return_value = success_resp
    
    session_instance = mock_aiohttp_session.return_value
    # First call returns rate limit, second call succeeds
    session_instance.get.side_effect = [rate_limited_context, success_context]
    
    with patch('mcp.polygon_rest_server.wait_for_rate_limit', AsyncMock()) as mock_wait:
        result = await polygon_request("/v2/aggs/ticker/AAPL/range/1/day/2021-07-01/2021-07-01")
        assert session_instance.get.call_count == 2
        assert result == {"status": "OK", "results": []}


@pytest.mark.asyncio
async def test_fetch_aggregates(mock_env_vars):
    """Test fetch_aggregates function formats request and handles response correctly"""
    sample_response = {
        "status": "OK",
        "ticker": "AAPL",
        "results": [
            {"c": 150.0, "h": 152.0, "l": 148.0, "o": 149.0, "t": 1625097600000, "v": 30000000}
        ]
    }
    
    with patch('mcp.polygon_rest_server.polygon_request', AsyncMock(return_value=sample_response)) as mock_request:
        result = await fetch_aggregates("AAPL", 1, "day", "2021-07-01", "2021-07-01")
        assert result["success"] == True
        assert result["ticker"] == "AAPL"
        assert "results" in result
        assert len(result["results"]) == 1
        
        # Check that polygon_request was called with correct parameters
        expected_endpoint = "/v2/aggs/ticker/AAPL/range/1/day/2021-07-01/2021-07-01"
        expected_params = {"adjusted": "true", "sort": "asc", "limit": "5000"}
        mock_request.assert_called_once_with(expected_endpoint, expected_params)


def test_extract_features_from_aggregates():
    """Test that feature engineering functions work on market data"""
    # Create sample market data
    sample_data = pd.DataFrame({
        'o': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0] * 10,
        'h': [105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0] * 10,
        'l': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0] * 10,
        'c': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0] * 10,
        'v': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000] * 10,
        't': [int((datetime(2021, 1, 1) + timedelta(days=i)).timestamp() * 1000) for i in range(70)]
    })
    
    df_features = extract_features_from_aggregates(sample_data)
    
    # Verify the expected features were calculated
    assert not df_features.empty
    expected_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'sma10', 'ema10', 'sma20', 'ema20', 'sma50', 'ema50',
        'rsi', 'atr', 'macd', 'macd_signal', 'macd_hist',
        'bb_sma', 'bb_std', 'bb_upper', 'bb_lower',
        'returns', 'log_returns', 'volatility',
        'relative_volume', 'volume_change', 'volume_sma5', 'volume_sma20', 'volume_ratio'
    ]
    for column in expected_columns:
        assert column in df_features.columns
    
    # Verify there's no NaN values in the result (after dropna())
    assert not df_features.isnull().values.any()


def test_identify_simple_support_resistance():
    """Test support and resistance identification"""
    # Create sample price data
    sample_data = pd.DataFrame({
        'high': [105.0, 106.0, 107.0, 108.5, 107.0, 106.0, 108.0, 109.0, 110.0, 109.5, 108.0] * 3,
        'low': [99.0, 100.0, 101.0, 102.0, 100.5, 99.0, 102.0, 103.0, 104.0, 103.0, 102.0] * 3,
        'close': [102.0, 103.0, 104.0, 105.0, 101.0, 100.0, 104.0, 106.0, 107.0, 104.0, 103.0] * 3
    })
    
    support, resistance = identify_simple_support_resistance(sample_data, window_size=3, num_levels=2)
    
    # Check that the function returned the expected type of results
    assert isinstance(support, list)
    assert isinstance(resistance, list)
    assert len(support) <= 2  # num_levels=2
    assert len(resistance) <= 2  # num_levels=2


def test_pattern_classifier():
    """Test PatternClassifier initialization and methods"""
    classifier = PatternClassifier()
    assert hasattr(classifier, 'model')
    assert hasattr(classifier, 'scaler')
    assert hasattr(classifier, 'is_trained')
    assert hasattr(classifier, 'predict')
    
    # Test prediction with untrained model
    test_data = pd.DataFrame({
        'close': [100.0, 101.0],
        'volume': [1000000, 1100000],
        'rsi': [55.0, 60.0],
        'atr': [2.5, 2.6],
        'macd': [0.5, 0.6]
    })
    predictions = classifier.predict(test_data)
    assert len(predictions) == 2
    assert all(p == "unknown_pattern" for p in predictions)


@pytest.mark.asyncio
async def test_analyze_historical_patterns_with_ml(mock_env_vars):
    """Test historical pattern analysis with ML classification"""
    sample_aggregates = {
        "success": True,
        "ticker": "AAPL",
        "results": [
            {"c": 150.0, "h": 152.0, "l": 148.0, "o": 149.0, "t": 1625097600000, "v": 30000000}
            for _ in range(100)  # Create 100 identical bars for testing
        ]
    }
    
    # Mock the fetch_aggregates function to return sample data
    with patch('mcp.polygon_rest_server.fetch_aggregates', AsyncMock(return_value=sample_aggregates)), \
         patch('mcp.polygon_rest_server.extract_features_from_aggregates') as mock_extract, \
         patch('mcp.polygon_rest_server.pattern_classifier.predict', return_value=["bullish_flag"]), \
         patch('mcp.polygon_rest_server.redis_client', None):
        
        # Configure mock to return a DataFrame with expected structure
        mock_df = pd.DataFrame({
            'open': [149.0] * 100,
            'high': [152.0] * 100,
            'low': [148.0] * 100,
            'close': [150.0] * 100,
            'volume': [30000000] * 100
        })
        mock_extract.return_value = mock_df
        
        result = await analyze_historical_patterns_with_ml("AAPL", "2021-07-01", "2021-07-31", ["1day"])
        
        assert result["success"] == True
        assert "analysis" in result
        assert result["analysis"]["symbol"] == "AAPL"
        assert "analysis_by_timeframe" in result["analysis"]
        assert "1day" in result["analysis"]["analysis_by_timeframe"]
        
        # Check pattern classification results are included
        day_analysis = result["analysis"]["analysis_by_timeframe"]["1day"]
        assert day_analysis["classified_pattern"] in ["bullish_flag", "N/A"]


def test_server_info_endpoint(test_client, mock_env_vars):
    """Test the server info endpoint"""
    response = test_client.get("/server_info")
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "polygon_rest"
    assert "version" in data
    assert "description" in data
    assert "tools" in data
    assert isinstance(data["tools"], list)
    assert len(data["tools"]) > 0


def test_fetch_aggregates_endpoint(test_client, mock_env_vars):
    """Test the fetch_aggregates endpoint requires authentication"""
    # Try without API key
    response = test_client.post("/fetch_aggregates", json={
        "ticker": "AAPL",
        "multiplier": 1,
        "timespan": "day",
        "from_date": "2021-07-01",
        "to_date": "2021-07-31",
        "adjusted": True,
        "sort": "asc",
        "limit": 5000
    })
    assert response.status_code == 401
    
    # With API key (mocked)
    with patch('mcp.polygon_rest_server.fetch_aggregates', AsyncMock(return_value={"success": True})):
        response = test_client.post(
            "/fetch_aggregates",
            json={
                "ticker": "AAPL",
                "multiplier": 1,
                "timespan": "day",
                "from_date": "2021-07-01",
                "to_date": "2021-07-31"
            },
            headers={"X-API-Key": "test-key-123"}
        )
        assert response.status_code == 200


def test_analyze_historical_patterns_endpoint(test_client, mock_env_vars):
    """Test the analyze_historical_patterns_with_ml endpoint requires authentication"""
    # Try without API key
    response = test_client.post("/analyze_historical_patterns_with_ml", json={
        "symbol": "AAPL",
        "start_date": "2021-07-01",
        "end_date": "2021-07-31",
        "timeframes": ["1day"]
    })
    assert response.status_code == 401
    
    # With API key (mocked)
    with patch('mcp.polygon_rest_server.analyze_historical_patterns_with_ml', AsyncMock(return_value={"success": True})):
        response = test_client.post(
            "/analyze_historical_patterns_with_ml",
            json={
                "symbol": "AAPL",
                "start_date": "2021-07-01",
                "end_date": "2021-07-31"
            },
            headers={"X-API-Key": "test-key-123"}
        )
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main(["-v", "test_polygon_rest_server.py"])