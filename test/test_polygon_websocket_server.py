"""
Tests for the Polygon WebSocket MCP FastAPI Server.
Tests core functionality by mocking external dependencies.
"""

import os
import sys
import json
import pytest
import torch
import numpy as np
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

# Add parent directory to path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path setup
from mcp.polygon_websocket_server import app, get_config, load_system_config, connect_redis
from mcp.polygon_websocket_server import polygon_ws_stream, StockScreener
from mcp.polygon_websocket_server import DeepARModel, TCNModel, InformerModel


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_env_vars():
    """Mock environment variables required by the application"""
    with patch.dict(os.environ, {
        "POLYGON_API_KEY": "test_api_key",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_DB": "0",
        "VALID_API_KEYS": "test-key-123"
    }):
        yield


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    with patch('mcp.polygon_websocket_server.aioredis.Redis', return_value=AsyncMock()) as mock:
        # Configure the mock
        mock.return_value.ping = AsyncMock(return_value=True)
        mock.return_value.get = AsyncMock(return_value=None)
        mock.return_value.set = AsyncMock(return_value=True)
        mock.return_value.delete = AsyncMock(return_value=True)
        mock.return_value.close = AsyncMock(return_value=None)
        mock.return_value.exists = AsyncMock(return_value=False)
        mock.return_value.keys = AsyncMock(return_value=[])
        mock.return_value.expire = AsyncMock(return_value=True)
        mock.return_value.sadd = AsyncMock(return_value=True)
        
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
        
        yield mock


@pytest.fixture
def mock_websockets():
    """Mock websockets library for testing"""
    with patch('mcp.polygon_websocket_server.websockets') as mock:
        # Create a mock WebSocket connection
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=[
            # Auth response
            json.dumps([{"ev": "status", "status": "auth_success", "message": "authenticated"}]),
            # Subscription response
            json.dumps([{"ev": "status", "status": "success", "message": "subscribed to: T.AAPL"}]),
            # Trade message
            json.dumps({"ev": "T", "sym": "AAPL", "p": 150.0, "s": 100, "t": 1625097600000}),
            # Quote message
            json.dumps({"ev": "Q", "sym": "AAPL", "bp": 149.9, "bs": 10, "ap": 150.1, "as": 5, "t": 1625097600000}),
        ])
        mock_ws.close = AsyncMock()
        
        # Mock the connect method to return our mock WebSocket
        mock.connect = AsyncMock(return_value=mock_ws)
        mock.connect.return_value.__aenter__ = AsyncMock(return_value=mock_ws)
        mock.connect.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock WebSocket exceptions
        mock.exceptions = MagicMock()
        mock.exceptions.ConnectionClosed = Exception
        
        yield mock


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for testing API calls"""
    with patch('aiohttp.ClientSession') as mock:
        # Configure AsyncMock responses
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "results": [{
                "v": 1000000,
                "c": 150.0,
                "vw": 150.5
            }]
        })
        
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_resp
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_context)
        mock_session.post = AsyncMock(return_value=mock_context)
        
        mock.return_value = mock_session
        mock.return_value.__aenter__.return_value = mock_session
        
        yield mock


@pytest.fixture
def mock_torch_models():
    """Mock torch models for testing"""
    with patch.object(DeepARModel, '__init__', return_value=None), \
         patch.object(TCNModel, '__init__', return_value=None), \
         patch.object(InformerModel, '__init__', return_value=None), \
         patch.object(DeepARModel, 'forward', return_value=torch.tensor([0.5])), \
         patch.object(TCNModel, 'forward', return_value=torch.tensor([0.7])), \
         patch.object(InformerModel, 'forward', return_value=torch.tensor([0.3])):
        yield


@pytest.fixture
def mock_asyncio():
    """Mock asyncio for testing"""
    with patch('asyncio.create_task', MagicMock()), \
         patch('asyncio.sleep', AsyncMock(return_value=None)), \
         patch('asyncio.CancelledError', Exception):
        yield


@pytest.mark.asyncio
async def test_load_system_config(mock_env_vars):
    """Test that system configuration loads correctly"""
    with patch('mcp.polygon_websocket_server.Path.exists', return_value=False):
        config = load_system_config()
        
        assert config["services"]["polygon"]["api_key"] == "test_api_key"
        assert config["redis"]["host"] == "localhost"
        assert config["redis"]["port"] == 6379
        assert "stock_screening" in config["services"]


@pytest.mark.asyncio
async def test_get_config(mock_env_vars):
    """Test that configuration merges correctly"""
    with patch('mcp.polygon_websocket_server.load_system_config') as mock_load:
        mock_load.return_value = {
            "services": {
                "polygon": {
                    "api_key": "test_api_key",
                    "buffer_size": 2000,
                    "websocket_url": "wss://test.polygon.io/stocks",
                    "min_volume": 1000000,
                    "min_rel_volume": 1.2,
                    "min_price_change": 0.02,
                    "min_atr": 0.2,
                    "model_dir": "test/models",
                    "rate_limit": 30,
                    "max_session_time": 7200
                },
                "stock_screening": {
                    "update_interval_minutes": 10
                }
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "password": None
            }
        }
        
        config = get_config()
        
        assert config["api_key"] == "test_api_key"
        assert config["buffer_size"] == 2000
        assert config["websocket_url"] == "wss://test.polygon.io/stocks"
        assert config["redis_host"] == "localhost"
        assert config["redis_port"] == 6379
        assert config["min_volume"] == 1000000
        assert config["update_interval_minutes"] == 10


@pytest.mark.asyncio
async def test_connect_redis(mock_redis, mock_env_vars):
    """Test Redis connection function"""
    await connect_redis()
    mock_redis.return_value.ping.assert_called_once()


@pytest.mark.asyncio
async def test_polygon_ws_stream(mock_websockets, mock_env_vars, mock_asyncio):
    """Test WebSocket streaming function"""
    buffer = []
    symbols = ["AAPL"]
    channels = ["T", "Q"]
    duration = 1  # Short duration for test
    
    with patch('mcp.polygon_websocket_server.CONFIG', {
            "api_key": "test_api_key", 
            "websocket_url": "wss://test.polygon.io/stocks",
            "buffer_size": 1000
        }):
        
        await polygon_ws_stream(symbols, channels, duration, buffer)
        
        # Verify WebSocket operations
        mock_websockets.connect.assert_called_once()
        mock_ws = mock_websockets.connect.return_value.__aenter__.return_value
        
        # Authentication
        mock_ws.send.assert_any_call(json.dumps({"action": "auth", "params": "test_api_key"}))
        
        # Subscription
        subscription_msg = json.dumps({"action": "subscribe", "params": "T.AAPL,Q.AAPL"})
        mock_ws.send.assert_any_call(subscription_msg)
        
        # Verify that the buffer contains data
        assert len(buffer) > 0
        
        # Verify unsubscribe
        unsubscribe_msg = json.dumps({"action": "unsubscribe", "params": "T.AAPL,Q.AAPL"})
        mock_ws.send.assert_any_call(unsubscribe_msg)


def test_deep_ar_model():
    """Test DeepARModel initialization and forward pass"""
    model = DeepARModel(input_dim=10, hidden_dim=64)
    assert model.lstm is not None
    assert model.fc is not None
    
    batch_size = 2
    seq_len = 5
    input_dim = 10
    x = torch.randn(batch_size, seq_len, input_dim)
    
    output = model(x)
    assert output.shape == torch.Size([batch_size, 1])


def test_tcn_model():
    """Test TCNModel initialization and forward pass"""
    model = TCNModel(input_dim=10, num_channels=[32, 64, 32])
    assert model.network is not None
    assert model.fc is not None
    
    batch_size = 2
    seq_len = 5
    input_dim = 10
    x = torch.randn(batch_size, seq_len, input_dim)
    
    output = model(x)
    assert output.shape == torch.Size([batch_size])


def test_informer_model():
    """Test InformerModel initialization and forward pass"""
    model = InformerModel(enc_in=10, d_model=64)
    assert model.embedding is not None
    assert model.attention is not None
    assert model.fc is not None
    
    batch_size = 2
    seq_len = 5
    input_dim = 10
    x = torch.randn(batch_size, seq_len, input_dim)
    
    output = model(x)
    assert output.shape == torch.Size([batch_size, 1])


def test_stock_screener_init(mock_torch_models):
    """Test StockScreener initialization"""
    config = {
        "min_volume": 1000000,
        "min_rel_volume": 1.5,
        "min_price_change": 0.03,
        "min_atr": 0.25,
        "model_dir": "models/pretrained"
    }
    
    with patch('os.path.exists', return_value=False), \
         patch('torch.load', return_value={}):
        screener = StockScreener(config)
        
        assert screener.min_volume == 1000000
        assert screener.min_rel_volume == 1.5
        assert screener.min_price_change == 0.03
        assert screener.min_atr == 0.25
        assert "deepar" in screener.models
        assert "tcn" in screener.models
        assert "informer" in screener.models


def test_stock_screener_preprocess_data(mock_torch_models):
    """Test StockScreener data preprocessing"""
    config = {
        "min_volume": 1000000,
        "min_rel_volume": 1.5,
        "min_price_change": 0.03,
        "min_atr": 0.25,
        "model_dir": "models/pretrained"
    }
    
    with patch('os.path.exists', return_value=False), \
         patch('torch.load', return_value={}):
        screener = StockScreener(config)
        
        # Test with sample trade data
        sample_data = [
            {"ev": "T", "sym": "AAPL", "p": 150.0, "s": 100, "t": 1625097600000},
            {"ev": "T", "sym": "AAPL", "p": 151.0, "s": 200, "t": 1625097660000},
            {"ev": "T", "sym": "AAPL", "p": 152.0, "s": 300, "t": 1625097720000},
            {"ev": "Q", "sym": "AAPL", "bp": 149.9, "bs": 10, "ap": 150.1, "as": 5, "t": 1625097600000}
        ]
        
        processed = screener.preprocess_data(sample_data)
        
        assert "AAPL" in processed
        assert processed["AAPL"]["open"] == 150.0
        assert processed["AAPL"]["high"] == 152.0
        assert processed["AAPL"]["low"] == 150.0
        assert processed["AAPL"]["close"] == 152.0
        assert processed["AAPL"]["volume"] == 600
        assert len(processed["AAPL"]["trades"]) == 3
        assert len(processed["AAPL"]["quotes"]) == 1


@pytest.mark.asyncio
async def test_stock_screener_get_historical_data(mock_torch_models, mock_aiohttp_session):
    """Test StockScreener historical data retrieval"""
    config = {
        "api_key": "test_api_key",
        "min_volume": 1000000,
        "model_dir": "models/pretrained"
    }
    
    with patch('os.path.exists', return_value=False), \
         patch('torch.load', return_value={}):
        screener = StockScreener(config)
        
        historical = await screener.get_historical_data(["AAPL"])
        
        assert "AAPL" in historical
        assert historical["AAPL"]["prev_volume"] == 1000000
        assert historical["AAPL"]["prev_close"] == 150.0
        assert historical["AAPL"]["prev_vwap"] == 150.5


@pytest.mark.asyncio
async def test_screen_stocks(mock_torch_models, mock_env_vars, mock_aiohttp_session):
    """Test stock screening functionality"""
    config = {
        "api_key": "test_api_key",
        "min_volume": 1000,  # Low threshold for test data
        "min_rel_volume": 0.5,  # Low threshold for test data
        "min_price_change": 0.01,  # Low threshold for test data
        "min_atr": 0.1,  # Low threshold for test data
        "model_dir": "models/pretrained"
    }
    
    with patch('os.path.exists', return_value=False), \
         patch('torch.load', return_value={}), \
         patch('torch.tensor', return_value=torch.zeros(1, 1, 10)), \
         patch('torch.sigmoid', return_value=torch.tensor([0.75])):
        
        screener = StockScreener(config)
        
        # Mock the _predict methods to return fixed scores
        screener._predict_deepar = MagicMock(return_value=0.8)
        screener._predict_tcn = MagicMock(return_value=0.7)
        screener._predict_informer = MagicMock(return_value=0.6)
        
        # Test with sample trade data
        sample_data = [
            {"ev": "T", "sym": "AAPL", "p": 150.0, "s": 1000, "t": 1625097600000},
            {"ev": "T", "sym": "AAPL", "p": 155.0, "s": 2000, "t": 1625097660000},
            {"ev": "Q", "sym": "AAPL", "bp": 149.9, "bs": 10, "ap": 150.1, "as": 5, "t": 1625097600000}
        ]
        
        candidates = await screener.screen_stocks(sample_data)
        
        assert len(candidates) == 1
        assert candidates[0]["symbol"] == "AAPL"
        assert candidates[0]["volume"] == 3000
        assert candidates[0]["last_price"] == 155.0
        assert candidates[0]["screening_score"] == 0.7  # Average of 0.8, 0.7, 0.6


def test_server_info_endpoint(test_client, mock_env_vars):
    """Test the server info endpoint"""
    with patch('mcp.polygon_websocket_server.CONFIG', {
        "api_key": "test_api_key",
        "min_volume": 1000000,
        "min_rel_volume": 1.5,
        "min_price_change": 0.03,
        "min_atr": 0.25
    }), patch('mcp.polygon_websocket_server.stock_screener.models', {"deepar": None, "tcn": None}):
        
        response = test_client.get("/server_info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "polygon_websocket"
        assert "version" in data
        assert "description" in data
        assert "tools" in data
        assert isinstance(data["tools"], list)
        assert len(data["tools"]) > 0
        assert "ai_models" in data


def test_health_endpoint(test_client, mock_env_vars):
    """Test the health check endpoint"""
    with patch('mcp.polygon_websocket_server.redis_client', None), \
         patch('mcp.polygon_websocket_server.CONFIG', {"api_key": "test_api_key"}), \
         patch('mcp.polygon_websocket_server.active_streams', {}):
        
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "redis_connected" in data
        assert data["api_key_configured"] is True


if __name__ == "__main__":
    pytest.main(["-v", "test_polygon_websocket_server.py"])