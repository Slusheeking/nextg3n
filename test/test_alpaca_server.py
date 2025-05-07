"""
Tests for the Alpaca Trading MCP FastAPI Server.
Tests core functionality by mocking external dependencies.
"""

import os
import sys
import json
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

# Add parent directory to path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path setup
from mcp.alpaca_server import app, load_config, connect_redis
from mcp.alpaca_server import get_account, get_positions, get_orders, place_order
from mcp.alpaca_server import get_bars, get_latest_quote, get_latest_trade
from mcp.alpaca_server import assess_trade_risk_detailed, get_portfolio_data


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app"""
    with patch('mcp.alpaca_server.CONFIG', {
        "api_key": "test_api_key",
        "secret_key": "test_secret_key",
        "trading_api_base": "https://paper-api.test-alpaca.markets/v2",
        "data_api_base": "https://data.test-alpaca.markets/v2",
        "paper_trading": True,
        "max_risk_per_trade_pct": 0.02,
        "min_reward_risk_ratio": 2.0,
        "max_position_value_pct": 0.10,
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 0,
        "portfolio_update_interval": 300,
    }):
        client = TestClient(app)
        return client


@pytest.fixture
def mock_env_vars():
    """Mock environment variables required by the application"""
    with patch.dict(os.environ, {
        "ALPACA_API_KEY": "test_api_key",
        "ALPACA_SECRET_KEY": "test_secret_key",
        "ALPACA_TRADING_API_BASE": "https://paper-api.test-alpaca.markets/v2",
        "ALPACA_DATA_API_BASE": "https://data.test-alpaca.markets/v2",
        "ALPACA_PAPER_TRADING": "True",
        "SYS_MAX_RISK_PER_TRADE_PCT": "0.02",
        "SYS_MIN_REWARD_RISK_RATIO": "2.0",
        "SYS_MAX_POSITION_VALUE_PCT": "0.10",
        "VALID_API_KEYS": "test-key-123"
    }):
        yield


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp client session for testing API calls"""
    with patch('aiohttp.ClientSession', autospec=True) as mock:
        # Create mock response for account endpoint
        mock_account_response = AsyncMock()
        mock_account_response.status = 200
        mock_account_response.json = AsyncMock(return_value={
            "account_number": "ABC123456",
            "buying_power": "100000",
            "cash": "50000",
            "portfolio_value": "150000",
            "equity": "150000",
            "equity_change": "1500",
            "status": "ACTIVE",
        })
        mock_account_context = AsyncMock()
        mock_account_context.__aenter__ = AsyncMock(return_value=mock_account_response)
        mock_account_context.__aexit__ = AsyncMock(return_value=None)
        
        # Create mock response for positions endpoint
        mock_positions_response = AsyncMock()
        mock_positions_response.status = 200
        mock_positions_response.json = AsyncMock(return_value=[
            {
                "symbol": "AAPL",
                "qty": "100",
                "market_value": "15000",
                "avg_entry_price": "140.50",
                "current_price": "150.00",
                "unrealized_pl": "950.00",
                "unrealized_plpc": "0.0675",
                "side": "long"
            },
            {
                "symbol": "MSFT",
                "qty": "50",
                "market_value": "18000",
                "avg_entry_price": "350.25",
                "current_price": "360.00",
                "unrealized_pl": "487.50",
                "unrealized_plpc": "0.0278",
                "side": "long"
            }
        ])
        mock_positions_context = AsyncMock()
        mock_positions_context.__aenter__ = AsyncMock(return_value=mock_positions_response)
        mock_positions_context.__aexit__ = AsyncMock(return_value=None)
        
        # Create mock response for orders endpoint
        mock_orders_response = AsyncMock()
        mock_orders_response.status = 200
        mock_orders_response.json = AsyncMock(return_value=[
            {
                "id": "order1",
                "client_order_id": "test-order-1",
                "symbol": "GOOG",
                "side": "buy",
                "type": "limit",
                "qty": "10",
                "limit_price": "2750.00",
                "status": "open"
            }
        ])
        mock_orders_context = AsyncMock()
        mock_orders_context.__aenter__ = AsyncMock(return_value=mock_orders_response)
        mock_orders_context.__aexit__ = AsyncMock(return_value=None)
        
        # Create mock response for (new) order placement
        mock_order_placement_response = AsyncMock()
        mock_order_placement_response.status = 200
        mock_order_placement_response.json = AsyncMock(return_value={
            "id": "new-order-id",
            "client_order_id": "test-client-order-id",
            "symbol": "AAPL",
            "side": "buy",
            "type": "market",
            "qty": "10",
            "status": "accepted"
        })
        mock_order_placement_context = AsyncMock()
        mock_order_placement_context.__aenter__ = AsyncMock(return_value=mock_order_placement_response)
        mock_order_placement_context.__aexit__ = AsyncMock(return_value=None)
        
        # Create mock response for bars endpoint
        mock_bars_response = AsyncMock()
        mock_bars_response.status = 200
        mock_bars_response.json = AsyncMock(return_value={
            "bars": {
                "AAPL": [
                    {
                        "t": "2025-05-07T09:30:00Z",
                        "o": 148.5,
                        "h": 152.0,
                        "l": 148.0,
                        "c": 151.5,
                        "v": 5000000,
                    },
                    {
                        "t": "2025-05-07T09:31:00Z",
                        "o": 151.5,
                        "h": 153.0,
                        "l": 151.0,
                        "c": 152.5,
                        "v": 3000000,
                    }
                ]
            },
            "next_page_token": None
        })
        mock_bars_context = AsyncMock()
        mock_bars_context.__aenter__ = AsyncMock(return_value=mock_bars_response)
        mock_bars_context.__aexit__ = AsyncMock(return_value=None)
        
        # Create mock response for latest quote
        mock_quote_response = AsyncMock()
        mock_quote_response.status = 200
        mock_quote_response.json = AsyncMock(return_value={
            "quote": {
                "t": "2025-05-07T13:30:00Z",
                "bp": 151.90,  # Bid price
                "bs": 100,     # Bid size
                "ap": 152.00,  # Ask price
                "as": 200      # Ask size
            }
        })
        mock_quote_context = AsyncMock()
        mock_quote_context.__aenter__ = AsyncMock(return_value=mock_quote_response)
        mock_quote_context.__aexit__ = AsyncMock(return_value=None)
        
        # Create mock response for latest trade
        mock_trade_response = AsyncMock()
        mock_trade_response.status = 200
        mock_trade_response.json = AsyncMock(return_value={
            "trade": {
                "t": "2025-05-07T13:30:00Z",
                "p": 151.95,  # Price
                "s": 150,     # Size
                "c": ["@", "T"],  # Conditions
                "i": 12345,   # Trade ID
                "x": "V",     # Exchange
                "v": 2000000  # Volume
            }
        })
        mock_trade_context = AsyncMock()
        mock_trade_context.__aenter__ = AsyncMock(return_value=mock_trade_response)
        mock_trade_context.__aexit__ = AsyncMock(return_value=None)
        
        # Assign contexts to get method
        mock_session = AsyncMock()
        
        # Configure get method based on URL
        def mock_get(url, **kwargs):
            if "/account" in url:
                return mock_account_context
            elif "/positions" in url:
                return mock_positions_context
            elif "/orders" in url:
                return mock_orders_context
            elif "/bars" in url:
                return mock_bars_context
            elif "/quotes/latest" in url:
                return mock_quote_context
            elif "/trades/latest" in url:
                return mock_trade_context
            else:
                mock_default = AsyncMock()
                mock_default.__aenter__ = AsyncMock(return_value=AsyncMock(status=404))
                mock_default.__aexit__ = AsyncMock(return_value=None)
                return mock_default
        
        # Configure post method
        def mock_post(url, **kwargs):
            if "/orders" in url:
                return mock_order_placement_context
            else:
                mock_default = AsyncMock()
                mock_default.__aenter__ = AsyncMock(return_value=AsyncMock(status=404))
                mock_default.__aexit__ = AsyncMock(return_value=None)
                return mock_default
        
        mock_session.get = mock_get
        mock_session.post = mock_post
        
        # Mock the ClientSession return value
        mock.return_value = mock_session
        mock.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock.return_value.__aexit__ = AsyncMock(return_value=None)
        
        yield mock


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
        
        # Patch the global variable
        with patch('mcp.alpaca_server.redis_client', mock.return_value):
            yield mock


@pytest.fixture
def mock_pandas():
    """Mock pandas for testing DataFrame operations"""
    with patch('pandas.DataFrame') as mock:
        mock_df = MagicMock()
        mock_df.__getitem__.return_value = mock_df
        mock_df.shift.return_value = mock_df
        mock_df.rolling.return_value.mean.return_value.iloc = [2.5]
        mock.return_value = mock_df
        yield mock


@pytest.fixture
def mock_scheduler():
    """Mock scheduler for testing async operations"""
    with patch('mcp.alpaca_server.AsyncIOScheduler') as mock:
        mock_instance = MagicMock()
        mock_instance.add_job = MagicMock()
        mock_instance.start = MagicMock()
        mock_instance.shutdown = MagicMock()
        mock.return_value = mock_instance
        yield mock


def test_load_config(mock_env_vars):
    """Test configuration loading"""
    with patch('os.path.join', return_value="nonexistent_path"), \
         patch('yaml.safe_load', return_value={}):
        config = load_config()
        
        assert config["api_key"] == "test_api_key"
        assert config["secret_key"] == "test_secret_key"
        assert config["trading_api_base"] == "https://paper-api.test-alpaca.markets/v2"
        assert config["paper_trading"] is True
        assert config["max_risk_per_trade_pct"] == 0.02
        assert config["min_reward_risk_ratio"] == 2.0
        assert config["max_position_value_pct"] == 0.10


@pytest.mark.asyncio
async def test_connect_redis(mock_redis, mock_env_vars):
    """Test Redis connection function"""
    await connect_redis()
    mock_redis.assert_called_once()
    mock_redis.return_value.ping.assert_called_once()


@pytest.mark.asyncio
async def test_get_account(mock_aiohttp_session):
    """Test getting account information"""
    result = await get_account()
    
    assert result["account_number"] == "ABC123456"
    assert result["buying_power"] == "100000"
    assert result["cash"] == "50000"
    assert result["portfolio_value"] == "150000"


@pytest.mark.asyncio
async def test_get_positions(mock_aiohttp_session):
    """Test getting current positions"""
    result = await get_positions()
    
    assert len(result) == 2
    assert result[0]["symbol"] == "AAPL"
    assert result[0]["qty"] == "100"
    assert result[1]["symbol"] == "MSFT"
    assert result[1]["qty"] == "50"


@pytest.mark.asyncio
async def test_get_orders(mock_aiohttp_session):
    """Test getting open orders"""
    result = await get_orders()
    
    assert len(result) == 1
    assert result[0]["symbol"] == "GOOG"
    assert result[0]["side"] == "buy"
    assert result[0]["limit_price"] == "2750.00"


@pytest.mark.asyncio
async def test_place_order(mock_aiohttp_session):
    """Test placing a new order"""
    result = await place_order(
        symbol="AAPL", 
        qty=10.0, 
        side="buy", 
        type="market", 
        time_in_force="day"
    )
    
    assert result["symbol"] == "AAPL"
    assert result["side"] == "buy"
    assert result["type"] == "market"
    assert result["qty"] == "10"
    assert result["status"] == "accepted"


@pytest.mark.asyncio
async def test_get_bars(mock_aiohttp_session):
    """Test getting historical bars data"""
    result = await get_bars(
        symbols="AAPL", 
        timeframe="1Min", 
        start="2025-05-07", 
        end="2025-05-08"
    )
    
    assert "bars" in result
    assert "AAPL" in result["bars"]
    assert len(result["bars"]["AAPL"]) == 2
    assert result["bars"]["AAPL"][0]["o"] == 148.5
    assert result["bars"]["AAPL"][0]["c"] == 151.5


@pytest.mark.asyncio
async def test_get_latest_quote(mock_aiohttp_session):
    """Test getting the latest quote for a symbol"""
    result = await get_latest_quote("AAPL")
    
    assert "quote" in result
    assert result["quote"]["bp"] == 151.90
    assert result["quote"]["ap"] == 152.00


@pytest.mark.asyncio
async def test_get_latest_trade(mock_aiohttp_session):
    """Test getting the latest trade for a symbol"""
    result = await get_latest_trade("AAPL")
    
    assert "trade" in result
    assert result["trade"]["p"] == 151.95
    assert result["trade"]["s"] == 150


@pytest.mark.asyncio
async def test_assess_trade_risk_detailed(mock_aiohttp_session, mock_pandas):
    """Test the risk assessment for a potential trade"""
    with patch('mcp.alpaca_server.calculate_stop_loss_atr', AsyncMock(return_value=145.0)), \
         patch('mcp.alpaca_server.calculate_position_size', AsyncMock(return_value=100)):
        result = await assess_trade_risk_detailed(
            symbol="AAPL",
            entry_price=150.0,
            side="long",
            stop_loss_price_manual=None,
            profit_target_manual=None
        )
        
        assert result["success"] is True
        assert "assessment_details" in result
        details = result["assessment_details"]
        assert details["symbol"] == "AAPL"
        assert details["entry_price"] == 150.0
        assert details["stop_loss_price"] == 145.0
        assert details["position_size_shares"] == 100


@pytest.mark.asyncio
async def test_get_portfolio_data(mock_aiohttp_session):
    """Test getting complete portfolio data"""
    with patch('mcp.alpaca_server.calculate_portfolio_metrics', AsyncMock(return_value={
            "portfolio_value": 150000.0,
            "cash": 50000.0,
            "invested_value": 100000.0,
            "daily_pl": 1500.0,
            "daily_pl_pct": 1.01,
            "position_count": 2,
            "largest_position": {
                "symbol": "MSFT",
                "value": 18000.0,
                "pct_of_portfolio": 12.0
            }
        })), \
        patch('mcp.alpaca_server.calculate_sector_exposure', AsyncMock(return_value={
            "sectors": {
                "Technology": {
                    "value": 33000.0,
                    "percentage": 22.0,
                    "positions": [
                        {"symbol": "AAPL", "value": 15000.0},
                        {"symbol": "MSFT", "value": 18000.0}
                    ]
                }
            },
            "total_value": 33000.0
        })):
        
        result = await get_portfolio_data()
        
        assert result["success"] is True
        assert "account_summary" in result
        assert "positions" in result
        assert "open_orders" in result
        assert "portfolio_metrics" in result
        assert "sector_exposure" in result
        assert result["positions"][0]["symbol"] == "AAPL"
        assert result["portfolio_metrics"]["portfolio_value"] == 150000.0
        assert result["sector_exposure"]["sectors"]["Technology"]["percentage"] == 22.0


def test_server_info_endpoint(test_client):
    """Test the server info endpoint"""
    response = test_client.get("/server_info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "alpaca"
    assert "version" in data
    assert "tools" in data
    assert len(data["tools"]) > 0


def test_health_endpoint(test_client):
    """Test the health check endpoint"""
    with patch('mcp.alpaca_server.redis_client', AsyncMock()), \
         patch('mcp.alpaca_server.get_account', AsyncMock(return_value={})), \
         patch('mcp.alpaca_server.scheduler', MagicMock(running=True)):
        
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "checks" in data
        assert "alpaca_api" in data["checks"]
        assert "redis" in data["checks"]


def test_api_authentication(test_client):
    """Test that endpoints require API key authentication"""
    # Without API key
    response = test_client.post("/get_account_info")
    assert response.status_code == 401
    
    # With valid API key
    with patch('mcp.alpaca_server.verify_api_key', MagicMock(return_value="test-key-123")):
        response = test_client.post(
            "/get_account_info",
            headers={"X-API-Key": "test-key-123"}
        )
        assert response.status_code != 401


if __name__ == "__main__":
    pytest.main(["-v", "test_alpaca_server.py"])