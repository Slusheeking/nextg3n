"""
Tests for the MCP Manager.
Tests core functionality by mocking external dependencies.
"""

import os
import sys
import json
import yaml
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock, mock_open
from pathlib import Path
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

# Add parent directory to path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path setup
from mcp.mcp_manager import MCPManager, app, mcp_manager


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for testing"""
    with patch('subprocess.Popen') as mock_popen:
        # Configure the mock process
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll = MagicMock(return_value=None)  # Process is running
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = MagicMock()
        mock_process.kill = MagicMock()
        
        # Set up the Popen mock to return our mock process
        mock_popen.return_value = mock_process
        yield mock_popen


@pytest.fixture
def mock_threading():
    """Mock threading for testing"""
    with patch('threading.Thread') as mock_thread:
        mock_thread_instance = MagicMock()
        mock_thread_instance.start = MagicMock()
        mock_thread.return_value = mock_thread_instance
        yield mock_thread


@pytest.fixture
def mock_asyncio():
    """Mock asyncio for testing"""
    with patch('asyncio.create_task', MagicMock()), \
         patch('asyncio.sleep', AsyncMock(return_value=None)), \
         patch('asyncio.CancelledError', Exception), \
         patch('asyncio.get_event_loop') as mock_loop, \
         patch('asyncio.run', MagicMock()):
        
        mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)
        yield mock_loop


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for testing API calls"""
    with patch('aiohttp.ClientSession') as mock:
        # Configure AsyncMock responses
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"name": "test_server", "version": "1.0.0"})
        mock_resp.text = AsyncMock(return_value=json.dumps({"name": "test_server", "version": "1.0.0"}))
        
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_resp
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_context)
        mock_session.post = AsyncMock(return_value=mock_context)
        mock_session.close = AsyncMock()
        
        mock.return_value = mock_session
        
        yield mock


@pytest.fixture
def mock_file_ops():
    """Mock file operations for testing"""
    mock_file = MagicMock()
    mock_file.write = MagicMock()
    mock_file.flush = MagicMock()
    mock_file.close = MagicMock()
    mock_file.seek = MagicMock()
    mock_file.readlines = MagicMock(return_value=["Test log line 1\n", "Test log line 2\n"])
    
    with patch('builtins.open', mock_open()), \
         patch('os.makedirs', MagicMock()), \
         patch('os.path.exists', MagicMock(return_value=True)), \
         patch('json.load', MagicMock(return_value={"server_stats": {}})), \
         patch('json.dump', MagicMock()):
        
        open.return_value = mock_file
        yield mock_file


@pytest.fixture(scope="function")
def test_config():
    """Provide test configuration for MCP Manager"""
    config = {
        "servers": [
            {
                "name": "test_server",
                "type": "test",
                "host": "localhost",
                "port": 8000,
                "enabled": True,
                "auto_start": True,
                "env": {
                    "TEST_ENV": "test_value"
                }
            },
            {
                "name": "disabled_server",
                "type": "disabled",
                "host": "localhost",
                "port": 8001,
                "enabled": False
            }
        ],
        "connection": {
            "timeout_seconds": 5,
            "retry_attempts": 2,
            "retry_delay_seconds": 1
        },
        "security": {
            "enable_auth": True,
            "api_key": "test_api_key"
        },
        "health_check": {
            "interval_seconds": 30,
            "auto_restart": True
        },
        "manager_port": 8080
    }
    return config


@pytest.fixture
def mcp_manager_instance(test_config, mock_file_ops):
    """Create an MCPManager instance with test configuration"""
    with patch('yaml.safe_load', MagicMock(return_value=test_config)), \
         patch('pathlib.Path.exists', MagicMock(return_value=True)):
        manager = MCPManager()
        yield manager


@pytest.mark.asyncio
async def test_load_config(mcp_manager_instance, test_config):
    """Test configuration loading"""
    config = mcp_manager_instance.config
    
    assert len(config["servers"]) == 2
    assert config["servers"][0]["name"] == "test_server"
    assert config["connection"]["timeout_seconds"] == 5
    assert config["security"]["api_key"] == "test_api_key"
    assert config["health_check"]["interval_seconds"] == 30


@pytest.mark.asyncio
async def test_validate_config():
    """Test configuration validation"""
    invalid_config = {
        "servers": [
            {
                # Missing name
                "type": "test",
                # Missing port
                "host": "localhost"
            }
        ],
        # Missing connection section
        "security": {
            "enable_auth": True
        },
        # Invalid health check interval
        "health_check": {
            "interval_seconds": -10
        }
    }
    
    with patch('yaml.safe_load', MagicMock(return_value=invalid_config)), \
         patch('pathlib.Path.exists', MagicMock(return_value=True)):
        manager = MCPManager()
        
        # Check that missing/invalid values have been corrected
        assert manager.config["servers"][0]["name"] == "server_0"
        assert manager.config["servers"][0]["port"] == 8000
        assert "connection" in manager.config
        assert manager.config["health_check"]["interval_seconds"] > 0


@pytest.mark.asyncio
async def test_start_server(mcp_manager_instance, mock_subprocess, mock_threading, mock_asyncio, mock_aiohttp_session, mock_file_ops):
    """Test starting an MCP server"""
    # Initialize the manager (would normally happen in FastAPI startup)
    mcp_manager_instance.session = mock_aiohttp_session.return_value
    
    result = await mcp_manager_instance.start_server("test_server")
    
    assert result is True
    assert "test_server" in mcp_manager_instance.processes
    assert "test_server" in mcp_manager_instance.servers
    assert "test_server" in mcp_manager_instance.output_threads
    assert "test_server" in mcp_manager_instance.server_stats
    
    # Verify the subprocess was started with correct arguments
    mock_subprocess.assert_called_once()
    call_args = mock_subprocess.call_args[0][0]
    assert "python" in call_args
    assert "mcp/test_server.py" in call_args or "mcp/test_server.py" == call_args[1]
    
    # Verify the output capture threads were started
    assert mock_threading.call_count == 2
    mock_threading.return_value.start.assert_called()


@pytest.mark.asyncio
async def test_stop_server(mcp_manager_instance, mock_subprocess, mock_file_ops):
    """Test stopping an MCP server"""
    # Set up a running server
    mock_process = mock_subprocess.return_value
    mcp_manager_instance.processes = {
        "test_server": mock_process
    }
    mcp_manager_instance.servers = {
        "test_server": {
            "info": {"name": "test_server", "version": "1.0.0"},
            "config": {"name": "test_server", "type": "test"},
            "last_health_check": datetime.utcnow().isoformat()
        }
    }
    mcp_manager_instance.server_stats = {
        "test_server": {
            "last_start": datetime.utcnow().isoformat(),
            "total_uptime": 3600,
            "start_count": 1
        }
    }
    
    # Mock file handles for output capture
    mock_stdout_file = MagicMock()
    mock_stderr_file = MagicMock()
    mcp_manager_instance.output_threads = {
        "test_server": {
            "stdout_file": mock_stdout_file,
            "stderr_file": mock_stderr_file
        }
    }
    
    result = await mcp_manager_instance.stop_server("test_server")
    
    assert result is True
    assert "test_server" not in mcp_manager_instance.processes
    assert "test_server" not in mcp_manager_instance.servers
    assert "test_server" not in mcp_manager_instance.output_threads
    
    # Verify the process was terminated
    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once()
    
    # Verify files were closed
    mock_stdout_file.close.assert_called_once()
    mock_stderr_file.close.assert_called_once()


@pytest.mark.asyncio
async def test_restart_server(mcp_manager_instance):
    """Test restarting an MCP server"""
    with patch.object(mcp_manager_instance, 'stop_server', AsyncMock(return_value=True)), \
         patch.object(mcp_manager_instance, 'start_server', AsyncMock(return_value=True)):
        
        result = await mcp_manager_instance.restart_server("test_server")
        
        assert result is True
        mcp_manager_instance.stop_server.assert_called_once_with("test_server")
        mcp_manager_instance.start_server.assert_called_once_with("test_server")


@pytest.mark.asyncio
async def test_execute_tool(mcp_manager_instance, mock_aiohttp_session):
    """Test executing a tool on an MCP server"""
    # Set up a running server
    mcp_manager_instance.session = mock_aiohttp_session.return_value
    mcp_manager_instance.servers = {
        "test_server": {
            "info": {"name": "test_server", "version": "1.0.0"},
            "config": {"name": "test_server", "host": "localhost", "port": 8000},
            "last_health_check": datetime.utcnow().isoformat()
        }
    }
    
    # Mock the tool response
    mock_response = {"result": "success", "data": {"value": 42}}
    mock_aiohttp_session.return_value.post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
    
    result = await mcp_manager_instance.execute_tool("test_server", "test_tool", {"param": "value"})
    
    # Verify the result
    assert result == mock_response
    
    # Verify the API call
    mock_aiohttp_session.return_value.post.assert_called_once()
    call_args = mock_aiohttp_session.return_value.post.call_args
    assert call_args[0][0] == "http://localhost:8000/execute_tool/test_tool"
    assert call_args[1]["json"] == {"param": "value"}


@pytest.mark.asyncio
async def test_get_resource(mcp_manager_instance, mock_aiohttp_session):
    """Test getting a resource from an MCP server"""
    # Set up a running server
    mcp_manager_instance.session = mock_aiohttp_session.return_value
    mcp_manager_instance.servers = {
        "test_server": {
            "info": {"name": "test_server", "version": "1.0.0"},
            "config": {"name": "test_server", "host": "localhost", "port": 8000},
            "last_health_check": datetime.utcnow().isoformat()
        }
    }
    
    # Mock the resource response
    mock_response = {"resource_data": {"key": "value"}}
    mock_aiohttp_session.return_value.get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
    
    result = await mcp_manager_instance.get_resource("test_server", "test/resource")
    
    # Verify the result
    assert result == mock_response
    
    # Verify the API call
    mock_aiohttp_session.return_value.get.assert_called_once()
    call_args = mock_aiohttp_session.return_value.get.call_args
    assert call_args[0][0] == "http://localhost:8000/resource/test/resource"


@pytest.mark.asyncio
async def test_health_check(mcp_manager_instance):
    """Test the health check functionality"""
    with patch.object(mcp_manager_instance, 'get_server_info', AsyncMock()) as mock_get_info, \
         patch.object(mcp_manager_instance, 'restart_server', AsyncMock()) as mock_restart:
        
        # Set up mock responses for get_server_info
        mock_get_info.side_effect = [
            # Healthy server
            {"name": "test_server", "version": "1.0.0"},
            # Unhealthy server
            {"error": "Connection failed"}
        ]
        
        # Add servers to the manager
        mcp_manager_instance.servers = {
            "test_server": {
                "info": {"name": "test_server", "version": "1.0.0"},
                "config": {"name": "test_server", "host": "localhost", "port": 8000},
                "last_health_check": datetime.utcnow().isoformat()
            },
            "failing_server": {
                "info": {"name": "failing_server", "version": "1.0.0"},
                "config": {"name": "failing_server", "host": "localhost", "port": 8001},
                "last_health_check": datetime.utcnow().isoformat()
            }
        }
        
        results = await mcp_manager_instance.health_check()
        
        # Verify the results
        assert "test_server" in results
        assert "failing_server" in results
        assert results["test_server"]["status"] == "ok"
        assert results["failing_server"]["status"] == "error"
        
        # Verify the unhealthy server was restarted
        mock_restart.assert_called_once_with("failing_server")


@pytest.mark.asyncio
async def test_get_server_logs(mcp_manager_instance, mock_file_ops):
    """Test getting logs from a server"""
    # Set up output files
    mcp_manager_instance.output_threads = {
        "test_server": {
            "stdout_file": mock_file_ops,
            "stderr_file": mock_file_ops
        }
    }
    
    # Call the method
    logs = mcp_manager_instance.get_server_logs("test_server", "both", 100)
    
    # Verify the result
    assert "stdout" in logs
    assert "stderr" in logs
    assert len(logs["stdout"]) == 2
    assert len(logs["stderr"]) == 2
    assert logs["stdout"][0] == "Test log line 1"
    assert logs["stderr"][0] == "Test log line 1"
    
    # Verify file operations
    assert mock_file_ops.seek.call_count == 2
    assert mock_file_ops.readlines.call_count == 2


def test_get_server_stats(mcp_manager_instance):
    """Test getting server statistics"""
    # Set up server stats
    now = datetime.utcnow()
    start_time = (now - timedelta(hours=1)).isoformat()
    mcp_manager_instance.servers = {
        "test_server": {
            "info": {"name": "test_server", "version": "1.0.0"},
            "config": {"name": "test_server"},
            "last_health_check": now.isoformat()
        }
    }
    mcp_manager_instance.server_stats = {
        "test_server": {
            "first_seen": (now - timedelta(days=7)).isoformat(),
            "total_uptime": 86400,  # 1 day in seconds
            "start_count": 10,
            "fail_count": 2,
            "last_start": start_time
        }
    }
    
    # Call the method
    stats = mcp_manager_instance.get_server_stats()
    
    # Verify the result
    assert "test_server" in stats
    assert stats["test_server"]["total_uptime"] == 86400
    assert stats["test_server"]["start_count"] == 10
    assert stats["test_server"]["fail_count"] == 2
    assert stats["test_server"]["last_start"] == start_time
    assert "current_uptime" in stats["test_server"]
    assert stats["test_server"]["current_uptime"] > 3500  # ~1 hour in seconds


def test_fastapi_endpoints():
    """Test the FastAPI endpoints"""
    with patch.object(mcp_manager, 'get_available_servers', return_value=[
            {"name": "test_server", "type": "test", "port": 8000},
            {"name": "disabled_server", "type": "disabled", "port": 8001, "enabled": False}
        ]), \
         patch.object(mcp_manager, 'get_running_servers', return_value={
            "test_server": {
                "info": {"name": "test_server", "version": "1.0.0"},
                "config": {"name": "test_server", "type": "test", "port": 8000},
            }
        }):
        
        client = TestClient(app)
        
        # Test GET /servers
        response = client.get("/servers")
        assert response.status_code == 200
        assert "available" in response.json()
        assert "running" in response.json()
        assert len(response.json()["available"]) == 2
        assert len(response.json()["running"]) == 1
        
        # Test GET /servers/{server_name}/info
        with patch.object(mcp_manager, 'get_server_info', AsyncMock(return_value={"name": "test_server", "version": "1.0.0"})):
            response = client.get("/servers/test_server/info")
            assert response.status_code == 200
            assert response.json()["name"] == "test_server"
        
        # Test for non-existent server
        response = client.get("/servers/non_existent_server/info")
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main(["-v", "test_mcp_manager.py"])