"""
MCP Manager for NextG3N Trading System

This module implements the MCPManager class for managing MCP servers,
including starting, stopping, and interacting with them.
Specifically configured for the Alpaca MCP Tool and AI-Redis MCP Tool.
"""

import os
import yaml
import logging
import asyncio
import subprocess
import signal
import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from copy import deepcopy
import aiohttp
from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn
import threading
import re

# Fallback logging if monitor.logging_utils is unavailable
try:
    from monitor.logging_utils import get_logger, get_logs, shutdown_logging
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
    
    def get_logs(level=None, service=None, limit=100, start_time=None, end_time=None):
        return []  # No logs available in fallback
    
    def shutdown_logging():
        logging.shutdown()

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger("mcp_manager")

# Load environment variables
load_dotenv()

class MCPManager:
    """
    Manager for MCP servers in the NextG3N Trading System.
    
    Provides functionality for:
    - Starting and stopping MCP servers
    - Executing tools on MCP servers
    - Accessing resources from MCP servers
    - Managing server configurations and health checks
    """
    
    def __init__(self, config_path: str = "config/mcp_config.yaml"):
        """
        Initialize the MCP Manager.
        
        Args:
            config_path: Path to the MCP configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.servers = {}
        self.processes = {}
        self.output_threads = {}
        self.health_check_task = None
        self.server_stats = {}
        self.state_file = "mcp/server_state.json"
        self.session = None
        self._load_persistent_state()
        logger.info(f"Initialized MCP Manager with config from {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load and merge MCP configuration from file with defaults.
        
        Returns:
            Configuration dictionary
        """
        default_config = {
            "servers": [
                {
                    "name": "alpaca_mcp",
                    "type": "alpaca_mcp",
                    "enabled": True,
                    "auto_start": True,
                    "host": "localhost",
                    "port": 8001,
                    "args": {
                        "config_path": "/home/ubuntu/nextg3n/config/llm_config.yaml"
                    },
                    "env": {
                        "ALPACA_API_KEY": os.environ.get("ALPACA_API_KEY", ""),
                        "ALPACA_API_SECRET": os.environ.get("ALPACA_API_SECRET", "")
                    }
                },
                {
                    "name": "ai_redis_mcp",
                    "type": "ai_redis_mcp",
                    "enabled": True,
                    "auto_start": True,
                    "host": "localhost",
                    "port": 8002,
                    "args": {
                        "config_path": "/home/ubuntu/nextg3n/config/llm_config.yaml"
                    }
                }
            ],
            "connection": {
                "timeout_seconds": 30,
                "retry_attempts": 3,
                "retry_delay_seconds": 1
            },
            "security": {
                "enable_auth": True,
                "api_key": os.environ.get("MCP_API_KEY", "")
            },
            "health_check": {
                "interval_seconds": 60,
                "auto_restart": True
            },
            "manager_port": 8080
        }
        
        try:
            config_path = Path(self.config_path)
            if not config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}, using default configuration")
                return default_config
            
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Deep merge default and loaded config
            def deep_merge(default: dict, update: dict) -> dict:
                merged = deepcopy(default)
                for key, value in update.items():
                    if isinstance(value, dict) and key in merged:
                        merged[key] = deep_merge(merged[key], value)
                    else:
                        merged[key] = value
                return merged
            
            config = deep_merge(default_config, config)
            
            # Process environment variables
            config_str = yaml.dump(config)
            for key, value in os.environ.items():
                placeholder = f"${{{key}}}"
                config_str = config_str.replace(placeholder, value)
            config = yaml.safe_load(config_str) or {}
            
            self._validate_config(config)
            logger.info(f"Loaded MCP configuration with {len(config.get('servers', []))} servers")
            return config
        except Exception as e:
            logger.error(f"Error loading MCP configuration: {str(e)}. Using default configuration")
            return default_config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration structure and values, fixing invalid entries.
        
        Args:
            config: Configuration dictionary to validate
        """
        required_sections = ["servers", "connection", "security", "health_check"]
        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing required configuration section: {section}")
                config[section] = {}
        
        for i, server in enumerate(config.get("servers", [])):
            if "name" not in server:
                logger.error(f"Server at index {i} missing required 'name' field")
                server["name"] = f"server_{i}"
            if "type" not in server:
                logger.warning(f"Server {server['name']} missing 'type' field")
                server["type"] = server["name"]
            if "port" not in server:
                logger.warning(f"Server {server['name']} missing 'port' field")
                server["port"] = 8000
            elif not isinstance(server["port"], int) or server["port"] <= 0:
                logger.warning(f"Server {server['name']} has invalid port: {server['port']}. Using default 8000")
                server["port"] = 8000
            if "host" not in server:
                server["host"] = "localhost"
        
        if config["connection"].get("timeout_seconds", 30) <= 0:
            config["connection"]["timeout_seconds"] = 30
        if config["connection"].get("retry_attempts", 3) <= 0:
            config["connection"]["retry_attempts"] = 3
        if config["connection"].get("retry_delay_seconds", 1) <= 0:
            config["connection"]["retry_delay_seconds"] = 1
        
        if config["health_check"].get("interval_seconds", 60) <= 0:
            config["health_check"]["interval_seconds"] = 60
    
    def _load_persistent_state(self) -> None:
        """
        Load persistent server state from file.
        """
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.server_stats = state.get("server_stats", {})
                    logger.info(f"Loaded persistent state for {len(self.server_stats)} servers")
        except Exception as e:
            logger.error(f"Error loading persistent state: {str(e)}")
            self.server_stats = {}
    
    def _save_persistent_state(self) -> None:
        """
        Save persistent server state to file.
        """
        try:
            state = {
                "server_stats": self.server_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            logger.debug("Saved persistent state")
        except Exception as e:
            logger.error(f"Error saving persistent state: {str(e)}")
    
    async def initialize(self):
        """Initialize the MCP Manager and start auto-start servers."""
        self.session = aiohttp.ClientSession()
        for server_config in self.config.get("servers", []):
            if server_config.get("enabled", True) and server_config.get("auto_start", False):
                await self.start_server(server_config["name"])
        await self._start_health_check_task()
        logger.info("MCP Manager initialization complete")
    
    async def shutdown(self):
        """Shutdown the MCP Manager and stop all servers."""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        for server_name in list(self.processes.keys()):
            await self.stop_server(server_name)
        
        if self.session:
            await self.session.close()
        
        self._save_persistent_state()
        logger.info("MCP Manager shutdown complete")
    
    async def _start_health_check_task(self):
        """Start the periodic health check task."""
        health_check_config = self.config.get("health_check", {})
        interval = health_check_config.get("interval_seconds", 60)
        
        async def health_check_loop():
            try:
                while True:
                    try:
                        await self.health_check()
                        for server_name, server in self.servers.items():
                            if server_name not in self.server_stats:
                                self.server_stats[server_name] = {
                                    "first_seen": datetime.utcnow().isoformat(),
                                    "total_uptime": 0,
                                    "start_count": 0,
                                    "fail_count": 0,
                                    "last_start": None
                                }
                        self._save_persistent_state()
                    except Exception as e:
                        logger.error(f"Error in health check loop: {str(e)}")
                    await asyncio.sleep(interval)
            finally:
                logger.info("Health check task terminated")
        
        self.health_check_task = asyncio.create_task(health_check_loop())
        logger.info(f"Started health check task with {interval} second interval")
    
    async def _capture_output(self, pipe, file, server_name: str, stream_type: str) -> None:
        """
        Capture and log output from a subprocess pipe asynchronously.
        
        Args:
            pipe: The subprocess pipe (stdout or stderr)
            file: The file object to write to
            server_name: Name of the server
            stream_type: Type of stream ('stdout' or 'stderr')
        """
        try:
            while True:
                line = await asyncio.get_event_loop().run_in_executor(None, pipe.readline)
                if not line:
                    break
                timestamp = datetime.now().isoformat()
                try:
                    file.write(f"[{timestamp}] {line}")
                    file.flush()
                    if stream_type == "stderr" and self._is_critical_error(line):
                        logger.error(f"Server {server_name} reported critical error: {line.strip()}")
                except UnicodeEncodeError:
                    logger.warning(f"Encoding error in {stream_type} for {server_name}: {line}")
        except Exception as e:
            logger.error(f"Error capturing {stream_type} from {server_name}: {str(e)}")
    
    def _is_critical_error(self, line: str) -> bool:
        """
        Check if a log line contains a critical error.
        
        Args:
            line: The log line to check
            
        Returns:
            True if the line contains a critical error indicator
        """
        critical_patterns = [r"error", r"exception", r"traceback", r"fail", r"critical"]
        lower_line = line.lower()
        if any(x in lower_line for x in ["debug", "info"]):
            if not any(x in lower_line for x in ["exception", "traceback"]):
                return False
        return any(re.search(pattern, lower_line) for pattern in critical_patterns)
    
    def get_server_logs(self, server_name: str, log_type: str = "both", lines: int = 100) -> Dict[str, Any]:
        """
        Get logs from a server.
        
        Args:
            server_name: Name of the server
            log_type: Type of logs to get ('stdout', 'stderr', or 'both')
            lines: Maximum number of lines to return
            
        Returns:
            Dictionary with logs
        """
        if server_name not in self.output_threads:
            return {"error": f"Server {server_name} is not running or has no log capture"}
        
        try:
            result = {"stdout": [], "stderr": []}
            if log_type in ["stdout", "both"]:
                stdout_file = self.output_threads[server_name]["stdout_file"]
                stdout_file.seek(0)
                result["stdout"] = self._get_last_lines(stdout_file, lines)
            if log_type in ["stderr", "both"]:
                stderr_file = self.output_threads[server_name]["stderr_file"]
                stderr_file.seek(0)
                result["stderr"] = self._get_last_lines(stderr_file, lines)
            return result
        except Exception as e:
            logger.error(f"Error getting logs for {server_name}: {str(e)}")
            return {"error": str(e)}
    
    def _get_last_lines(self, file, n: int) -> List[str]:
        """
        Get the last n lines from a file.
        
        Args:
            file: File object
            n: Number of lines to get
            
        Returns:
            List of lines
        """
        lines = file.readlines()
        return [line.rstrip() for line in lines[-n:] if line.strip()]
    
    def get_server_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all tracked servers.
        
        Returns:
            Dictionary of server statistics
        """
        for server_name in self.servers.keys():
            if server_name in self.server_stats:
                last_start = self.server_stats[server_name].get("last_start")
                if last_start:
                    try:
                        start_time = datetime.fromisoformat(last_start)
                        current_uptime = (datetime.utcnow() - start_time).total_seconds()
                        self.server_stats[server_name]["current_uptime"] = current_uptime
                    except (ValueError, TypeError):
                        pass
        return self.server_stats
    
    async def start_server(self, server_name: str) -> bool:
        """
        Start an MCP server.
        
        Args:
            server_name: Name of the server to start
            
        Returns:
            True if server started successfully, False otherwise
        """
        server_config = next((config for config in self.config.get("servers", []) if config["name"] == server_name), None)
        if not server_config:
            logger.error(f"Server {server_name} not found in configuration")
            return False
        
        if not server_config.get("enabled", True):
            logger.warning(f"Server {server_name} is disabled in configuration")
            return False
        
        if server_name in self.processes:
            logger.warning(f"Server {server_name} is already running")
            return True
        
        try:
            server_type = server_config.get("type", server_name)
            script_path = f"mcp/{server_type}_tool.py"
            if not os.path.exists(script_path):
                script_path = f"mcp/{server_name}_tool.py"
            if not os.path.exists(script_path):
                logger.error(f"Script for server {server_name} (type: {server_type}) not found at {script_path}")
                return False
            
            cmd = ["python", script_path]
            if "args" in server_config:
                for arg_name, arg_value in server_config["args"].items():
                    cmd.extend([f"--{arg_name}", str(arg_value)])
            
            env = os.environ.copy()
            if "env" in server_config:
                env.update({str(k): str(v) for k, v in server_config["env"].items()})
            
            log_dir = "logs/servers"
            os.makedirs(log_dir, exist_ok=True)
            stdout_path = f"{log_dir}/{server_name}_stdout.log"
            stderr_path = f"{log_dir}/{server_name}_stderr.log"
            stdout_file = open(stdout_path, "a+", encoding="utf-8")
            stderr_file = open(stderr_path, "a+", encoding="utf-8")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            await asyncio.sleep(0.1)
            if process.poll() is not None:
                error_output = process.stderr.read() if process.stderr else ""
                logger.error(f"Server {server_name} failed to start: {error_output}")
                stdout_file.close()
                stderr_file.close()
                return False
            
            self.processes[server_name] = process
            logger.info(f"Started MCP server {server_name} (PID: {process.pid})")
            
            stdout_thread = threading.Thread(
                target=lambda: asyncio.run(self._capture_output(process.stdout, stdout_file, server_name, "stdout")),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=lambda: asyncio.run(self._capture_output(process.stderr, stderr_file, server_name, "stderr")),
                daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()
            
            self.output_threads[server_name] = {
                "stdout": stdout_thread,
                "stderr": stderr_thread,
                "stdout_file": stdout_file,
                "stderr_file": stderr_file
            }
            
            if server_name not in self.server_stats:
                self.server_stats[server_name] = {
                    "first_seen": datetime.utcnow().isoformat(),
                    "total_uptime": 0,
                    "start_count": 0,
                    "fail_count": 0,
                    "last_start": None
                }
            
            self.server_stats[server_name]["start_count"] += 1
            self.server_stats[server_name]["last_start"] = datetime.utcnow().isoformat()
            
            host = server_config.get("host", "localhost")
            port = server_config.get("port", 8000)
            connection_config = self.config.get("connection", {})
            retry_attempts = connection_config.get("retry_attempts", 3)
            retry_delay = connection_config.get("retry_delay_seconds", 1)
            
            for attempt in range(retry_attempts):
                try:
                    async with self.session.get(f"http://{host}:{port}/server_info", timeout=2) as response:
                        if response.status == 200:
                            server_info = await response.json()
                            self.servers[server_name] = {
                                "info": server_info,
                                "config": server_config,
                                "last_health_check": datetime.utcnow().isoformat()
                            }
                            logger.info(f"MCP server {server_name} is ready")
                            return True
                except Exception as e:
                    logger.debug(f"Server {server_name} not ready (attempt {attempt+1}/{retry_attempts}): {str(e)}")
                    await asyncio.sleep(retry_delay)
            
            logger.error(f"MCP server {server_name} failed to start after {retry_attempts} attempts")
            self.server_stats[server_name]["fail_count"] += 1
            await self.stop_server(server_name)
            return False
        except Exception as e:
            logger.error(f"Error starting MCP server {server_name}: {str(e)}")
            stdout_file.close()
            stderr_file.close()
            return False
    
    async def stop_server(self, server_name: str) -> bool:
        """
        Stop an MCP server.
        
        Args:
            server_name: Name of the server to stop
            
        Returns:
            True if server stopped successfully, False otherwise
        """
        if server_name not in self.processes:
            logger.warning(f"Server {server_name} is not running")
            return True
        
        try:
            process = self.processes[server_name]
            if server_name in self.servers and server_name in self.server_stats:
                last_start = self.server_stats[server_name]["last_start"]
                if last_start:
                    try:
                        start_time = datetime.fromisoformat(last_start)
                        uptime_seconds = (datetime.utcnow() - start_time).total_seconds()
                        self.server_stats[server_name]["total_uptime"] += uptime_seconds
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error calculating uptime for {server_name}: {str(e)}")
            
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"Force killed server {server_name}")
            
            if server_name in self.output_threads:
                thread_data = self.output_threads[server_name]
                if "stdout_file" in thread_data:
                    thread_data["stdout_file"].close()
                if "stderr_file" in thread_data:
                    thread_data["stderr_file"].close()
                del self.output_threads[server_name]
            
            if server_name in self.servers:
                del self.servers[server_name]
            
            del self.processes[server_name]
            self._save_persistent_state()
            logger.info(f"Stopped MCP server {server_name}")
            return True
        except Exception as e:
            logger.error(f"Error stopping MCP server {server_name}: {str(e)}")
            return False
    
    async def restart_server(self, server_name: str) -> bool:
        """
        Restart an MCP server.
        
        Args:
            server_name: Name of the server to restart
            
        Returns:
            True if server restarted successfully, False otherwise
        """
        await self.stop_server(server_name)
        return await self.start_server(server_name)
    
    async def execute_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool on an MCP server.
        
        Args:
            server_name: Name of the server
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        if server_name not in self.servers:
            logger.error(f"Server {server_name} is not running")
            return {"error": f"Server {server_name} is not running"}
        
        server_info = self.servers[server_name]
        host = server_info["config"].get("host", "localhost")
        port = server_info["config"].get("port", 8000)
        
        try:
            headers = {}
            if "api_key" in server_info["config"]:
                headers["Authorization"] = f"Bearer {server_info['config']['api_key']}"
            
            async with self.session.post(
                f"http://{host}:{port}/execute_tool/{tool_name}",
                json=arguments,
                headers=headers,
                timeout=self.config.get("connection", {}).get("timeout_seconds", 30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                error_text = await response.text()
                logger.error(f"Error executing tool {tool_name} on server {server_name}: {error_text}")
                return {"error": f"HTTP {response.status}: {error_text}"}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name} on server {server_name}: {str(e)}")
            return {"error": str(e)}
    
    async def get_server_info(self, server_name: str) -> Dict[str, Any]:
        """
        Get information about an MCP server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            Server information
        """
        if not any(config["name"] == server_name for config in self.config.get("servers", [])):
            logger.error(f"Server {server_name} not found in configuration")
            return {"error": f"Server {server_name} not found in configuration"}
        
        if server_name not in self.servers:
            logger.error(f"Server {server_name} is not running")
            return {"error": f"Server {server_name} is not running"}
        
        server_info = self.servers[server_name]
        host = server_info["config"].get("host", "localhost")
        port = server_info["config"].get("port", 8000)
        
        try:
            headers = {}
            if "api_key" in server_info["config"]:
                headers["Authorization"] = f"Bearer {server_info['config']['api_key']}"
            
            async with self.session.get(
                f"http://{host}:{port}/server_info",
                headers=headers,
                timeout=self.config.get("connection", {}).get("timeout_seconds", 30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.servers[server_name]["info"] = result
                    self.servers[server_name]["last_health_check"] = datetime.utcnow().isoformat()
                    return result
                error_text = await response.text()
                logger.error(f"Error getting server info for {server_name}: {error_text}")
                return {"error": f"HTTP {response.status}: {error_text}"}
        except Exception as e:
            logger.error(f"Error getting server info for {server_name}: {str(e)}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health checks on all running MCP servers.
        
        Returns:
            Health check results
        """
        results = {}
        for server_name in list(self.servers.keys()):
            try:
                server_info = await self.get_server_info(server_name)
                if "error" in server_info:
                    results[server_name] = {
                        "status": "error",
                        "error": server_info["error"],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    logger.warning(f"Restarting unresponsive server {server_name}")
                    await self.restart_server(server_name)
                else:
                    results[server_name] = {
                        "status": "ok",
                        "info": server_info,
                        "timestamp": datetime.utcnow().isoformat()
                    }
            except Exception as e:
                results[server_name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
        return results
    
    def get_available_servers(self) -> List[Dict[str, Any]]:
        """
        Get a list of available MCP servers from the configuration.
        
        Returns:
            List of server configurations
        """
        return self.config.get("servers", [])
    
    def get_running_servers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of running MCP servers.
        
        Returns:
            Dictionary of server information
        """
        return self.servers
    
    async def execute_alpaca_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool on the Alpaca MCP server.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        return await self.execute_tool("alpaca_mcp", tool_name, arguments)
    
    async def execute_ai_redis_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool on the AI-Redis MCP server.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        return await self.execute_tool("ai_redis_mcp", tool_name, arguments)

# Create singleton instance
mcp_manager = MCPManager()

if __name__ == "__main__":
    app = FastAPI(
        title="MCP Manager API",
        description="API for managing MCP servers in the NextG3N Trading System",
        version="1.0.0"
    )
    
    # Security
    security = HTTPBearer(auto_error=False)
    
    async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
        """Verify API key if authentication is enabled."""
        security_config = mcp_manager.config.get("security", {})
        if not security_config.get("enable_auth", True):
            return
        
        if not credentials:
            raise HTTPException(status_code=401, detail="API key required")
        
        valid_api_keys = os.getenv("VALID_API_KEYS", security_config.get("api_key", "")).split(",")
        if not valid_api_keys or credentials.credentials not in valid_api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return credentials.credentials
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {str(exc)}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    
    @app.on_event("startup")
    async def startup():
        await mcp_manager.initialize()
    
    @app.on_event("shutdown")
    async def shutdown():
        await mcp_manager.shutdown()
        shutdown_logging()
    
    @app.get("/servers", tags=["Servers"])
    async def get_servers():
        """Get available and running servers."""
        return {
            "available": mcp_manager.get_available_servers(),
            "running": mcp_manager.get_running_servers()
        }
    
    @app.post("/servers/{server_name}/start", tags=["Servers"])
    async def start_server(server_name: str):
        """Start a specific server."""
        result = await mcp_manager.start_server(server_name)
        if result:
            return {"status": "success", "message": f"Server {server_name} started"}
        raise HTTPException(status_code=500, detail=f"Failed to start server {server_name}")
    
    @app.post("/servers/{server_name}/stop", tags=["Servers"])
    async def stop_server(server_name: str):
        """Stop a specific server."""
        result = await mcp_manager.stop_server(server_name)
        if result:
            return {"status": "success", "message": f"Server {server_name} stopped"}
        raise HTTPException(status_code=500, detail=f"Failed to stop server {server_name}")
    
    @app.post("/servers/{server_name}/restart", tags=["Servers"], dependencies=[Depends(verify_api_key)])
    async def restart_server(server_name: str, background_tasks: BackgroundTasks):
        """Schedule a server restart in the background."""
        if not any(config["name"] == server_name for config in mcp_manager.get_available_servers()):
            raise HTTPException(status_code=404, detail=f"Server {server_name} not found in configuration")
        task_id = f"restart_{server_name}_{int(time.time())}"
        background_tasks.add_task(mcp_manager.restart_server, server_name)
        return {"status": "success", "message": f"Server {server_name} restart scheduled", "task_id": task_id}
    
    @app.get("/servers/{server_name}/info", tags=["Servers"])
    async def get_server_info(server_name: str):
        """Get information about a specific server."""
        if not any(config["name"] == server_name for config in mcp_manager.get_available_servers()):
            raise HTTPException(status_code=404, detail=f"Server {server_name} not found in configuration")
        return await mcp_manager.get_server_info(server_name)
    
    @app.post("/alpaca/tools/{tool_name}", tags=["Alpaca Tools"], dependencies=[Depends(verify_api_key)])
    async def execute_alpaca_tool(tool_name: str, request: Request):
        """Execute a tool on the Alpaca MCP server."""
        arguments = await request.json()
        return await mcp_manager.execute_alpaca_tool(tool_name, arguments)
    
    @app.post("/ai_redis/tools/{tool_name}", tags=["AI-Redis Tools"], dependencies=[Depends(verify_api_key)])
    async def execute_ai_redis_tool(tool_name: str, request: Request):
        """Execute a tool on the AI-Redis MCP server."""
        arguments = await request.json()
        return await mcp_manager.execute_ai_redis_tool(tool_name, arguments)
    
    @app.get("/health", tags=["Health"], dependencies=[Depends(verify_api_key)])
    async def health_check():
        """Perform health checks on all running servers."""
        return await mcp_manager.health_check()
    
    @app.get("/servers/{server_name}/logs", tags=["Servers"], dependencies=[Depends(verify_api_key)])
    async def get_server_logs(server_name: str, log_type: str = "both", lines: int = 100):
        """Get logs from a specific server."""
        return mcp_manager.get_server_logs(server_name, log_type, lines)
    
    @app.get("/servers/stats", tags=["Servers"], dependencies=[Depends(verify_api_key)])
    async def get_server_stats():
        """Get statistics for all tracked servers."""
        return mcp_manager.get_server_stats()
    
    @app.get("/market_data/{symbol}", tags=["Market Data"], dependencies=[Depends(verify_api_key)])
    async def get_market_data(symbol: str):
        """Get market data for a specific symbol using the AI-Redis MCP Tool."""
        return await mcp_manager.execute_ai_redis_tool("get_market_data", {"symbol": symbol})
    
    @app.get("/positions", tags=["Portfolio"], dependencies=[Depends(verify_api_key)])
    async def get_positions():
        """Get all active positions using the AI-Redis MCP Tool."""
        return await mcp_manager.execute_ai_redis_tool("get_all_positions", {})
    
    @app.get("/account", tags=["Portfolio"], dependencies=[Depends(verify_api_key)])
    async def get_account_info():
        """Get account information using the AI-Redis MCP Tool."""
        return await mcp_manager.execute_ai_redis_tool("get_account_info", {})
    
    @app.post("/orders", tags=["Trading"], dependencies=[Depends(verify_api_key)])
    async def submit_order(request: Request):
        """Submit a trading order using the Alpaca MCP Tool."""
        order_data = await request.json()
        return await mcp_manager.execute_alpaca_tool("submit_order", order_data)
    
    port = mcp_manager.config.get("manager_port", 8080)
    uvicorn.run(app, host="0.0.0.0", port=port)