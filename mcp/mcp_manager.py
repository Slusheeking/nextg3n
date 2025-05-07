"""
MCP Manager for NextG3N Trading System

This module implements the MCPManager class for managing MCP servers,
including starting, stopping, and interacting with them.
"""

import os
import yaml
import logging
import asyncio
import subprocess
import signal
import json
import time
import sys
import threading
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import aiohttp
from dotenv import load_dotenv
from pathlib import Path
from monitor.logging_utils import get_logger, get_logs, shutdown_logging

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Get logger from centralized logging system
logger = get_logger("mcp_manager")

# Load environment variables
load_dotenv()

class MCPManager:
    """
    Manager for MCP servers.
    
    Provides functionality for:
    - Starting and stopping MCP servers
    - Executing tools on MCP servers
    - Accessing resources from MCP servers
    - Managing server configurations
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
        Load the MCP configuration from file.
        
        Returns:
            Configuration dictionary
        """
        default_config = {
            "servers": [],
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
            }
        }
        
        try:
            # Check if config file exists
            config_path = Path(self.config_path)
            if not config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}, using default configuration")
                return default_config
                
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not config:
                logger.warning("Empty config file, using default configuration")
                return default_config
            
            # Process environment variables in config
            config_str = yaml.dump(config)
            for key, value in os.environ.items():
                placeholder = f"${{{key}}}"
                if placeholder in config_str:
                    config_str = config_str.replace(placeholder, value)
            
            config = yaml.safe_load(config_str)
            
            # Validate configuration
            self._validate_config(config)
            
            logger.info(f"Loaded MCP configuration with {len(config.get('servers', []))} servers")
            return config
        except Exception as e:
            logger.error(f"Error loading MCP configuration: {str(e)}")
            return default_config
            
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration structure and values.
        
        Args:
            config: Configuration dictionary to validate
        """
        # Check if required sections exist
        required_sections = ["servers", "connection", "security"]
        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing required configuration section: {section}")
                config[section] = {}
        
        # Validate servers configuration
        for i, server in enumerate(config.get("servers", [])):
            if "name" not in server:
                logger.warning(f"Server at index {i} missing required 'name' field")
            if "type" not in server:
                logger.warning(f"Server {server.get('name', f'at index {i}')} missing 'type' field")
            if "port" in server and not isinstance(server["port"], int):
                logger.warning(f"Server {server.get('name')} has non-integer port: {server['port']}")

    def _load_persistent_state(self) -> None:
        """
        Load persistent server state from file.
        """
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
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
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug("Saved persistent state")
        except Exception as e:
            logger.error(f"Error saving persistent state: {str(e)}")
    
    async def initialize(self):
        """Initialize the MCP Manager and start auto-start servers."""
        self.session = aiohttp.ClientSession()
        
        # Start auto-start servers
        for server_config in self.config.get("servers", []):
            if server_config.get("enabled", True) and server_config.get("auto_start", False):
                await self.start_server(server_config["name"])
        
        # Start health check task
        await self._start_health_check_task()
        logger.info("MCP Manager initialization complete")
    
    async def shutdown(self):
        """Shutdown the MCP Manager and stop all servers."""
        # Cancel health check task if running
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Stop all running servers
        for server_name in list(self.processes.keys()):
            await self.stop_server(server_name)
        
        # Close session
        if self.session:
            await self.session.close()
        
        # Save persistent state
        self._save_persistent_state()
        
        logger.info("MCP Manager shutdown complete")
        
    async def _start_health_check_task(self):
        """Start the periodic health check task."""
        health_check_config = self.config.get("health_check", {})
        interval = health_check_config.get("interval_seconds", 60)
        
        async def health_check_loop():
            while True:
                try:
                    await self.health_check()
                    # Update and save server stats
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
        
        self.health_check_task = asyncio.create_task(health_check_loop())
        logger.info(f"Started health check task with {interval} second interval")
    
    def _capture_output(self, pipe, file, server_name: str, stream_type: str) -> None:
        """
        Capture and log output from a subprocess pipe.
        
        Args:
            pipe: The subprocess pipe (stdout or stderr)
            file: The file object to write to
            server_name: Name of the server
            stream_type: Type of stream ('stdout' or 'stderr')
        """
        try:
            for line in iter(pipe.readline, ''):
                # Write to log file
                timestamp = datetime.now().isoformat()
                file.write(f"[{timestamp}] {line}")
                file.flush()
                
                # Log critical errors to main log
                if stream_type == "stderr" and self._is_critical_error(line):
                    logger.error(f"Server {server_name} reported critical error: {line.strip()}")
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
        critical_patterns = [
            r"error",
            r"exception",
            r"traceback",
            r"fail",
            r"critical"
        ]
        
        lower_line = line.lower()
        # Skip debug and info messages that contain error-like terms
        if any(x in lower_line for x in ["debug", "info"]):
            # But still check for actual exceptions
            if not any(x in lower_line for x in ["exception", "traceback"]):
                return False
                
        return any(re.search(pattern, lower_line) for pattern in critical_patterns)
    
    def get_server_logs(self, server_name: str, log_type: str = "both",
                      lines: int = 100) -> Dict[str, Any]:
        """
        Get logs from a server.
        
        Args:
            server_name: Name of the server
            log_type: Type of logs to get ('stdout', 'stderr', or 'both')
            lines: Maximum number of lines to return
            
        Returns:
            Dictionary with logs
        """
        result = {"stdout": [], "stderr": []}
        
        if server_name not in self.output_threads:
            return {"error": f"Server {server_name} is not running or has no log capture"}
        
        try:
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
        # Simple implementation that reads all lines and returns the last n
        lines = file.readlines()
        return [line.rstrip() for line in lines[-n:] if line.strip()]
        
    def get_server_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all tracked servers.
        
        Returns:
            Dictionary of server statistics
        """
        # Update uptime for all currently running servers
        for server_name in self.servers.keys():
            if server_name in self.server_stats:
                last_start = self.server_stats[server_name].get("last_start")
                if last_start:
                    try:
                        start_time = datetime.fromisoformat(last_start)
                        current_uptime = (datetime.utcnow() - start_time).total_seconds()
                        # For running servers, include current session in total
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
        # Find server config
        server_config = None
        for config in self.config.get("servers", []):
            if config["name"] == server_name:
                server_config = config
                break
        
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
            # Get server type from config
            server_type = server_config.get("type", server_name)
            
            # Dynamically determine the script path
            script_path = f"mcp/{server_type}_server.py"
            
            # Fallback to using the server name if type-based script doesn't exist
            if not os.path.exists(script_path):
                script_path = f"mcp/{server_name}_server.py"
            
            if not os.path.exists(script_path):
                logger.error(f"Script for server {server_name} (type: {server_type}) not found at {script_path}")
                return False

            # Build command with arguments from config
            cmd = ["python", script_path]
            
            # Add any additional arguments from configuration
            if "args" in server_config:
                for arg_name, arg_value in server_config["args"].items():
                    cmd.extend([f"--{arg_name}", str(arg_value)])
            
            # Set environment variables for the process
            env = os.environ.copy()
            if "env" in server_config:
                for env_name, env_value in server_config["env"].items():
                    env[env_name] = str(env_value)
                    
            # Create log directory if needed
            log_dir = "logs/servers"
            os.makedirs(log_dir, exist_ok=True)
            
            # Open log files
            stdout_path = f"{log_dir}/{server_name}_stdout.log"
            stderr_path = f"{log_dir}/{server_name}_stderr.log"
            stdout_file = open(stdout_path, "a+")
            stderr_file = open(stderr_path, "a+")
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            self.processes[server_name] = process
            logger.info(f"Started MCP server {server_name} (PID: {process.pid})")
            
            # Start output capturing threads
            stdout_thread = threading.Thread(
                target=self._capture_output,
                args=(process.stdout, stdout_file, server_name, "stdout"),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=self._capture_output,
                args=(process.stderr, stderr_file, server_name, "stderr"),
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
            
            # Update server stats
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
            
            # Wait for server to start
            host = server_config.get("host", "localhost")
            port = server_config.get("port", 8000)
            
            # Get connection settings
            connection_config = self.config.get("connection", {})
            retry_attempts = connection_config.get("retry_attempts", 3)
            retry_delay = connection_config.get("retry_delay_seconds", 1)
            
            # Try multiple times based on configuration
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
            
            # If we get here, server didn't start properly
            logger.error(f"MCP server {server_name} failed to start after {retry_attempts} attempts")
            
            # Update failure stats
            if server_name in self.server_stats:
                self.server_stats[server_name]["fail_count"] += 1
            
            # Clean up
            await self.stop_server(server_name)
            return False
            
        except Exception as e:
            logger.error(f"Error starting MCP server {server_name}: {str(e)}")
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
            
            # Update uptime stats if server exists in our tracking
            if server_name in self.servers and server_name in self.server_stats:
                last_start = self.server_stats[server_name]["last_start"]
                if last_start:
                    try:
                        start_time = datetime.fromisoformat(last_start)
                        uptime_seconds = (datetime.utcnow() - start_time).total_seconds()
                        self.server_stats[server_name]["total_uptime"] += uptime_seconds
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error calculating uptime for {server_name}: {str(e)}")
            
            # Try graceful shutdown first
            process.terminate()
            
            # Wait for process to terminate
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if not terminated
                process.kill()
                logger.warning(f"Force killed server {server_name}")
            
            # Clean up output threads and files
            if server_name in self.output_threads:
                thread_data = self.output_threads[server_name]
                
                # Close files
                if "stdout_file" in thread_data:
                    thread_data["stdout_file"].close()
                if "stderr_file" in thread_data:
                    thread_data["stderr_file"].close()
                
                # Remove thread data
                del self.output_threads[server_name]
            
            # Clean up server data
            if server_name in self.servers:
                del self.servers[server_name]
            
            del self.processes[server_name]
            
            # Save updated stats
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
                else:
                    error_text = await response.text()
                    logger.error(f"Error executing tool {tool_name} on server {server_name}: {error_text}")
                    return {"error": f"HTTP {response.status}: {error_text}"}
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name} on server {server_name}: {str(e)}")
            return {"error": str(e)}
    
    async def get_resource(self, server_name: str, resource_uri: str) -> Dict[str, Any]:
        """
        Get a resource from an MCP server.
        
        Args:
            server_name: Name of the server
            resource_uri: URI of the resource
            
        Returns:
            Resource content
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
            
            async with self.session.get(
                f"http://{host}:{port}/resource/{resource_uri}",
                headers=headers,
                timeout=self.config.get("connection", {}).get("timeout_seconds", 30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Error getting resource {resource_uri} from server {server_name}: {error_text}")
                    return {"error": f"HTTP {response.status}: {error_text}"}
                
        except Exception as e:
            logger.error(f"Error getting resource {resource_uri} from server {server_name}: {str(e)}")
            return {"error": str(e)}
    
    async def get_server_info(self, server_name: str) -> Dict[str, Any]:
        """
        Get information about an MCP server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            Server information
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
            
            async with self.session.get(
                f"http://{host}:{port}/server_info",
                headers=headers,
                timeout=self.config.get("connection", {}).get("timeout_seconds", 30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    # Update cached server info
                    self.servers[server_name]["info"] = result
                    self.servers[server_name]["last_health_check"] = datetime.utcnow().isoformat()
                    return result
                else:
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
                    
                    # Try to restart server if it's not responding
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


# Create singleton instance
mcp_manager = MCPManager()

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(title="MCP Manager API", version="1.0.0")
    security = HTTPBearer()
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Verify API key based on configuration
    async def verify_api_key(
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ):
        """Verify API key if authentication is enabled"""
        security_config = mcp_manager.config.get("security", {})
        
        # Skip auth check if disabled
        if not security_config.get("enable_auth", True):
            return
            
        api_key = security_config.get("api_key", "")
        if not api_key or credentials.credentials != api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    @app.on_event("startup")
    async def startup():
        await mcp_manager.initialize()
    
    @app.on_event("shutdown")
    async def shutdown():
        await mcp_manager.shutdown()
        # Shutdown logging system
        shutdown_logging()
    
    @app.get("/servers")
    async def get_servers():
        return {
            "available": mcp_manager.get_available_servers(),
            "running": mcp_manager.get_running_servers()
        }
    
    @app.post("/servers/{server_name}/start")
    async def start_server(server_name: str):
        result = await mcp_manager.start_server(server_name)
        if result:
            return {"status": "success", "message": f"Server {server_name} started"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to start server {server_name}")
        
    def background_restart(server_name: str):
        """Background task for server restart to allow endpoint to return quickly"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(mcp_manager.restart_server(server_name))
    
    @app.post("/servers/{server_name}/stop")
    async def stop_server(server_name: str):
        result = await mcp_manager.stop_server(server_name)
        if result:
            return {"status": "success", "message": f"Server {server_name} stopped"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to stop server {server_name}")
    
    @app.post("/servers/{server_name}/restart", dependencies=[Depends(verify_api_key)])
    async def restart_server(server_name: str, background_tasks: BackgroundTasks):
        # Validate that server exists in config
        server_exists = False
        for config in mcp_manager.get_available_servers():
            if config["name"] == server_name:
                server_exists = True
                break
        
        if not server_exists:
            raise HTTPException(status_code=404, detail=f"Server {server_name} not found in configuration")
            
        # Schedule restart in background to allow endpoint to return quickly
        background_tasks.add_task(background_restart, server_name)
        return {"status": "success", "message": f"Server {server_name} restart scheduled"}
    
    @app.get("/servers/{server_name}/info")
    async def get_server_info(server_name: str):
        return await mcp_manager.get_server_info(server_name)
    
    @app.post("/servers/{server_name}/tools/{tool_name}", dependencies=[Depends(verify_api_key)])
    async def execute_tool(server_name: str, tool_name: str, request: Request):
        arguments = await request.json()
        return await mcp_manager.execute_tool(server_name, tool_name, arguments)
    
    @app.get("/servers/{server_name}/resources/{resource_uri:path}", dependencies=[Depends(verify_api_key)])
    async def get_resource(server_name: str, resource_uri: str):
        return await mcp_manager.get_resource(server_name, resource_uri)
    
    @app.get("/health", dependencies=[Depends(verify_api_key)])
    async def health_check():
        return await mcp_manager.health_check()
    
    # Logging API endpoints
    @app.get("/logs", dependencies=[Depends(verify_api_key)])
    async def get_system_logs(
        level: Optional[str] = None,
        service: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ):
        """
        Get system logs with optional filtering.
        
        Args:
            level: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            service: Filter by service name
            limit: Maximum number of logs to return
            start_time: Filter logs after this time (ISO format)
            end_time: Filter logs before this time (ISO format)
        """
        logs = get_logs(level, service, limit, start_time, end_time)
        return {"logs": logs, "count": len(logs), "timestamp": datetime.utcnow().isoformat()}
    
    @app.get("/logs/services", dependencies=[Depends(verify_api_key)])
    async def get_log_services():
        """Get a list of all services that have logged messages"""
        all_logs = get_logs(limit=10000)
        services = sorted(list(set(log["logger"] for log in all_logs)))
        return {"services": services, "count": len(services)}
    
    @app.get("/logs/levels", dependencies=[Depends(verify_api_key)])
    async def get_log_levels():
        """Get a list of all log levels used"""
        all_logs = get_logs(limit=10000)
        levels = sorted(list(set(log["level"] for log in all_logs)))
        return {"levels": levels}
    
    # Server logs and stats endpoints
    @app.get("/servers/{server_name}/logs", dependencies=[Depends(verify_api_key)])
    async def get_server_logs(
        server_name: str,
        log_type: str = "both",
        lines: int = 100
    ):
        """
        Get logs from a specific server.
        
        Args:
            server_name: Name of the server
            log_type: Type of logs to get ('stdout', 'stderr', or 'both')
            lines: Maximum number of lines to return
        """
        return mcp_manager.get_server_logs(server_name, log_type, lines)
    
    @app.get("/servers/stats", dependencies=[Depends(verify_api_key)])
    async def get_server_stats():
        """Get statistics for all tracked servers."""
        return mcp_manager.get_server_stats()
    
    # Documentation endpoints
    @app.get("/docs/servers")
    async def get_server_docs():
        """Get documentation about available servers and their tools/resources."""
        available_servers = mcp_manager.get_available_servers()
        running_servers = mcp_manager.get_running_servers()
        
        result = {}
        for server_config in available_servers:
            server_name = server_config["name"]
            server_info = {
                "config": server_config,
                "status": "running" if server_name in running_servers else "stopped",
            }
            
            # Include tool and resource info if server is running
            if server_name in running_servers:
                server_info["info"] = running_servers[server_name]["info"]
                
            result[server_name] = server_info
            
        return result
    
    # Main entry point - get port from config or default to 8080
    port = mcp_manager.config.get("manager_port", 8080)
    uvicorn.run(app, host="0.0.0.0", port=port)
