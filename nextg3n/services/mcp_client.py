"""
MCP Client for NextG3N Trading System

Implements a custom client for JSON-RPC over HTTP+SSE and Stdio to interact with MCP servers.
Supports AutoGen 0.9.0 for agent-based data access.
"""

import aiohttp
import json
import logging
import time
import asyncio
import subprocess
import sys
from typing import Dict, Any, List, Optional
from monitoring.metrics_logger import MetricsLogger

class StdioTransport:
    def __init__(self, command: List[str]):
        self.process: Optional[subprocess.Popen] = None
        self.command = command
        self.logger = logging.getLogger("StdioTransport")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    async def connect(self):
        if self.process is None or self.process.poll() is not None:
            self.logger.info(f"Starting stdio process: {' '.join(self.command)}")
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True, # Use text mode for universal newlines
                bufsize=1 # Line buffering
            )
            # Give the process a moment to start
            await asyncio.sleep(0.1)
            if self.process.poll() is not None:
                 raise Exception(f"Failed to start stdio process: {' '.join(self.command)}. Return code: {self.process.returncode}")


    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        await self.connect() # Ensure process is running
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
             raise Exception("Stdio process not started or pipes not available")

        request = {
            "jsonrpc": "2.0",
            "id": int(time.time()),
            "method": method,
            "params": params
        }
        request_str = json.dumps(request) + "\n"
        self.logger.debug(f"Sending stdio request: {request_str.strip()}")

        try:
            self.process.stdin.write(request_str)
            self.process.stdin.flush()

            # Read response line by line until a complete JSON object is received
            response_str = ""
            while True:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, self.process.stdout.readline
                )
                if not line:
                    # Process exited or pipe closed
                    raise Exception("Stdio process exited unexpectedly")
                response_str += line
                try:
                    response = json.loads(response_str)
                    self.logger.debug(f"Received stdio response: {response_str.strip()}")
                    return response
                except json.JSONDecodeError:
                    # Not a complete JSON object yet, continue reading
                    pass

        except Exception as e:
            self.logger.error(f"Error sending/receiving stdio request: {e}")
            self.shutdown() # Attempt to clean up
            raise

    def shutdown(self):
        if self.process and self.process.poll() is None:
            self.logger.info("Shutting down stdio process")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


class MCPClient:
    def __init__(self, config: Dict[str, Any]):
        self.logger = MetricsLogger(component_name="mcp_client")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.config = config
        self.mcp_servers = config.get("mcp_servers", {})
        self.http_session = aiohttp.ClientSession()
        self.stdio_transports: Dict[str, StdioTransport] = {}

    async def _get_transport(self, server_name: str):
        server_config = self.mcp_servers.get(server_name, {})
        transport_type = server_config.get("transport", "sse") # Default to sse

        if transport_type == "sse":
            url = server_config.get("url")
            if not url:
                 # Fallback to endpoint if url is not provided for sse
                 url = server_config.get("endpoint", f"http://localhost:8002/{server_name}")
                 self.logger.warning(f"Using 'endpoint' as fallback for 'url' for SSE server {server_name}. Please update config to use 'url'.")

            if not url:
                 raise ValueError(f"URL or endpoint must be provided for SSE transport for server {server_name}")

            # For SSE, we use the aiohttp session directly, no separate transport object needed here
            return "sse", url, server_config.get("headers", {})

        elif transport_type == "stdio":
            command = server_config.get("command")
            if not command or not isinstance(command, list):
                raise ValueError(f"Command (list of strings) must be provided for stdio transport for server {server_name}")

            if server_name not in self.stdio_transports:
                self.stdio_transports[server_name] = StdioTransport(command)

            await self.stdio_transports[server_name].connect()
            return "stdio", self.stdio_transports[server_name], None # Return the transport object

        else:
            raise ValueError(f"Unsupported transport type: {transport_type} for server {server_name}")


    async def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        transport_type, transport_info, headers = await self._get_transport(server_name)

        if transport_type == "sse":
            endpoint = f"{transport_info}/tools/list"
            try:
                async with self.http_session.get(endpoint, headers=headers) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to list tools for {server_name} (SSE): {response.status}")
                        return []
                    data = await response.json()
                    return data.get("tools", [])
            except Exception as e:
                self.logger.error(f"Error listing tools for {server_name} (SSE): {e}")
                return []

        elif transport_type == "stdio":
            stdio_transport: StdioTransport = transport_info
            try:
                response = await stdio_transport.send_request("list_tools", {})
                if response.get("error"):
                    self.logger.error(f"Failed to list tools for {server_name} (Stdio): {response['error']}")
                    return []
                return response.get("result", {}).get("tools", [])
            except Exception as e:
                self.logger.error(f"Error listing tools for {server_name} (Stdio): {e}")
                return []
        else:
             return [] # Should not reach here


    async def call_tool(self, server_name: str, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        transport_type, transport_info, headers = await self._get_transport(server_name)

        if transport_type == "sse":
            endpoint = f"{transport_info}/tools/call"
            payload = {
                "jsonrpc": "2.0",
                "id": int(time.time()),
                "method": tool_name,
                "params": params
            }
            try:
                async with self.http_session.post(endpoint, json=payload, headers=headers) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to call {tool_name} on {server_name} (SSE): {response.status}")
                        return {"success": False, "error": f"HTTP {response.status}"}
                    data = await response.json()
                    # Assuming SSE tool call returns a JSON-RPC response structure
                    if data.get("error"):
                         return {"success": False, "error": data["error"]}
                    return {"success": True, "result": data.get("result", {})}
            except Exception as e:
                self.logger.error(f"Error calling {tool_name} on {server_name} (SSE): {e}")
                return {"success": False, "error": str(e)}

        elif transport_type == "stdio":
            stdio_transport: StdioTransport = transport_info
            try:
                response = await stdio_transport.send_request(tool_name, params)
                if response.get("error"):
                    self.logger.error(f"Failed to call {tool_name} on {server_name} (Stdio): {response['error']}")
                    return {"success": False, "error": response["error"]}
                return {"success": True, "result": response.get("result", {})}
            except Exception as e:
                self.logger.error(f"Error calling {tool_name} on {server_name} (Stdio): {e}")
                return {"success": False, "error": str(e)}
        else:
             return {"success": False, "error": "Unsupported transport type"} # Should not reach here


    async def read_resource(self, server_name: str, uri: str) -> Dict[str, Any]:
        transport_type, transport_info, headers = await self._get_transport(server_name)

        if transport_type == "sse":
            # Assuming SSE resource access is via a specific endpoint, e.g., /resources/{uri}
            # The document doesn't specify the exact SSE resource endpoint format,
            # so I'll make a reasonable assumption based on the tool endpoints.
            # A more robust implementation might require a specific resource endpoint config.
            resource_path = uri.replace("://", "/") # Simple conversion, might need refinement
            endpoint = f"{transport_info}/resources/{resource_path}"
            try:
                async with self.http_session.get(endpoint, headers=headers) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to read resource {uri} on {server_name} (SSE): {response.status}")
                        return {"success": False, "error": f"HTTP {response.status}"}
                    # Assuming resource content is returned directly or in a 'content' field
                    try:
                        data = await response.json()
                        return {"success": True, "content": data.get("content", data)} # Try 'content' field, fallback to full response
                    except aiohttp.ContentTypeError:
                         # Not JSON, return raw text
                         text = await response.text()
                         return {"success": True, "content": text}

            except Exception as e:
                self.logger.error(f"Error reading resource {uri} on {server_name} (SSE): {e}")
                return {"success": False, "error": str(e)}

        elif transport_type == "stdio":
            stdio_transport: StdioTransport = transport_info
            try:
                # Assuming stdio resource access is via a specific method, e.g., 'read_resource'
                response = await stdio_transport.send_request("read_resource", {"uri": uri})
                if response.get("error"):
                    self.logger.error(f"Failed to read resource {uri} on {server_name} (Stdio): {response['error']}")
                    return {"success": False, "error": response["error"]}
                # Assuming stdio resource response has a 'content' field in the result
                return {"success": True, "content": response.get("result", {}).get("content", {})}
            except Exception as e:
                self.logger.error(f"Error reading resource {uri} on {server_name} (Stdio): {e}")
                return {"success": False, "error": str(e)}
        else:
             return {"success": False, "error": "Unsupported transport type"} # Should not reach here


    async def shutdown(self):
        await self.http_session.close()
        for transport in self.stdio_transports.values():
            transport.shutdown()
        self.logger.info("MCPClient shutdown")