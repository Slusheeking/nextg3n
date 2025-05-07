"""
MCP-enabled AutoGen Agents for NextG3N Trading System

Implements BaseMCPAgent and AssistantMCPAgent classes for integrating MCP tools
with AutoGen agents, following the pattern described in the integration guide.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from autogen import ConversableAgent, GroupChat, GroupChatManager, register_function
# Assuming mcp_client.py is in services/
from services.mcp_client import MCPClient

# Configuration class for MCP server parameters


class MCPServerParams:
    def __init__(self,
                 name: str,
                 transport: str,
                 command: Optional[List[str]] = None,
                 url: Optional[str] = None,
                 headers: Optional[Dict[str,
                                        str]] = None):
        self.name = name
        self.transport = transport
        self.command = command
        self.url = url
        self.headers = headers


class BaseMCPAgent:
    def __init__(self,
                 name: str,
                 mcp_client: MCPClient,
                 server_params: MCPServerParams,
                 llm_config: Dict[str,
                                  Any],
                 system_message: str):
        self.name = name
        self.mcp_client = mcp_client
        self.server_params = server_params
        self.llm_config = llm_config
        self.system_message = system_message
        self.logger = logging.getLogger(f"BaseMCPAgent.{name}")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Create caller and executor agents
        self.caller_agent = ConversableAgent(
            name=f"{name}_Caller",
            llm_config={
                "config_list": [llm_config]},
            system_message=system_message,
            is_termination_msg=lambda x: x.get(
                "content",
                "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER")
        self.executor_agent = ConversableAgent(
            name=f"{name}_Executor",
            llm_config=False,  # Executor does not use LLM
            is_termination_msg=lambda x: x.get(
                "content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            code_execution_config={"use_docker": False}
        )

        # Register MCP interface functions
        register_function(
            self.list_tools_sync,
            caller=self.caller_agent,
            executor=self.executor_agent,
            description="List available tools from the MCP server."
        )
        register_function(
            self.call_tool_sync,
            caller=self.caller_agent,
            executor=self.executor_agent,
            description="Call a tool on the MCP server. Available tools can be found using list_tools."
        )
        register_function(
            self.read_resource_sync,
            caller=self.caller_agent,
            executor=self.executor_agent,
            description="Read a resource from the MCP server."
        )

        self.logger.info(
            f"BaseMCPAgent {name} initialized with server {server_params.name}")

    # Synchronous wrappers for async MCPClient methods
    def list_tools_sync(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        return asyncio.run(self.mcp_client.list_tools(self.server_params.name))

    def call_tool_sync(self, tool_name: str, **kwargs) -> Any:
        """Call a tool on the MCP server."""
        # The kwargs will contain the parameters for the tool call
        return asyncio.run(self.mcp_client.call_tool(self.server_params.name, tool_name, kwargs))

    def read_resource_sync(self, uri: str) -> str:
        """Read a resource from the MCP server."""
        # The read_resource method in MCPClient returns a dict, extract content
        result = asyncio.run(self.mcp_client.read_resource(self.server_params.name, uri))
        return json.dumps(result)  # Return as JSON string for agent to parse


class AssistantMCPAgent(BaseMCPAgent):
    def __init__(self,
                 name: str,
                 mcp_client: MCPClient,
                 server_params: MCPServerParams,
                 llm_config: Dict[str,
                                  Any],
                 system_message: str):
        super().__init__(name, mcp_client, server_params, llm_config, system_message)
        self.logger = logging.getLogger(f"AssistantMCPAgent.{name}")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Enhance system message with tool descriptions
        self._enhance_system_message()
        self.caller_agent.update_system_message(self.system_message)
        self.logger.info(
            f"AssistantMCPAgent {name} initialized with enhanced system message")

    def _enhance_system_message(self):
        """Fetches tools and enhances the system message with their descriptions."""
        self.logger.info(
            f"Fetching tools for {self.server_params.name} to enhance system message")
        try:
            available_tools = self.list_tools_sync()  # Use the sync wrapper
            if not available_tools:
                self.logger.warning(
                    f"No tools found for server {self.server_params.name}. System message not enhanced.")
                return

            tool_descriptions = []
            for tool in available_tools:
                tool_name = tool.get("name", "unknown")
                tool_description = tool.get(
                    "description", "No description available")
                parameters = tool.get("parameters", {}).get("properties", {})
                param_infos = []

                for param_name, param_info in parameters.items():
                    param_type = param_info.get("type", "unknown")
                    param_description = param_info.get("description", "")
                    if param_description:
                        param_infos.append(
                            f"- {param_name} ({param_type}): {param_description}")
                    else:
                        param_infos.append(f"- {param_name} ({param_type})")

                tool_descriptions.append(
                    f"Tool: {tool_name}\nDescription: {tool_description}\nParameters:\n" +
                    "\n".join(param_infos) +
                    "\n")

            tool_info_string = "\n".join(tool_descriptions)

            self.system_message += f"\n\nYou have access to the following tools from the '{self.server_params.name}' server:\n\n{tool_info_string}"
            self.logger.info(
                f"System message enhanced with tools from {self.server_params.name}")

        except Exception as e:
            self.logger.error(
                f"Error enhancing system message with tools: {e}")
            # Continue with the original system message if tool fetching fails

    def initiate_chat(
            self,
            recipient: ConversableAgent,
            message: str,
            **kwargs):
        """Initiates a chat using the caller agent."""
        self.logger.info(
            f"Initiating chat from {self.name} with message: {message}")
        return self.caller_agent.initiate_chat(
            recipient, message=message, **kwargs)

    def send(self, message: str, recipient: ConversableAgent, **kwargs):
        """Sends a message using the caller agent."""
        self.logger.info(
            f"Agent {self.name} sending message to {recipient.name}: {message}")
        return self.caller_agent.send(message, recipient, **kwargs)
