"""
Metrics Logger for NextG3N Trading System

This module implements the MetricsLogger class, logging system events, tracking metrics,
and sending alerts for health check failures.
"""

import logging
import time
from typing import Any, Dict
from datetime import datetime
import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

class MetricsLogger(logging.Logger):
    """
    Custom logger for tracking system events, metrics, and sending alerts.
    """

    def __init__(self, component_name: str):
        """
        Initialize the MetricsLogger with component name.

        Args:
            component_name: Name of the component (e.g., 'trade_flow_orchestrator')
        """
        super().__init__(f"nextg3n.{component_name}")
        self.metrics = {}
        self.llm_usage = {"requests": 0, "tokens": 0, "estimated_cost": 0.0}  # Track LLM usage
        load_dotenv()
        self.slack_client = WebClient(token=os.getenv("SLACK_TOKEN")) if os.getenv("SLACK_TOKEN") else None

    def counter(self, metric_name: str, value: int):
        """
        Increment a counter metric.

        Args:
            metric_name: Name of the metric
            value: Value to increment by
        """
        self.metrics[metric_name] = self.metrics.get(metric_name, 0) + value
        self.debug(f"Counter {metric_name} incremented by {value}, total: {self.metrics[metric_name]}")

    def timing(self, metric_name: str, duration_ms: float):
        """
        Record a timing metric.

        Args:
            metric_name: Name of the metric
            duration_ms: Duration in milliseconds
        """
        self.metrics[metric_name] = duration_ms
        self.debug(f"Timing {metric_name}: {duration_ms:.2f} ms")

    def track_llm_usage(self, tokens: int, model: str):
        """
        Track LLM usage and estimated cost.

        Args:
            tokens: Number of tokens used
            model: LLM model name (e.g., 'openai/gpt-4')
        """
        self.llm_usage["requests"] += 1
        self.llm_usage["tokens"] += tokens
        # Simplified cost estimation (adjust based on OpenRouter pricing)
        cost_per_token = 0.00001 if "gpt-4" in model.lower() else 0.000005  # Example rates
        cost = tokens * cost_per_token
        self.llm_usage["estimated_cost"] += cost
        self.info(f"LLM usage: requests={self.llm_usage['requests']}, tokens={self.llm_usage['tokens']}, cost=${self.llm_usage['estimated_cost']:.6f}")

    async def send_alert(self, message: str):
        """
        Send an alert to Slack for critical failures.

        Args:
            message: Alert message
        """
        if not self.slack_client:
            self.warning("Slack client not initialized; skipping alert")
            return
        try:
            self.slack_client.chat_postMessage(
                channel="#nextg3n-alerts",  # Configure your Slack channel
                text=f"NextG3N Alert: {message}"
            )
            self.info(f"Sent Slack alert: {message}")
        except SlackApiError as e:
            self.error(f"Failed to send Slack alert: {str(e)}")