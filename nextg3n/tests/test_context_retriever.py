"""
Unit tests for ContextRetriever in NextG3N Trading System

Tests RAG-based context retrieval using Sentence Transformers and FinBERT with MCP tools.
Ensures integration with Yahoo Finance news and Reddit posts for market context.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from models.context.context_retriever import ContextRetriever

@pytest.mark.asyncio
class TestContextRetriever:
    @pytest.fixture
    def config(self):
        return {
            "kafka": {
                "bootstrap_servers": "localhost:9092",
                "topic_prefix": "nextg3n-"
            },
            "storage": {
                "redis": {"host": "localhost", "port": 6379, "db": 0},
                "vector_db": {
                    "provider": "chromadb",
                    "collection_name": "nextg3n_vectors",
                    "embedding_dim": 384
                }
            }
        }

    @pytest.fixture
    def context_retriever(self, config):
        return ContextRetriever(config)

    async def test_retrieve_context_success(self, context_retriever):
        with patch("services.mcp_client.MCPClient.call_tool", side_effect=[
            AsyncMock(return_value={
                "success": True,
                "news": [
                    {"title": "AAPL reports strong earnings", "summary": "Apple exceeded expectations."}
                ]
            }),
            AsyncMock(return_value={
                "success": True,
                "posts": [
                    {"title": "Bullish on AAPL!", "text": "Great Q2 results."}
                ]
            })
        ]):
            result = await context_retriever.retrieve_context("AAPL")
            assert result["success"]
            assert "context_summary" in result
            assert result["symbol"] == "AAPL"
            assert result["context_count"] > 0

    async def test_retrieve_context_no_data(self, context_retriever):
        with patch("services.mcp_client.MCPClient.call_tool", AsyncMock(return_value={"success": False, "error": "No data"})):
            result = await context_retriever.retrieve_context("AAPL")
            assert not result["success"]
            assert "error" in result
            assert result["symbol"] == "AAPL"