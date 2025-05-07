"""
Unit tests for the ContextRetriever class in the NextG3N Trading System.
"""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from nextg3n.models.context.context_retriever import ContextRetriever
import asyncio

class TestContextRetriever(unittest.TestCase):
    def setUp(self):
        self.config = {
            "models": {
                "context": {
                    "embedder_model": "all-MiniLM-L6-v2",
                    "embedding_dim": 384,
                    "batch_size": 32
                }
            },
            "kafka": {"bootstrap_servers": "localhost:9092"},
            "llm": {
                "enabled": True,
                "provider": "openrouter",
                "model": "openai/gpt-4",
                "max_tokens": 512,
                "temperature": 0.7,
                "retry_attempts": 3,
                "retry_delay": 1000
            }
        }
        self.context_retriever = ContextRetriever(self.config)
        self.context_retriever.logger = MagicMock()

    @patch('sentence_transformers.SentenceTransformer')
    @patch('nextg3n.storage.vector_db.VectorDB')
    async def test_store_context_success(self, mock_vector_db, mock_embedder):
        # Mock dependencies
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_embedder_instance.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_vector_db_instance = MagicMock()
        mock_vector_db.return_value = mock_vector_db_instance
        mock_vector_db_instance.store_texts = AsyncMock(return_value={"success": True, "text_count": 1})
        
        # Test data
        texts = ["Test news article"]
        metadatas = [{"source": "news"}]
        
        # Run store_context
        result = await self.context_retriever.store_context(texts, metadatas)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["text_count"], 1)
        self.context_retriever.logger.info.assert_called()

    @patch('sentence_transformers.SentenceTransformer')
    @patch('nextg3n.storage.vector_db.VectorDB')
    @patch('aiohttp.ClientSession.post')
    async def test_retrieve_context_with_summary_success(self, mock_post, mock_vector_db, mock_embedder):
        # Mock dependencies
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_embedder_instance.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_vector_db_instance = MagicMock()
        mock_vector_db.return_value = mock_vector_db_instance
        mock_vector_db_instance.query_texts = AsyncMock(return_value={
            "success": True,
            "texts": ["Test news article"],
            "metadatas": [{"source": "news"}],
            "ids": ["doc_1"],
            "distances": [0.1],
            "count": 1
        })
        
        # Mock OpenRouter response for summary
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "Summary of financial news"}}]
        })
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Run retrieve_context with summary
        result = await self.context_retriever.retrieve_context("Test query", k=5, generate_summary=True)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["summary"], "Summary of financial news")
        self.context_retriever.logger.info.assert_called()
        self.context_retriever.logger.track_llm_usage.assert_called()

if __name__ == '__main__':
    unittest.main()