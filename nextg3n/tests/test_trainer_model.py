"""
Unit tests for TrainerModel in NextG3N Trading System

Tests LLM-driven hyperparameter tuning with Optuna.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch
from models.trainer.trainer_model import TrainerModel

@pytest.mark.asyncio
class TestTrainerModel:
    @pytest.fixture
    def config(self):
        return {
            "kafka": {"bootstrap_servers": "localhost:9092", "topic_prefix": "nextg3n-"},
            "llm": {"model": "gpt-4", "base_url": "https://openrouter.ai/api/v1"},
            "storage": {"redis": {"host": "localhost", "port": 6379, "db": 0}}
        }

    @pytest.fixture
    def trainer_model(self, config):
        return TrainerModel(config)

    async def test_optimize_training_success(self, trainer_model):
        with patch("aiohttp.ClientSession.post", new=AsyncMock()) as mock_post:
            mock_post.return_value.status = 200
            mock_post.return_value.json = AsyncMock(return_value={
                "choices": [{"message": {"content": json.dumps({"learning_rate": 0.001, "batch_size": 32})}}]
            })
            result = await trainer_model.optimize_training("LSTM")
            assert result["success"]
            assert "parameters" in result
            assert result["model_name"] == "LSTM"

    async def test_optimize_training_llm_failure(self, trainer_model):
        with patch("aiohttp.ClientSession.post", AsyncMock(side_effect=Exception("API error"))):
            result = await trainer_model.optimize_training("LSTM")
            assert not result["success"]
            assert "error" in result