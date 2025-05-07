"""
Unit tests for TrainerModel in the NextG3N Trading System.

Tests LLM-driven hyperparameter optimization, preprocessing, training, evaluation,
and Kafka event publishing.
"""

import pytest
import pandas as pd
import json
from unittest.mock import AsyncMock, patch
from datetime import datetime
from models.trainer.trainer_model import TrainerModel

@pytest.mark.asyncio
class TestTrainerModel:
    @pytest.fixture
    def config(self):
        return {
            "kafka": {
                "bootstrap_servers": "localhost:9092",
                "topic_prefix": "nextg3n-"
            },
            "llm": {
                "model": "openai/gpt-4",
                "base_url": "https://openrouter.ai/api/v1",
                "max_tokens": 300
            },
            "models": {
                "trainer": {
                    "max_epochs": 10,
                    "batch_size": 32,
                    "checkpoint_dir": "./checkpoints"
                }
            }
        }

    @pytest.fixture
    def trainer_model(self, config):
        return TrainerModel(config)

    @pytest.fixture
    def mock_model(self):
        class MockModel:
            learning_rate = 0.001
            dropout = 0.1
            def train_model(self, data):
                pass
        return MockModel()

    @pytest.fixture
    def training_data(self):
        return pd.DataFrame({
            "text": ["Test news", "Another news"],
            "label": [1, 0]
        })

    @pytest.fixture
    def validation_data(self):
        return pd.DataFrame({
            "text": ["Valid news"],
            "label": [1]
        })

    async def test_optimize_training_success(self, trainer_model):
        with patch("aiohttp.ClientSession.post", new=AsyncMock()) as mock_post:
            mock_post.return_value.status = 200
            mock_post.return_value.json = AsyncMock(return_value={
                "choices": [{"message": {"content": json.dumps({
                    "parameters": {"max_epochs": 15, "batch_size": 64, "learning_rate": 0.0001, "dropout": 0.2},
                    "preprocessing": ["normalize", "remove_outliers"]
                })}}]
            })
            result = await trainer_model.optimize_training("sentiment")
            assert result["parameters"] == {
                "max_epochs": 15,
                "batch_size": 64,
                "learning_rate": 0.0001,
                "dropout": 0.2
            }
            assert result["preprocessing"] == ["normalize", "remove_outliers"]

    async def test_optimize_training_invalid_response(self, trainer_model):
        with patch("aiohttp.ClientSession.post", new=AsyncMock()) as mock_post:
            mock_post.return_value.status = 200
            mock_post.return_value.json = AsyncMock(return_value={
                "choices": [{"message": {"content": "Invalid JSON"}}]
            })
            result = await trainer_model.optimize_training("sentiment")
            assert result["parameters"] == {
                "max_epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "dropout": 0.1
            }
            assert result["preprocessing"] == []

    async def test_train_model_success(self, trainer_model, mock_model, training_data):
        with patch("aiohttp.ClientSession.post", new=AsyncMock()) as mock_post, \
             patch("kafka.KafkaProducer.send") as mock_kafka, \
             patch("pytorch_lightning.Trainer.fit") as mock_fit:
            mock_post.return_value.status = 200
            mock_post.return_value.json = AsyncMock(return_value={
                "choices": [{"message": {"content": json.dumps({
                    "parameters": {"max_epochs": 12, "batch_size": 32, "learning_rate": 0.0005, "dropout": 0.15},
                    "preprocessing": ["normalize"]
                })}}]
            })
            result = await trainer_model.train_model(mock_model, "sentiment", training_data)
            assert result["success"] is True
            assert "checkpoint_path" in result
            assert result["llm_parameters"] == {
                "max_epochs": 12,
                "batch_size": 32,
                "learning_rate": 0.0005,
                "dropout": 0.15
            }
            assert result["preprocessing"] == ["normalize"]
            mock_kafka.assert_called_once()

    async def test_train_model_failure(self, trainer_model, mock_model, training_data):
        with patch("aiohttp.ClientSession.post", new=AsyncMock()) as mock_post, \
             patch("kafka.KafkaProducer.send") as mock_kafka:
            mock_post.side_effect = Exception("API error")
            result = await trainer_model.train_model(mock_model, "sentiment", training_data)
            assert result["success"] is False
            assert "error" in result
            mock_kafka.assert_not_called()

    async def test_evaluate_model_success(self, trainer_model, mock_model, validation_data):
        with patch("aiohttp.ClientSession.post", new=AsyncMock()) as mock_post, \
             patch("kafka.KafkaProducer.send") as mock_kafka:
            mock_post.return_value.status = 200
            mock_post.return_value.json = AsyncMock(return_value={
                "choices": [{"message": {"content": json.dumps({
                    "parameters": {"max_epochs": 10, "batch_size": 16, "learning_rate": 0.001, "dropout": 0.1},
                    "preprocessing": []
                })}}]
            })
            result = await trainer_model.evaluate_model(mock_model, "sentiment", validation_data)
            assert result["success"] is True
            assert "metrics" in result
            assert "llm_suggestions" in result
            mock_kafka.assert_called_once()

    async def test_evaluate_model_invalid_data(self, trainer_model, mock_model):
        with patch("aiohttp.ClientSession.post", new=AsyncMock()) as mock_post, \
             patch("kafka.KafkaProducer.send") as mock_kafka:
            mock_post.return_value.status = 200
            mock_post.return_value.json = AsyncMock(return_value={
                "choices": [{"message": {"content": json.dumps({
                    "parameters": {"max_epochs": 10, "batch_size": 16, "learning_rate": 0.001, "dropout": 0.1},
                    "preprocessing": []
                })}}]
            })
            result = await trainer_model.evaluate_model(mock_model, "sentiment", pd.DataFrame({"invalid": [1]}))
            assert result["success"] is True
            assert result["metrics"] == {"accuracy": 0.0, "loss": 0.0}
            mock_kafka.assert_called_once()