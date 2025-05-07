"""
Unit tests for ForecastModel in NextG3N Trading System

Tests the forecasting process of the ForecastModel.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from models.forecast.forecast_model import ForecastModel
import torch

@pytest.mark.asyncio
class TestForecastModel:
    @pytest.fixture
    def config(self):
        return {
            "kafka": {"bootstrap_servers": "localhost:9092", "topic_prefix": "nextg3n-"},
            "storage": {"redis": {"host": "localhost", "port": 6379, "db": 0}},
            "forecast": {"input_size": 5, "hidden_size": 64, "num_layers": 2, "model_path": "path/to/model.pth", "up_threshold": 0.05, "down_threshold": -0.05}
        }

    @pytest.fixture
    def forecast_model(self, config):
        # Create a dummy model file for testing
        with open("path/to/model.pth", "w") as f:
            f.write("dummy model")
        return ForecastModel(config)

    async def test_predict_success(self, forecast_model):
        # Generate some dummy data for testing
        test_data = torch.randn(1, 60, 5)  # Batch size 1, sequence length 60, input size 5

        # Call the predict method
        result = await forecast_model.predict(test_data)

        # Assertions
        assert result["success"]
        assert "prediction" in result
        assert "direction" in result

    async def test_predict_no_data(self, forecast_model):
        # Call the predict method with no data
        result = await forecast_model.predict(None)

        # Assertions
        assert not result["success"]
        assert "error" in result