"""
Forecast Model for NextG3N Trading System

Implements intraday price forecasting using LSTM.
Designed for low-latency inference when integrated directly.
Requires a pre-trained model file.
"""

import os
import logging
import datetime
import torch
import torch.nn as nn
import time
from typing import Dict, Any, List

from monitoring.metrics_logger import MetricsLogger


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class ForecastModel:
    def __init__(self, config: Dict[str, Any]):
        self.logger = MetricsLogger(component_name="forecast_model")
        self.logger.setLevel(logging.WARNING)  # Reduce logging level in production
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.config = config
        self.forecast_config = config.get("forecast", {})

        # Initialize the LSTM model and load weights
        input_size = self.forecast_config.get("input_size")
        hidden_size = self.forecast_config.get("hidden_size")
        num_layers = self.forecast_config.get("num_layers")
        model_path = self.forecast_config.get("model_path")

        # Validate required configuration parameters
        if input_size is None or hidden_size is None or num_layers is None or not model_path:
            error_msg = "ForecastModel requires 'input_size', 'hidden_size', 'num_layers', and 'model_path' in configuration."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if not os.path.exists(model_path):
            error_msg = f"ForecastModel: Model file not found at specified path: {model_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            self.model = LSTMModel(input_size, hidden_size, num_layers)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()  # Set model to evaluation mode

            # Move model to GPU if available and configured
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and self.forecast_config.get(
                    "use_gpu", False) else "cpu")
            self.model.to(self.device)
            self.logger.info(
                f"ForecastModel initialized and model loaded from {model_path} on device: {self.device}")

        except Exception as e:
            self.logger.error(
                f"Failed to initialize ForecastModel or load model: {e}")
            self.model = None  # Ensure model is None on failure
            # Ensure device is set even on failure
            self.device = torch.device("cpu")
            raise  # Re-raise the exception after logging

    async def predict(self, data: torch.Tensor) -> Dict[str, Any]:
        """
        Predicts the next price movement based on input data.

        Args:
            data: A torch.Tensor containing the input data for the LSTM model.
                  Expected shape: (batch_size, sequence_length, input_size)

        Returns:
            A dictionary containing the prediction result.
        """
        operation_id = f"forecast_prediction_{int(time.time())}"
        self.logger.info(
            f"Predicting price with input data shape: {data.shape} - Operation: {operation_id}")

        if not self.model:
            self.logger.error(
                "Forecast model not initialized. Cannot perform prediction.")
            return {
                "success": False,
                "error": "Forecast model not ready",
                "operation_id": operation_id}

        if data is None or data.numel() == 0:
            self.logger.warning("No input data provided for forecasting.")
            return {"success": False, "error": "No input data provided",
                    "operation_id": operation_id}  # Return failure on no data

        try:
            # Move data to the appropriate device
            data = data.to(self.device)

            with torch.no_grad():
                prediction_tensor = self.model(data)
                prediction = prediction_tensor.cpu().item()  # Get scalar prediction

            # Determine prediction direction based on configurable thresholds
            up_threshold = self.forecast_config.get("up_threshold")
            down_threshold = self.forecast_config.get("down_threshold")

            if up_threshold is None or down_threshold is None:
                error_msg = "ForecastModel prediction requires 'up_threshold' and 'down_threshold' in configuration."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            prediction_direction = "up" if prediction > up_threshold else "down" if prediction < down_threshold else "neutral"

            result = {
                "success": True,
                "prediction": float(f"{prediction:.4f}"),
                "direction": prediction_direction,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            self.logger.info(f"Forecast prediction complete. Result: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error during forecast prediction: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id}

    async def shutdown(self):
        """Shuts down the ForecastModel."""
        self.logger.info("ForecastModel shutdown.")
        # No specific shutdown needed for the model in this context
