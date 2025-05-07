"""
Forecast Model for NextG3N Trading System

This module implements the ForecastModel class, using a Temporal Fusion Transformer (TFT)
to predict stock price movements. It supports the PredictorAgent in TradeFlowOrchestrator
by generating price forecasts.
"""

import os
import json
import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import torch
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning import LightningModule
import pytorch_lightning as pl

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

# Kafka imports
from kafka import KafkaProducer

class ForecastTFT(LightningModule):
    """
    Temporal Fusion Transformer model for stock price forecasting.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        tft_config = {
            "hidden_size": config.get("hidden_size", 64),
            "lstm_layers": config.get("lstm_layers", 2),
            "attention_head_size": config.get("attention_head_size", 4),
            "dropout": config.get("dropout", 0.1),
            "hidden_continuous_size": config.get("hidden_continuous_size", 32),
            "output_size": config.get("output_size", 1),  # Predict price change
            "loss": config.get("loss", "quantile"),
            "log_interval": config.get("log_interval", 10),
            "reduce_on_plateau_patience": config.get("reduce_on_plateau_patience", 4)
        }
        self.model = TemporalFusionTransformer.from_dataset(
            dataset=None,  # Dataset initialized in ForecastModel
            **tft_config
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.to('cuda')
        
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for price prediction.

        Args:
            x: Input dictionary with time-series features

        Returns:
            Predicted price change
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step for PyTorch Lightning.

        Args:
            batch: Batch of training data
            batch_idx: Batch index

        Returns:
            Loss value
        """
        x, y = batch
        y_hat = self(x)
        loss = self.model.loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizer for training.
        """
        return torch.optim.Adam(self.parameters(), lr=self.config.get("learning_rate", 0.001))

class ForecastModel:
    """
    Model for forecasting stock price movements using a Temporal Fusion Transformer.
    Supports the PredictorAgent in TradeFlowOrchestrator.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ForecastModel with configuration and model settings.

        Args:
            config: Configuration dictionary with model and Kafka settings
        """
        init_start_time = datetime.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="forecast_model")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.model_config = config.get("models", {}).get("forecast", {})
        self.kafka_config = config.get("kafka", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize TFT model and dataset
        self.model = None
        self.dataset = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialize_model()
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        init_duration = (datetime.time() - init_start_time) * 1000
        self.logger.timing("forecast_model.initialization_time_ms", init_duration)
        self.logger.info("ForecastModel initialized")

    def _initialize_model(self):
        """
        Initialize the TFT model and dataset.
        """
        try:
            # Placeholder dataset configuration (requires actual data for initialization)
            max_encoder_length = self.model_config.get("max_encoder_length", 30)
            max_prediction_length = self.model_config.get("max_prediction_length", 1)
            
            # Define dataset (requires actual data; placeholder for schema)
            self.dataset = TimeSeriesDataSet(
                data=pd.DataFrame({"time_idx": [], "group_id": [], "target": []}),  # Placeholder
                time_idx="time_idx",
                target="target",
                group_ids=["group_id"],
                max_encoder_length=max_encoder_length,
                max_prediction_length=max_prediction_length,
                static_categoricals=[],
                time_varying_known_reals=["rsi", "macd", "news_sentiment", "social_sentiment"],
                time_varying_unknown_reals=["close"],
                target_normalizer=GroupNormalizer(groups=["group_id"])
            )
            
            self.model = ForecastTFT(self.model_config)
            
            # Load pre-trained checkpoint if provided
            checkpoint_path = self.model_config.get("checkpoint_path")
            if checkpoint_path and os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                self.logger.info(f"Loaded pre-trained checkpoint from {checkpoint_path}")
            
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TFT model: {e}")
            self.model = None
            raise

    async def predict_price(
        self,
        symbol: str,
        timeframe: str = "1d",
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Predict the next price movement for a stock.

        Args:
            symbol: Stock symbol
            timeframe: Prediction timeframe (e.g., '1d')
            data: Optional input data (OHLCV, indicators, sentiment)

        Returns:
            Dictionary containing price prediction
        """
        start_time = datetime.time()
        operation_id = f"predict_price_{int(start_time)}"
        self.logger.info(f"Predicting price for {symbol}, timeframe={timeframe} - Operation: {operation_id}")

        if not self.model or not self.dataset:
            self.logger.error("TFT model or dataset not initialized")
            return {
                "success": False,
                "error": "TFT model or dataset not initialized",
                "symbol": symbol,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                
                # Process input data
                input_data = await loop.run_in_executor(
                    self.executor,
                    lambda: self._process_input_data(symbol, timeframe, data)
                )
                
                # Convert to tensor
                input_tensor = self.dataset.to_dataloader(
                    input_data, batch_size=1, train=False
                ).__iter__().__next__()
                
                # Move to device
                input_tensor = {k: v.to(self.device) for k, v in input_tensor.items()}
                
                # Generate prediction
                with torch.no_grad():
                    prediction = await loop.run_in_executor(
                        self.executor,
                        lambda: self.model(input_tensor)
                    )
                    predicted_price_change = prediction[0].item()  # Scalar price change
                    confidence = torch.sigmoid(prediction).item()  # Simplified confidence
                    
                    # Determine direction
                    direction = "up" if predicted_price_change > 0 else "down"

            result = {
                "success": True,
                "symbol": symbol,
                "predicted_price_change": predicted_price_change,
                "direction": direction,
                "confidence": confidence,
                "timeframe": timeframe,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}forecast_events",
                {"event": "price_predicted", "data": result}
            )

            duration = (datetime.time() - start_time) * 1000
            self.logger.timing("forecast_model.predict_price_time_ms", duration)
            self.logger.info(f"Price predicted for {symbol}: {direction} (change: {predicted_price_change:.4f}, confidence: {confidence:.2f})")
            self.logger.counter("forecast_model.predictions_made", 1)
            return result

        except Exception as e:
            self.logger.error(f"Error predicting price for {symbol}: {e}")
            self.logger.counter("forecast_model.prediction_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def _process_input_data(self, symbol: str, timeframe: str, data: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process input data into a format suitable for the TFT model.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            data: Input data dictionary (OHLCV, indicators, sentiment)

        Returns:
            Processed DataFrame
        """
        try:
            # Example processing (customize based on actual data structure)
            if data is None:
                data = {}
            
            # OHLCV data
            bars = data.get("bars", [])
            df = pd.DataFrame(bars) if bars else pd.DataFrame(columns=["timestamp", "close"])
            
            # Add time index and group ID
            df["time_idx"] = range(len(df))
            df["group_id"] = symbol
            
            # Target (close price or price change)
            df["target"] = df["close"].pct_change().shift(-1).fillna(0)  # Next period change
            
            # Technical indicators
            indicators = data.get("technical_indicators", {})
            df["rsi"] = indicators.get("rsi", 50.0)
            df["macd"] = indicators.get("macd", 0.0)
            
            # Sentiment scores
            sentiment = data.get("sentiment", {})
            df["news_sentiment"] = sentiment.get("news_sentiment", 0.0)
            df["social_sentiment"] = sentiment.get("social_sentiment", 0.0)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error processing input data: {e}")
            return pd.DataFrame()

    async def train_model(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train or fine-tune the TFT model.

        Args:
            training_data: Training data DataFrame

        Returns:
            Dictionary indicating training result
        """
        start_time = datetime.time()
        operation_id = f"train_model_{int(start_time)}"
        self.logger.info(f"Training TFT model - Operation: {operation_id}")

        if not self.model:
            self.logger.error("TFT model not initialized")
            return {
                "success": False,
                "error": "TFT model not initialized",
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                
                # Update dataset with training data
                self.dataset = TimeSeriesDataSet(
                    training_data,
                    time_idx="time_idx",
                    target="target",
                    group_ids=["group_id"],
                    max_encoder_length=self.model_config.get("max_encoder_length", 30),
                    max_prediction_length=self.model_config.get("max_prediction_length", 1),
                    static_categoricals=[],
                    time_varying_known_reals=["rsi", "macd", "news_sentiment", "social_sentiment"],
                    time_varying_unknown_reals=["close"],
                    target_normalizer=GroupNormalizer(groups=["group_id"])
                )
                
                # Create dataloader
                train_dataloader = self.dataset.to_dataloader(train=True, batch_size=32)
                
                # Train model
                trainer = pl.Trainer(
                    max_epochs=self.model_config.get("max_epochs", 10),
                    gpus=1 if torch.cuda.is_available() else 0,
                    gradient_clip_val=0.1
                )
                
                await loop.run_in_executor(
                    self.executor,
                    lambda: trainer.fit(self.model, train_dataloader)
                )

            result = {
                "success": True,
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Publish to Kafka
            self.producer.send(
                f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}forecast_events",
                {"event": "model_trained", "data": result}
            )

            duration = (datetime.time() - start_time) * 1000
            self.logger.timing("forecast_model.train_model_time_ms", duration)
            self.logger.info("TFT model trained successfully")
            self.logger.counter("forecast_model.trainings_completed", 1)
            return result

        except Exception as e:
            self.logger.error(f"Error training TFT model: {e}")
            self.logger.counter("forecast_model.training_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def shutdown(self):
        """
        Shutdown the ForecastModel and close resources.
        """
        self.logger.info("Shutting down ForecastModel")
        self.executor.shutdown(wait=True)
        self.producer.close()
        self.logger.info("Kafka producer closed")