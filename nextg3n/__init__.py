"""
Initialize the NextG3N Trading System package.

This module makes the nextg3n directory a Python package, enabling imports of key classes and
modules for orchestrating trading workflows, managing models, services, storage, monitoring,
and visualization components.
"""

# Orchestration
from .orchestration.trade_flow_orchestrator import TradeFlowOrchestrator

# Models
from .models import (
    StockRanker,
    SentimentModel,
    ForecastModel,
    ContextRetriever,
    DecisionModel,
    TradeExecutor,
    BacktestEngine,
    TrainerModel
)

# Services
from .services import (
    MarketDataService,
    TradeService,
    NewsService,
    SocialService,
    OptionsService,
    CacheService
)

# Storage
from .storage import RedisCluster, VectorDB

# Monitoring
from .monitoring import MetricsLogger, SystemAnalyzer, ResourceTracker

# Visualization
from .visualization import ChartGenerator, MetricsApi, TradeDashboard

__all__ = [
    # Orchestration
    "TradeFlowOrchestrator",
    # Models
    "StockRanker",
    "SentimentModel",
    "ForecastModel",
    "ContextRetriever",
    "DecisionModel",
    "TradeExecutor",
    "BacktestEngine",
    "TrainerModel",
    # Services
    "MarketDataService",
    "TradeService",
    "NewsService",
    "SocialService",
    "OptionsService",
    "CacheService",
    # Storage
    "RedisCluster",
    "VectorDB",
    # Monitoring
    "MetricsLogger",
    "SystemAnalyzer",
    "ResourceTracker",
    # Visualization
    "ChartGenerator",
    "MetricsApi",
    "TradeDashboard",
]
