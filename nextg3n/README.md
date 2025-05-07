# NextG3N Trading System

## Overview
NextG3N is an advanced, automated trading system designed for high-accuracy stock trading, leveraging state-of-the-art AI/ML models, large language models (LLMs), and real-time data processing. It achieves ~60-70% prediction accuracy and ~52-62% trading win rate (improving to ~55-65% with fine-tuning) by integrating:
- **AI/ML Models**: RoBERTa for sentiment analysis, Temporal Fusion Transformer (TFT) for forecasting, Decision Transformer for trading decisions.
- **LLM Integration**: OpenRouter for sentiment analysis, context summarization, and decision explanations.
- **AutoGen**: Orchestrates LLM-powered agents for dynamic trading workflows.
- **Real-Time Data**: Alpaca for market data, NewsAPI and Reddit for sentiment, processed via Kafka and stored in Redis/ChromaDB.
- **24/7 Operation**: Scheduled trading (9:30 AM - 4:00 PM ET), training, data syncing, and health checks.
- **System Health Checks**: Automated checks on startup and 3x daily (8:00 AM, 2:00 PM, 8:00 PM ET) to ensure component reliability.
- **Visualization**: Flask-based dashboard for real-time metrics and charts.

The system is optimized for NVIDIA A100 GPUs, uses Backtrader for backtesting, and operates without Docker/Kubernetes, ensuring flexibility and performance on Ubuntu 20.04/22.04.

## Features
- **High Accuracy**: Combines price predictions, sentiment analysis, and technical indicators for robust trading decisions.
- **Real-Time Processing**: Kafka for event-driven communication, Redis for caching, ChromaDB/FAISS for vector storage.
- **LLM-Enhanced Analytics**: OpenRouter-powered sentiment scoring, context summarization, and decision explanations.
- **Automated Workflow**: APScheduler for 24/7 trading, training, and health checks; PM2 for process management.
- **Comprehensive Monitoring**: System health checks, Slack alerts, and a web dashboard for performance tracking.
- **Extensible Architecture**: Modular design with models, services, and orchestration components.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd nextg3n/nextg3n

   nextg3n/
├── .env                          # Environment variables (API keys, Slack token)
├── requirements.txt              # Python dependencies
├── ecosystem.config.js           # PM2 process configuration
├── install.sh                    # Installation script
├── run.sh                        # System startup script
├── scheduler.py                  # APScheduler for 24/7 tasks
├── config/
│   ├── system_config.yaml        # System-wide configuration
│   └── kafka_config.yaml         # Kafka configuration
├── docs/
│   ├── system_architecture.md    # System architecture details
│   ├── system_flow.md            # System workflow flowchart
│   └── system_hardware.md        # Hardware requirements
├── models/
│   ├── sentiment/
│   │   └── sentiment_model.py    # RoBERTa and LLM-based sentiment analysis
│   ├── context/
│   │   └── context_retriever.py  # RAG and LLM-based context retrieval
│   ├── decision/
│   │   └── decision_model.py     # Decision Transformer and LLM explanations
│   ├── forecast/
│   │   └── forecast_model.py     # TFT for price forecasting
│   ├── stock_ranker/
│   │   └── stock_ranker.py       # Stock ranking logic
│   ├── trade/
│   │   └── trade_executor.py     # Trade execution with Alpaca
│   ├── backtest/
│   │   └── backtest_engine.py    # Backtrader-based backtesting
│   ├── trainer/
│   │   └── trainer_model.py      # Model training logic
├── services/
│   ├── market_data_service.py    # Alpaca market data
│   ├── trade_service.py          # Alpaca trade execution
│   ├── news_service.py           # NewsAPI integration
│   ├── social_service.py         # Reddit sentiment data
│   └── cache_service.py          # Redis caching
├── orchestration/
│   ├── trade_flow_orchestrator.py # AutoGen-based workflow orchestration
├── storage/
│   ├── redis_cluster.py          # Redis storage
│   └── vector_db.py              # ChromaDB/FAISS vector storage
├── monitoring/
│   ├── metrics_logger.py         # Logging and Slack alerts
│   ├── system_health_checker.py  # System health checks
│   ├── system_analyzer.py        # System performance analysis
│   ├── resource_tracker.py       # Resource usage tracking
├── visualization/
│   ├── chart_generator.py        # Chart generation
│   ├── metrics_api.py            # Metrics API
│   ├── trade_dashboard.py        # Web dashboard
│   └── templates/
│       └── index.html            # Dashboard template
└── tests/
    ├── test_sentiment_model.py   # Sentiment model tests
    ├── test_context_retriever.py # Context retriever tests
    ├── test_decision_model.py    # Decision model tests
    ├── test_trade_flow_orchestrator.py # Orchestrator tests