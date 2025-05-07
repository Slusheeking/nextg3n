# NextG3N System Flow

## Overview
The NextG3N trading system operates as an event-driven, 24/7 automated trading platform, coordinating data ingestion, model inference, trading decisions, execution, and monitoring. The workflow integrates AI/ML models, LLMs via OpenRouter, AutoGen for orchestration, and real-time data processing through Kafka, Redis, and ChromaDB/FAISS. Health checks ensure system reliability, and a web dashboard visualizes performance.

## System Flowchart
Below is a Mermaid flowchart depicting the system’s workflow, covering data ingestion, processing, trading, health checks, and visualization.

```mermaid
graph TD
    A[Start] --> B{Scheduler}
    B -->|Startup Health Check| C[SystemHealthChecker]
    B -->|Periodic Health Check<br>8:00 AM, 2:00 PM, 8:00 PM ET| C
    B -->|Data Sync<br>4:00 AM - 6:00 AM ET| D[Services]
    B -->|Trading<br>9:30 AM - 4:00 PM ET| E[TradeFlowOrchestrator]
    B -->|Training<br>12:00 AM - 4:00 AM ET| F[TrainerModel]
    B -->|Maintenance<br>6:00 PM - 10:00 PM ET| G[System Maintenance]

    C --> H{Check Components}
    H -->|Models| I[SentimentModel, ContextRetriever,<br>DecisionModel, ForecastModel]
    H -->|Services| J[MarketDataService, NewsService,<br>SocialService, TradeService]
    H -->|Storage| K[Redis, ChromaDB/FAISS]
    H -->|Messaging| L[Kafka]
    H -->|Orchestration| M[TradeFlowOrchestrator]
    H -->|Visualization| N[TradeDashboard, MetricsApi]
    H --> O[Log Results to Kafka]
    O --> P[Slack Alerts for Failures]

    D --> Q[Fetch Data]
    Q -->|Market Data| R[Alpaca]
    Q -->|News| S[NewsAPI]
    Q -->|Social| T[Reddit]
    Q --> U[Store in Redis/ChromaDB]

    E --> V{AutoGen Agents}
    V -->|StockPicker| W[StockRanker]
    V -->|SentimentAgent| X[SentimentModel]
    V -->|PredictorAgent| Y[ForecastModel]
    V -->|ContextAgent| Z[ContextRetriever]
    V -->|TradeAgent| AA[DecisionModel]
    V -->|MonitorAgent| AB[MetricsLogger]
    V --> AC[Kafka Events]

    W --> AD[Rank Stocks]
    X --> AE[Analyze Sentiment<br>(RoBERTa/LLM)]
    Y --> AF[Forecast Prices]
    Z --> AG[Retrieve Context<br>(RAG/LLM)]
    AA --> AH[Make Trading Decision<br>(Transformer/LLM)]
    AH --> AI[TradeExecutor]
    AI --> AJ[Execute Trades via Alpaca]

    F --> AK[Fine-Tune Models]
    AK --> AL[Save Checkpoints]

    G --> AM[Backups, Cleanup]
    AM --> AN[Log Maintenance]

    N --> AO[TradeDashboard]
    AO --> AP[Display Metrics/Charts]
    AO --> AQ[LLM-Generated Summaries]

    AC --> AR[Redis Cache]
    AC --> AS[ChromaDB/FAISS Storage]
    AC --> AT[MetricsApi]
    AT --> AU[Expose Metrics]

    AP --> AV[End: User Monitoring]



##### 4. `/home/ubuntu/nextg3n/nextg3n/docs/system_hardware.md`
**Updated**: Specifies hardware requirements, focusing on NVIDIA A100 GPUs.

```markdown
# NextG3N System Hardware Requirements

## Overview
The NextG3N trading system is optimized for high-performance computing, leveraging NVIDIA A100 GPUs for AI/ML model training and inference. It runs on Ubuntu 20.04/22.04, utilizing a single-node setup for Kafka, Redis, and ChromaDB. Below are the recommended hardware specifications to ensure optimal performance for 24/7 trading, health checks, and data processing.

## Hardware Specifications

### Server
- **CPU**: 16-core AMD EPYC or Intel Xeon (e.g., AMD EPYC 7313P, Intel Xeon Silver 4314)
  - Rationale: Supports parallel processing for model inference, data processing, and health checks.
- **RAM**: 128 GB DDR4 ECC
  - Rationale: Handles large datasets, model training, and caching in Redis (2GB limit).
- **Storage**: 2 TB NVMe SSD
  - Rationale: Stores historical data, model checkpoints, and logs; NVMe ensures low-latency access.
- **OS**: Ubuntu 20.04 or 22.04 LTS
  - Rationale: Compatible with Python 3.8+, NVIDIA drivers, and system dependencies.

### GPU
- **Model**: NVIDIA A100 40GB or 80GB (1-2 GPUs)
  - Rationale: Accelerates training and inference for RoBERTa, TFT, and Decision Transformer models.
- **CUDA**: Version 11.2 or higher
  - Rationale: Required for PyTorch and GPU-accelerated libraries.
- **Driver**: NVIDIA driver 450.80.02 or later
  - Rationale: Ensures compatibility with A100 GPUs.

### Networking
- **Bandwidth**: 1 Gbps Ethernet
  - Rationale: Supports real-time data fetching from Alpaca, NewsAPI, Reddit, and OpenRouter.
- **Firewall**: Open ports 9092 (Kafka), 6379 (Redis), 8001 (ChromaDB), 3050 (TradeDashboard), 8000 (MetricsApi)
  - Rationale: Enables internal and external communication.

## Additional Requirements
- **Power Supply**: 1000W+ with 80+ Platinum efficiency
  - Rationale: Supports high-power A100 GPUs and server components.
- **Cooling**: Enterprise-grade air or liquid cooling
  - Rationale: Prevents thermal throttling during intensive model training.
- **UPS**: Uninterruptible Power Supply (1500VA+)
  - Rationale: Ensures system uptime during power outages.

## Deployment Considerations
- **Single Node**: Current setup assumes a single server for simplicity. For production, consider multi-node Kafka and Redis clusters for redundancy.
- **GPU Utilization**: Use `nvidia-smi` to monitor A100 usage, prioritizing training during off-hours (12:00 AM - 4:00 AM ET).
- **Storage**: Allocate 500GB for model checkpoints, 1TB for historical data, and 500GB for logs and temporary files.
- **Monitoring**: Implement Prometheus/Grafana for advanced hardware monitoring (optional).

## Verification
After deployment, verify hardware compatibility:
1. Check GPU availability:
   ```bash
   nvidia-smi

   lscpu
free -h

df -h

systemctl status redis kafka



]