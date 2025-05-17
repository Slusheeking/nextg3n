# NextG3n Trading System

A comprehensive algorithmic trading system that integrates with Interactive Brokers (IB) Gateway, featuring market-making capabilities and machine learning models for trading decisions.

## Project Structure

- **data/**: Contains market data and training data for machine learning models
  - **market-data/**: Historical and real-time market data
  - **training-data/**: Processed data for training ML models

- **ib-gateway/**: Docker configuration for Interactive Brokers Gateway
  - Contains Dockerfiles, configuration templates, and scripts for running IB Gateway in a containerized environment

- **market-making/**: Market making strategies and algorithms

- **ml-models/**: Machine learning models for market prediction
  - **trained_models/**: Saved trained models

- **order-processor/**: Order execution and management system

- **reports/**: Trading performance reports and analytics

- **tests/**: Test scripts for various components
  - Includes tests for IB API connectivity (sync, async, simple, order)

- **trainer/**: Training pipelines for machine learning models

## Requirements

See `requirements.txt` for a list of Python dependencies.

## Environment Variables

The system uses environment variables stored in `.env` for configuration. Make sure to set up your environment variables before running the system.

## Getting Started

1. Set up the IB Gateway using Docker:
   ```
   cd ib-gateway
   docker-compose up -d
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the tests to ensure connectivity with IB Gateway:
   ```
   python -m tests.ib_simple_test
   ```

## Usage

[Add specific usage instructions here]

## Development

[Add development guidelines here]

## License

[Add license information here]
