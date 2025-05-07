# NextG3N MCP Deployment Guide

This directory contains deployment configurations and scripts for running the NextG3N MCP servers in production environments.

## Deployment Options

### Docker Deployment (Recommended)

The Docker deployment is the recommended method for running the NextG3N MCP servers in production. It provides:

- Containerized environment for consistent execution
- Automatic restart on failure
- Health checks and monitoring
- Centralized logging system
- Easy deployment across different environments

#### Prerequisites

- Docker and Docker Compose installed
- API keys for various services configured in `.env` file

#### Deployment Steps

1. Make sure your `.env` file is properly configured with all required API keys
2. Run the deployment script:

```bash
chmod +x deploy/docker_deploy.sh
./deploy/docker_deploy.sh
```

3. Verify the deployment:

```bash
docker ps
```

4. Check the logs:

```bash
docker-compose -f deploy/docker-compose.yml logs -f
```

#### Stopping the Services

```bash
./deploy/docker_deploy.sh stop
```

### Systemd Service Deployment (Alternative)

For environments where Docker is not available, you can use the systemd service deployment.

#### Prerequisites

- Linux system with systemd
- Python 3.10+ installed
- API keys for various services configured in `.env` file

#### Deployment Steps

1. Make sure your `.env` file is properly configured with all required API keys
2. Run the service setup script:

```bash
chmod +x deploy/setup_service.sh
sudo ./deploy/setup_service.sh
```

3. Verify the service is running:

```bash
systemctl status nextg3n-mcp.service
```

4. Check the logs:

```bash
journalctl -u nextg3n-mcp.service -f
```

## Centralized Logging System

The NextG3N MCP system includes a centralized logging system that provides:

- Low-latency logging with in-memory buffer
- Asynchronous file writing to minimize performance impact
- JSON-formatted logs for structured analysis
- FastAPI endpoints for log access and filtering
- Configurable log levels and retention

### Accessing Logs via API

The MCP Manager exposes several endpoints for accessing logs:

#### Get Filtered Logs

```
GET /logs?level=INFO&service=alpaca_server&limit=100&start_time=2025-01-01T00:00:00&end_time=2025-01-02T00:00:00
```

Parameters:
- `level`: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `service`: Filter by service name
- `limit`: Maximum number of logs to return
- `start_time`: Filter logs after this time (ISO format)
- `end_time`: Filter logs before this time (ISO format)

#### Get Available Services

```
GET /logs/services
```

Returns a list of all services that have logged messages.

#### Get Available Log Levels

```
GET /logs/levels
```

Returns a list of all log levels used in the logs.

### Log File Location

Logs are stored in the `logs` directory in the project root. Each day gets a new log file named `nextg3n_YYYYMMDD.log`.

### Using the Logging System in Code

To use the centralized logging system in your code:

```python
from mcp.logging_utils import get_logger

# Get a logger with a specific name
logger = get_logger("your_service_name")

# Log messages at different levels
logger.debug("Debug message with detailed information")
logger.info("Informational message about normal operation")
logger.warning("Warning about potential issues")
logger.error("Error message about problems that need attention")
logger.critical("Critical message about severe problems")

# Log with additional context
logger.info("Processing request", extra={"request_id": "123", "user": "admin"})
```

## Health Checks

The system includes health checks to ensure all services are running properly:

- Docker health check via HTTP endpoint
- Internal health check mechanism in MCP Manager
- Automatic restart of failed services

Access the health check endpoint:

```
GET /health
```

This returns the status of all MCP servers and can be used for monitoring and alerting.