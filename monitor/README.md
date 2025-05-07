# NextG3N Monitoring System

This directory contains the monitoring and logging system for the NextG3N Trading Platform.

## Centralized Logging System

The logging system provides:

- Low-latency logging with in-memory buffer
- Asynchronous file writing to minimize performance impact
- JSON-formatted logs for structured analysis
- ANSI colored console output
- Centralized error log file for all WARNING, ERROR, and CRITICAL messages
- FastAPI endpoints for log access and filtering

### Key Features

- **Colored Console Output**: Logs are color-coded by level for better visibility
- **Centralized Error Log**: All warnings and errors across services are collected in a single file
- **Datetime Stamps**: All logs include ISO-format timestamps
- **Low Latency**: In-memory buffering and async writing ensure minimal performance impact
- **API Access**: Logs can be accessed and filtered via API endpoints

## Log Files

The logging system creates two types of log files:

1. **Service-specific logs**: `logs/nextg3n_YYYYMMDD.log` - Contains all logs from all services
2. **Centralized error log**: `logs/errors.log` - Contains only WARNING, ERROR, and CRITICAL logs from all services

## Using the Log Viewer

The `log_viewer.py` script provides a command-line interface for viewing and filtering logs:

```bash
# View the last 50 lines from the error log
./monitor/log_viewer.py

# View the last 100 lines from a specific log file
./monitor/log_viewer.py -f logs/nextg3n_20250507.log -n 100

# Filter by log level
./monitor/log_viewer.py -l ERROR

# Filter by service
./monitor/log_viewer.py -s alpaca_server

# Show logs from the last hour
./monitor/log_viewer.py --since 1h

# Follow logs in real-time (like tail -f)
./monitor/log_viewer.py --follow

# Output in JSON format
./monitor/log_viewer.py --json
```

## Using the Logging System in Code

To use the centralized logging system in your code:

```python
from monitor.logging_utils import get_logger

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

## API Access to Logs

The MCP Manager exposes several endpoints for accessing logs:

### Get Filtered Logs

```
GET /logs?level=INFO&service=alpaca_server&limit=100&start_time=2025-01-01T00:00:00&end_time=2025-01-02T00:00:00
```

Parameters:
- `level`: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `service`: Filter by service name
- `limit`: Maximum number of logs to return
- `start_time`: Filter logs after this time (ISO format)
- `end_time`: Filter logs before this time (ISO format)

### Get Available Services

```
GET /logs/services
```

Returns a list of all services that have logged messages.

### Get Available Log Levels

```
GET /logs/levels
```

Returns a list of all log levels used in the logs.