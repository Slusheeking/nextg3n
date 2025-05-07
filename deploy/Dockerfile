FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for historical data cache
RUN mkdir -p historical/cache

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose ports for all MCP servers
EXPOSE 8000 8001 8002 8003 8004 8005 8006 8080

# Run the MCP manager
CMD ["python", "-m", "mcp.mcp_manager"]