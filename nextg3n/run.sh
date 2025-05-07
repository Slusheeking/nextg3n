#!/bin/bash

# run.sh
# Run script for NextG3N Trading System
# Starts Kafka, Redis, ChromaDB, ngrok, and system processes via PM2
# Run as: bash run.sh

set -e

# Define variables
NEXTG3N_DIR="/home/ubuntu/nextg3n/nextg3n"
VENV_DIR="/home/ubuntu/nextg3n/venv"
KAFKA_DIR="/home/ubuntu/kafka"
USER="ubuntu"
LOG_DIR="$NEXTG3N_DIR/logs"
HEALTH_LOG="$LOG_DIR/health_check.log"
NGROK_LOG="$LOG_DIR/ngrok.log"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    log "ERROR: Virtual environment not found at $VENV_DIR. Run install.sh first."
    exit 1
fi

# Activate virtual environment
log "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Ensure log directory exists
mkdir -p $LOG_DIR
chown $USER:$USER $LOG_DIR
chmod 755 $LOG_DIR

# Start Redis if not running
log "Checking Redis status..."
if ! pgrep -x "redis-server" > /dev/null; then
    log "Starting Redis..."
    sudo systemctl start redis
    sudo systemctl enable redis
else
    log "Redis is already running"
fi

# Start Kafka if not running
log "Checking Kafka status..."
if ! pgrep -f "kafka.Kafka" > /dev/null; then
    log "Starting Zookeeper..."
    $KAFKA_DIR/bin/zookeeper-server-start.sh -daemon $KAFKA_DIR/config/zookeeper.properties
    sleep 5
    log "Starting Kafka server..."
    $KAFKA_DIR/bin/kafka-server-start.sh -daemon $KAFKA_DIR/config/server.properties
    sleep 5
else
    log "Kafka is already running"
fi

# Start ChromaDB server if not running
log "Checking ChromaDB status..."
if ! pgrep -f "chromadb" > /dev/null; then
    log "Starting ChromaDB server..."
    $VENV_DIR/bin/chromadb run --host localhost --port 8001 --path $NEXTG3N_DIR/chromadb_data &
    sleep 5
else
    log "ChromaDB is already running"
fi

# Start ngrok for front-end services
log "Starting ngrok for front-end services..."
if ! pgrep -f "ngrok" > /dev/null; then
    # Kill any existing ngrok processes
    pkill -f ngrok || true
    sleep 2
    
    # Start ngrok for metrics API
    log "Starting ngrok for Metrics API..."
    ngrok http 8000 > $LOG_DIR/ngrok_metrics.log 2>&1 &
    sleep 5
    METRICS_URL=$(grep -o 'https://[^ ]*.ngrok-free.app' $LOG_DIR/ngrok_metrics.log | head -1)
    
    # Start ngrok for dashboard
    log "Starting ngrok for TradeDashboard..."
    ngrok http 3050 > $NGROK_LOG 2>&1 &
    sleep 5
    DASHBOARD_URL=$(grep -o 'https://[^ ]*.ngrok-free.app' $NGROK_LOG | head -1)
    
    if [ -n "$DASHBOARD_URL" ]; then
        log "TradeDashboard accessible at: $DASHBOARD_URL"
    else
        log "WARNING: Failed to retrieve dashboard ngrok URL. Check $NGROK_LOG for details."
    fi
    
    if [ -n "$METRICS_URL" ]; then
        log "Metrics API accessible at: $METRICS_URL"
    else
        log "WARNING: Failed to retrieve metrics API ngrok URL. Check $LOG_DIR/ngrok_metrics.log for details."
    fi
else
    log "ngrok is already running"
fi

# Start services
log "Starting services..."

# Start metrics API
log "Starting Metrics API..."
cd $NEXTG3N_DIR && PYTHONPATH=$NEXTG3N_DIR nohup python start_metrics.py > $LOG_DIR/metrics_api.log 2>&1 &
METRICS_PID=$!
log "Metrics API started with PID: $METRICS_PID"
sleep 2

# Start trade dashboard
log "Starting Trade Dashboard..."
cd $NEXTG3N_DIR && PYTHONPATH=$NEXTG3N_DIR nohup python start_dashboard.py > $LOG_DIR/trade_dashboard.log 2>&1 &
DASHBOARD_PID=$!
log "Trade Dashboard started with PID: $DASHBOARD_PID"
sleep 2

# Start other PM2 processes
log "Starting other PM2 processes..."
pm2 start $NEXTG3N_DIR/ecosystem.config.js
pm2 save

# Wait for initial health check (up to 2 minutes)
log "Waiting for initial health check..."
sleep 60
if [ -f "$HEALTH_LOG" ]; then
    if grep -q "System health check passed" $HEALTH_LOG; then
        log "Initial health check passed"
    else
        log "WARNING: Initial health check failed. Check $HEALTH_LOG for details."
    fi
else
    log "WARNING: Health check log not found at $HEALTH_LOG. Check PM2 logs."
fi

# Post-run instructions
log "NextG3N system started successfully!"
echo "Next steps:"
echo "1. Monitor system logs:"
echo "   pm2 logs"
echo "2. Check health check logs:"
echo "   tail -f $HEALTH_LOG"
echo "3. Access services via ngrok URLs:"
echo "   TradeDashboard: $DASHBOARD_URL"
echo "   Metrics API: $METRICS_URL"
echo "4. Monitor Kafka topics:"
echo "   $KAFKA_DIR/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic nextg3n-health-events --from-beginning"
echo "   $KAFKA_DIR/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic nextg3n-trainer-events --from-beginning"
echo "   $KAFKA_DIR/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic nextg3n-backtest-events --from-beginning"
echo "5. Check service logs:"
echo "   tail -f $LOG_DIR/metrics_api.log"
echo "   tail -f $LOG_DIR/trade_dashboard.log"
echo "6. If issues arise, check PM2 status:"
echo "   pm2 list"
echo "7. Stop the system:"
echo "   pm2 stop all && kill $METRICS_PID $DASHBOARD_PID"

# Deactivate virtual environment
deactivate