#!/bin/bash

# install.sh
# Install script for NextG3N Trading System
# Sets up system dependencies, Python environment, Node.js, PM2, Redis, Kafka, and directories
# Run as: sudo bash install.sh

set -e

# Define variables
NEXTG3N_DIR="/home/ubuntu/nextg3n/nextg3n"
VENV_DIR="/home/ubuntu/nextg3n/venv"
PYTHON_VERSION="3.8"
NODE_VERSION="16"
KAFKA_VERSION="3.7.0"
REDIS_MEMORY_LIMIT="2gb"
USER="ubuntu"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log "ERROR: This script must be run as root (use sudo)"
    exit 1
fi

# Update package lists
log "Updating package lists..."
apt-get update -y

# Install system dependencies
log "Installing system dependencies..."
apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    redis-server \
    openjdk-11-jdk \
    curl \
    gnupg \
    wget \
    unzip \
    git \
    build-essential

# Install Node.js
log "Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash -
apt-get install -y nodejs

# Install PM2
log "Installing PM2..."
npm install -g pm2

# Set up Python virtual environment
log "Setting up Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python${PYTHON_VERSION} -m venv $VENV_DIR
fi
source $VENV_DIR/bin/activate

# Install Python dependencies
log "Installing Python dependencies..."
pip install --upgrade pip
pip install -r $NEXTG3N_DIR/requirements.txt

# Configure Redis
log "Configuring Redis..."
echo "appendonly yes" >> /etc/redis/redis.conf
echo "maxmemory $REDIS_MEMORY_LIMIT" >> /etc/redis/redis.conf
systemctl restart redis
systemctl enable redis
log "Redis configured with AOF persistence and memory limit"

# Install and configure Kafka
log "Installing Kafka..."
KAFKA_DIR="/home/ubuntu/kafka"
if [ ! -d "$KAFKA_DIR" ]; then
    wget https://downloads.apache.org/kafka/${KAFKA_VERSION}/kafka_2.13-${KAFKA_VERSION}.tgz -O /tmp/kafka.tgz
    tar -xzf /tmp/kafka.tgz -C /home/ubuntu
    mv /home/ubuntu/kafka_2.13-${KAFKA_VERSION} $KAFKA_DIR
    rm /tmp/kafka.tgz
fi

# Configure Kafka topics
log "Configuring Kafka topics..."
$KAFKA_DIR/bin/kafka-topics.sh --create --topic nextg3n_trade_events --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092 || true
$KAFKA_DIR/bin/kafka-topics.sh --create --topic nextg3n_sentiment_events --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092 || true
$KAFKA_DIR/bin/kafka-topics.sh --create --topic nextg3n_context_events --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092 || true
$KAFKA_DIR/bin/kafka-topics.sh --create --topic nextg3n_decision_events --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092 || true
$KAFKA_DIR/bin/kafka-topics.sh --create --topic nextg3n_dashboard_events --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092 || true
$KAFKA_DIR/bin/kafka-topics.sh --create --topic nextg3n_health_events --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092 || true

# Start Kafka (Zookeeper and Kafka server)
log "Starting Kafka..."
$KAFKA_DIR/bin/zookeeper-server-start.sh -daemon $KAFKA_DIR/config/zookeeper.properties
sleep 5
$KAFKA_DIR/bin/kafka-server-start.sh -daemon $KAFKA_DIR/config/server.properties
log "Kafka started"

# Create necessary directories
log "Creating directories..."
mkdir -p $NEXTG3N_DIR/charts
mkdir -p $NEXTG3N_DIR/checkpoints
mkdir -p $NEXTG3N_DIR/logs
chown -R $USER:$USER $NEXTG3N_DIR
chmod -R 755 $NEXTG3N_DIR

# Set up PM2 startup
log "Setting up PM2 startup..."
pm2 startup systemd -u $USER
pm2 save

# Post-installation instructions
log "Installation complete!"
echo "Next steps:"
echo "1. Configure API keys in $NEXTG3N_DIR/.env:"
echo "   - OPENROUTER_API_KEY: Your OpenRouter API key"
echo "   - ALPACA_API_KEY: Your Alpaca API key"
echo "   - ALPACA_SECRET_KEY: Your Alpaca secret key"
echo "   - NEWS_API_KEY: Your NewsAPI key"
echo "   - REDDIT_API_KEY: Your Reddit API key"
echo "   - REDDIT_API_SECRET: Your Reddit API secret"
echo "   - SLACK_TOKEN: Your Slack bot token for alerts"
echo "2. Start the system with PM2:"
echo "   pm2 start $NEXTG3N_DIR/ecosystem.config.js"
echo "3. Monitor logs:"
echo "   pm2 logs"
echo "4. Access TradeDashboard at http://localhost:3050"
echo "5. Verify health checks in $NEXTG3N_DIR/logs or via Kafka topic 'nextg3n_health_events'"

# Deactivate virtual environment
deactivate