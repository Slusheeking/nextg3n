#!/bin/bash

# install.sh
# Install script for NextG3N Trading System
# Sets up system dependencies, Python environment, Node.js, PM2, Redis, Kafka, ngrok, and directories
# Run as: sudo bash install.sh

set -e

# Define variables
NEXTG3N_DIR="/home/ubuntu/nextg3n/nextg3n"
VENV_DIR="/home/ubuntu/nextg3n/venv"
PYTHON_VERSION="3.10"
NODE_VERSION="22"
KAFKA_VERSION="3.7.0"
KAFKA_FALLBACK_VERSION="3.6.1"
REDIS_MEMORY_LIMIT="2gb"
USER="ubuntu"
KAFKA_URL="https://downloads.apache.org/kafka/${KAFKA_VERSION}/kafka_2.13-${KAFKA_VERSION}.tgz"
KAFKA_ALT_URL="https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/kafka_2.13-${KAFKA_VERSION}.tgz"
KAFKA_FALLBACK_URL="https://archive.apache.org/dist/kafka/${KAFKA_FALLBACK_VERSION}/kafka_2.13-${KAFKA_FALLBACK_VERSION}.tgz"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log "ERROR: This script must be run as root (use sudo)"
    exit 1
fi

# Check disk space (require at least 1GB free)
log "Checking disk space..."
FREE_SPACE=$(df -m /home/ubuntu | tail -1 | awk '{print $4}')
if [ "$FREE_SPACE" -lt 1000 ]; then
    log "ERROR: Insufficient disk space ($FREE_SPACE MB available, 1000 MB required)"
    exit 1
fi

# Check /tmp permissions
log "Checking /tmp permissions..."
if [ ! -w /tmp ]; then
    log "ERROR: /tmp is not writable"
    sudo chmod 1777 /tmp || { log "ERROR: Failed to set /tmp permissions"; exit 1; }
fi

# Check /home/ubuntu permissions
log "Checking /home/ubuntu permissions..."
sudo chown -R $USER:$USER /home/ubuntu
sudo chmod -R 755 /home/ubuntu

# Update package lists
log "Updating package lists..."
apt-get update -y || { log "ERROR: Failed to update package lists"; exit 1; }

# Install system dependencies
log "Installing system dependencies..."
apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    redis-server \
    openjdk-21-jdk \
    curl \
    gnupg \
    wget \
    unzip \
    git \
    build-essential \
    snapd || { log "ERROR: Failed to install system dependencies"; exit 1; }

# Install Node.js
log "Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - || { log "ERROR: Failed to set up Node.js repository"; exit 1; }
apt-get install -y nodejs || { log "ERROR: Failed to install Node.js"; exit 1; }

# Install PM2
log "Installing PM2..."
npm install -g pm2 || { log "ERROR: Failed to install PM2"; exit 1; }

# Install ngrok
log "Installing ngrok..."
if ! command -v ngrok >/dev/null 2>&1; then
    sudo snap install ngrok || { log "ERROR: Failed to install ngrok"; exit 1; }
fi

# Authenticate ngrok
log "Authenticating ngrok..."
source $NEXTG3N_DIR/.env
if [ -n "$NGROK_AUTH_TOKEN" ]; then
    ngrok authtoken $NGROK_AUTH_TOKEN || { log "ERROR: Failed to authenticate ngrok"; exit 1; }
else
    log "ERROR: NGROK_AUTH_TOKEN not found in .env"
    exit 1
fi

# Set up Python virtual environment
log "Setting up Python virtual environment..."
if [ -d "$VENV_DIR" ]; then
    log "Removing existing virtual environment to ensure Python $PYTHON_VERSION..."
    sudo rm -rf $VENV_DIR
fi
python${PYTHON_VERSION} -m venv $VENV_DIR || { log "ERROR: Failed to create virtual environment"; exit 1; }
source $VENV_DIR/bin/activate

# Verify Python version
PYTHON_ACTUAL_VERSION=$(python --version)
if [[ ! "$PYTHON_ACTUAL_VERSION" =~ "Python 3.10" ]]; then
    log "ERROR: Virtual environment is using $PYTHON_ACTUAL_VERSION, expected Python 3.10"
    exit 1
fi

# Install Python dependencies
log "Installing Python dependencies..."
pip install --upgrade pip || { log "ERROR: Failed to upgrade pip"; exit 1; }
pip install -r $NEXTG3N_DIR/requirements.txt || { log "ERROR: Failed to install Python dependencies"; exit 1; }

# Configure Redis
log "Configuring Redis..."
echo "appendonly yes" >> /etc/redis/redis.conf
echo "maxmemory $REDIS_MEMORY_LIMIT" >> /etc/redis/redis.conf
systemctl restart redis-server || { log "ERROR: Failed to restart Redis"; exit 1; }
systemctl enable redis-server || { log "ERROR: Failed to enable Redis"; exit 1; }
log "Redis configured with AOF persistence and memory limit"

# Install and configure Kafka
log "Checking Kafka installation..."
KAFKA_DIR="/home/ubuntu/kafka"
if [ -d "$KAFKA_DIR" ] && [ -f "$KAFKA_DIR/bin/zookeeper-server-start.sh" ]; then
    log "Kafka already installed at $KAFKA_DIR, verifying setup..."
else
    log "Installing Kafka..."
    if [ -f "/tmp/kafka.tgz" ]; then
        log "Using existing Kafka tarball at /tmp/kafka.tgz..."
    else
        log "Testing network connectivity to Kafka archive..."
        curl -I $KAFKA_URL >/dev/null 2>&1 || { log "WARNING: Cannot reach $KAFKA_URL"; }
        
        log "Downloading Kafka ${KAFKA_VERSION}..."
        wget --no-verbose $KAFKA_URL -O /tmp/kafka.tgz
        if [ $? -ne 0 ]; then
            log "WARNING: Failed to download Kafka ${KAFKA_VERSION}, trying alternative URL..."
            wget --no-verbose $KAFKA_ALT_URL -O /tmp/kafka.tgz
            if [ $? -ne 0 ]; then
                log "WARNING: Alternative URL failed, falling back to version ${KAFKA_FALLBACK_VERSION}..."
                wget --no-verbose $KAFKA_FALLBACK_URL -O /tmp/kafka.tgz
                if [ $? -ne 0 ]; then
                    log "ERROR: Failed to download any Kafka version. Please download manually to /tmp/kafka.tgz and re-run."
                    log "URLs tried:"
                    log "- $KAFKA_URL"
                    log "- $KAFKA_ALT_URL"
                    log "- $KAFKA_FALLBACK_URL"
                    exit 1
                fi
                KAFKA_VERSION=$KAFKA_FALLBACK_VERSION
            fi
        fi
    fi

    # Verify tarball
    log "Verifying Kafka tarball..."
    if [ ! -s /tmp/kafka.tgz ]; then
        log "ERROR: Kafka tarball is empty or corrupted"
        exit 1
    fi

    log "Extracting Kafka..."
    sudo tar -xzf /tmp/kafka.tgz -C /home/ubuntu || { log "ERROR: Failed to extract Kafka archive"; exit 1; }
    
    log "Moving Kafka to $KAFKA_DIR..."
    sudo mv /home/ubuntu/kafka_2.13-${KAFKA_VERSION} $KAFKA_DIR || { log "ERROR: Failed to move Kafka directory"; exit 1; }
    rm /tmp/kafka.tgz
    log "Kafka ${KAFKA_VERSION} installed successfully at $KAFKA_DIR"
fi

# Set permissions for Kafka
log "Setting Kafka permissions..."
sudo chown -R $USER:$USER $KAFKA_DIR
sudo chmod -R 755 $KAFKA_DIR

# Start Kafka (Zookeeper and Kafka server)
log "Starting Kafka..."
if ! pgrep -f "kafka.Kafka" > /dev/null; then
    $KAFKA_DIR/bin/zookeeper-server-start.sh -daemon $KAFKA_DIR/config/zookeeper.properties || { log "ERROR: Failed to start Zookeeper"; exit 1; }
    sleep 10
    $KAFKA_DIR/bin/kafka-server-start.sh -daemon $KAFKA_DIR/config/server.properties || { log "ERROR: Failed to start Kafka server"; exit 1; }
    log "Kafka started"
    sleep 10  # Ensure Kafka is fully initialized
else
    log "Kafka is already running"
fi

# Configure Kafka topics
log "Configuring Kafka topics..."
for topic in trade-events sentiment-events context-events decision-events dashboard-events health-events trainer-events backtest-events; do
    if ! $KAFKA_DIR/bin/kafka-topics.sh --describe --topic nextg3n-$topic --bootstrap-server localhost:9092 >/dev/null 2>&1; then
        log "Creating topic nextg3n-$topic..."
        $KAFKA_DIR/bin/kafka-topics.sh --create --topic nextg3n-$topic --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092 || log "WARNING: Failed to create topic nextg3n-$topic"
    else
        log "Topic nextg3n-$topic already exists, skipping creation"
    fi
done

# Create necessary directories
log "Creating directories..."
mkdir -p $NEXTG3N_DIR/charts
mkdir -p $NEXTG3N_DIR/checkpoints
mkdir -p $NEXTG3N_DIR/logs
sudo chown -R $USER:$USER $NEXTG3N_DIR
sudo chmod -R 755 $NEXTG3N_DIR

# Set up PM2 startup
log "Setting up PM2 startup..."
pm2 startup systemd -u $USER || { log "ERROR: Failed to set up PM2 startup"; exit 1; }
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
echo "4. Access TradeDashboard via ngrok URL (check $NEXTG3N_DIR/logs/ngrok.log)"
echo "5. Verify health checks in $NEXTG3N_DIR/logs or via Kafka topic 'nextg3n-health-events'"

# Deactivate virtual environment
deactivate