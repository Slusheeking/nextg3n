#!/bin/bash
# Install script for NextG3N Trading System
set -e
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

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }
if [ "$EUID" -ne 0 ]; then log "ERROR: Run as root"; exit 1; fi
FREE_SPACE=$(df -m /home/ubuntu | tail -1 | awk '{print $4}')
if [ "$FREE_SPACE" -lt 1000 ]; then log "ERROR: Insufficient disk space"; exit 1; fi
if [ ! -w /tmp ]; then sudo chmod 1777 /tmp || { log "ERROR: /tmp permissions"; exit 1; }; fi
sudo chown -R $USER:$USER /home/ubuntu
sudo chmod -R 755 /home/ubuntu
apt-get update -y || { log "ERROR: Package lists"; exit 1; }
apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python3-pip redis-server openjdk-21-jdk curl gnupg wget unzip git build-essential snapd || { log "ERROR: Dependencies"; exit 1; }
curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - || { log "ERROR: Node.js setup"; exit 1; }
apt-get install -y nodejs || { log "ERROR: Node.js install"; exit 1; }
npm install -g pm2 || { log "ERROR: PM2 install"; exit 1; }
if ! command -v ngrok >/dev/null 2>&1; then sudo snap install ngrok || { log "ERROR: ngrok install"; exit 1; }; fi
source $NEXTG3N_DIR/.env
if [ -n "$NGROK_AUTH_TOKEN" ]; then ngrok authtoken $NGROK_AUTH_TOKEN || { log "ERROR: ngrok auth"; exit 1; }; else log "ERROR: NGROK_AUTH_TOKEN missing"; exit 1; fi
if [ -d "$VENV_DIR" ]; then sudo rm -rf $VENV_DIR; fi
python${PYTHON_VERSION} -m venv $VENV_DIR || { log "ERROR: Venv creation"; exit 1; }
source $VENV_DIR/bin/activate
PYTHON_ACTUAL_VERSION=$(python --version)
if [[ ! "$PYTHON_ACTUAL_VERSION" =~ "Python 3.10" ]]; then log "ERROR: Wrong Python version"; exit 1; fi
pip install --upgrade pip || { log "ERROR: Pip upgrade"; exit 1; }
pip install -r $NEXTG3N_DIR/requirements.txt || { log "ERROR: Python dependencies"; exit 1; }
echo "appendonly yes" >> /etc/redis/redis.conf
echo "maxmemory $REDIS_MEMORY_LIMIT" >> /etc/redis/redis.conf
systemctl restart redis-server || { log "ERROR: Redis restart"; exit 1; }
systemctl enable redis-server || { log "ERROR: Redis enable"; exit 1; }
KAFKA_DIR="/home/ubuntu/kafka"
if [ ! -d "$KAFKA_DIR" ] || [ ! -f "$KAFKA_DIR/bin/zookeeper-server-start.sh" ]; then
    wget --no-verbose $KAFKA_URL -O /tmp/kafka.tgz || {
        wget --no-verbose $KAFKA_ALT_URL -O /tmp/kafka.tgz || {
            wget --no-verbose $KAFKA_FALLBACK_URL -O /tmp/kafka.tgz || { log "ERROR: Kafka download"; exit 1; }
            KAFKA_VERSION=$KAFKA_FALLBACK_VERSION
        }
    }
    if [ ! -s /tmp/kafka.tgz ]; then log "ERROR: Kafka tarball empty"; exit 1; fi
    sudo tar -xzf /tmp/kafka.tgz -C /home/ubuntu || { log "ERROR: Kafka extract"; exit 1; }
    sudo mv /home/ubuntu/kafka_2.13-${KAFKA_VERSION} $KAFKA_DIR || { log "ERROR: Kafka move"; exit 1; }
    rm /tmp/kafka.tgz
fi
sudo chown -R $USER:$USER $KAFKA_DIR
sudo chmod -R 755 $KAFKA_DIR
if ! pgrep -f "kafka.Kafka" > /dev/null; then
    $KAFKA_DIR/bin/zookeeper-server-start.sh -daemon $KAFKA_DIR/config/zookeeper.properties || { log "ERROR: Zookeeper start"; exit 1; }
    sleep 10
    $KAFKA_DIR/bin/kafka-server-start.sh -daemon $KAFKA_DIR/config/server.properties || { log "ERROR: Kafka start"; exit 1; }
    sleep 10
fi
for topic in trade-events sentiment-events context-events decision-events dashboard-events health-events trainer-events backtest-events orchestration-events realtime-quotes options-flow-events; do
    if ! $KAFKA_DIR/bin/kafka-topics.sh --describe --topic nextg3n-$topic --bootstrap-server localhost:9092 >/dev/null 2>&1; then
        $KAFKA_DIR/bin/kafka-topics.sh --create --topic nextg3n-$topic --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092 || log "WARNING: Topic creation failed"
    fi
done
mkdir -p $NEXTG3N_DIR/charts $NEXTG3N_DIR/checkpoints $NEXTG3N_DIR/logs
sudo chown -R $USER:$USER $NEXTG3N_DIR
sudo chmod -R 755 $NEXTG3N_DIR
pm2 startup systemd -u $USER || { log "ERROR: PM2 startup"; exit 1; }
pm2 save
log "Installation complete!"
echo "Next steps:"
echo "1. Configure API keys in $NEXTG3N_DIR/.env:"
echo "   - OPENROUTER_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY, POLYGON_API_KEY, NEWS_API_KEY, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, UNUSUAL_WHALES_API_KEY, SLACK_TOKEN"
echo "2. Start system: pm2 start $NEXTG3N_DIR/ecosystem.config.js"
echo "3. Monitor logs: pm2 logs"
echo "4. Access dashboard: cat $NEXTG3N_DIR/logs/ngrok.log | grep 'https://.*.ngrok-free.app'"
deactivate