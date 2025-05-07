#!/bin/bash
# NextG3N MCP Docker Deployment Script
# This script builds and runs the NextG3N MCP Docker container

set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo -e "Project directory: ${PROJECT_DIR}"

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
        echo -e "Visit https://docs.docker.com/get-docker/ for installation instructions."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
        echo -e "Visit https://docs.docker.com/compose/install/ for installation instructions."
        exit 1
    fi
}

# Function to check if .env file exists
check_env_file() {
    if [ ! -f "${PROJECT_DIR}/.env" ]; then
        echo -e "${YELLOW}Warning: .env file not found in project directory.${NC}"
        echo -e "Creating a sample .env file. Please update it with your API keys."
        
        cat > "${PROJECT_DIR}/.env" << EOL
# Environment variables for the NextG3N Trading System
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
POLYGON_API_KEY=your_polygon_api_key_here
UNUSUAL_WHALES_API_KEY=your_unusual_whales_api_key_here
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USERNAME=your_reddit_username_here
REDDIT_PASSWORD=your_reddit_password_here
REDDIT_USER_AGENT=NextG3N Trading Bot v1.0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
EOL
        
        echo -e "${YELLOW}Please edit the .env file before continuing.${NC}"
        read -p "Press Enter to continue after updating the .env file..."
    fi
}

# Function to build and start the Docker container
build_and_start() {
    echo -e "${GREEN}Building and starting NextG3N MCP Docker container...${NC}"
    
    cd "${PROJECT_DIR}"
    
    # Build and start the container using docker-compose
    docker-compose -f deploy/docker-compose.yml up -d --build
    
    echo -e "${GREEN}NextG3N MCP Docker container is now running!${NC}"
    echo -e "You can check the logs with: ${YELLOW}docker-compose -f deploy/docker-compose.yml logs -f${NC}"
}

# Function to stop the Docker container
stop_container() {
    echo -e "${YELLOW}Stopping NextG3N MCP Docker container...${NC}"
    
    cd "${PROJECT_DIR}"
    
    # Stop the container using docker-compose
    docker-compose -f deploy/docker-compose.yml down
    
    echo -e "${GREEN}NextG3N MCP Docker container has been stopped.${NC}"
}

# Main script execution
echo -e "${GREEN}NextG3N MCP Docker Deployment${NC}"
echo -e "--------------------------------"

# Check if Docker is installed
check_docker

# Check if .env file exists
check_env_file

# Parse command line arguments
if [ "$1" == "stop" ]; then
    stop_container
else
    build_and_start
fi

echo -e "${GREEN}Done!${NC}"