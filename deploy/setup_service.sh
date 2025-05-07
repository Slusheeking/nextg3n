#!/bin/bash
# NextG3N MCP Service Setup Script
# This script sets up the NextG3N MCP Manager as a systemd service

set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up NextG3N MCP Manager as a systemd service...${NC}"

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo -e "Project directory: ${PROJECT_DIR}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}This script needs to be run as root to install the systemd service.${NC}"
  echo -e "${YELLOW}Running with sudo...${NC}"
  sudo "$0" "$@"
  exit $?
fi

# Copy the service file to systemd directory
echo -e "Copying service file to /etc/systemd/system/"
cp "${PROJECT_DIR}/deploy/nextg3n-mcp.service" /etc/systemd/system/

# Update the service file with the correct path
echo -e "Updating service file with correct project path..."
sed -i "s|WorkingDirectory=.*|WorkingDirectory=${PROJECT_DIR}|g" /etc/systemd/system/nextg3n-mcp.service

# Reload systemd to recognize the new service
echo -e "Reloading systemd daemon..."
systemctl daemon-reload

# Enable the service to start on boot
echo -e "Enabling service to start on boot..."
systemctl enable nextg3n-mcp.service

# Start the service
echo -e "Starting NextG3N MCP service..."
systemctl start nextg3n-mcp.service

# Check service status
echo -e "Checking service status..."
systemctl status nextg3n-mcp.service

echo -e "${GREEN}NextG3N MCP Manager service has been set up successfully!${NC}"
echo -e "You can check the service logs with: ${YELLOW}journalctl -u nextg3n-mcp.service -f${NC}"