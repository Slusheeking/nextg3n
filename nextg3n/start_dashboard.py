#!/usr/bin/env python3
"""
Script to start the TradeDashboard server
"""

import asyncio
import yaml
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization.trade_dashboard import TradeDashboard

async def main():
    """
    Main function to start the TradeDashboard server
    """
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/system_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize and start the TradeDashboard
    dashboard = TradeDashboard(config)
    await dashboard.start_dashboard()

if __name__ == "__main__":
    asyncio.run(main())