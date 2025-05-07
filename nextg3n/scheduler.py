"""
Scheduler for NextG3N Trading System

This module implements the NextG3NScheduler class, using APScheduler to manage a 24/7
schedule for trading, model training, data syncing, system maintenance, and health checks.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from pytz import timezone
from pandas_market_calendars import get_calendar
import pandas as pd
import yaml
import os
from dotenv import load_dotenv
from nextg3n.orchestration.trade_flow_orchestrator import TradeFlowOrchestrator
from nextg3n.monitoring.system_health_checker import SystemHealthChecker

class NextG3NScheduler:
    """
    Scheduler for coordinating 24/7 operations of the NextG3N trading system.
    """

    def __init__(self, config):
        """
        Initialize the scheduler with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.orchestrator = TradeFlowOrchestrator(config)
        self.health_checker = SystemHealthChecker(config)
        self.logger = logging.getLogger("nextg3n.scheduler")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.scheduler = AsyncIOScheduler(timezone=timezone('US/Eastern'))
        self.nyse = get_calendar("NYSE")

    def is_market_open(self):
        """Check if NYSE is open at the current time."""
        now = datetime.now(timezone('US/Eastern'))
        schedule = self.nyse.schedule(start_date=now.date(), end_date=now.date())
        if schedule.empty:
            return False
        market_open = schedule.iloc[0]['market_open'].tz_convert('US/Eastern')
        market_close = schedule.iloc[0]['market_close'].tz_convert('US/Eastern')
        return market_open.time() <= now.time() <= market_close.time()

    async def start_trading(self):
        """Start trading session during market hours."""
        if self.is_market_open():
            self.logger.info("Starting trading session")
            await self.orchestrator.start()
        else:
            self.logger.info("Market closed, skipping trading session")

    async def train_models(self):
        """Train models with new data."""
        self.logger.info("Starting model training")
        trainer = self.orchestrator.models.get("trainer")
        if trainer:
            # Placeholder: Train SentimentModel with sample data
            training_data = pd.DataFrame({"text": ["Sample news"], "label": [2]})
            await trainer.train_model(trainer, "sentiment", training_data)
        else:
            self.logger.warning("TrainerModel not initialized")

    async def sync_data(self):
        """Sync data from external sources."""
        self.logger.info("Syncing data")
        context_retriever = self.orchestrator.models.get("context")
        if context_retriever:
            await context_retriever.store_context(["New news article"], [{"source": "news"}])
        else:
            self.logger.warning("ContextRetriever not initialized")

    async def run_maintenance(self):
        """Perform system maintenance tasks."""
        self.logger.info("Running system maintenance")
        # Placeholder: Implement backups, cleanup
        analyzer = self.orchestrator.monitoring.get("analyzer")
        if analyzer:
            analyzer.check_system_health()
        else:
            self.logger.warning("SystemAnalyzer not initialized")

    async def run_health_check(self):
        """Run system health check."""
        self.logger.info("Running system health check")
        result = await self.health_checker.check_health()
        if not result["success"]:
            self.logger.error(f"Health check failed: {result['errors']}")
        else:
            self.logger.info("Health check passed")

    def setup_schedule(self):
        """Configure the 24/7 schedule."""
        # Startup health check: Run 1 minute after startup
        self.scheduler.add_job(
            self.run_health_check,
            trigger=DateTrigger(run_date=datetime.now(timezone('US/Eastern')) + timedelta(minutes=1)),
            id="startup_health_check",
            max_instances=1
        )
        # Periodic health checks: 8:00 AM, 2:00 PM, 8:00 PM ET, daily
        self.scheduler.add_job(
            self.run_health_check,
            trigger=CronTrigger(hour="8,14,20", minute="0", day_of_week="mon-sun", timezone="US/Eastern"),
            id="periodic_health_check",
            max_instances=1
        )
        # Trading: 9:30 AM - 4:00 PM ET, Monday-Friday
        self.scheduler.add_job(
            self.start_trading,
            trigger=CronTrigger(hour="9-15", minute="30-59/5", day_of_week="mon-fri", timezone="US/Eastern"),
            id="trading",
            max_instances=1
        )
        # Training: 12:00 AM - 4:00 AM ET, daily
        self.scheduler.add_job(
            self.train_models,
            trigger=CronTrigger(hour="0-3", minute="0", day_of_week="mon-sun", timezone="US/Eastern"),
            id="training",
            max_instances=1
        )
        # Data Sync: 4:00 AM - 6:00 AM ET, daily
        self.scheduler.add_job(
            self.sync_data,
            trigger=CronTrigger(hour="4-5", minute="0", day_of_week="mon-sun", timezone="US/Eastern"),
            id="data_sync",
            max_instances=1
        )
        # Maintenance: 6:00 PM - 10:00 PM ET, daily
        self.scheduler.add_job(
            self.run_maintenance,
            trigger=CronTrigger(hour="18-21", minute="0", day_of_week="mon-sun", timezone="US/Eastern"),
            id="maintenance",
            max_instances=1
        )

    async def start(self):
        """Start the scheduler."""
        self.logger.info("Starting NextG3N scheduler")
        self.setup_schedule()
        self.scheduler.start()
        try:
            while True:
                await asyncio.sleep(3600)  # Keep running
        except (KeyboardInterrupt, SystemExit):
            await self.shutdown()

    async def shutdown(self):
        """Shutdown the scheduler and orchestrator."""
        self.logger.info("Shutting down NextG3N scheduler")
        self.scheduler.shutdown()
        await self.orchestrator.shutdown()

if __name__ == "__main__":
    load_dotenv()
    with open("config/system_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("config/kafka_config.yaml", "r") as f:
        config["kafka"] = yaml.safe_load(f)
    scheduler = NextG3NScheduler(config)
    asyncio.run(scheduler.start())