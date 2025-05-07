"""
Resource Tracker for NextG3N Trading System

This module implements the ResourceTracker class, continuously tracking resource usage
metrics (CPU, memory, GPU, disk, network) across the NextG3N system. It supports the
MonitorAgent in TradeFlowOrchestrator by collecting and publishing resource data.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
import psutil
try:
    import pynvml
    HAVE_PYNVML = True
except ImportError:
    HAVE_PYNVML = False
    logging.warning("pynvml not installed. GPU monitoring will be unavailable.")

# Kafka imports
from kafka import KafkaProducer

# Monitoring imports
from monitoring.metrics_logger import MetricsLogger

class ResourceTracker:
    """
    Class for tracking resource usage in the NextG3N system.
    Collects metrics for CPU, memory, GPU, disk, and network.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ResourceTracker with configuration and monitoring settings.

        Args:
            config: Configuration dictionary with monitoring and Kafka settings
        """
        init_start_time = datetime.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="resource_tracker")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.monitoring_config = config.get("monitoring", {}).get("resource_tracker", {})
        self.kafka_config = config.get("kafka", {})
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize GPU monitoring
        self.gpu_available = HAVE_PYNVML
        self.gpu_handles = []
        if self.gpu_available:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)]
                self.logger.info(f"Initialized GPU monitoring for {self.gpu_count} GPUs")
            except pynvml.NVMLError as e:
                self.logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.gpu_available = False
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialize monitoring state
        self.running = True
        self.monitor_task = None
        self.last_disk_io = psutil.disk_io_counters()
        self.last_net_io = psutil.net_io_counters()
        
        init_duration = (datetime.time() - init_start_time) * 1000
        self.logger.timing("resource_tracker.initialization_time_ms", init_duration)
        self.logger.info("ResourceTracker initialized")

    async def start_tracking(self, interval: float = 30.0):
        """
        Start continuous resource tracking in a background task.

        Args:
            interval: Tracking interval in seconds (default: 30)
        """
        self.logger.info(f"Starting resource tracking with interval {interval}s")
        self.monitor_task = asyncio.create_task(self._track_loop(interval))

    async def _track_loop(self, interval: float):
        """
        Continuous tracking loop to collect resource usage metrics.

        Args:
            interval: Tracking interval in seconds
        """
        while self.running:
            try:
                metrics = await self.collect_metrics()
                if metrics.get("success"):
                    # Publish metrics to Kafka
                    self.producer.send(
                        f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}monitoring_events",
                        {"event": "resource_metrics", "data": metrics}
                    )
                
                await asyncio.sleep(interval)
            
            except Exception as e:
                self.logger.error(f"Error in tracking loop: {e}")
                self.logger.counter("resource_tracker.track_errors", 1)
                await asyncio.sleep(interval)

    async def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect resource usage metrics (CPU, memory, GPU, disk, network).

        Returns:
            Dictionary containing resource metrics
        """
        start_time = datetime.time()
        operation_id = f"collect_metrics_{int(start_time)}"
        self.logger.info(f"Collecting resource metrics - Operation: {operation_id}")

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                
                # CPU metrics
                cpu_percent = await loop.run_in_executor(self.executor, psutil.cpu_percent)
                cpu_count = psutil.cpu_count()
                
                # Memory metrics
                memory = await loop.run_in_executor(self.executor, psutil.virtual_memory)
                memory_percent = memory.percent
                memory_used = memory.used / (1024 ** 3)  # Convert to GB
                memory_total = memory.total / (1024 ** 3)  # Convert to GB
                
                # Disk I/O metrics
                disk_io = await loop.run_in_executor(self.executor, psutil.disk_io_counters)
                disk_read_rate = (disk_io.read_bytes - self.last_disk_io.read_bytes) / (1024 ** 2) / interval  # MB/s
                disk_write_rate = (disk_io.write_bytes - self.last_disk_io.write_bytes) / (1024 ** 2) / interval  # MB/s
                self.last_disk_io = disk_io
                
                # Network I/O metrics
                net_io = await loop.run_in_executor(self.executor, psutil.net_io_counters)
                net_sent_rate = (net_io.bytes_sent - self.last_net_io.bytes_sent) / (1024 ** 2) / interval  # MB/s
                net_recv_rate = (net_io.bytes_recv - self.last_net_io.bytes_recv) / (1024 ** 2) / interval  # MB/s
                self.last_net_io = net_io
                
                # GPU metrics
                gpu_metrics = []
                if self.gpu_available:
                    for i, handle in enumerate(self.gpu_handles):
                        try:
                            util = await loop.run_in_executor(
                                self.executor,
                                lambda: pynvml.nvmlDeviceGetUtilizationRates(handle)
                            )
                            mem = await loop.run_in_executor(
                                self.executor,
                                lambda: pynvml.nvmlDeviceGetMemoryInfo(handle)
                            )
                            gpu_metrics.append({
                                "gpu_id": i,
                                "utilization_percent": util.gpu,
                                "memory_used_gb": mem.used / (1024 ** 3),
                                "memory_total_gb": mem.total / (1024 ** 3),
                                "memory_free_percent": (mem.free / mem.total) * 100
                            })
                        except pynvml.NVMLError as e:
                            self.logger.warning(f"Error collecting GPU {i} metrics: {e}")
                
                metrics = {
                    "success": True,
                    "cpu_percent": cpu_percent,
                    "cpu_count": cpu_count,
                    "memory_percent": memory_percent,
                    "memory_used_gb": memory_used,
                    "memory_total_gb": memory_total,
                    "disk_read_rate_mbps": disk_read_rate,
                    "disk_write_rate_mbps": disk_write_rate,
                    "net_sent_rate_mbps": net_sent_rate,
                    "net_recv_rate_mbps": net_recv_rate,
                    "gpu_metrics": gpu_metrics,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

                self.logger.gauge("resource_tracker.cpu_percent", cpu_percent)
                self.logger.gauge("resource_tracker.memory_percent", memory_percent)
                self.logger.gauge("resource_tracker.disk_read_rate_mbps", disk_read_rate)
                self.logger.gauge("resource_tracker.disk_write_rate_mbps", disk_write_rate)
                self.logger.gauge("resource_tracker.net_sent_rate_mbps", net_sent_rate)
                self.logger.gauge("resource_tracker.net_recv_rate_mbps", net_recv_rate)
                for gpu in gpu_metrics:
                    self.logger.gauge(f"resource_tracker.gpu_{gpu['gpu_id']}_utilization", gpu["utilization_percent"])
                    self.logger.gauge(f"resource_tracker.gpu_{gpu['gpu_id']}_memory_free_percent", gpu["memory_free_percent"])
                
                duration = (datetime.time() - start_time) * 1000
                self.logger.timing("resource_tracker.collect_metrics_time_ms", duration)
                self.logger.info("Resource metrics collected")
                self.logger.counter("resource_tracker.metrics_collected", 1)
                return metrics

        except Exception as e:
            self.logger.error(f"Error collecting resource metrics: {e}")
            self.logger.counter("resource_tracker.metrics_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def shutdown(self):
        """
        Shutdown the ResourceTracker and close resources.
        """
        self.logger.info("Shutting down ResourceTracker")
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
        self.executor.shutdown(wait=True)
        self.producer.close()
        if self.gpu_available:
            pynvml.nvmlShutdown()
        self.logger.info("Kafka producer and GPU monitoring closed")