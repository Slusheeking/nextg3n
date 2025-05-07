"""
System Analyzer for NextG3N Trading System

This module implements the SystemAnalyzer class, monitoring the health and performance of the
NextG3N system. It collects system metrics (CPU, memory, GPU), tracks operation latencies,
detects anomalies, and generates alerts, supporting the MonitorAgent in TradeFlowOrchestrator.
"""

import os
import json
import logging
import asyncio
import aiohttp
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
from nextg3n.monitoring.metrics_logger import MetricsLogger

class SystemAnalyzer:
    """
    Class for monitoring system health and performance in the NextG3N system.
    Collects metrics, detects anomalies, and generates alerts.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SystemAnalyzer with configuration and monitoring settings.

        Args:
            config: Configuration dictionary with monitoring and Kafka settings
        """
        init_start_time = datetime.time()
        
        # Initialize logging
        self.logger = MetricsLogger(component_name="system_analyzer")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(handler)

        # Load configuration
        self.config = config
        self.monitoring_config = config.get("monitoring", {}).get("system_analyzer", {})
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
                self.gpu_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)
                ]
                self.logger.info("Initialized GPU monitoring for %s GPUs", self.gpu_count)
            except pynvml.NVMLError as e:
                self.logger.warning("Failed to initialize GPU monitoring: %s", e)
                self.gpu_available = False
        
        # Initialize thresholds for anomaly detection
        self.thresholds = self.monitoring_config.get("thresholds", {
            "cpu_percent": 90.0,  # Alert if CPU usage > 90%
            "memory_percent": 85.0,  # Alert if memory usage > 85%
            "gpu_utilization": 95.0,  # Alert if GPU utilization > 95%
            "gpu_memory_free_percent": 10.0,  # Alert if GPU free memory < 10%
            "latency_ms": 1000.0  # Alert if operation latency > 1000ms
        })
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialize monitoring state
        self.running = True
        self.monitor_task = None
        
        init_duration = (datetime.time() - init_start_time) * 1000
        self.logger.timing("system_analyzer.initialization_time_ms", init_duration)
        self.logger.info("SystemAnalyzer initialized")

    async def start_monitoring(self, interval: float = 60.0):
        """
        Start continuous system monitoring in a background task.

        Args:
            interval: Monitoring interval in seconds (default: 60)
        """
        self.logger.info(f"Starting system monitoring with interval {interval}s")
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))

    async def _monitor_loop(self, interval: float):
        """
        Continuous monitoring loop to collect metrics and detect anomalies.

        Args:
            interval: Monitoring interval in seconds
        """
        while self.running:
            try:
                metrics = await self.collect_metrics()
                anomalies = self.detect_anomalies(metrics)
                if anomalies:
                    await self._publish_alerts(anomalies)
                
                # Publish metrics to Kafka
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}monitoring_events",
                    {"event": "system_metrics", "data": metrics}
                )
                
                await asyncio.sleep(interval)
            
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self.logger.counter("system_analyzer.monitor_errors", 1)
                await asyncio.sleep(interval)

    async def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect system metrics (CPU, memory, GPU, latency).

        Returns:
            Dictionary containing system metrics
        """
        start_time = datetime.time()
        operation_id = f"collect_metrics_{int(start_time)}"
        self.logger.info(f"Collecting system metrics - Operation: {operation_id}")

        try:
            async with aiohttp.ClientSession() as session:
                loop = asyncio.get_event_loop()
                
                # CPU and memory metrics
                cpu_percent = await loop.run_in_executor(self.executor, psutil.cpu_percent)
                memory = await loop.run_in_executor(self.executor, psutil.virtual_memory)
                memory_percent = memory.percent
                memory_used = memory.used / (1024 ** 3)  # Convert to GB
                memory_total = memory.total / (1024 ** 3)  # Convert to GB
                
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
                
                # Latency metrics (example: aggregate from MetricsLogger)
                latency_metrics = self._aggregate_latency_metrics()
                
                metrics = {
                    "success": True,
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "memory_used_gb": memory_used,
                    "memory_total_gb": memory_total,
                    "gpu_metrics": gpu_metrics,
                    "latency_metrics": latency_metrics,
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

                self.logger.gauge("system_analyzer.cpu_percent", cpu_percent)
                self.logger.gauge("system_analyzer.memory_percent", memory_percent)
                for gpu in gpu_metrics:
                    self.logger.gauge(
                        "system_analyzer.gpu_%s_utilization", gpu["gpu_id"], gpu["utilization_percent"]
                    )
                    self.logger.gauge(
                        "system_analyzer.gpu_%s_memory_free_percent",
                        gpu["gpu_id"],
                        gpu["memory_free_percent"],
                    )

                duration = (datetime.datetime() - start_time) * 1000
                self.logger.timing("system_analyzer.collect_metrics_time_ms", duration)
                self.logger.info("System metrics collected")
                self.logger.counter("system_analyzer.metrics_collected", 1)
                return metrics

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            self.logger.counter("system_analyzer.metrics_errors", 1)
            return {
                "success": False,
                "error": str(e),
                "operation_id": operation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def _aggregate_latency_metrics(self) -> Dict[str, float]:
        """
        Aggregate latency metrics from MetricsLogger.

        Returns:
            Dictionary of latency metrics
        """
        latency_metrics = {}
        # Retrieve latency metrics from MetricsLogger
        for metric_name, metric_value in self.logger.metrics.items():
            if "latency" in metric_name:
                latency_metrics[metric_name] = metric_value
        return latency_metrics

    def detect_anomalies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in system metrics using threshold-based rules.

        Args:
            metrics: System metrics dictionary

        Returns:
            List of anomaly dictionaries
        """
        anomalies = []
        if not metrics.get("success"):
            return anomalies

        try:
            # CPU usage
            if metrics["cpu_percent"] > self.thresholds["cpu_percent"]:
                anomalies.append(
                    {
                        "metric": "cpu_percent",
                        "value": metrics["cpu_percent"],
                        "threshold": self.thresholds["cpu_percent"],
                        "severity": "high",
                        "message": (
                            f"High CPU usage: {metrics['cpu_percent']}% > "
                            f"{self.thresholds['cpu_percent']}%"
                        ),
                    }
                )

            # Memory usage
            if metrics["memory_percent"] > self.thresholds["memory_percent"]:
                anomalies.append(
                    {
                        "metric": "memory_percent",
                        "value": metrics["memory_percent"],
                        "threshold": self.thresholds["memory_percent"],
                        "severity": "high",
                        "message": (
                            f"High memory usage: {metrics['memory_percent']}% > "
                            f"{self.thresholds['memory_percent']}%"
                        ),
                    }
                )

            # GPU metrics
            for gpu in metrics["gpu_metrics"]:
                if gpu["utilization_percent"] > self.thresholds["gpu_utilization"]:
                    anomalies.append(
                        {
                            "metric": f"gpu_{gpu['gpu_id']}_utilization",
                            "value": gpu["utilization_percent"],
                            "threshold": self.thresholds["gpu_utilization"],
                            "severity": "medium",
                            "message": (
                                f"High GPU {gpu['gpu_id']} utilization: "
                                f"{gpu['utilization_percent']}% > {self.thresholds['gpu_utilization']}%"
                            ),
                        }
                    )
                if gpu["memory_free_percent"] < self.thresholds["gpu_memory_free_percent"]:
                    anomalies.append(
                        {
                            "metric": f"gpu_{gpu['gpu_id']}_memory_free_percent",
                            "value": gpu["memory_free_percent"],
                            "threshold": self.thresholds["gpu_memory_free_percent"],
                            "severity": "high",
                            "message": (
                                f"Low GPU {gpu['gpu_id']} free memory: "
                                f"{gpu['memory_free_percent']}% < "
                                f"{self.thresholds['gpu_memory_free_percent']}%"
                            ),
                        }
                    )

            # Latency metrics
            for metric, value in metrics["latency_metrics"].items():
                if value > self.thresholds["latency_ms"]:
                    anomalies.append(
                        {
                            "metric": metric,
                            "value": value,
                            "threshold": self.thresholds["latency_ms"],
                            "severity": "medium",
                            "message": (
                                f"High latency for {metric}: {value}ms > "
                                f"{self.thresholds['latency_ms']}ms"
                            ),
                        }
                    )

            self.logger.info(f"Detected {len(anomalies)} anomalies")
            self.logger.counter("system_analyzer.anomalies_detected", len(anomalies))
            return anomalies

        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return []

    async def _publish_alerts(self, anomalies: List[Dict[str, Any]]):
        """
        Publish alerts for detected anomalies to Kafka.

        Args:
            anomalies: List of anomaly dictionaries
        """
        operation_id = f"alert_{int(datetime.time())}"
        self.logger.info(f"Publishing {len(anomalies)} alerts - Operation: {operation_id}")

        try:
            for anomaly in anomalies:
                alert = {
                    "success": True,
                    "metric": anomaly["metric"],
                    "value": anomaly["value"],
                    "threshold": anomaly["threshold"],
                    "severity": anomaly["severity"],
                    "message": anomaly["message"],
                    "operation_id": operation_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                self.producer.send(
                    f"{self.kafka_config.get('topic_prefix', 'nextg3n_')}monitoring_events",
                    {"event": "system_alert", "data": alert}
                )
                self.logger.info(f"Published alert: {anomaly['message']}")
                self.logger.counter("system_analyzer.alerts_published", 1)
        
        except Exception as e:
            self.logger.error(f"Error publishing alerts: {e}")
            self.logger.counter("system_analyzer.alert_errors", len(anomalies))

    def shutdown(self):
        """
        Shutdown the SystemAnalyzer and close resources.
        """
        self.logger.info("Shutting down SystemAnalyzer")
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
        self.executor.shutdown(wait=True)
        self.producer.close()
        if self.gpu_available:
            pynvml.nvmlShutdown()
        self.logger.info("Kafka producer and GPU monitoring closed")