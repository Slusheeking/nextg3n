"""
Centralized logging system for NextG3N Trading System.
Provides low-latency logging with FastAPI export capabilities.
"""

import os
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from threading import Lock
import queue
import threading
from functools import lru_cache

# Constants
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
MAX_LOG_ENTRIES = 10000  # Maximum number of log entries to keep in memory
FLUSH_INTERVAL = 5  # Seconds between log flushes to disk

# ANSI color codes for terminal output
ANSI_COLORS = {
    "RESET": "\033[0m",
    "BLACK": "\033[30m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
    "BACKGROUND_RED": "\033[41m",
    "BACKGROUND_YELLOW": "\033[43m"
}

# Centralized error log file path
ERROR_LOG_FILE = os.path.join(LOG_DIR, "errors.log")

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# In-memory log storage for fast access
log_buffer = queue.Queue(maxsize=MAX_LOG_ENTRIES)
log_list = []  # For API access
log_lock = Lock()  # Thread safety for log_list

# Configure logging levels
LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to logs based on level"""
    
    COLORS = {
        "DEBUG": ANSI_COLORS["BLUE"],
        "INFO": ANSI_COLORS["GREEN"],
        "WARNING": ANSI_COLORS["YELLOW"],
        "ERROR": ANSI_COLORS["RED"],
        "CRITICAL": ANSI_COLORS["BACKGROUND_RED"] + ANSI_COLORS["WHITE"] + ANSI_COLORS["BOLD"]
    }
    
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{ANSI_COLORS['RESET']}"
            record.msg = f"{self.COLORS[levelname]}{record.msg}{ANSI_COLORS['RESET']}"
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if available
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if available
        if hasattr(record, "extra"):
            log_entry.update(record.extra)
            
        return json.dumps(log_entry)

class MemoryHandler(logging.Handler):
    """Custom handler that stores logs in memory for fast API access"""
    
    def emit(self, record):
        try:
            log_entry = self.format(record)
            
            # Add to queue for file writing
            try:
                log_buffer.put_nowait(log_entry)
            except queue.Full:
                # If buffer is full, remove oldest entry
                try:
                    log_buffer.get_nowait()
                    log_buffer.put_nowait(log_entry)
                except:
                    pass
            
            # Add to list for API access (with lock for thread safety)
            with log_lock:
                log_list.append(json.loads(log_entry))
                if len(log_list) > MAX_LOG_ENTRIES:
                    log_list.pop(0)
                    
        except Exception:
            self.handleError(record)

class AsyncFileWriter(threading.Thread):
    """Asynchronous file writer that flushes logs to disk periodically"""
    
    def __init__(self, log_file):
        super().__init__(daemon=True)
        self.log_file = log_file
        self.running = True
        
    def run(self):
        while self.running:
            entries = []
            # Collect all available entries
            while not log_buffer.empty() and len(entries) < 1000:
                try:
                    entries.append(log_buffer.get_nowait())
                except queue.Empty:
                    break
            
            # Write collected entries to file
            if entries:
                try:
                    with open(self.log_file, 'a') as f:
                        for entry in entries:
                            f.write(f"{entry}\n")
                except Exception as e:
                    print(f"Error writing to log file: {e}")
            
            # Sleep before next flush
            time.sleep(FLUSH_INTERVAL)
    
    def stop(self):
        self.running = False

@lru_cache(maxsize=32)
def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger with the specified name.
    Uses LRU cache to avoid creating duplicate loggers.
    
    Args:
        name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set level
    logger.setLevel(LEVEL_MAP.get(LOG_LEVEL, logging.INFO))
    
    # Create console handler with colored output
    console_handler = logging.StreamHandler()
    console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_handler.setFormatter(ColoredFormatter(console_format))
    logger.addHandler(console_handler)
    
    # Create memory handler with JSON formatter
    memory_handler = MemoryHandler()
    memory_handler.setFormatter(JSONFormatter())
    logger.addHandler(memory_handler)
    
    # Create centralized error log file handler (only for WARNING, ERROR, CRITICAL)
    try:
        error_handler = logging.FileHandler(ERROR_LOG_FILE)
        error_handler.setLevel(logging.WARNING)  # Only log WARNING and above
        error_format = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s"
        error_handler.setFormatter(logging.Formatter(error_format))
        logger.addHandler(error_handler)
    except Exception as e:
        print(f"Error setting up error log file: {e}")
    
    return logger

# Start the async file writer
log_file = os.path.join(LOG_DIR, f"nextg3n_{datetime.now().strftime('%Y%m%d')}.log")
file_writer = AsyncFileWriter(log_file)
file_writer.start()

# Initialize the centralized error log file with a header
try:
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(f"# NextG3N Error Log - Started {datetime.now().isoformat()}\n")
        f.write("# This file contains WARNING, ERROR, and CRITICAL log messages from all services\n")
        f.write("#" + "-" * 80 + "\n\n")
except Exception as e:
    print(f"Error initializing error log file: {e}")

def get_logs(
    level: Optional[str] = None,
    service: Optional[str] = None,
    limit: int = 100,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get filtered logs for API access.
    
    Args:
        level: Filter by log level
        service: Filter by service name
        limit: Maximum number of logs to return
        start_time: Filter logs after this time (ISO format)
        end_time: Filter logs before this time (ISO format)
        
    Returns:
        List of log entries
    """
    with log_lock:
        filtered_logs = log_list.copy()
    
    # Apply filters
    if level:
        filtered_logs = [log for log in filtered_logs if log["level"] == level.upper()]
    
    if service:
        filtered_logs = [log for log in filtered_logs if service.lower() in log["logger"].lower()]
    
    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time)
            filtered_logs = [log for log in filtered_logs if datetime.fromisoformat(log["timestamp"]) >= start_dt]
        except ValueError:
            pass
    
    if end_time:
        try:
            end_dt = datetime.fromisoformat(end_time)
            filtered_logs = [log for log in filtered_logs if datetime.fromisoformat(log["timestamp"]) <= end_dt]
        except ValueError:
            pass
    
    # Sort by timestamp (newest first) and limit
    filtered_logs.sort(key=lambda x: x["timestamp"], reverse=True)
    return filtered_logs[:limit]

def shutdown_logging():
    """Properly shutdown logging system"""
    if file_writer and file_writer.is_alive():
        file_writer.stop()
        file_writer.join(timeout=5)
        
        # Flush remaining logs
        entries = []
        while not log_buffer.empty():
            try:
                entries.append(log_buffer.get_nowait())
            except queue.Empty:
                break
        
        if entries:
            try:
                with open(log_file, 'a') as f:
                    for entry in entries:
                        f.write(f"{entry}\n")
            except Exception as e:
                print(f"Error writing to log file during shutdown: {e}")