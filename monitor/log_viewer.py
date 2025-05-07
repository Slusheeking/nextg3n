#!/usr/bin/env python3
"""
NextG3N Log Viewer Utility

This script provides a command-line interface for viewing and filtering logs
from the centralized logging system.
"""

import os
import sys
import argparse
import json
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any, Optional

# Add parent directory to path to allow importing from monitor package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.logging_utils import ANSI_COLORS, LOG_DIR, ERROR_LOG_FILE

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="NextG3N Log Viewer")
    parser.add_argument(
        "-f", "--file", 
        help="Log file to view (default: errors.log)",
        default=ERROR_LOG_FILE
    )
    parser.add_argument(
        "-l", "--level", 
        help="Filter by log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    parser.add_argument(
        "-s", "--service", 
        help="Filter by service name"
    )
    parser.add_argument(
        "-n", "--lines", 
        help="Number of lines to show",
        type=int,
        default=50
    )
    parser.add_argument(
        "--since", 
        help="Show logs since time (e.g. '1h', '30m', '1d')"
    )
    parser.add_argument(
        "--follow", 
        help="Follow log file in real-time",
        action="store_true"
    )
    parser.add_argument(
        "--json", 
        help="Output in JSON format",
        action="store_true"
    )
    return parser.parse_args()

def parse_time_spec(time_spec: str) -> Optional[datetime]:
    """Parse a time specification like '1h', '30m', '1d'"""
    if not time_spec:
        return None
        
    units = {
        's': 'seconds',
        'm': 'minutes',
        'h': 'hours',
        'd': 'days',
        'w': 'weeks'
    }
    
    try:
        value = int(time_spec[:-1])
        unit = time_spec[-1].lower()
        
        if unit not in units:
            print(f"Invalid time unit: {unit}. Valid units are: {', '.join(units.keys())}")
            return None
            
        delta_args = {units[unit]: value}
        since_time = datetime.now() - timedelta(**delta_args)
        return since_time
    except (ValueError, IndexError):
        print(f"Invalid time specification: {time_spec}. Format should be like '1h', '30m', '1d'")
        return None

def colorize_log_line(line: str) -> str:
    """Add ANSI colors to log lines based on level"""
    if "[WARNING]" in line:
        return f"{ANSI_COLORS['YELLOW']}{line}{ANSI_COLORS['RESET']}"
    elif "[ERROR]" in line:
        return f"{ANSI_COLORS['RED']}{line}{ANSI_COLORS['RESET']}"
    elif "[CRITICAL]" in line:
        return f"{ANSI_COLORS['BACKGROUND_RED']}{ANSI_COLORS['WHITE']}{ANSI_COLORS['BOLD']}{line}{ANSI_COLORS['RESET']}"
    else:
        return line

def filter_log_line(line: str, level: Optional[str], service: Optional[str], since_time: Optional[datetime]) -> bool:
    """Check if a log line matches the filter criteria"""
    if not line or line.startswith('#'):
        return False
        
    # Filter by level
    if level and f"[{level}]" not in line:
        return False
        
    # Filter by service
    if service and service.lower() not in line.lower():
        return False
        
    # Filter by time
    if since_time:
        try:
            # Extract timestamp from the beginning of the line
            timestamp_str = line.split('[')[0].strip()
            log_time = datetime.fromisoformat(timestamp_str)
            if log_time < since_time:
                return False
        except (ValueError, IndexError):
            # If we can't parse the timestamp, include the line
            pass
            
    return True

def display_logs(args):
    """Display logs based on command line arguments"""
    log_file = args.file
    level = args.level
    service = args.service
    max_lines = args.lines
    since_time = parse_time_spec(args.since)
    follow = args.follow
    json_output = args.json
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    # Initial display
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    filtered_lines = []
    for line in reversed(lines):  # Start with most recent
        if filter_log_line(line.strip(), level, service, since_time):
            filtered_lines.append(line.strip())
            if len(filtered_lines) >= max_lines and not follow:
                break
                
    # Display in reverse order (oldest first)
    for line in reversed(filtered_lines):
        if json_output:
            try:
                # Try to parse as JSON
                data = json.loads(line)
                print(json.dumps(data, indent=2))
            except json.JSONDecodeError:
                print(line)
        else:
            print(colorize_log_line(line))
    
    # Follow mode
    if follow:
        print("\nFollowing log file. Press Ctrl+C to exit.\n")
        
        # Get current file size
        file_size = os.path.getsize(log_file)
        
        try:
            while True:
                # Check if file has been modified
                current_size = os.path.getsize(log_file)
                
                if current_size > file_size:
                    # File has grown, read new content
                    with open(log_file, 'r') as f:
                        f.seek(file_size)
                        new_lines = f.readlines()
                        
                    for line in new_lines:
                        if filter_log_line(line.strip(), level, service, since_time):
                            if json_output:
                                try:
                                    data = json.loads(line)
                                    print(json.dumps(data, indent=2))
                                except json.JSONDecodeError:
                                    print(line.strip())
                            else:
                                print(colorize_log_line(line.strip()))
                    
                    file_size = current_size
                
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nExiting log viewer.")

if __name__ == "__main__":
    args = parse_args()
    display_logs(args)