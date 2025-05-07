#!/usr/bin/env python3
"""
Simple Git automation script for the nextg3n system.
Performs: git status, git add, git commit with timestamp, and git push.
"""

import subprocess
import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command):
    """Run a shell command and return the output."""
    logger.info(f"Executing: {command}")
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True)

    if result.stdout:
        logger.info(f"Output:\n{result.stdout}")

    if result.stderr and result.returncode != 0:
        logger.error(f"Error:\n{result.stderr}")
        raise Exception(f"Command failed with exit code {result.returncode}")

    return result.stdout


def main():
    """Main function to execute git commands."""
    try:
        # Get current git status
        logger.info("\n--- Checking Git Status ---")
        status_output = run_command("git status")

        # Check if there are changes to commit
        if "nothing to commit, working tree clean" in status_output:
            logger.info("No changes to commit. Exiting.")
            return

        # Add all changes
        logger.info("\n--- Adding All Changes ---")
        run_command("git add .")

        # Create commit message with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Automated update: {timestamp}"

        # Commit changes
        logger.info(f"\n--- Committing Changes with message: '{commit_message}' ---")
        run_command(f'git commit -m "{commit_message}"')

        # Push changes
        logger.info("\n--- Pushing to Remote Repository ---")
        run_command("git push")

        logger.info("\n--- Git Operations Completed Successfully ---")

    except Exception as e:
        logger.error(f"\nError occurred: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
