#!/usr/bin/env python3
"""
Test runner for all NextG3N MCP server tests.
This script executes all test files in the test directory.

Usage:
    python run_tests.py                # Run all tests
    python run_tests.py test_file.py   # Run a specific test file
    python run_tests.py -k keyword     # Run tests matching keyword
"""

import os
import sys
import pytest
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run NextG3N MCP server tests.')
    parser.add_argument('test_file', nargs='?', help='Specific test file to run (optional)')
    parser.add_argument('-k', '--keyword', help='Only run tests matching the given keyword expression')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-xvs', '--xvs', action='store_true', help='Exit on first error, show output')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage report')
    args = parser.parse_args()

    # Change to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Build pytest arguments
    pytest_args = []
    
    # Add test file if specified
    if args.test_file:
        if not os.path.exists(f"test/{args.test_file}") and not os.path.exists(args.test_file):
            print(f"Error: Test file '{args.test_file}' not found.")
            sys.exit(1)
        
        test_path = f"test/{args.test_file}" if not os.path.exists(args.test_file) else args.test_file
        pytest_args.append(test_path)
    else:
        # Run all test files in the test directory
        pytest_args.append("test/")

    # Add keyword filter if specified
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])

    # Add verbosity
    if args.verbose:
        pytest_args.append("-v")

    # Exit on first error and show output
    if args.xvs:
        pytest_args.extend(["-xvs"])

    # Add coverage if requested
    if args.coverage:
        pytest_args.extend(["--cov=mcp", "--cov-report=term", "--cov-report=html"])

    # Print the command being run
    print(f"Running: pytest {' '.join(pytest_args)}")

    # Run the tests
    result = pytest.main(pytest_args)
    
    sys.exit(result)


if __name__ == "__main__":
    main()