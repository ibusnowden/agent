"""
Test Runner for COEVOLVE

Runs all tests and generates a report.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def main():
    """Run all tests with coverage and reporting."""

    print("="*80)
    print("COEVOLVE Test Suite")
    print("="*80)

    # Test configuration
    args = [
        "tests/",
        "-v",  # Verbose
        "--tb=short",  # Short traceback
        "-s",  # Show print statements
        "--color=yes",  # Colored output
    ]

    # Run tests
    exit_code = pytest.main(args)

    print("\n" + "="*80)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
