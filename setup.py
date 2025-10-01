"""
Office Seat Occupancy Detection System
=====================================

A comprehensive data science project for detecting and analyzing office seat occupancy
using computer vision and machine learning techniques.

Author: Academic Project
Date: 2024
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Create necessary directories
def create_directories():
    """Create necessary project directories"""
    directories = [
        "data/videos",
        "data/processed", 
        "data/raw",
        "output",
        "results",
        "logs",
        "notebooks",
        "tests",
        "src",
        "src/models",
        "src/utils",
        "src/analysis",
        "src/visualization"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    create_directories()
