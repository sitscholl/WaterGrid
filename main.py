#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Climatic Water Balance Calculator

This application calculates climatic water balance following the methodology
described in http://dx.doi.org/10.1002/gdj3.70007.

It handles multi-resolution data efficiently and supports flexible output configurations.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import yaml

from src.data_io import load_config, setup_output_directory
from src.calculations import calculate_water_balance
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate climatic water balance using the Thornthwaite method."
    )
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for the application."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Setup output directory
        output_dir = setup_output_directory(config["output"]["directory"])
        logger.info(f"Output will be saved to {output_dir}")
        
        # Calculate water balance
        logger.info("Starting water balance calculation")
        calculate_water_balance(config)
        
        logger.info("Water balance calculation completed successfully")
        return 0
    
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())