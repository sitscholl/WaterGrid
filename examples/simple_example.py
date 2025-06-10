#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple example of using the Climatic Water Balance Calculator.

This example demonstrates how to use the application programmatically
rather than through the command line interface.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_io import load_config
from src.calculations import calculate_water_balance
from src.utils import setup_logging


def main():
    """Run a simple example calculation."""
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    config_path = Path("../config.yaml")
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    config = load_config(config_path)
    
    # Override configuration for example
    # This would typically be done by editing the config file,
    # but we do it here for demonstration purposes
    
    # Set input paths relative to this script
    data_dir = Path("../data")
    config["input"]["temperature"]["path"] = str(data_dir / "temperature.zarr")
    config["input"]["precipitation"]["path"] = str(data_dir / "precipitation.zarr")
    config["input"]["landuse"]["path"] = str(data_dir / "landuse.tif")
    config["input"]["kc_coefficients"]["path"] = str(data_dir / "kc_values.xlsx")
    
    # Set output directory
    results_dir = Path("../results/example")
    results_dir.mkdir(parents=True, exist_ok=True)
    config["output"]["directory"] = str(results_dir)
    
    # Calculate water balance
    try:
        logger.info("Starting water balance calculation")
        output_paths = calculate_water_balance(config)
        
        logger.info("Water balance calculation completed successfully")
        logger.info(f"Output files: {output_paths}")
        
        return 0
    except Exception as e:
        logger.exception(f"Error during calculation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())