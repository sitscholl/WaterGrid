#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for the Climatic Water Balance Calculator.

This module provides helper functions for logging, date handling,
and other miscellaneous tasks.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd
import xarray as xr


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"water_balance_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Silence noisy libraries
    logging.getLogger('rasterio').setLevel(logging.WARNING)
    logging.getLogger('numcodecs').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
    logging.getLogger('flox').setLevel(logging.WARNING)
    
    logger = logging.getLogger("water_balance")
    logger.info(f"Logging initialized at level {logging.getLevelName(level)}")
    logger.info(f"Log file: {log_file}")
    
    return logger


def get_season(date: Union[datetime, pd.Timestamp], season_months: Dict[str, List[str]]) -> str:
    """Determine the season for a given date based on the configuration.
    
    Args:
        date: Date to determine season for
        season_months: Dictionary mapping season names to lists of month numbers (as strings)
        
    Returns:
        Season name (winter, spring, summer, autumn)
        
    Raises:
        ValueError: If the month is not assigned to any season
    """
    month_str = str(date.month).zfill(2)  # Convert to two-digit string (01, 02, etc.)
    
    for season, months in season_months.items():
        if month_str in months:
            return season
    
    # If we get here, the month is not assigned to any season
    raise ValueError(f"Month {month_str} is not assigned to any season in the configuration")


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in a human-readable format.
    
    Args:
        seconds: Elapsed time in seconds
        
    Returns:
        Formatted time string (e.g., "2h 30m 45s")
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{int(minutes)}m")
    parts.append(f"{seconds:.1f}s")
    
    return " ".join(parts)

def align_chunks(data: xr.DataArray | xr.Dataset, target_chunks: dict[tuple]):

    if target_chunks is None:
        return data

    if not isinstance(target_chunks, dict):
        raise ValueError(f"target_chunks must be a dictionary mapping dimensions to chunk sizes, got {type(target_chunks)} instead.")
        
    common_dims = set(target_chunks.keys()) & set(data.dims)
    data_chunks = {dim: target_chunks[dim] for dim in common_dims}
   
    return data.chunk(data_chunks)

def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return a list of validation errors.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check required sections
    required_sections = ["input", "output", "spatial", "temporal", "processing", "calculation"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # If any required sections are missing, return errors
    if errors:
        return errors
    
    # Check input configuration
    input_config = config["input"]
    required_inputs = ["temperature", "precipitation", "landuse", "kc_coefficients"]
    for input_type in required_inputs:
        if input_type not in input_config:
            errors.append(f"Missing required input: {input_type}")
        elif "path" not in input_config[input_type]:
            errors.append(f"Missing path for input: {input_type}")
        elif not os.path.exists(input_config[input_type]["path"]):
            errors.append(f"Input file not found: {input_config[input_type]['path']}")
    
    # Check temporal configuration
    temporal_config = config["temporal"]
    if "start_date" not in temporal_config or "end_date" not in temporal_config:
        errors.append("Missing start_date or end_date in temporal configuration")
    else:
        # Validate date format
        try:
            start_date = pd.Timestamp(temporal_config["start_date"])
            end_date = pd.Timestamp(temporal_config["end_date"])
            if end_date < start_date:
                errors.append(f"End date {end_date} is before start date {start_date}")
        except ValueError as e:
            errors.append(f"Invalid date format: {e}")
    
    # Check output frequency
    valid_frequencies = ["daily", "monthly", "seasonal", "annual"]
    if "output_frequency" not in temporal_config:
        errors.append("Missing output_frequency in temporal configuration")
    elif temporal_config["output_frequency"] not in valid_frequencies:
        errors.append(f"Invalid output_frequency: {temporal_config['output_frequency']}. "
                     f"Valid values: {valid_frequencies}")
    
    # Check seasons configuration if output frequency is seasonal
    if temporal_config.get("output_frequency") == "seasonal" and "seasons" not in config:
        errors.append("Missing seasons configuration for seasonal output frequency")
    
    return errors