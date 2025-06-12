#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data I/O module for the Climatic Water Balance Calculator.

This module handles loading and saving data from various formats:
- zarr datasets for temperature and precipitation
- GeoTIFF for land-use data
- Excel for Kc coefficients
- YAML for configuration
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
import yaml
from rasterio.enums import Resampling

from .config import BOUNDING_BOXES

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        yaml.YAMLError: If the configuration file is not valid YAML
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding = 'utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.debug(f"Loaded configuration from {config_path}")
    return config


def setup_output_directory(output_dir: str) -> Path:
    """Create output directory if it doesn't exist.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        Path object for the output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def apply_spatial_filter(data: Union[xr.Dataset, xr.DataArray, rioxarray.raster_array.RasterArray], 
                     config: Dict[str, Any]) -> Union[xr.Dataset, xr.DataArray, rioxarray.raster_array.RasterArray]:
    """Apply spatial filtering to a dataset or data array based on region configuration.
    
    Args:
        data: Dataset or DataArray to filter
        config: Configuration dictionary
        
    Returns:
        Filtered dataset or data array
        
    Raises:
        KeyError: If the specified region is not found in BOUNDING_BOXES
    """
    region_name = config['spatial'].get('region')
    if region_name is None:
        return data
        
    if region_name not in BOUNDING_BOXES.keys():
        raise KeyError(f"Region '{region_name}' not found in BOUNDING_BOXES. Use one of {list(BOUNDING_BOXES.keys())}")
    
    minx, miny, maxx, maxy = BOUNDING_BOXES[region_name]
    
    # Determine if latitude is in ascending or descending order
    if 'lat' in data.coords:
        lat_values = data.lat.values
        if len(lat_values) > 1:
            lat_ascending = lat_values[0] < lat_values[-1]
            
            # Select appropriate slice order based on latitude direction
            if lat_ascending:
                # For ascending latitude (min to max), use slice(miny, maxy)
                data = data.sel(lon=slice(minx, maxx), lat=slice(miny, maxy))
            else:
                # For descending latitude (max to min), use slice(maxy, miny)
                data = data.sel(lon=slice(minx, maxx), lat=slice(maxy, miny))
        else:
            # If there's only one latitude value, just use regular slice
            data = data.sel(lon=slice(minx, maxx), lat=slice(miny, maxy))
    
    logger.debug(f"Applied spatial filter for region '{region_name}'")
    return data


def load_climate_data(config: Dict[str, Any], data_type: str) -> xr.Dataset:
    """Load climate data (temperature or precipitation) from zarr dataset.
    
    Args:
        config: Configuration dictionary
        data_type: Type of climate data to load ('temperature' or 'precipitation')
        
    Returns:
        xarray Dataset containing climate data
        
    Raises:
        FileNotFoundError: If the climate data file does not exist
        ValueError: If the data_type is not supported or variable not found in dataset
    """
    if data_type not in ["temperature", "precipitation"]:
        raise ValueError(f"Unsupported data type: {data_type}. Use 'temperature' or 'precipitation'")
    
    data_config = config["input"][data_type]
    data_path = data_config["path"]
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_type.capitalize()} data not found: {data_path}")
    
    # Load with dask for chunked processing
    ds = xr.open_zarr(data_path, chunks=config["processing"]["chunk_size"], decode_coords='all')
    
    # Check for the variable
    var_name = data_config["variable"]
    if var_name not in ds:
        raise ValueError(f"{data_type.capitalize()} variable '{var_name}' not found in dataset")
   
    # Filter by date range if specified
    if "temporal" in config and "start_date" in config["temporal"] and "end_date" in config["temporal"]:
        start_date = config["temporal"]["start_date"]
        end_date = config["temporal"]["end_date"]
        ds = ds.sel(time=slice(start_date, end_date))

    # Apply spatial filtering
    ds = apply_spatial_filter(ds, config)

    # Compute if not using dask
    if not config['processing'].get('use_dask', True):
        ds = ds.compute()

    # Drop time coordinates where all values are NaN
    # ds = ds.dropna(dim="time", how="all", subset=[var_name])
    
    logger.info(f"Loaded {data_type} data from {data_path}")
    logger.debug(f"{data_type.capitalize()} data shape: {ds[var_name].shape}")
    logger.debug(f"{data_type.capitalize()} data crs: {ds[var_name].rio.crs}")
    logger.debug(f"{data_type.capitalize()} data resolution: {ds[var_name].rio.resolution()}")
    
    return ds[var_name]

def save_water_balance(water_balance: xr.DataArray, config: Dict[str, Any], 
                      period: str, start_date: str, end_date: str) -> str:
    """Save water balance data to GeoTIFF.
    
    Args:
        water_balance: DataArray containing water balance data
        config: Configuration dictionary
        period: Time period (daily, monthly, seasonal, annual)
        start_date: Start date of the period
        end_date: End date of the period
        
    Returns:
        Path to the saved file
    """
    output_config = config["output"]
    output_dir = Path(output_config["directory"])
    
    # Create filename from pattern
    filename_pattern = output_config["filename_pattern"]
    filename = filename_pattern.format(
        frequency=period,
        start_date=start_date,
        end_date=end_date
    )
    output_path = output_dir / filename
    
    # Set export parameters
    export_kwargs = {
        "driver": output_config.get("format", "GTiff"),
    }
    
    # Add compression if specified
    if "compression" in output_config:
        export_kwargs["compress"] = output_config["compression"]
    
    # Save to file
    if 'lon' in water_balance.dims or 'lat' in water_balance.dims:
        water_balance = water_balance.rename({"lon": "x", "lat": "y"})

    water_balance.rio.to_raster(output_path, **export_kwargs)
    
    logger.info(f"Saved water balance to {output_path}")
    
    return str(output_path)


def save_metadata(metadata: Dict[str, Any], output_path: str) -> None:
    """Save metadata to YAML file.
    
    Args:
        metadata: Dictionary containing metadata
        output_path: Path to the output file (will append .meta.yaml)
    """
    meta_path = f"{output_path}.meta.yaml"
    
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    logger.info(f"Saved metadata to {meta_path}")