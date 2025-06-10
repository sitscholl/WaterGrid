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
    
    with open(config_path, "r") as f:
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


def load_temperature_data(config: Dict[str, Any]) -> xr.Dataset:
    """Load temperature data from zarr dataset.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        xarray Dataset containing temperature data
        
    Raises:
        FileNotFoundError: If the temperature data file does not exist
    """
    temp_config = config["input"]["temperature"]
    temp_path = temp_config["path"]
    
    if not os.path.exists(temp_path):
        raise FileNotFoundError(f"Temperature data not found: {temp_path}")
    
    # Load with dask for chunked processing
    ds = xr.open_zarr(temp_path, chunks=config["processing"]["chunk_size"], decode_coords = 'all')
    
    # Check for the temperature variable
    temp_var = temp_config["variable"]
    if temp_var not in ds:
        raise ValueError(f"Temperature variable '{temp_var}' not found in dataset")
   
    # Filter by date range if specified
    if "temporal" in config and "start_date" in config["temporal"] and "end_date" in config["temporal"]:
        start_date = config["temporal"]["start_date"]
        end_date = config["temporal"]["end_date"]
        ds = ds.sel(time=slice(start_date, end_date))

    # Drop time coordinates where all values are NaN
    # ds = ds.dropna(dim="time", how="all", subset=[temp_var])
    
    logger.info(f"Loaded temperature data from {temp_path}")
    logger.debug(f"Temperature data shape: {ds[temp_var].shape}")
    logger.debug(f"Temperature data crs: {ds[temp_var].rio.crs}")
    logger.debug(f"Temperature data resolution: {ds[temp_var].rio.resolution()}")
    
    return ds


def load_precipitation_data(config: Dict[str, Any]) -> xr.Dataset:
    """Load precipitation data from zarr dataset.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        xarray Dataset containing precipitation data
        
    Raises:
        FileNotFoundError: If the precipitation data file does not exist
    """
    precip_config = config["input"]["precipitation"]
    precip_path = precip_config["path"]
    
    if not os.path.exists(precip_path):
        raise FileNotFoundError(f"Precipitation data not found: {precip_path}")
    
    # Load with dask for chunked processing
    ds = xr.open_zarr(precip_path, chunks=config["processing"]["chunk_size"], decode_coords='all')
    
    # Check for the precipitation variable
    precip_var = precip_config["variable"]
    if precip_var not in ds:
        raise ValueError(f"Precipitation variable '{precip_var}' not found in dataset")
   
    # Filter by date range if specified
    if "temporal" in config and "start_date" in config["temporal"] and "end_date" in config["temporal"]:
        start_date = config["temporal"]["start_date"]
        end_date = config["temporal"]["end_date"]
        ds = ds.sel(time=slice(start_date, end_date))

    # Drop time coordinates where all values are NaN
    # ds = ds.dropna(dim="time", how="all", subset=[precip_var])
    
    logger.info(f"Loaded precipitation data from {precip_path}")
    logger.debug(f"Precipitation data shape: {ds[precip_var].shape}")
    logger.debug(f"Precipitation data crs: {ds[precip_var].rio.crs}")
    logger.debug(f"Precipitation data resolution: {ds[precip_var].rio.resolution()}")
    
    return ds


def load_landuse_data(config: Dict[str, Any]) -> rioxarray.raster_array.RasterArray:
    """Load land-use data from GeoTIFF.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RasterArray containing land-use data
        
    Raises:
        FileNotFoundError: If the land-use data file does not exist
    """
    landuse_config = config["input"]["landuse"]
    landuse_path = landuse_config["path"]
    target_res = config['spatial']['target_resolution']
    
    if not os.path.exists(landuse_path):
        raise FileNotFoundError(f"Land-use data not found: {landuse_path}")
    
    # Load with rioxarray
    landuse = rioxarray.open_rasterio(landuse_path).squeeze(drop = True)

    if "x" in landuse.coords:
        landuse = landuse.rename({"x": "lon"})
    if "y" in landuse.coords:
        landuse = landuse.rename({"y": "lat"})
    landuse = landuse.rio.set_spatial_dims(x_dim = 'lon', y_dim = 'lat')
    
    # Check CRS
    expected_crs = landuse_config.get("crs", "EPSG:32632")
    if (landuse.rio.crs.to_string() != expected_crs) or (landuse.rio.resolution()[0] != target_res):
        logger.info(f"Reprojecting land-use data to {expected_crs} and resolution {target_res} using Resampling.nearest")
        
        target_bounds = landuse.rio.bounds()
        minx, miny, maxx, maxy = target_bounds
        width = int((maxx - minx) / target_res)
        height = int((maxy - miny) / target_res)
        
        # Resample to target grid
        landuse = landuse.rio.reproject(
            expected_crs,
            shape=(height, width),
            bounds=target_bounds,
            resampling=Resampling.nearest
        )
    
    logger.info(f"Loaded land-use data from {landuse_path}")
    logger.debug(f"Land-use data shape: {landuse.shape}")
    logger.debug(f"Land-use data crs: {landuse.rio.crs}")
    logger.debug(f"Land-use data resolution: {landuse.rio.resolution()}")
    
    return landuse


def load_kc_coefficients(config: Dict[str, Any]) -> pd.DataFrame:
    """Load Kc coefficients from Excel file.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame containing Kc coefficients for different land-use types and seasons
        
    Raises:
        FileNotFoundError: If the Kc coefficients file does not exist
    """
    kc_config = config["input"]["kc_coefficients"]
    kc_path = kc_config["path"]
    
    if not os.path.exists(kc_path):
        raise FileNotFoundError(f"Kc coefficients file not found: {kc_path}")
    
    # Load Excel file
    sheet_name = kc_config.get("sheet_name", 0)
    kc_df = pd.read_excel(kc_path, sheet_name=sheet_name)
    
    # Validate DataFrame structure
    required_columns = ["landuse_code", *list(config['seasons'].keys())]
    missing_columns = [col for col in required_columns if col not in kc_df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in Kc coefficients file: {missing_columns}")
    
    logger.info(f"Loaded Kc coefficients from {kc_path}")
    logger.debug(f"Kc coefficients shape: {kc_df.shape}")
    
    return kc_df


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