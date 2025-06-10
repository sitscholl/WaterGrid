#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resampling module for the Climatic Water Balance Calculator.

This module handles spatial resampling between different resolution grids,
specifically for handling the difference between the 250m climate data
and the 5m land-use data.
"""

import logging
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)


def resample_to_target_grid(source: xr.DataArray, target: xr.DataArray, method: str = "bilinear") -> xr.DataArray:
    """Resample a source grid to match a target grid while preserving chunking.
    
    Args:
        source: DataArray to resample
        target: DataArray with the target grid
        method: Resampling method (nearest, bilinear, cubic)
        
    Returns:
        Resampled DataArray with exact same x and y coordinates as target grid
    """
    # Check if source has rio accessor
    if not hasattr(source, "rio"):
        # Convert to rioxarray
        source = rioxarray.open_rasterio(source)
    
    # Get target bounds and CRS
    target_crs = target.rio.crs
      
    # Map interp methods
    resampling_methods_rasterio = {        
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }

    resampling_methods = {
        "nearest": "nearest",
        "bilinear": "linear",
        "cubic": "cubic",
    }    

    if method not in resampling_methods:
        raise ValueError(f"Unsupported resampling method: {method}. "
                       f"Supported methods: {list(resampling_methods.keys())}")
    
    # First ensure the source is in the same CRS as the target
    if source.rio.crs != target_crs:
        # This step still loads into memory, but we need it for CRS transformation
        logger.warning(f"Source CRS ({source.rio.crs}) does not match target CRS ({target_crs}). Reprojecting source to match target CRS (requires array to be loaded into memory).")
        source = source.rio.reproject(
            target_crs,
            resampling=resampling_methods_rasterio[method]
        )
                
    # Use xarray's interp which preserves chunking
    resampled = source.interp_like(target, method = resampling_methods[method])
    
    return resampled


def align_grids(grids: Dict[str, xr.DataArray], target_resolution: float, 
               method: str = "bilinear") -> Dict[str, xr.DataArray]:
    """Align multiple grids to a common grid with the specified resolution.
    
    Args:
        grids: Dictionary of named DataArrays to align
        target_resolution: Target resolution in meters
        method: Resampling method (nearest, bilinear, cubic)
        
    Returns:
        Dictionary of aligned DataArrays
    """
    # Find the grid with the highest resolution (smallest cell size)
    highest_res_grid = None
    highest_res_name = None
    smallest_cell_size = float('inf')
    
    for name, grid in grids.items():
        # Get cell size
        if hasattr(grid, "rio"):
            transform = grid.rio.transform()
            cell_size = abs(transform[0])  # Absolute value of pixel width
            
            if cell_size < smallest_cell_size:
                smallest_cell_size = cell_size
                highest_res_grid = grid
                highest_res_name = name
    
    if highest_res_grid is None:
        raise ValueError("No valid grid found with rio accessor")
    
    logger.info(f"Using {highest_res_name} as reference grid with resolution {smallest_cell_size}m")
    
    # If target resolution is higher than the highest resolution grid,
    # we need to upsample the reference grid first
        # Ensure chunking is preserved or re-applied if it was lost
    if hasattr(source, "chunks") and source.chunks is not None:
        # Try to preserve original chunking pattern along non-spatial dimensions
        chunks = {}
        for dim in resampled.dims:
            if dim in ["lat", "lon"]:
                # For spatial dimensions, we might need different chunking
                # based on the new dimensions
                continue
            elif dim in source.dims and source.chunks is not None:
                # Find the chunk size for this dimension in the source
                dim_index = source.dims.index(dim)
                if dim_index < len(source.chunks):
                    chunks[dim] = source.chunks[dim_index]
        
        # Apply chunking if we have any dimensions to chunk
        if chunks:
            resampled = resampled.chunk(chunks)
            logger.warning(f"Target resolution {target_resolution}m is higher than "
                     f"the highest resolution grid ({smallest_cell_size}m). "
                     f"This may lead to artificial precision.")
        
        # Calculate new dimensions for the reference grid
        bounds = highest_res_grid.rio.bounds()
        minx, miny, maxx, maxy = bounds
        width = int((maxx - minx) / target_resolution)
        height = int((maxy - miny) / target_resolution)
        
        # Resample the reference grid to the target resolution
        reference_grid = highest_res_grid.rio.reproject(
            highest_res_grid.rio.crs,
            shape=(height, width),
            bounds=bounds,
            resampling=Resampling.bilinear
        )
    else:
        # Use the highest resolution grid as reference
        reference_grid = highest_res_grid
    
    # Align all grids to the reference grid
    aligned_grids = {}
    
    for name, grid in grids.items():
        if grid is highest_res_grid and target_resolution >= smallest_cell_size:
            # No need to resample the reference grid if target resolution is not higher
            aligned_grids[name] = grid
        else:
            # Resample to match the reference grid
            aligned_grids[name] = resample_to_target_grid(
                grid, reference_grid, method
            )
    
    return aligned_grids