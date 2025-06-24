#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculations module for the Climatic Water Balance Calculator.

This module implements the core calculations for the water balance:
- Thornthwaite method for potential evapotranspiration
- Adjustment of evapotranspiration using land-use specific crop coefficients
- Water balance calculation (P - ET)
"""

import logging
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import xarray as xr

from src.core import Landuse, Precipitation, Temperature
from src.validation import Validator, Watersheds
from src.data_io import (
    save_water_balance,
    save_metadata
)
from src.calc.evapotranspiration import calculate_thornthwaite_pet, adjust_pet_with_kc
from src.correction.utils import construct_interstation_watersheds
from src.correction import PrCorrection
# from src.cluster import start_dask_cluster
from src.utils import get_season
from src.config import DATETIME_FREQUENCY_MAPPING

logger = logging.getLogger(__name__)

def calculate_water_balance(config: Dict[str, Any]) -> List[str]:
    """Calculate water balance using the Thornthwaite method.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of paths to saved output files
    """

    # if config['processing'].get('use_dask', False):
    #     client, cluster = start_dask_cluster()

    # Load input data
    temperature = Temperature(config)
    # temperature.correct() #TODO: Improve this calculation as dask graph seems very inefficient
    temperature.to_geotiff()

    precipitation = Precipitation(config)
    precipitation.to_geotiff()

    landuse = Landuse(config)
    landuse.load()
        
    # Resample data to target grid if needed
    target_resolution = config["spatial"].get("target_resolution", 5)  # Default to 5m
    resampling_method = config["spatial"].get("resampling_method", "bilinear")
    
    # Create target grid based on landuse (high resolution) data
    logger.info(f"Resampling data to target resolution of {target_resolution}m")
    
    # Resample temperature and precipitation to target grid
    temperature.resample_match(landuse.data, resampling_method)
    precipitation.resample_match(landuse.data, resampling_method)
    
    # Calculate potential evapotranspiration using Thornthwaite method
    logger.info("Calculating potential evapotranspiration using Thornthwaite method")
    pet = calculate_thornthwaite_pet(temperature.data, config)
    
    # Adjust PET using land-use specific crop coefficients
    logger.info("Adjusting potential evapotranspiration using crop coefficients")
    landuse.correct()
    et = adjust_pet_with_kc(pet, landuse.kc_grid, config)

    # Initialize watersheds and interstation watersheds
    logger.info("Initializing watersheds and interstation watersheds")
    watersheds = Watersheds(config)
    watersheds.load(target = landuse.data)
    interstation_regions = Watersheds(config, data=construct_interstation_watersheds(watersheds))

    # Initialize validator
    logger.info("Initializing validator")
    validator = Validator(config)
    # Use precipitation here, because we want to compare summed modeled precipitation 
    # with expected precipitation from discharge stations
    validation_tbl = validator.validate(interstation_regions, precipitation.data, compute_for_interstation_regions=True)

    # Precipitation Correction
    logger.info("Correcting Precipitation")
    pr_correction = PrCorrection(config)
    correction_factors = pr_correction.calculate_correction_factors(
        interstation_regions, precipitation.data, pet, validation_tbl
        )
    corr_raster = pr_correction.initialize_correction_grids(interstation_regions, correction_factors)
    pr_corr = pr_correction.apply_correction(precipitation.data, corr_raster)
    
    # Calculate water balance (P - ET)
    logger.info("Calculating water balance")
    ##TODO: Change fixed frequency here and allow dynamic frequency
    et_yearly = et.resample(time='YE-SEP').sum()
    water_balance_corrected = calculate_p_minus_et(pr_corr, et_yearly)

    # Validate results
    logger.info('Validating results after correction')
    validation_tbl_after_correction = validator.validate(watersheds, water_balance_corrected)
    validator.plot_timeseries(validation_tbl_after_correction)
    
    # Aggregate results based on output frequency
    output_frequency = config["temporal"].get("output_frequency", "monthly")
    if isinstance(output_frequency, str):
        output_frequency = [output_frequency]
    
    # Save results
    logger.info(f"Aggregating results to {output_frequency} frequency")
    output_paths = save_results(water_balance_corrected, output_frequency, config)
    
    return output_paths

def calculate_p_minus_et(precipitation: xr.DataArray, et: xr.DataArray) -> xr.DataArray:
    """Calculate water balance as P - ET.
    
    Args:
        precipitation: DataArray containing precipitation data (mm)
        et: DataArray containing evapotranspiration data (mm)
        
    Returns:
        DataArray containing water balance (mm)
    """
    # Ensure precipitation and ET have the same dimensions
    if set(precipitation.dims) != set(et.dims):
        raise ValueError("Precipitation and ET must have the same dimensions")
    
    # Check if both arrays have a time dimension
    if 'time' in precipitation.dims and 'time' in et.dims:
        # Infer the frequency of both time series
        precip_freq = xr.infer_freq(precipitation.time) if len(precipitation.time) > 2 else "YE-SEP" #Assume hydrological year
        et_freq = xr.infer_freq(et.time) if len(et.time) > 2 else "YE-SEP"
        
        logger.debug(f"Precipitation time frequency: {precip_freq}, ET time frequency: {et_freq}")
        
        # If frequencies don't match, resample precipitation to match ET
        if precip_freq != et_freq:
            logger.info(f"Resampling precipitation from {precip_freq} to {et_freq} frequency")
            
            # Determine the target frequency (use ET's frequency)
            if et_freq in DATETIME_FREQUENCY_MAPPING['daily']:
                # Resample to daily
                precipitation = precipitation.resample(time="1D").sum()
            elif et_freq in DATETIME_FREQUENCY_MAPPING['monthly']:
                # Resample to monthly
                precipitation = precipitation.resample(time="1ME").sum()
            elif et_freq in DATETIME_FREQUENCY_MAPPING['annual']:
                # Resample to yearly
                precipitation = precipitation.resample(time="1YE").sum()
            else:
                # For other frequencies, use the exact frequency string
                precipitation = precipitation.resample(time=et_freq).sum()
                
    # Calculate water balance
    water_balance = precipitation - et
    
    # Add metadata
    water_balance.attrs["long_name"] = "Water Balance (P - ET)"
    water_balance.attrs["units"] = "mm"
    water_balance.attrs["description"] = "Climatic water balance calculated as precipitation minus evapotranspiration"

    water_balance = water_balance.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    
    return water_balance


def save_results(water_balance: xr.DataArray, frequency: str, 
                config: Dict[str, Any]) -> List[str]:
    """Save water balance results at the specified frequency.
    
    Args:
        water_balance: DataArray containing water balance data
        frequency: Output frequency (daily, monthly, seasonal, annual)
        config: Configuration dictionary
        
    Returns:
        List of paths to saved output files
    """
    output_paths = []

    if not all([i in ['daily', 'monthly', 'seasonal', 'annual'] for i in frequency]):
        raise ValueError(f"Unsupported output frequency: {frequency}. Choose one of: ['daily', 'monthly', 'seasonal', 'annual']")
    
    if not config['output'].get('save_grids', False):
        return output_paths

    # Check if water_balance has a time dimension
    if "time" not in water_balance.dims:
        # If no time dimension, save as a single file
        start_date = config["temporal"].get("start_date", "")
        end_date = config["temporal"].get("end_date", "")
        
        output_path = save_water_balance(
            water_balance, config, frequency, start_date, end_date
        )
        output_paths.append(output_path)
        
        # Save metadata if requested
        if config["output"].get("create_report", True):
            metadata = {
                "calculation_time": datetime.now().isoformat(),
                "frequency": frequency,
                "start_date": start_date,
                "end_date": end_date,
                "statistics": {
                    "min": float(water_balance.min().values),
                    "max": float(water_balance.max().values),
                    "mean": float(water_balance.mean().values),
                    "std": float(water_balance.std().values)
                }
            }
            save_metadata(metadata, output_path)
    else:
        out_grids = []

        # Resample to the requested frequency
        if "daily" in frequency:
            # No resampling needed for daily output if input is daily
            if xr.infer_freq(water_balance.time) in DATETIME_FREQUENCY_MAPPING['daily']:
                wb_resampled = water_balance
            else:
                # If input is not daily, cannot produce daily output
                logger.warning("Cannot produce daily output from non-daily input")
                return output_paths
            out_grids.append((wb_resampled, 'daily'))

        if "monthly" in frequency:
            if xr.infer_freq(water_balance.time) in DATETIME_FREQUENCY_MAPPING['monthly']:
                wb_resampled = water_balance
            else:
                # Resample to monthly
                wb_resampled = water_balance.resample(time="1M").sum()
            out_grids.append((wb_resampled, 'monthly'))   

        ##TODO: Check seasonal aggregation here            
        if "seasonal" in frequency:
            # Get season definitions from config
            season_months = config["seasons"]
            
            # Create a season coordinate
            season_coord = xr.DataArray(
                [get_season(dt, season_months) for dt in water_balance.time.values],
                dims=["time"],
                coords={"time": water_balance.time}
            )
            
            # Group by season and sum
            wb_resampled = water_balance.groupby(season_coord).sum()
            
            # Convert to DataArray with time dimension for consistent processing
            # Use the middle month of each season as the representative time
            season_times = {
                "winter": "2020-01-15",
                "spring": "2020-04-15",
                "summer": "2020-07-15",
                "autumn": "2020-10-15"
            }
            
            # Create a new time coordinate
            new_times = [pd.Timestamp(season_times[s]) for s in wb_resampled.season.values]
            
            # Create a new DataArray with time dimension
            wb_resampled = xr.DataArray(
                wb_resampled.values,
                dims=["time"] + list(wb_resampled.dims[1:]),
                coords={
                    "time": new_times,
                    **{dim: wb_resampled[dim] for dim in wb_resampled.dims[1:]}
                },
                attrs=wb_resampled.attrs
            )

            out_grids.append((wb_resampled, 'seasonal'))

        if "annual" in frequency:
            if xr.infer_freq(water_balance.time) in DATETIME_FREQUENCY_MAPPING['annual']:
                wb_resampled = water_balance
            else:
                # Resample to annual
                wb_resampled = water_balance.resample(time="1YE").sum()
            out_grids.append((wb_resampled, 'annual'))
        
        # Save each time step as a separate file
        for grid, freq in out_grids:
            for i, time_val in enumerate(grid.time.values):
                # Extract the time step
                wb_time_slice = grid.isel(time=i)
                
                # Format dates for filename
                time_dt = pd.Timestamp(time_val)
                
                if freq == "daily":
                    start_date = end_date = time_dt.strftime("%Y-%m-%d")
                elif freq == "monthly":
                    start_date = time_dt.strftime("%Y-%m-01")
                    end_date = time_dt.strftime("%Y-%m-%d")
                elif freq == "seasonal":
                    # Use season name
                    season = get_season(time_dt, season_months)
                    start_date = season
                    end_date = season
                elif freq == "annual":
                    start_date = time_dt.strftime("%Y-01-01")
                    end_date = time_dt.strftime("%Y-12-31")
                
                # Save the time slice
                output_path = save_water_balance(
                    wb_time_slice, config, freq, start_date, end_date
                )
                output_paths.append(output_path)
                
                # Save metadata if requested
                if config["output"].get("create_report", True):
                    metadata = {
                        "calculation_time": datetime.now().isoformat(),
                        "frequency": frequency,
                        "start_date": start_date,
                        "end_date": end_date,
                        "statistics": {
                            "min": float(wb_time_slice.min().values),
                            "max": float(wb_time_slice.max().values),
                            "mean": float(wb_time_slice.mean().values),
                            "std": float(wb_time_slice.std().values)
                        }
                    }
                    save_metadata(metadata, output_path)
    
    return output_paths