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
from typing import Dict, Any, Tuple, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import dask

from src.data_io import (
    load_temperature_data,
    load_precipitation_data,
    load_landuse_data,
    load_kc_coefficients,
    save_water_balance,
    save_metadata
)
from src.calc.day_length import day_lengths, get_lat_in_4326
from src.resampling import resample_to_target_grid
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
    # Load input data
    temp_ds = load_temperature_data(config)
    precip_ds = load_precipitation_data(config)
    landuse = load_landuse_data(config)
    kc_df = load_kc_coefficients(config)
    
    # Extract variables from datasets
    temp_var = config["input"]["temperature"]["variable"]
    precip_var = config["input"]["precipitation"]["variable"]
    
    temperature = temp_ds[temp_var]
    precipitation = precip_ds[precip_var]
    
    # Resample data to target grid if needed
    target_resolution = config["spatial"].get("target_resolution", 5)  # Default to 5m
    resampling_method = config["spatial"].get("resampling_method", "bilinear")
    
    # Create target grid based on landuse (high resolution) data
    logger.info(f"Resampling data to target resolution of {target_resolution}m")
    
    # Resample temperature and precipitation to target grid
    temperature = resample_to_target_grid(
        temperature, landuse, resampling_method
    )
    precipitation = resample_to_target_grid(
        precipitation, landuse, resampling_method
    )
    
    # Calculate potential evapotranspiration using Thornthwaite method
    logger.info("Calculating potential evapotranspiration using Thornthwaite method")
    pet = calculate_thornthwaite_pet(temperature, config)
    
    # Adjust PET using land-use specific crop coefficients
    logger.info("Adjusting potential evapotranspiration using crop coefficients")
    et = adjust_pet_with_kc(pet, landuse, kc_df, config)
    
    # Calculate water balance (P - ET)
    logger.info("Calculating water balance")
    water_balance = calculate_p_minus_et(precipitation, et)

    # Compute chunks
    logger.info('Computing chunks...')
    water_balance = water_balance.compute()
    
    # Aggregate results based on output frequency
    output_frequency = config["temporal"].get("output_frequency", "monthly")
    if isinstance(output_frequency, str):
        output_frequency = [output_frequency]
    logger.info(f"Aggregating results to {output_frequency} frequency")
    
    # Save results
    output_paths = save_results(water_balance, output_frequency, config)
    
    return output_paths


def calculate_thornthwaite_pet(temperature: xr.DataArray, config: Dict[str, Any]) -> xr.DataArray:
    """Calculate potential evapotranspiration using the Thornthwaite method.
    
    The Thornthwaite method calculates PET based on mean monthly temperature and
    day length (which depends on latitude).
    
    Args:
        temperature: DataArray containing temperature data (°C)
        config: Configuration dictionary
        
    Returns:
        DataArray containing potential evapotranspiration (mm)
    """
    # Extract calculation parameters
    thornthwaite_config = config["calculation"]["thornthwaite"]
    apply_lat_correction = thornthwaite_config.get("latitude_correction", False)
    downscale_to_daily = thornthwaite_config.get("daily_downscaling", False)
    
    # Ensure temperature is in Celsius
    if config["input"]["temperature"]["units"].lower() != "celsius":
        raise ValueError("Temperature must be in Celsius for Thornthwaite method")
    
    # Step 1: Calculate monthly mean temperature if input is daily
    temperature_freq = xr.infer_freq(temperature.time)
    if temperature_freq in DATETIME_FREQUENCY_MAPPING['hourly']:
        temp_daily = temperature.resample(time="1D").mean()
        temp_monthly = temp_daily.resample(time="1ME").mean()

    elif temperature_freq in DATETIME_FREQUENCY_MAPPING['daily']:
        temp_monthly = temperature.resample(time="1ME").mean()

    elif temperature_freq in DATETIME_FREQUENCY_MAPPING['monthly']:
        temp_monthly = temperature
    else:
        raise ValueError(f"Unknown temporal frequency in temperature array. Got {temperature_freq}")
    
    # Step 2: Calculate heat index (I)
    # I is the sum of 12 monthly index values i, where i = (T/5)^1.514
    # Only include months where T > 0°C
    monthly_heat_index = xr.where(
        temp_monthly > 0,
        (temp_monthly / 5) ** 1.514,
        0
    )
    
    heat_index = monthly_heat_index.groupby('time.year').sum()
    
    # Step 3: Calculate unadjusted PET
    # PET (mm/month) = 16 * (10 * T / I)^a
    # where a = (6.75e-7 * I^3) - (7.71e-5 * I^2) + (1.792e-2 * I) + 0.49239
    
    # Calculate exponent 'a'
    a = (
        6.75e-7 * heat_index**3 -
        7.71e-5 * heat_index**2 +
        1.792e-2 * heat_index +
        0.49239
    )
    
    # Calculate unadjusted PET
    unadjusted_pet = xr.where(
        temp_monthly > 0,
        (16 * (10 * (temp_monthly.groupby('time.year') / heat_index))).groupby('time.year') ** a,
        0
    ).drop('year')
    unadjusted_pet = unadjusted_pet.rio.write_crs(temperature.rio.crs)
    
    # Step 4: Apply latitude correction if requested
    if apply_lat_correction:
        dates = xr.DataArray(
            unadjusted_pet.time.values, 
            dims=['time'], 
            coords={'time': unadjusted_pet.time.values}
        )
        lat = get_lat_in_4326(unadjusted_pet.isel(time = 0))

        day_length = day_lengths(dates, lat)
        day_length = day_length.assign_coords({'lat': unadjusted_pet.lat.values})
        pet = unadjusted_pet * day_length
    else:
        pet = unadjusted_pet
    
    # Convert monthly PET to daily if input was daily
    ##TODO: Fix this or keep monthly frequency
    if downscale_to_daily:
        raise NotImplementedError("Daily downscaling is not yet implemented.")
        # Distribute monthly PET evenly across days in each month
        
        # Create a daily PET dataset with the same time index as the original temperature
        pet_daily = xr.full_like(temperature, np.nan)
        
        # For each month, distribute the monthly PET evenly across days
        for month in range(1, 13):
            # Get the monthly PET for this month
            month_pet = pet.sel(time=pet.time.dt.month == month)
            
            # Get the days in this month from the original temperature dataset
            month_days = temperature.sel(time=temperature.time.dt.month == month)
            
            # Count the number of days in each month
            days_count = month_days.time.dt.daysinmonth
            
            # Distribute monthly PET evenly across days
            # First, calculate the daily value (PET / days in month)
            daily_value = month_pet / days_count.values[0]  # Use scalar value to avoid dimension mismatch
            
            # Then broadcast this value to all days in this month
            month_mask = temperature.time.dt.month == month
            pet_daily = xr.where(
                month_mask,
                daily_value.values,  # Use .values to avoid dimension alignment issues
                pet_daily
            )
        
        pet = pet_daily
    
    return pet


def adjust_pet_with_kc(pet: xr.DataArray, landuse: xr.DataArray, 
                      kc_df: pd.DataFrame, config: Dict[str, Any]) -> xr.DataArray:
    """Adjust potential evapotranspiration using land-use specific crop coefficients.
    
    Args:
        pet: DataArray containing potential evapotranspiration (mm)
        landuse: DataArray containing land-use data
        kc_df: DataFrame containing Kc coefficients for different land-use types and seasons
        config: Configuration dictionary
        
    Returns:
        DataArray containing adjusted evapotranspiration (mm)
    """
    # Get season definitions from config
    season_months = config["seasons"]
    
    # Create a dictionary mapping landuse codes to seasonal Kc values
    kc_mapping = {}
    for _, row in kc_df.iterrows():
        landuse_code = row["landuse_code"]
        kc_mapping[landuse_code] = {season: row[season] for season in list(season_months.keys())}
        
    # Create a time-varying Kc grid based on landuse and season
    # This is a computationally intensive step, so we'll use dask for parallelization
    
    # First, create a season DataArray with the same time dimension as pet
    if "time" in pet.dims:
        # Extract month information from pet time coordinate
        months = pet.time.dt.month.values
        
        # Map each month to its season
        month_to_season = {}
        for season, month_list in season_months.items():
            for month in month_list:
                month_to_season[int(month)] = season
        
        # Create a season DataArray
        season_data = np.array([month_to_season[m] for m in months])
        
        # Create a Kc grid for each time step
        # This is a memory-intensive operation, so we'll process in chunks
        
        # Initialize the adjusted ET DataArray with the same shape as pet
        et = xr.zeros_like(pet)
        
        # Process each time step
        for i, (time_val, season) in enumerate(zip(pet.time.values, season_data)):
            # Create a Kc grid for this time step based on landuse and season
            kc_grid = xr.zeros_like(landuse)
            
            # Assign Kc values based on landuse codes
            for landuse_code, kc_values in kc_mapping.items():
                kc_value = kc_values[season]
                kc_grid = xr.where(landuse == landuse_code, kc_value, kc_grid)
            
            # Apply Kc to PET for this time step
            et_time_slice = pet.isel(time=i) * kc_grid
            
            # Assign to the adjusted ET DataArray
            et = et.where(et.time != time_val, et_time_slice)
    else:
        # If pet has no time dimension, use annual average Kc values
        kc_grid = xr.zeros_like(landuse)
        
        # Calculate annual average Kc for each landuse type
        for landuse_code, kc_values in kc_mapping.items():
            annual_kc = sum(kc_values.values()) / len(kc_values)
            kc_grid = xr.where(landuse == landuse_code, annual_kc, kc_grid)
        
        # Apply Kc to PET
        et = pet * kc_grid
    
    return et


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
        precip_freq = xr.infer_freq(precipitation.time)
        et_freq = xr.infer_freq(et.time)
        
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
                wb_resampled = water_balance.resample(time="1Y").sum()
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