import numpy as np
import xarray as xr
import pandas as pd

from typing import Dict, Any
import logging

from ..config import DATETIME_FREQUENCY_MAPPING
from .day_length import day_lengths, get_lat_in_4326
from ..utils import get_season

logger = logging.getLogger(__name__)

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


def adjust_pet_with_kc(pet: xr.DataArray, 
                       kc_grid: xr.DataArray, 
                       config: Dict[str, Any]) -> xr.DataArray:
    """Adjust potential evapotranspiration using land-use specific crop coefficients.
    
    Args:
        pet: DataArray containing potential evapotranspiration (mm)
        landuse: DataArray containing land-use data
        kc_df: DataFrame containing Kc coefficients for different land-use types and seasons
        config: Configuration dictionary
        
    Returns:
        DataArray containing adjusted evapotranspiration (mm)
    """

    ##Assign seasons to pet
    season_months = config["seasons"]
    time_coords = [pd.to_datetime(i) for i in pet['time'].values]
    seasons = [get_season(i, season_months) for i in time_coords]

   # Apply the appropriate seasonal Kc value to each timestep
    et = xr.zeros_like(pet)
    for i, (time_val, season) in enumerate(zip(pet.time.values, seasons)):
        # Select the correct seasonal Kc grid
        seasonal_kc = kc_grid.sel(season=season)
        
        # Apply to the corresponding time slice of pet
        time_slice = pet.sel(time=time_val)
        et[i, :, :] = time_slice * seasonal_kc

        logger.debug('Adjusted PET for time step %s and season %s', time_val, season)
    
    return et
