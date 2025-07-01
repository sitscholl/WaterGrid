import xarray as xr
import pandas as pd
import numpy as np

import logging
from typing import Union, Optional

from .utils import construct_interstation_watersheds
# from ..validation import Watersheds
from ..core import StationDistance, WindEffect
from ..resampling import resample_to_target_grid
from ..utils import align_chunks
from ..config import SECONDS_PER_YEAR, SECONDS_PER_MONTH

logger = logging.getLogger(__name__)

class PrCorrection:
    """
    Precipitation correction class.

    This class provides methods to correct precipitation data using various factors
    like station distance, wind effect, and validation data.
    """

    def __init__(self, config, target):
        """
        Initialize the PrCorrection class.

        Parameters:
        -----------
        config : dict
            Configuration dictionary
        station_distance_path : str, optional
            Path to the station distance raster
        wind_effect_path : str, optional
            Path to the wind effect raster

        Raises:
        -------
        ValueError
            If station_distance_path is not provided and not in config
        """
        self.config = config

        # Use provided wind_effect_path or get from config
        wind_effect_path = config['input'].get('wind_effect', {}).get('path')

        try:
            station_distance = StationDistance(config, var_name = 'station_distance', target = target).data

            if wind_effect_path is not None:
                wind_effect = WindEffect(config, var_name = 'wind_effect', target = target).data

                # Calculate distance raster
                distance_raster = station_distance * np.power(wind_effect, 4)
            else:
                distance_raster = station_distance

            if target is not None:
                distance_raster = align_chunks(distance_raster, dict(zip(target.dims, target.chunks)))

            self.distance_raster = distance_raster
            self.correction_factors = None
        except Exception as e:
            logger.error(f"Error initializing PrCorrection: {str(e)}")
            raise ValueError(f"Failed to initialize PrCorrection: {str(e)}")

    def add_glaciers(self, glacier_data: xr.DataArray) -> None:
        """
        Add glacier data to the correction model.

        Parameters:
        -----------
        glacier_data : xr.DataArray
            Glacier data to add to the model

        Note:
        -----
        This method is currently a placeholder for future implementation.
        """
        # TODO: Implement glacier correction logic
        logger.warning("add_glaciers method is not yet implemented")
        pass

    def calculate_correction_factors(
            self,
            watersheds,
            precipitation: Union[xr.DataArray, xr.Dataset],
            et: Union[xr.DataArray, xr.Dataset],
            validation_tbl: pd.DataFrame,
            freq: Optional[str] = None
        ) -> pd.DataFrame:
        """
        Calculate correction factors for precipitation based on water balance validation.

        Parameters:
        -----------
        watersheds : Watersheds
            Watersheds object containing watershed masks
        precipitation : xr.DataArray or xr.DataSet
            Precipitation data
        et : xr.DataArray or xr.DataSet
            Evapotranspiration data
        validation_tbl : pd.DataFrame
            Validation table containing measured discharge values
        freq : str, optional
            Frequency for aggregation, by default uses the default_freq from config

        Returns:
        --------
        pd.DataFrame
            DataFrame containing correction factors (preci_factor and preci_diff)
        """

        if freq not in ['YE-SEP', 'ME']:
            raise ValueError(f"Frequency {freq} is not supported. Please use 'YE-SEP' or 'ME'.")
        seconds = SECONDS_PER_YEAR if freq == 'YE-SEP' else SECONDS_PER_MONTH

        try:
            
            grouper = [pd.Grouper(freq=freq, level='time'), pd.Grouper(level = 'Code')]
            target_res = precipitation.rio.resolution()[0]

            # Calculate modeled precipitation for interstation regions
            modeled_interstation_precipitation = watersheds.aggregate(precipitation)['modeled_values']
            modeled_interstation_precipitation = modeled_interstation_precipitation.groupby(grouper).sum() #mm/year over entire watershed

            # Calculate modeled evapotranspiration for interstation regions
            modeled_interstation_evaporation = watersheds.aggregate(et)['modeled_values']
            modeled_interstation_evaporation = modeled_interstation_evaporation.groupby(grouper).sum() #mm/year over entire watershed

            #measured_interstation_discharge = get_measured_discharge_for_interstation_regions(validation_tbl)['measured_values'] #in m³/s
            measured_interstation_discharge = (validation_tbl['measured_values'] * (seconds * 1000)) / target_res**2 # Convert from m³/s to mm/year or mm/month over watershed.
            measured_interstation_discharge = measured_interstation_discharge.groupby(grouper).sum()

            # Calculate expected precipitation based on water balance equation
            expected_interstation_precipitation = (
                measured_interstation_discharge + modeled_interstation_evaporation
                )
            expected_interstation_precipitation.dropna(inplace=True)

            preci_factor = modeled_interstation_precipitation / expected_interstation_precipitation
            preci_diff = expected_interstation_precipitation - modeled_interstation_precipitation #* (1000*365*24*60*60)) / target_res**2  # mm/year

            correction_factors = pd.DataFrame({
                'preci_factor': preci_factor,
                'preci_diff': preci_diff
            })
            
            return correction_factors
        except Exception as e:
            logger.error(f"Error calculating correction factors: {str(e)}")
            raise ValueError(f"Error calculating correction factors: {str(e)}")

    def initialize_correction_grids_vectorized(self, watersheds, correction_factors):
        """
        Vectorized version of initialize_correction_grids for maximum performance.

        This method uses xarray's advanced indexing and broadcasting to process
        all watersheds and time steps simultaneously, avoiding Python loops.

        Parameters:
        -----------
        watersheds : Watersheds
            Watersheds object containing watershed masks
        correction_factors : pd.DataFrame
            DataFrame containing correction factors for each watershed.

        Returns:
        --------
        xr.DataArray
            Correction raster
        """

        if correction_factors.index.names != ['time', 'Code']:
            raise ValueError(f"Index names of correction_factors must be ['time', 'Code']. Got {correction_factors.index.names}")

        # Get unique time steps and watershed codes
        unique_times = correction_factors.index.get_level_values('time').unique()
        unique_codes = correction_factors.index.get_level_values('Code').unique()
        watershed_integers = np.arange(unique_codes)

        logger.debug(f"Processing {len(unique_times)} time steps and {len(unique_codes)} watersheds")

        # Create a 3D array: (time, watersheds, spatial)
        # This allows us to process all combinations at once

        # Pre-load all watershed masks and stack them
        logger.debug("Creating watershed mask stack...")
        watersheds_masks = xr.concat([watersheds.get_mask(i) for i in watersheds.get_ids() if i in unique_codes], dim='watershed')
        watersheds_masks.assign_coords(watershed = watersheds.get_ids()).argmax()
        watershed_masks_list = []
        valid_codes = []

        for w_int, w_id in zip(watershed_integers, unique_codes):
            try:
                ws = watersheds.get_mask(w_id)
                mask = ws != ws.attrs.get('_FillValue', -999)
                watershed_masks_list.append(mask)
                valid_codes.append(w_id)
            except Exception as e:
                logger.warning(f"Could not load watershed {w_id}: {str(e)}")
                continue

        if not watershed_masks_list:
            raise ValueError("No valid watersheds found")

        # Stack all watershed masks along a new dimension
        watershed_stack = xr.concat(watershed_masks_list, dim='watershed')
        watershed_stack = watershed_stack.assign_coords(watershed=valid_codes)

        # Create correction factors array aligned with time and watershed dimensions
        logger.debug("Creating correction factors array...")
        correction_array = xr.DataArray(
            np.full((len(unique_times), len(valid_codes)), np.nan),
            dims=['time', 'watershed'],
            coords={'time': unique_times, 'watershed': valid_codes}
        )

        # Fill correction factors array
        for (ts, w_id), row in correction_factors.iterrows():
            if w_id in valid_codes:
                correction_array.loc[dict(time=ts, watershed=w_id)] = row['preci_diff']

        # Calculate distance weights for all watersheds at once
        logger.debug("Calculating distance weights...")
        distance_masked = self.distance_raster.where(watershed_stack)
        distance_sums = distance_masked.sum(dim=['lat', 'lon'])

        # Avoid division by zero
        distance_sums = distance_sums.where(distance_sums > 0, 1)
        distance_weights = distance_masked / distance_sums

        # Apply corrections: broadcast correction factors across spatial dimensions
        logger.debug("Applying corrections...")
        corrections = distance_weights * correction_array

        # Sum across watersheds to get final correction grid
        # Use nansum to handle overlapping watersheds properly
        final_corrections = corrections.sum(dim='watershed', skipna=True)

        # Set name and ensure proper coordinate order
        final_corrections.name = 'Correction Grid'
        final_corrections = final_corrections.rio.write_crs(self.distance_raster.rio.crs)

        logger.debug("Vectorized correction grid initialization completed")
        return final_corrections.transpose('time', 'lat', 'lon')

    def initialize_correction_grids(self, watersheds, correction_factors):
        """
        Initialize correction grids based on watershed correction factors and distance raster.

        This method automatically chooses between the standard and vectorized implementation
        based on the data size for optimal performance.

        Parameters:
        -----------
        watersheds : Watersheds
            Watersheds object containing watershed masks
        correction_factors : pd.DataFrame
            DataFrame containing correction factors for each watershed.

        Returns:
        --------
        xr.DataArray
            Correction raster
        """

        # Use vectorized version for better performance
        # Fall back to iterative version if vectorized fails (e.g., memory issues)
        try:
            return self.initialize_correction_grids_vectorized(watersheds, correction_factors)
        except Exception as e:
            logger.warning(f"Vectorized method failed ({str(e)}), falling back to iterative method")
            return self.initialize_correction_grids_iterative(watersheds, correction_factors)

    def initialize_correction_grids_iterative(self, watersheds, correction_factors):
        """
        Iterative version of initialize_correction_grids (original implementation with optimizations).

        This is the fallback method when the vectorized version fails due to memory constraints.
        """

        if correction_factors.index.names != ['time', 'Code']:
            raise ValueError(f"Index names of correction_factors must be ['time', 'Code']. Got {correction_factors.index.names}")

        # Get unique time steps and watershed codes
        unique_times = correction_factors.index.get_level_values('time').unique()
        unique_codes = correction_factors.index.get_level_values('Code').unique()

        # Pre-load all watershed masks to avoid repeated I/O
        logger.debug("Pre-loading watershed masks...")
        watershed_masks = {}
        for w_id in unique_codes:
            try:
                ws = watersheds.get_mask(w_id)
                mask = ws != ws.attrs.get('_FillValue', -999)
                watershed_masks[w_id] = mask
            except Exception as e:
                logger.warning(f"Could not load watershed {w_id}: {str(e)}")
                continue

        # Initialize the output array with proper dimensions
        corr_raster = xr.zeros_like(self.distance_raster).expand_dims(time=unique_times)
        corr_raster = corr_raster.rio.write_crs(self.distance_raster.rio.crs)
        corr_raster.name = 'Correction Grid'

        # Group correction factors by time for efficient processing
        correction_by_time = correction_factors.groupby(level='time')

        logger.debug("Processing correction grids by time step...")
        for ts, time_group in correction_by_time:
            logger.debug(f"Processing time step: {ts}")

            # Initialize accumulator for this time step
            time_correction = xr.zeros_like(self.distance_raster)

            # Process all watersheds for this time step
            for w_id in time_group.index.get_level_values('Code'):
                if w_id not in watershed_masks:
                    continue

                try:
                    corr_amount = time_group.xs((ts, w_id))['preci_diff'].item()
                    mask = watershed_masks[w_id]

                    # Calculate distance weights for this watershed
                    dist_mask = self.distance_raster.where(mask)
                    dist_sum = dist_mask.sum()

                    # Skip if no valid distances (avoid division by zero)
                    if dist_sum == 0:
                        logger.warning(f"No valid distances for watershed {w_id} at time {ts}")
                        continue

                    # Calculate weighted correction for this watershed
                    dist_weights = dist_mask / dist_sum
                    watershed_correction = dist_weights * corr_amount

                    # Add to time step accumulator (only where mask is valid)
                    time_correction = xr.where(
                        mask,
                        watershed_correction.fillna(0),
                        time_correction
                    )

                except Exception as e:
                    logger.warning(f"Error processing watershed {w_id} at time {ts}: {str(e)}")
                    continue

            # Assign the complete time step to the output array
            corr_raster.loc[dict(time=ts)] = time_correction

        logger.debug("Iterative correction grid initialization completed")
        return corr_raster.transpose('time', 'lat', 'lon')
        
    def apply_correction(
            self,
            precipitation: xr.DataArray,
            correction_grid: Optional[xr.DataArray] = None,
        ) -> xr.DataArray:
        """
        Apply correction to precipitation data.

        Parameters:
        -----------
        precipitation : xr.DataArray
            Precipitation data to correct
        correction_grid : xr.DataArray, optional
            Correction grid to apply. If None, uses the result of initialize_correction_grids
        freq : str, optional
            Frequency for resampling precipitation data, by default uses the default_freq from config

        Returns:
        --------
        xr.DataArray
            Corrected precipitation data
        """   

        if not np.array_equal(precipitation.time, correction_grid.time):
            raise ValueError("Got mismatched time dimensions between precipitation and correction grid. Cannot apply correction.")

        # Ensure the correction grid matches the precipitation grid
        correction_grid = correction_grid.rename({'lon': 'x', 'lat': 'y'}).rio.reproject_match(precipitation.rename({'lon': 'x', 'lat': 'y'}))
        correction_grid = correction_grid.rename({'x': 'lon', 'y': 'lat'})
        
        # Apply the correction
        corrected_precipitation = precipitation + correction_grid
        
        # Ensure no negative precipitation
        corrected_precipitation = corrected_precipitation.clip(min=0)
        
        return corrected_precipitation

if __name__ == "__main__":
    import yaml
    from src.core import Precipitation, Temperature
    from src.validation import Validator
    from src.validation.watersheds import Watersheds
    from src.calc.evapotranspiration import calculate_thornthwaite_pet
    from src.calculations import calculate_p_minus_et

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    with open("config.yaml", "r", encoding = 'utf-8') as f:
        config = yaml.safe_load(f)

    precipitation = Precipitation(config)
    temperature = Temperature(config)
    pet = calculate_thornthwaite_pet(temperature.data, config)
    water_balance = calculate_p_minus_et(precipitation.data, pet)

    watersheds = Watersheds(config)
    watersheds.load(precipitation.data.isel(time = 0))
    interstation_regions = Watersheds(config, data=construct_interstation_watersheds(watersheds))

    validator = Validator(config)
    #Use precipitation here, because we want to compare summed modeled precipitation with expected precipitation from discharge stations
    validation_tbl = validator.validate(interstation_regions, precipitation.data, compute_for_interstation_regions=True)
    #validator.plot_timeseries(validation_tbl)

    pr_correction = PrCorrection(config)
    correction_factors = pr_correction.calculate_correction_factors(
        interstation_regions, precipitation.data, pet, validation_tbl
        )
    corr_raster = pr_correction.initialize_correction_grids(interstation_regions, correction_factors)
    pr_corr = pr_correction.apply_correction(precipitation.data, corr_raster)

    ##TODO: Change fixed frequency here and allow dynamic frequency
    pet_yearly = pet.resample(time='YE-SEP').sum()
    wb_corr = calculate_p_minus_et(pr_corr, pet_yearly)

    validation_tbl_after_correction = validator.validate(watersheds, wb_corr)
    validator.plot_timeseries(validation_tbl_after_correction)
  
