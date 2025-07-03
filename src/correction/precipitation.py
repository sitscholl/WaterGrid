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

            if target is not None and target.chunks is not None:
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
            modeled_interstation_precipitation = modeled_interstation_precipitation.groupby(grouper).sum(min_count=1) #mm/year over entire watershed

            # Calculate modeled evapotranspiration for interstation regions
            modeled_interstation_evaporation = watersheds.aggregate(et)['modeled_values']
            modeled_interstation_evaporation = modeled_interstation_evaporation.groupby(grouper).sum(min_count=1) #mm/year over entire watershed

            #measured_interstation_discharge = get_measured_discharge_for_interstation_regions(validation_tbl)['measured_values'] #in m³/s
            measured_interstation_discharge = (validation_tbl['measured_values'] * (seconds * 1000)) / target_res**2 # Convert from m³/s to mm/year or mm/month over watershed.
            measured_interstation_discharge = measured_interstation_discharge.groupby(grouper).sum(min_count=1)

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

    def initialize_correction_grids(self, watersheds, correction_factors):
        """
        Vectorized version of initialize_correction_grids for maximum performance.

        This method uses xarray's advanced indexing and broadcasting to process
        all watersheds and time steps simultaneously, avoiding Python loops.

        CAREFUL: THIS METHOD CURRENTLY REQUIRES THE WATERSHEDS TO NOT OVERLAP EACH OTHER!

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
        watersheds_to_int = {code: idx for idx, code in enumerate(unique_codes)}

        logger.debug(f"Processing {len(unique_times)} time steps and {len(unique_codes)} watersheds to calculate correction grids")

        # Create a 3D array: (time, watersheds, spatial)
        # This allows us to process all combinations at once

        # Pre-load all watershed masks and stack them
        logger.debug("Creating watershed mask stack...")
        watersheds_masks = xr.concat([watersheds.get_mask(i) for i in watersheds.get_ids() if i in unique_codes], dim='id')
        watersheds_masks = watersheds_masks.assign_coords(id = [watersheds_to_int[i] for i in watersheds_masks['id'].values])
        watersheds_masks = watersheds_masks.idxmax('id')

        # Create correction factors array aligned with time and watershed dimensions
        logger.debug("Creating correction factors array...")
        correction_array = watersheds_masks.expand_dims(time = unique_times)

        # Map the precipitation correction amount to the watershed masks
        mapping_df = correction_factors.reset_index().pivot(index='Code', columns='time', values='preci_diff')
        mapping_df.index = mapping_df.index.map(watersheds_to_int)

        def map_single_time(block, mapping, max_id):
            # block: numpy or dask array for a single time slice
            # mapping: dict {int: float} for this time

            # Create a lookup array
            lookup = np.full(max_id, np.nan, dtype=float)
            for k, v in mapping.items():
                lookup[k] = v
            # Map values using lookup
            return lookup[block]

        def map_func(block, time_idx):
            # block: numpy array for this time
            # time_idx: the time value
            mapping = mapping_df[time_idx].squeeze().dropna().to_dict()
            return map_single_time(block, mapping, max_id = len(watersheds_to_int))
            logger.debug(f"Remapped correction factors for timestep {time_idx:%Y-%m-%d %H:%M:%S}")

        correction_per_watershed = correction_array.groupby('time').map(lambda x: map_func(x, x['time'].values))

        # Calculate distance weights for all watersheds at once
        logger.debug("Calculating distance weights...")
        distance_sums = self.distance_raster.groupby(watersheds_masks.compute()).sum().to_dataframe()
        watershed_distance = xr.full_like(watersheds_masks, fill_value = np.nan)
        for watershed_id, dist in zip(distance_sums.index, distance_sums['band_data']):
            watershed_distance = xr.where(watersheds_masks == watershed_id, dist, watershed_distance)

        # Avoid division by zero
        watershed_distance = watershed_distance.where(watershed_distance > 0, 1)
        distance_weights = self.distance_raster / watershed_distance

        # Apply corrections: broadcast correction factors across spatial dimensions
        logger.debug("Applying corrections...")
        corrections = distance_weights * correction_per_watershed

        # Set name and ensure proper coordinate order
        corrections.name = 'Correction Grid'
        corrections = corrections.rio.write_crs(self.distance_raster.rio.crs)

        logger.debug("Vectorized correction grid initialization completed")
        return corrections.transpose('time', 'lat', 'lon')
        
    def apply_correction(
            self,
            precipitation: xr.DataArray,
            correction_grid: Optional[xr.DataArray] = None,
            clip_precipitation: bool = False
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
        #correction_grid = correction_grid.rename({'lon': 'x', 'lat': 'y'}).rio.reproject_match(precipitation.rename({'lon': 'x', 'lat': 'y'}))
        #correction_grid = correction_grid.rename({'x': 'lon', 'y': 'lat'})
        
        # Apply the correction
        corrected_precipitation = precipitation + correction_grid
        
        # Ensure no negative precipitation
        if clip_precipitation:
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
  
