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

    def initialize_correction_grids(self, watersheds, correction_factors):
        """
        Initialize correction grids based on watershed correction factors and distance raster.

        Parameters:
        -----------
        watersheds : Watersheds
            Watersheds object containing watershed masks
            correction_factors : pd.DataFrame, optional
            DataFrame containing correction factors for each watershed.
            If None, uses self.correction_factors
        method : str, optional
            Method to use for correction, by default 'chelsa'

        Returns:
        --------
        xr.DataArray
            Correction raster

        Raises:
        -------
        ValueError
            If the difference between the correction amount and the sum of the correction raster is too high
        """

        if correction_factors.index.names != ['time', 'Code']:
            raise ValueError(f"Index names of correction_factors must be ['time', 'Code']. Got {correction_factors.index.names}")

        # Initialize an empty DataArray with the same dimensions as the distance raster
        # to store the combined correction grid
        corr_raster = None
        for ts, w_id in correction_factors.index:

            ws = watersheds.get_mask(w_id)
            mask = ws != ws.attrs.get('_FillValue', -999)
            corr_amount = correction_factors.xs((ts, w_id))['preci_diff'].item()

            # dist_raster = resample_to_target_grid(self.distance_raster, ws)
            dist_mask = self.distance_raster.where(mask)

            dist_weights = dist_mask / dist_mask.sum()
            _corr_raster = dist_weights * corr_amount
            _corr_raster.name = 'Correction Grid'
            _corr_raster = _corr_raster.assign_coords(time = ts)

            # Initialize the template grid
            if corr_raster is None:
                corr_raster = xr.zeros_like(_corr_raster)
                corr_raster = corr_raster.expand_dims(time = np.unique(correction_factors.index.get_level_values('time')))
                corr_raster = corr_raster.rio.write_crs(_corr_raster.rio.crs)

            corr_raster.loc[dict(time = ts)] = xr.where(
                _corr_raster.isnull(), 
                corr_raster.sel(time = ts), 
                _corr_raster, 
                keep_attrs = True
            )

            logger.debug(f"Calculated correction grid for time {ts} and watershed {w_id}")

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
  
