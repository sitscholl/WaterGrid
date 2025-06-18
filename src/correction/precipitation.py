import xarray as xr
import pandas as pd
import numpy as np

import logging

from src.config import DATETIME_FREQUENCY_MAPPING
from src.correction.utils import construct_interstation_watersheds, get_measured_discharge_for_interstation_regions
from src.validation import Watersheds
from src.data_io import load_static_data

logger = logging.getLogger(__name__)

class PrCorrection:
    """
    Precipitation correction class.
    
    This class provides methods to correct precipitation data using various factors
    like station distance, wind effect, and validation data.
    """

    def __init__(self, config, station_distance_path=None, wind_effect_path=None):
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
        wind_effect_path = config['input'].get('wind_effect', {}).get('path')

        station_distance = load_static_data(config, 'station_distance')

        if wind_effect_path is not None:
            wind_effect = load_static_data(config, 'wind_effect')
            wind_effect = wind_effect.rio.reproject_match(station_distance).rename({'x': 'lon', 'y': 'lat'})

            # Calculate distance raster
            distance_raster = station_distance * np.power(wind_effect, 4)
        else:
            distance_raster = station_distance

        # TODO: Align distance raster to landuse
        self.distance_raster = distance_raster
        self.correction_factors = None

    def add_glaciers(self, glacier_data):
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
        pass

    def calculate_correction_factors(
            self,
            watersheds: Watersheds,
            precipitation: xr.DataArray | xr.Dataset,
            et: xr.DataArray | xr.Dataset,
            validation_tbl: pd.DataFrame,
            freq: str = 'YE-SEP'
        ):
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
            Frequency for aggregation, by default 'ME' (month end)
            
        Returns:
        --------
        tuple
            (precipitation_factor, precipitation_difference)
        """
        #TODO: At the moment, correction factors are calculated fixed for hydrological year. Make this dynamic by allowing different frequencies
        try:
            interstation_regions = Watersheds(self.config, data=construct_interstation_watersheds(watersheds))

            ##TODO: Include also Code into grouping? 
            ##TODO: Fix transformation from m³/s to mm/year
            ##TODO: Check get_area method if it gets correct resolution
            modeled_interstation_precipitation = interstation_regions.aggregate(precipitation)['modeled_values']
            modeled_interstation_precipitation = modeled_interstation_precipitation.groupby(pd.Grouper(freq='YE-SEP', level='time')).sum() #mm/year

            modeled_interstation_evaporation = interstation_regions.aggregate(et)['modeled_values']
            modeled_interstation_evaporation = modeled_interstation_evaporation.groupby(pd.Grouper(freq='YE-SEP', level='time')).sum()
           
            measured_interstation_discharge = get_measured_discharge_for_interstation_regions(validation_tbl)['measured_values'] #in m³/s
            measured_interstation_discharge *= (365*24*60*60 * 1000) / interstation_regions.get_area()  # Convert from m³/s to mm/year
            measured_interstation_discharge = measured_interstation_discharge.groupby(pd.Grouper(freq='YE-SEP', level='time')).sum()

            expected_interstation_precipitation = (
                measured_interstation_discharge + modeled_interstation_evaporation
                )
            expected_interstation_precipitation.dropna(inplace = True)

            preci_factor = modeled_interstation_precipitation / expected_interstation_precipitation
            preci_diff = expected_interstation_precipitation - modeled_interstation_precipitation #* (1000*365*24*60*60)) / 625  # mm/year

            self.correction_factors = pd.DataFrame({
                'preci_factor': preci_factor,
                'preci_diff': preci_diff
            })
            
            return preci_factor, preci_diff
        except Exception as e:
            raise ValueError(f"Error calculating correction factors: {str(e)}")

    def initialize_correction_grids(self, watersheds, correction_factors=None, method='chelsa'):
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
            If correction_factors is None and self.correction_factors is None
            If the difference between the correction amount and the sum of the correction raster is too high
        """
        if correction_factors is None:
            if self.correction_factors is None:
                raise ValueError("No correction factors available. Run calculate_correction_factors first.")
            correction_factors = self.correction_factors
        
        corr_raster = []
        for w_id in correction_factors.index:
            
            mask = np.isfinite(watersheds.get_mask(w_id))
            corr_amount = correction_factors.loc[int(w_id), 'preci_diff']
            
            dist_mask = self.distance_raster.where(mask)

            if (method == 'chelsa') and (corr_amount < 0):
                dist_mask = dist_mask.max() - dist_mask
                
            dist_weights = dist_mask / dist_mask.sum()
            corr_raster1 = dist_weights * corr_amount
            
            perc_diff = ((corr_amount - corr_raster1.sum().values) / corr_amount) * 100 if corr_amount != 0 else 0
            
            if abs(perc_diff) > 0.001:
                raise ValueError(f'Difference too high! {perc_diff}%')

            corr_raster.append(corr_raster1)
            
        corr_raster = xr.merge(corr_raster).band_data.fillna(0)
        corr_raster = corr_raster.rio.write_nodata(0)
        
        return corr_raster
        
    def apply_correction(self, precipitation, correction_grid=None):
        """
        Apply correction to precipitation data.
        
        Parameters:
        -----------
        precipitation : xr.DataArray
            Precipitation data to correct
        correction_grid : xr.DataArray, optional
            Correction grid to apply. If None, uses the result of initialize_correction_grids
            
        Returns:
        --------
        xr.DataArray
            Corrected precipitation data
        """
        if correction_grid is None:
            raise ValueError("Correction grid must be provided")
            
        # Ensure the correction grid matches the precipitation grid
        correction_grid = correction_grid.rio.reproject_match(precipitation)
        
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

    validator = Validator(config)
    validation_tbl = validator.validate(watersheds, precipitation.data)
    
    pr_correction = PrCorrection(config)
    correction_factors = pr_correction.calculate_correction_factors(
        watersheds, precipitation.data, pet, validation_tbl
        )
