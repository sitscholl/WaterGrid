import xarray as xr
import numpy as np

import logging

from .base import BaseProcessor
from ..data_io import load_climate_data, apply_spatial_filter
from ..resampling import resample_to_target_grid

logger = logging.getLogger(__name__)

class Temperature(BaseProcessor):

    def __init__(self, config):
        super().__init__(config)
        self.corrected = False
        self.load(var_name = self.load(var_name = config['input']['temperature']['variable']))

    def load(self, var_name: str = 'temperature'):
        """Load temperature data from zarr dataset. """
        self.data = load_climate_data(self.config, var_name)
        self.var_name = var_name

    def _load_radiation(self):

        if self.data is None:
            raise ValueError("Temperature data must be loaded before loading radiation data.")

        rad_corr = xr.open_dataset(
            self.config['calculation']['temperature']["radiation_files"], 
            decode_coords='all')

        rad_corr = rad_corr[list(rad_corr.keys())[0]]
        rad_corr = rad_corr - 1

        if 'x' in rad_corr.dims or 'y' in rad_corr.dims:
            rad_corr = rad_corr.rename({'x': 'lon', 'y': 'lat'})

        rad_corr = apply_spatial_filter(rad_corr, self.config)
                
        return rad_corr

    def correct(self):
        if self.config['calculation'].get('temperature', {}).get('temperature_correction', False):
            logger.info('Correcting temperature data')

            radiation_data = self._load_radiation()
                        
            # Resample temperature data to radiation grid
            tair_resampled = resample_to_target_grid(
                source = self.data,
                target = radiation_data,
                method = self.config["spatial"].get("resampling_method", "bilinear")
            )
            
            # Apply consistent chunking to both datasets
            # chunk_size = self.config["processing"]["chunk_size"]
            # tair_resampled = tair_resampled.chunk(chunk_size)
            # radiation_data = radiation_data.chunk(chunk_size)
            
            # 1. Calculate absolute value and scaling
            abs_scaled = np.abs(tair_resampled) * 0.93
            
            # 2. Group by month
            grouped = abs_scaled.groupby('time.month')
            
            # 3. Multiply with radiation data
            correction = grouped * radiation_data
            
            # 4. Add original temperature data
            tas_corr = correction + tair_resampled
            
            # 5. Compute the result with optimized parallelism
            logger.info('Computing corrected temperature data')
            self.data = tas_corr.compute()
            self.corrected = True
            logger.info('Temperature correction completed')

