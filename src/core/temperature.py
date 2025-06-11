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

        self.data = None
        self.corrected = False

    def load(self, var_name: str = 'temperature'):
        """Load temperature data from zarr dataset. """
        self.data = load_climate_data(self.config, var_name)

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

        rad_corr = resample_to_target_grid(
            source = rad_corr, 
            target = self.data, 
            method = self.config["spatial"].get("resampling_method", "bilinear")
            )

        rad_corr = rad_corr.chunk(self.config["processing"]["chunk_size"])

        return rad_corr

    def correct(self):
        if self.config['calculation'].get('temperature', {}).get('temperature_correction', False):
            logger.info('Correcting temperature data')

            radiation_data = self._load_radiation()
            tas_corr = ((np.abs(self.data) * 0.93).groupby('time.month') * radiation_data) + self.data

            self.data = tas_corr.compute()
            self.corrected = True

