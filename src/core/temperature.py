import xarray as xr
import numpy as np

from .base import BaseProcessor
from ..data_io import load_climate_data

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

        rad_corr = xr.open_dataset(self.config['calculation']['temperature']["radiation_files"])
        rad_corr = rad_corr[list(rad_corr.keys())[0]]
        rad_corr = rad_corr - 1

        rad_corr = self.resample_to_target_grid(
            source = rad_corr, 
            target = self.data, 
            resampling_method = self.config["spatial"].get("resampling_method", "bilinear")
            )
        
        return rad_corr

    def correct(self):
        if self.config['calculation'].get('temperature', {}).get('temperature_correction', False):
            radiation_data = self._load_radiation()
            tas_corr = ((np.abs(self.data) * 0.93).groupby('time.month') * radiation_data) + self.data

            self.data = tas_corr
            self.corrected = True

