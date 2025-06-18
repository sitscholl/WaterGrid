import xarray as xr
import numpy as np

from .base import BaseProcessor
from ..data_io import load_climate_data

class Precipitation(BaseProcessor):

    def __init__(self, config):
        super().__init__(config)
        self.corrected = False
        self.load(var_name = config['input']['precipitation']['variable'])

    def load(self, var_name: str = 'precipitation'):
        """Load precipitation data from zarr dataset."""
        self.data = load_climate_data(self.config, var_name)
        self.var_name = var_name

    def correct(self):
        if self.config['calculation'].get('precipitation', {}).get('precipitation_correction', False):
            pass
