import xarray as xr
import numpy as np

from .base import BaseProcessor
from ..data_io import load_climate_data

class Precipitation(BaseProcessor):

    def load(self, var_name):
        """Load precipitation data from zarr dataset."""
        return load_climate_data(self.config, var_name)
