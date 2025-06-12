from abc import ABC, abstractmethod
import logging
from datetime import datetime

import xarray as xr
from pandas import to_datetime

from ..resampling import resample_to_target_grid

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):

    def __init__(self, config):
        self.config = config
        self.data = None
        self.var_name = None

    def resample_match(self, target: xr.DataArray, method: str = "bilinear") -> xr.DataArray:
        """Resample a source grid to match a target grid while preserving chunking.
        
        Args:
            source: DataArray to resample
            target: DataArray with the target grid
            method: Resampling method (nearest, bilinear, cubic)

        """

        if self.data is None:
            raise ValueError("No data loaded. Please load data before resampling.")
        source = self.data
        
        resampled = resample_to_target_grid(source, target, method)
        
        self.data = resampled

    def to_geotiff(self):

        if self.var_name is None:
            logger.warning("Variable name is not set. Please set var_name before calling to_geotiff.")
        else:
            out_pattern = self.config['output'].get('intermediate', {}).get(self.var_name, {}).get('pattern')
            if out_pattern is None:
                logger.warning(f"No configuration for var_name found. Got {self.var_name}")
                return

            if "time" not in self.data.dims:
                self.data.rio.to_raster(out_pattern.format(timestamp = '').replace('_.', '.'))
            else:
                start_date = self.config['output'].get('intermediate', {}).get(self.var_name, {}).get('start_date', datetime(1980, 1, 1))
                end_date = self.config['output'].get('intermediate', {}).get(self.var_name, {}).get('end_date', datetime(2030, 12, 31))
                data_to_write = self.data.sel(time=slice(start_date, end_date))
                data_to_write = data_to_write.rename({'lat': 'y', 'lon': 'x'})

                for timestamp, data in data_to_write.groupby("time"):
                    timestamp_str = to_datetime(timestamp).strftime('%Y%m%d_%H%M%S')
                    data.rio.to_raster(out_pattern.format(timestamp=timestamp_str))
                    logger.debug(f"Saved file {out_pattern.format(timestamp=timestamp_str)}")

    @abstractmethod
    def load():
        """Abstract method to load data, must be implemented by subclasses."""
        pass
    