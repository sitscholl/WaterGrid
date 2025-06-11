from abc import ABC, abstractmethod
import logging

import xarray as xr

from ..resampling import resample_to_target_grid

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):

    def __init__(self, config):
        self.config = config
        self.data = None

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

    @abstractmethod
    def load():
        """Abstract method to load data, must be implemented by subclasses."""
        pass

    @abstractmethod
    def correct():
        """Abstract method to correct data, must be implemented by subclasses."""
        pass
