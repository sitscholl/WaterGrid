from abc import ABC, abstractmethod
import logging

from pandas.errors import AbstractMethodError
import xarray as xr
import rioxarray
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):

    def __init__(self, config):
        self.config = config

    def resample_to_target_grid(self, source: xr.DataArray, target: xr.DataArray, method: str = "bilinear") -> xr.DataArray:
        """Resample a source grid to match a target grid while preserving chunking.
        
        Args:
            source: DataArray to resample
            target: DataArray with the target grid
            method: Resampling method (nearest, bilinear, cubic)
            
        Returns:
            Resampled DataArray with exact same x and y coordinates as target grid
        """
        # Check if source has rio accessor
        if not hasattr(source, "rio"):
            # Convert to rioxarray
            source = rioxarray.open_rasterio(source)
        
        # Get target bounds and CRS
        target_crs = target.rio.crs
        
        # Map interp methods
        resampling_methods_rasterio = {        
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
        }

        resampling_methods = {
            "nearest": "nearest",
            "bilinear": "linear",
            "cubic": "cubic",
        }    

        if method not in resampling_methods:
            raise ValueError(f"Unsupported resampling method: {method}. "
                        f"Supported methods: {list(resampling_methods.keys())}")
        
        # First ensure the source is in the same CRS as the target
        if source.rio.crs != target_crs:
            # This step still loads into memory, but we need it for CRS transformation
            logger.warning(f"Source CRS ({source.rio.crs}) does not match target CRS ({target_crs}). Reprojecting source to match target CRS (requires array to be loaded into memory).")
            source = source.rio.reproject(
                target_crs,
                resampling=resampling_methods_rasterio[method]
            )
                    
        # Use xarray's interp which preserves chunking
        resampled = source.interp_like(target, method = resampling_methods[method])
        
        return resampled

    @abstractmethod
    def load():
        """Abstract method to load data, must be implemented by subclasses."""
        pass

    @abstractmethod
    def correct():
        """Abstract method to correct data, must be implemented by subclasses."""
        pass
