import xarray as xr
import numpy as np

import logging
from pathlib import Path

from src.core.base import BaseProcessor
from src.resampling import resample_to_target_grid
from src.data_io import apply_spatial_filter

logger = logging.getLogger(__name__)


class Watersheds(BaseProcessor):

    def __init__(self, config):
        super().__init__(config)

        self.var_name = 'watersheds'
        self.data = {}
        self.aggregated_data = {}

        watersheds_config = config['input'].get('watersheds')
        if watersheds_config is None:
            logger.warning("Watersheds configuration is missing in the input configuration. No validation will be carried out.")
            self.validate = False
        else:
            self.validate = True

    def load(self, target: xr.DataArray | xr.Dataset):
        if not self.validate:
            logger.warning("No watersheds configuration to load.")
            return

        ws_config = self.config['input']['watersheds']
        ws_root = Path(ws_config.get('root', '.'))
        ws_files = list(ws_root.glob(ws_config.get('pattern', '*.tif')))[:3]
        fill_value = ws_config.get('fill_value', -999)

        logger.debug(f"Found {len(ws_files)} watershed files")

        for ws_file in ws_files:
            ws = xr.open_dataset(ws_file)
            ws = ws[list(ws.keys())[0]]
            ws = ws.squeeze(drop = True)

            if 'x' in ws.dims or 'y' in ws.dims:
                ws = ws.rename({'x': 'lon', 'y': 'lat'})

            # Apply spatial filter
            ws = apply_spatial_filter(ws, self.config)

            if any([i == 0 for j, i in ws.sizes.items()]):
                logger.warning(f"Watershed {ws_file.name} contains zero-sized dimensions after spatial filtering.")
                continue
            
            # Resample to target grid
            ws_re = resample_to_target_grid(
                ws, 
                target, 
                method = 'nearest'
            )

            # Apply fill value to nodata areas (assuming 0 or NaN are nodata)
            # Watersheds should have value 1 for valid areas
            ws_re = ws_re.where(ws_re > 0, fill_value)
            
            # Assign fill value attribute for metadata
            ws_re = ws_re.assign_attrs(_FillValue=fill_value)
            
            # Ensure data is of integer type
            if not np.issubdtype(ws.dtype, np.integer):
                logger.debug(f"File {ws_file} has non-integer data type. Converting to int")
                ws_re = ws_re.astype(int)
            
            self.data[ws_file.stem] = ws_re

    def aggregate(self, data: xr.DataArray, method: str = 'sum', dim = ['lon', 'lat']) -> xr.DataArray:

        if len(self.data) == 0:
            raise ValueError("No watersheds available for aggregation. Load watersheds first.")
        
        for ws_name, ws_data in self.data.items():
            # Create a proper boolean mask where valid watershed values exist
            fill_value = ws_data.attrs.get('_FillValue', None)
            # Create a boolean mask (True where watershed data is valid, False elsewhere)
            ws_mask = ws_data != fill_value
            
            # Apply the mask to the data
            masked_data = data.where(ws_mask)
            
            if method == 'sum':
                self.aggregated_data[ws_name] = masked_data.sum(dim=dim)
            elif method == 'mean':
                self.aggregated_data[ws_name] = masked_data.mean(dim=dim)
            else:
                raise ValueError(f"Unsupported aggregation method: {method}")
                
if __name__ == "__main__":
    import yaml
    from src.core import Landuse, Precipitation, Temperature

    with open("config.yaml", "r", encoding = 'utf-8') as f:
        config = yaml.safe_load(f)

    landuse = Landuse(config)
    landuse.load()

    watersheds = Watersheds(config)
    watersheds.load(landuse.data)
    watersheds.aggregate(landuse.data)