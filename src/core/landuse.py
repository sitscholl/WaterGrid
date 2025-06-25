import xarray as xr
import pandas as pd

import os
import logging

from .base import BaseProcessor
from ..data_io import load_static_data
from ..utils import align_chunks

logger = logging.getLogger(__name__)

class Landuse(BaseProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.coefficients = None
        self.kc_grid = None

    def load(self, var_name):
        """Load land-use data from GeoTIFF."""
        return load_static_data(self.config, var_name = 'landuse')

    def align_chunks(self, target: xr.DataArray | xr.Dataset):
        target_chunks = dict(zip(target.dims, target.chunks))
        self.data = align_chunks(self.data, target_chunks)

    def _load_kc_coefficients(self):
        """Load Kc coefficients from Excel file."""

        kc_config = self.config["input"]["kc_coefficients"]
        kc_path = kc_config["path"]
        
        if not os.path.exists(kc_path):
            raise FileNotFoundError(f"Kc coefficients file not found: {kc_path}")
        
        # Load Excel file
        sheet_name = kc_config.get("sheet_name", 0)
        kc_df = pd.read_excel(kc_path, sheet_name=sheet_name, na_values = [' ']).fillna(0)
        
        # Validate DataFrame structure
        required_columns = ["landuse_code", *list(self.config['seasons'].keys())]
        missing_columns = [col for col in required_columns if col not in kc_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in Kc coefficients file: {missing_columns}")
        
        logger.info(f"Loaded Kc coefficients from {kc_path}")
        logger.debug(f"Kc coefficients shape: {kc_df.shape}")
        
        self.coefficients =  kc_df

    def correct(self):
        """Assign Kc values based on landuse codes"""

        if self.data is None:
            raise ValueError("Data must be loaded before correction can be applied.") 

        if self.kc_grid is None:

            if self.coefficients is None:
                self._load_kc_coefficients()

            # Get season definitions from config
            season_months = self.config["seasons"]
            seasons = list(season_months.keys())
            
            # Create a dictionary mapping landuse codes to seasonal Kc values
            kc_mapping = {}
            for _, row in self.coefficients.iterrows():
                landuse_code = row["landuse_code"]
                kc_mapping[landuse_code] = {season: row[season] for season in list(season_months.keys())}
            
            kc_base = xr.zeros_like(self.data)
            kc_grids = []
            for season in seasons:
                season_grid = kc_base.copy()
                season_grid = season_grid.assign_coords(season = season)

                for landuse_code, kc_values in kc_mapping.items():
                    kc_value = kc_values[season]
                    season_grid = xr.where(self.data == landuse_code, kc_value, season_grid)
            
                kc_grids.append(season_grid)

            self.kc_grid = xr.concat(kc_grids, dim='season')
            
            logger.info(f'Initialized kc_grid with shape {self.kc_grid.shape}')

        else:
            logger.info("kc_grid has already been initalized. Skipping...")

