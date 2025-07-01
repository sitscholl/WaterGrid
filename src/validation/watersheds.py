import xarray as xr
import numpy as np
import pandas as pd
from flox.xarray import xarray_reduce

import logging
from pathlib import Path

from ..core import WatershedLoader
from ..utils import align_chunks

logger = logging.getLogger(__name__)

class Watersheds:

    def __init__(self, config, var_name: str, data: dict | None = None, **kwargs):

        self.config = config
        self.var_name = var_name

        if data is None:
            self.data = self.load(var_name=self.var_name, **kwargs)
        else:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary")
            self.data = data

    def load(self, var_name: str, target: xr.DataArray | xr.Dataset | None = None, method: str = 'nearest'):

        ws_config = self.config['input'].get(var_name)

        if ws_config is None:
            raise ValueError(f"No configuration found for variable name: {var_name}")

        ws_root = Path(ws_config.get('root', '.'))
        ws_files = list(ws_root.glob(ws_config.get('pattern', '*.tif')))
        fill_value = ws_config.get('fill_value', -999)

        logger.debug(f"Found {len(ws_files)} watershed files")

        watersheds = {}
        for ws_file in ws_files:
            # WatershedLoader expects a dictionary with the configuration  
            ws_file_config = {'input': {var_name: {'path': ws_file}},
                              'spatial': self.config['spatial']}   
            ws = WatershedLoader(ws_file_config, var_name = var_name, target = target, method = method).data

            if any([i == 0 for j, i in ws.sizes.items()]):
                continue

            # Apply fill value to nodata areas (assuming 0 or NaN are nodata)
            # Watersheds should have value 1 for valid areas
            ws = ws.where(ws > 0, fill_value)
            
            # Assign fill value attribute for metadata
            ws = ws.assign_attrs(_FillValue=fill_value)

            # Assign coordinate
            ws = ws.assign_coords({'id': ws_file.stem})
            
            # Ensure data is of integer type
            if not np.issubdtype(ws.dtype, np.integer):
                logger.debug(f"File {ws_file} has non-integer data type. Converting to int")
                ws = ws.astype(int)
            
            watersheds[ws_file.stem] = ws
        
        return watersheds

    def align_chunks(self, target: xr.DataArray | xr.Dataset):
        target_chunks = dict(zip(target.dims, target.chunks))

        for i, data in self.data.items():
            self.data[i] = align_chunks(data, target_chunks)

    def get_ids(self) -> list | None:
        return list(self.data.keys())

    def get_mask(self, id: str) -> xr.DataArray | None:
        return self.data.get(id)

    def get_area(self) -> pd.Series:

        ws_areas = []
        for ws_id in self.get_ids():
            ws = self.get_mask(ws_id)
            ws_areas.append(ws.rio.resolution()[0]**2 * (ws == 1).sum().item())

        return pd.Series(ws_areas, index=pd.Index(self.get_ids(), name = 'Code'))

    def aggregate(self, data: xr.DataArray, method: str = 'sum', dim = ['lon', 'lat']) -> pd.DataFrame:
        """
        Efficiently aggregate water balance over overlapping watersheds using flox.
        
        Parameters:
        -----------
        water_balance : xarray.DataArray
            Water balance data with dimensions (time, lat, lon)
        watershed_masks : list of xarray.DataArray
            List of watershed masks where 1=inside watershed, -999=outside
        watershed_names : list of str, optional
            Names for each watershed
        
        Returns:
        --------
        xarray.DataArray
            Aggregated water balance for each watershed
        """        
        logger.debug('Aggregating over watersheds...')

        # Create label arrays for each watershed
        label_arrays = []
        
        for i, id in enumerate(self.get_ids()):
            # Create labels: watershed_id where mask==1, -1 elsewhere
            mask = self.get_mask(id)
            labels = xr.where(mask == 1, i, -1)
            label_arrays.append(labels)
        
        # Stack all label arrays along a new dimension
        combined_labels = xr.concat(label_arrays, dim='id')
        combined_labels.name = 'Code'

        # Transform coordinate to integer, otherwise error
        watershed_id_to_int = dict(zip(combined_labels.id.values, range(len(combined_labels.id.values))))
        combined_labels = combined_labels.assign_coords(id = [watershed_id_to_int[i] for i in combined_labels.id.values])
        
        # Perform the aggregation
        result = xarray_reduce(
            data, 
            combined_labels, 
            func="sum",
            expected_groups = combined_labels.id.values
            # Specify dimensions to reduce over (lat, lon are implicit from labels)
            # time dimension is preserved
        )

        # Add watershed names as index
        int_to_watershed_id = {v: k for k, v in watershed_id_to_int.items()}
        result = result.assign_coords(Code = [int_to_watershed_id[i] for i in result.Code.values])

        if result.name is None:
            result.name = 'modeled_values'

        aggregated_values = result.to_dataframe().drop(columns = ['spatial_ref'], errors = 'ignore')
        aggregated_values = aggregated_values.replace(0, np.nan).dropna()

        if isinstance(data, xr.DataArray) and data.name is not None:
            aggregated_values.rename(columns = {data.name: 'modeled_values'}, inplace = True)

        return aggregated_values

    def _aggregate(self, data: xr.DataArray, method: str = 'sum', dim = ['lon', 'lat']) -> pd.DataFrame:

        if len(self.data) == 0:
            raise ValueError("No watersheds available for aggregation. Load watersheds first.")
        
        # Check if time dimension exists in the data
        has_time_dim = 'time' in data.dims
        
        # Dictionary to store results for each watershed
        results = {}
        
        for ws_id in self.get_ids():

            # Create a boolean mask (True where watershed data is valid, False elsewhere)
            ws_mask = self.get_mask(ws_id)
            ws_mask = ws_mask != ws_mask.attrs.get('_FillValue', -999)
            
            # Apply the mask to the data
            masked_data = data.where(ws_mask)
            
            # Aggregate data based on the specified method
            if method == 'sum':
                #Sum means to calculate discharge
                aggregated = masked_data.sum(dim=dim)
            elif method == 'mean':
                aggregated = masked_data.mean(dim=dim)
            else:
                raise ValueError(f"Unsupported aggregation method: {method}")
            
            # Store the result for this watershed
            results[ws_id] = aggregated

        # Create DataFrame based on whether time dimension exists
        if has_time_dim:
            # For multiple timesteps: create a DataFrame with time as index and watersheds as columns
            df_data = {}
            time_values = data.time.values
            
            for ws_name, result in results.items():
                # Extract values for each timestep
                df_data[ws_name] = result.values
                logger.debug(f"Aggregated watershed {ws_name}")
            
            # Create DataFrame with time as index and watersheds as columns
            model_tbl = pd.DataFrame(df_data, index=time_values)
            model_tbl.index = model_tbl.index.set_names('time')

            model_tbl = model_tbl.melt(
                ignore_index = False, value_name = 'modeled_values', var_name = 'Code'
                )
            model_tbl.set_index('Code', append = True, inplace = True)
            return model_tbl
        else:
            # For single timestep: create a DataFrame with watersheds as index
            return pd.DataFrame(
                {"modeled_values": [result.values.item() for result in results.values()]},
                index=results.keys()
            )
                
if __name__ == "__main__":
    import yaml
    from src.core import Landuse, Precipitation, Temperature

    with open("config.yaml", "r", encoding = 'utf-8') as f:
        config = yaml.safe_load(f)

    landuse = Landuse(config)
    landuse.load()

    watersheds = Watersheds(config)
    watersheds.load(landuse.data)
    aggregated_data = watersheds.aggregate(landuse.data)

    print(aggregated_data)