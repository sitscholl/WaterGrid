import pandas as pd
import xarray as xr
import numpy as np

INTERSTATION_NETWORK = {
    "67350PG": ["59450PG", "64550PG"],
    "69790PG": ["67350PG"],
    "85550PG": ["19850PG", "69790PG"]
}

def construct_interstation_watersheds(watersheds):
    """
    Construct interstation watershed regions.
    
    Parameters:
    -----------
    watersheds : Watersheds
        Watersheds object containing watershed masks
        
    Returns:
    --------
    dict
        Dictionary of interstation regions, keyed by watershed ID
    """
    interstation_regions = {}

    for ws_id in watersheds.get_ids():
        if ws_id in INTERSTATION_NETWORK:
            original_region = watersheds.get_mask(ws_id)
            fill_value = original_region.attrs.get('_FillValue', -999)

            nested_watersheds = INTERSTATION_NETWORK[ws_id]
            nested_watersheds = xr.concat([watersheds.get_mask(nw) for nw in nested_watersheds if watersheds.get_mask(nw) is not None], dim="watershed").max('watershed')

            interstation_region = original_region.where(nested_watersheds != 1)

            # Watersheds should have value 1 for valid areas
            interstation_region = interstation_region.where(interstation_region == 1, fill_value)
            
            # Assign fill value attribute for metadata
            interstation_region = interstation_region.assign_attrs(_FillValue=fill_value)
            
            # Ensure data is of integer type
            interstation_region = interstation_region.astype(int)
        else:
            interstation_region = watersheds.get_mask(ws_id)

        interstation_regions[ws_id] = interstation_region
    return interstation_regions

def get_measured_discharge_for_interstation_regions(validation_tbl):
    """
    Calculate measured discharge for interstation regions.
    
    Parameters:
    -----------
    validation_tbl : pd.DataFrame
        Validation table containing measured discharge values
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing water budget for each interstation region
        
    Raises:
    -------
    ValueError
        If required columns are missing from validation_tbl
    """
        
    if validation_tbl.index.names != ['time', 'Code']:
        raise ValueError("Index must have 'time' and 'Code' as names.")

    interstation_discharge = []

    for st_code, data in validation_tbl.groupby(level=1):
        if st_code in INTERSTATION_NETWORK:
            nested_watersheds = INTERSTATION_NETWORK[st_code]
            q_nested = validation_tbl.loc[validation_tbl.index.get_level_values(1).isin(nested_watersheds)]

            # Identify years where all nested watersheds have a value
            nested_size = q_nested.groupby(level=0).size()
            nested_count = len(nested_watersheds)
            nested_valid = nested_size[nested_size == nested_count].index
                
            q_agg = q_nested.groupby(level=0).sum()
            q_agg.loc[~q_agg.index.isin(nested_valid)] = np.nan
            
            # Subtract
            diff = data.sub(q_agg)
            
            interstation_discharge.append(diff)
        else:
            interstation_discharge.append(data)
        
    interstation_discharge = pd.concat(interstation_discharge)
    
    return interstation_discharge