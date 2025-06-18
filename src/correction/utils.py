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
            nested_watersheds = INTERSTATION_NETWORK[ws_id]
            nested_watersheds = xr.concat([watersheds.get_mask(nw) for nw in nested_watersheds], dim="watershed").max('watershed')

            interstation_region = original_region.mask(nested_watersheds != 1)
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
    if validation_tbl is None:
        raise ValueError("validation_tbl cannot be None")
        
    if 'measured_values' not in validation_tbl.columns and 'modeled_values' not in validation_tbl.columns:
        # Check if the table already has the right structure
        if not all(col in validation_tbl.index.names for col in ['Code', 'Hyd_year']):
            raise ValueError(f"Columns 'measured_values' and 'modeled_values' are required or index must contain 'Code' and 'Hyd_year'. Got {validation_tbl.columns}")
        # If we're here, assume the table is already in the right format
        data_to_process = validation_tbl
    else:
        data_to_process = validation_tbl['measured_values']

    interstation_discharge = []

    for st_code, data in data_to_process.groupby(level=1):
        if st_code in INTERSTATION_NETWORK:
            nested_watersheds = INTERSTATION_NETWORK[st_code]
            q_nested = data_to_process.loc[data_to_process.index.get_level_values(1).isin(nested_watersheds)]

            # Identify years where all nested watersheds have a value
            nested_size = q_nested.groupby('Zeitstempel').size() if 'Zeitstempel' in q_nested.columns else q_nested.groupby(level=0).size()
            nested_count = len(nested_watersheds)
            nested_valid = nested_size[nested_size == nested_count].index
                
            # Sum nested watersheds and filter
            if 'Zeitstempel' in q_nested.columns:
                q_agg = q_nested.groupby('Zeitstempel')[['qm', 'Abfluss']].sum() if 'qm' in q_nested.columns else q_nested.groupby('Zeitstempel').sum()
                q_agg = q_agg.loc[q_agg.index.isin(nested_valid)]
                
                # Subtract
                if 'Zeitstempel' in data.columns:
                    diff = data.set_index(['Zeitstempel', 'Code', 'Hyd_year']).sub(q_agg).dropna(how='all')
                else:
                    diff = data.sub(q_agg).dropna(how='all')
            else:
                q_agg = q_nested.groupby(level=0).sum()
                q_agg = q_agg.loc[q_agg.index.isin(nested_valid)]
                
                # Subtract
                diff = data.sub(q_agg).dropna(how='all')
            
            interstation_discharge.append(diff.reset_index() if hasattr(diff, 'reset_index') else diff)
        else:
            interstation_discharge.append(data)
        
    interstation_discharge = pd.concat(interstation_discharge)
    
    # Ensure the DataFrame has the right structure
    if 'Code' in interstation_discharge.columns and 'Hyd_year' in interstation_discharge.columns:
        interstation_discharge.set_index(['Code', 'Hyd_year'], inplace=True)
    
    # Drop unnecessary columns
    if 'Zeitstempel' in interstation_discharge.columns:
        interstation_discharge.drop('Zeitstempel', axis=1, inplace=True)
    
    # Rename columns if needed
    if 'qm' in interstation_discharge.columns:
        interstation_discharge = interstation_discharge[['qm']].rename(columns={'qm': 'water_budget'})
    elif not any(col == 'water_budget' for col in interstation_discharge.columns):
        interstation_discharge = interstation_discharge.rename(columns={interstation_discharge.columns[0]: 'water_budget'})

    return interstation_discharge