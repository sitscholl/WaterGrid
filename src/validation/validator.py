import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

import logging
from pathlib import Path

from src.cluster import start_dask_cluster
from ..correction.utils import get_measured_discharge_for_interstation_regions

logger = logging.getLogger(__name__)

class Validator:

    def __init__(self, config):
        self.config = config
        self.data = None  # Changed from self.discharge_data to self.data for consistency
        self.load()

    def load(self):
        """
        Load station data from CSV files, preprocess it, and concatenate into a single DataFrame.
        
        This method performs the following operations:
        1. Loads data from CSV files matching the pattern in the configuration
        2. Resamples the data to daily frequency
        3. Interpolates missing values up to a specified maximum gap
        4. Replaces values with NaN if consecutive missing values exceed the maximum gap
        5. Concatenates all processed data into a single DataFrame
        
        Raises:
            ValueError: If no files are found matching the pattern
            KeyError: If required columns are missing in the CSV files
        """
        stations_config = self.config['input']['stations']
        root = Path(stations_config.get('root', '.'))
        pattern = stations_config.get('pattern', '*.csv')
        files = list(root.glob(pattern))

        if len(files) == 0:
            raise ValueError(f'No files found matching the pattern "{pattern}" in {root}')

        tables = []
        for file_path in files:
            try:
                processed_data = self._process_station_file(file_path, stations_config)
                tables.append(processed_data)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
        
        if not tables:
            logger.warning("No valid data could be processed from any of the files")
            self.data = None
        else:
            self.data = pd.concat(tables)
            logger.info(f"Loaded {len(tables)} tables into the validator.")
                        
    def _process_station_file(self, file_path, stations_config):
        """
        Process a single station file.
        
        Args:
            file_path: Path to the CSV file
            stations_config: Configuration for station data processing
            
        Returns:
            pd.DataFrame: Processed data for the station
        """
        # Read station metadata
        try:
            info = pd.read_csv(file_path, encoding='latin-1', nrows=20, header=None, index_col=0).to_dict()[1]
        except Exception as e:
            logger.error(f"Error reading station metadata from {file_path}: {str(e)}")
            raise
            
        logger.debug(f"Loading Station name: {info.get('Station name', 'Unknown')}")
        
        # Read time series data
        try:
            data = pd.read_csv(
                file_path, 
                encoding='latin-1', 
                skiprows=20, 
                sep='\t', 
                decimal=',',
                usecols=['Zeitstempel', 'Wert[m³/s]', 'Status des Werts']
            )
            
            # Filter out invalid status values
            data = data.loc[~data['Status des Werts'].isin(['S', 'I']), ['Zeitstempel', 'Wert[m³/s]']]
        except Exception as e:
            logger.error(f"Error reading time series data from {file_path}: {str(e)}")
            raise
        
        data.rename(columns = {'Zeitstempel': 'time'}, inplace = True)

        # Parse dates and set as index
        try:
            data['time'] = pd.to_datetime(data['time'], format='%d/%m/%Y %H:%M:%S')
            data.set_index('time', inplace=True)
        except Exception as e:
            logger.error(f"Error parsing dates in {file_path}: {str(e)}")
            raise
        
        # Resample to daily frequency
        data_fill = data.resample('D').first()
        logger.debug(f"\tnans before: {data_fill.isna().sum().item()}")
        
        # Get maximum gap for interpolation from config
        maxgap = stations_config.get('maxgap', 3)
        
        # Process missing values
        data_fill = self._handle_missing_values(data_fill, maxgap)
        logger.debug(f"\tnans after: {data_fill.isna().sum().item()}")
        
        # Add station code and rename columns
        data_fill['Code'] = info.get('Station number', 'Unknown')
        data_fill.rename(columns={'Wert[m³/s]': 'Abfluss'}, inplace=True)
        
        return data_fill
    
    def _handle_missing_values(self, data, maxgap):
        """
        Handle missing values in the data by interpolating up to a maximum gap.
        
        Args:
            data: DataFrame with time series data
            maxgap: Maximum number of consecutive missing values to interpolate
            
        Returns:
            pd.DataFrame: Data with handled missing values
        """
        # Count number of consecutive NaNs or non-NaNs
        # Fixed: Use the correct column name based on whether it's been renamed yet
        flow_column = 'Wert[m³/s]' if 'Wert[m³/s]' in data.columns else 'Abfluss'
        
        # Create a copy to avoid SettingWithCopyWarning
        result = data.copy()
        
        # Count consecutive NaN values
        cons_count = (result.notnull() != result.notnull().shift()).cumsum().groupby(flow_column).transform('size')
        
        # Find all dates with more than maxgap consecutive NaNs
        mask = (result[flow_column].isnull()) & (cons_count > maxgap)
        
        # Interpolate missing values
        result.interpolate(method='time', limit=maxgap, inplace=True)
        
        # Insert NaNs where more than maxgap consecutive NaNs
        result.loc[mask, flow_column] = np.nan
        
        return result

    def aggregate(self, freq: str = 'YE-SEP', method = "mean", min_size: int = 365, compute_for_interstation_regions = False):
        """
        Aggregate the discharge data to the specified frequency.
        
        Args:
            freq: Frequency string for resampling (default: 'YE-SEP' for hydrological year ending in September)
            method: Aggregation method (default: "mean")
            min_size: Minimum number of non-NaN values required for aggregation (default: 365)
            
        Returns:
            pd.DataFrame: Aggregated discharge data
            
        Raises:
            ValueError: If data has not been loaded
        """
        if self.data is None:
            raise ValueError("Data has not been loaded. Please load data first before calculating discharge.")

        data_agg = (
            self.data
            .groupby('Code')
            .resample(freq, include_groups=False)
            .agg(measured_values = ('Abfluss', method), #Abfluss is in m³/s per day
                 size = ('Abfluss', lambda x: len(x.dropna())),
            ))

        data_agg = data_agg.loc[data_agg['size'] >= min_size].drop('size', axis = 1).reset_index()

        # Convert from m³/s to m³/day if summing
        if method == "sum" or method == np.sum:
            data_agg['measured_values'] = data_agg['measured_values'] * 86400
        data_agg.set_index(['time', 'Code'], inplace = True)

        if compute_for_interstation_regions:
            data_agg = get_measured_discharge_for_interstation_regions(data_agg)

        return data_agg

    def validate(self, watersheds, water_balance, freq: str = 'YE-SEP', compute_for_interstation_regions = False):
        """
        Validate the water balance model against measured discharge data.
        
        Args:
            watersheds: Watersheds object containing watershed data
            water_balance: Water balance data (xarray.DataArray)
            freq: Frequency string for resampling (default: 'YE-SEP' for hydrological year)
            
        Returns:
            pd.DataFrame: Validation table with modeled and measured values
            
        Raises:
            NotImplementedError: If frequency other than 'YE-SEP' is specified
            ValueError: If rio accessor is not available
        """

        if freq != 'YE-SEP':
            raise NotImplementedError("Only 'YE-SEP' frequency is currently supported.")
            
        # Aggregate water balance data to the specified frequency
        if len(water_balance.time) > 2 and xr.infer_freq(water_balance.time) != freq:
            balance_agg = water_balance.resample(time = freq).sum(min_count = 12)
        else:
            #Assume correct frequency
            balance_agg = water_balance
        
        # Aggregate watershed data
        modeled_data = watersheds.aggregate(balance_agg) #unit is in mm/year
        modeled_data.replace(0, np.nan, inplace=True) #years with less values than min_count have values of 0

        ## Transform from mm/year to m³/year 
        if not hasattr(water_balance, 'rio'):
            raise ValueError('Rio accessor not available. Cannot calculate discharge')
        res = water_balance.rio.resolution()[0]
        modeled_discharge = modeled_data * (res**2) / 1000

        ## Transform from m³/year to m³/s
        seconds_per_year = 365.25 * 24 * 60 * 60  # More accurate seconds per year
        modeled_discharge /= seconds_per_year

        # Get measured discharge data
        measured_discharge = self.aggregate(freq = freq, compute_for_interstation_regions = compute_for_interstation_regions) #unit is in m³/s (average discharge per hydrological year)

        # Join modeled and measured data
        validation_tbl = modeled_discharge.join(measured_discharge, how='outer')
        
        # Log warning for stations without matching data
        # missing_stations = validation_tbl[validation_tbl['measured_values'].isna()].index.get_level_values('Code').unique()
        # if len(missing_stations) > 0:
        #     logger.warning(f"No measured data available for stations: {', '.join(missing_stations)}")
            
        # missing_modeled = validation_tbl[validation_tbl['modeled_values'].isna()].index.get_level_values('Code').unique()
        # if len(missing_modeled) > 0:
        #     logger.warning(f"No modeled data available for stations: {', '.join(missing_modeled)}")
        
        return validation_tbl
    
    def plot_timeseries(self, validation_tbl):
        """
        Plot time series of modeled and measured discharge for each station.
        
        Args:
            validation_tbl: Validation table with modeled and measured values
        """
        out_dir = Path(self.config['output'].get('directory', '.'), 'figures')
        out_dir.mkdir(parents=True, exist_ok=True)

        for code, data in validation_tbl.groupby(level = 1):
            
            plot_data = data.melt(ignore_index=False, var_name='variable', value_name='value').reset_index()

            if plot_data['value'].isna().all():
                logger.warning(f"No validation data available for station {code}. Skipping plot generation.")
                continue
            
            g = sns.relplot(data = plot_data, x = 'time', y = 'value', hue = 'variable', kind = 'line', height = 2)
            g.set_titles('{col_name}')
            g.set_ylabels('Abfluss [m³/s]')
            g.set_xlabels('Hydrologisches Jahr')

            plt.tight_layout()
            sns.move_legend(
                g, "lower center",
                bbox_to_anchor=(.5, .93), ncol=2, title=None, frameon=False,
            )
            
            # Add error metrics
            # Only calculate metrics if both modeled and measured values exist
            if not data['measured_values'].isna().all() and not data['modeled_values'].isna().all():
                # Calculate correlation only on rows where both values exist
                valid_data = data.dropna()
                if len(valid_data) > 1:  # Need at least 2 points for correlation
                    ws_correlation = valid_data['measured_values'].corr(valid_data['modeled_values'])
                    
                    # Calculate relative error as percentage
                    relative_error = ((valid_data['modeled_values'] - valid_data['measured_values']) / 
                                     valid_data['measured_values']).mean() * 100
                    
                    ax = g.axes[0][0]
                    label = f"p = {ws_correlation:.2f}\ne = {relative_error:.2f}%"
                    ax.annotate(text = label, xy = (.05, .85), xycoords = 'axes fraction')
                else:
                    logger.warning(f"Not enough valid data points for station {code} to calculate correlation.")

            plt.savefig(f'{out_dir}/{code}.png', dpi = 300, bbox_inches = 'tight')
            plt.close()

            logger.debug(f"Saved figure to {out_dir}/{code}.png")
            


if __name__ == "__main__":
    import yaml
    from src.core import Precipitation
    from src.validation.watersheds import Watersheds

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    with open("config.yaml", "r", encoding = 'utf-8') as f:
        config = yaml.safe_load(f)

    client, cluster = start_dask_cluster()

    precipitation = Precipitation(config)
    precipitation.load()

    watersheds = Watersheds(config)
    watersheds.load(precipitation.data)

    validator = Validator(config)
    validation_tbl = validator.validate(watersheds, precipitation.data)
    validator.plot_timeseries(validation_tbl)