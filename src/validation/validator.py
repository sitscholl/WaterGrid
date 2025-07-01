import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

import logging
from pathlib import Path

from src.cluster import start_dask_cluster
from ..correction.utils import get_measured_discharge_for_interstation_regions
from ..config import SECONDS_PER_YEAR, SECONDS_PER_MONTH

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
            logger.debug(f"Loaded {len(tables)} tables into the validator.")
                        
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
        nans_before = data_fill.isna().sum().item()
        
        # Get maximum gap for interpolation from config
        maxgap = stations_config.get('maxgap', 3)
        
        # Process missing values
        data_fill = self._handle_missing_values(data_fill, maxgap)
        nans_after = data_fill.isna().sum().item()

        logger.debug(f"Loaded Station {info.get('Station name', 'Unknown')} with {nans_before - nans_after} NA values filled.")
        
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

    def aggregate(self, freq: str = 'YE-SEP', method = "mean", compute_for_interstation_regions = False):
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

        if freq not in ['YE-SEP', 'ME']:
            raise ValueError(f"Frequency {freq} is not supported. Please use 'YE-SEP' or 'ME'.")

        min_size = 365 if freq == 'YE-SEP' else 28

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

        # Aggregate water balance data to the specified frequency
        if len(water_balance.time) > 2 and xr.infer_freq(water_balance.time) != freq:
            balance_agg = water_balance.resample(time = freq).sum()
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

        ## Transform from m³/year or m³/month to m³/s
        if freq == 'YE-SEP':
            modeled_discharge /= SECONDS_PER_YEAR
        elif freq == 'ME':
            modeled_discharge /= SECONDS_PER_MONTH
        else:
            raise NotImplementedError(f"Frequency {freq} is not supported. Please use 'YE-SEP' or 'ME'.")

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
    
    def plot_timeseries(self, validation_tbl, clip: bool = True):
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

            # Create figure with appropriate size
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot data with better colors and styling
            measured = plot_data[plot_data['variable'] == 'measured_values']
            modeled = plot_data[plot_data['variable'] == 'modeled_values']

            # Plot measured data
            ax.plot(measured['time'], measured['value'],
                   color='#1f77b4', label='Measured', marker='o',
                   markersize=4, alpha=0.8)

            # Plot modeled data
            ax.plot(modeled['time'], modeled['value'],
                   color='#ff7f0e', label='Modeled', marker='s',
                   markersize=4, alpha=0.8, linestyle='--')

            # Add shaded area to highlight differences
            if not measured.empty and not modeled.empty:
                # Create a common time index
                common_times = pd.merge(measured, modeled, on='time', how='inner').dropna()['time']
                if not common_times.empty:
                    measured_values = measured[measured['time'].isin(common_times)].set_index('time')['value']
                    modeled_values = modeled[modeled['time'].isin(common_times)].set_index('time')['value']

                    # Sort by time to ensure correct plotting
                    measured_values = measured_values.sort_index()
                    modeled_values = modeled_values.sort_index()

                    # Plot the area between the curves
                    ax.fill_between(measured_values.index, measured_values, modeled_values,
                                   alpha=0.2, color='gray', label='Difference')
            
            if clip:
                common_times = pd.merge(measured, modeled, on='time', how='inner').dropna()['time']
                if not common_times.empty:
                    xmin, xmax = common_times.min(), common_times.max()
                    ax.set_xlim(xmin, xmax)

            # Improve axis labels and title
            ax.set_xlabel('Hydrological Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('Discharge [m³/s]', fontsize=12, fontweight='bold')
            #ax.set_title(f"{station_name} (Code: {code})", fontsize=14, fontweight='bold', pad=20)

            # Format x-axis for better date display
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.xticks(rotation=45)

            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)

            # Improve legend
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                     ncol=3, frameon=True, fontsize=10, shadow=True)

            # Add error metrics with better formatting
            if not data['measured_values'].isna().all() and not data['modeled_values'].isna().all():
                # Calculate correlation only on rows where both values exist
                valid_data = data.dropna()
                if len(valid_data) > 1:  # Need at least 2 points for correlation
                    ws_correlation = valid_data['measured_values'].corr(valid_data['modeled_values'])

                    # Calculate relative error as percentage
                    relative_error = ((valid_data['modeled_values'] - valid_data['measured_values']) /
                                     valid_data['measured_values']).mean() * 100

                    # Calculate RMSE
                    rmse = np.sqrt(((valid_data['modeled_values'] - valid_data['measured_values']) ** 2).mean())

                    # Add metrics in a nice text box
                    metrics_text = (f"Correlation: {ws_correlation:.2f}\n"
                                   f"Relative Error: {relative_error:.2f}%\n"
                                   f"RMSE: {rmse:.2f} m³/s")

                    # Create a text box for metrics
                    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
                    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                           verticalalignment='top', bbox=props)
                else:
                    logger.warning(f"Not enough valid data points for station {code} to calculate correlation.")

            # Adjust layout
            plt.tight_layout()

            # Save with high quality
            plt.savefig(f'{out_dir}/{code}.png', dpi=300, bbox_inches='tight')
            plt.close()
            


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