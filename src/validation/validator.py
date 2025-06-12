import pandas as pd
import numpy as np

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Validator:

    def __init__(self, config):
        self.config = config
        self.discharge_data = None
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
        flow_column = 'Wert[m³/s]'  # Original column name before renaming
        
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

    def aggregate(self, freq: str = 'YE-SEP', method = np.sum, min_size: int = 365):

        if self.data is None:
            raise ValueError("Data has not been loaded. Please load data first before calculating discharge.")

        data_agg = (
            self.data
            .groupby('Code')
            .resample(freq, include_groups=False)
            .agg(measured_values = ('Abfluss', method), 
                 size = ('Abfluss', lambda x: len(x.dropna())),
            ))

        data_agg = data_agg.loc[data_agg['size'] >= min_size].drop('size', axis = 1).reset_index()

        if method == np.sum:
            data_agg['measured_values'] = data_agg['measured_values'] * 86400
        # data_agg['Hyd_year'] = data_agg['time'].dt.year

        return data_agg.set_index(['Code', 'time'])

    def validate(self, watersheds, water_balance, freq: str = 'YE-SEP'):

        modeled_data = watersheds.aggregate(water_balance)
        modeled_data = modeled_data.groupby('Code').resample(freq, include_groups = False).sum()

        measured_data = self.aggregate(freq = freq)

        validation_tbl = modeled_data.join(measured_data)
        print(measured_data)



if __name__ == "__main__":
    import yaml
    from src.core import Landuse, Precipitation, Temperature
    from src.validation.watersheds import Watersheds

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    with open("config.yaml", "r", encoding = 'utf-8') as f:
        config = yaml.safe_load(f)

    precipitation = Precipitation(config)
    precipitation.load()

    watersheds = Watersheds(config)
    watersheds.load(precipitation.data)

    validator = Validator(config)
    validator.validate(watersheds, precipitation.data)
