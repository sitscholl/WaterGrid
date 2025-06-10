# Climatic Water Balance Calculator

A Python application for calculating climatic water balance following the methodology described in [http://dx.doi.org/10.1002/gdj3.70007](http://dx.doi.org/10.1002/gdj3.70007).

## Features

- Calculate water balance as: Precipitation - Evapotranspiration
- Compute potential evapotranspiration using the Thornthwaite method from temperature data (°C)
- Adjust evapotranspiration using land-use specific crop coefficients (Kc values) applied as discrete seasonal values
- Memory-efficient processing using chunked operations (xarray/dask)
- Support for multi-resolution data handling (250m climate data, 5m land-use data)
- Flexible output configurations (daily/monthly/seasonal/annual)

## Installation

### Using uv (recommended)

```bash
uv venv
uv pip install -e .
```

### Using pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Quick Start

1. Generate synthetic test data:

```bash
python examples/generate_test_data.py
```

2. Run the water balance calculation:

```bash
python main.py --config config.yaml
```

3. Visualize the results:

```bash
python examples/visualize_results.py
```

## Input Data

The application requires the following input data:

- **Temperature data**: Daily temperature grids in .zarr format (EPSG:32632, 250m resolution)
- **Precipitation data**: Daily precipitation grids in .zarr format (EPSG:32632, 250m resolution)
- **Land-use data**: Land-use classification in GeoTIFF format (EPSG:32632, 5m resolution)
- **Kc coefficients**: Excel (.xlsx) file with land-use types and seasonal Kc coefficients

### Kc Coefficients Format

The Kc coefficients file should have the following columns:

- `landuse_code`: Numeric code matching the values in the land-use data
- `landuse_name`: Descriptive name of the land-use type
- `winter`, `spring`, `summer`, `autumn`: Seasonal Kc coefficients

## Usage

### Command Line Interface

Run the application with a configuration file:

```bash
python main.py --config config.yaml
```

For verbose logging:

```bash
python main.py --config config.yaml --verbose
```

### Programmatic Usage

You can also use the application programmatically:

```python
from src.data_io import load_config
from src.calculations import calculate_water_balance

# Load configuration
config = load_config("config.yaml")

# Calculate water balance
output_paths = calculate_water_balance(config)

print(f"Output files: {output_paths}")
```

See `examples/simple_example.py` for a complete example.

## Configuration

The application is configured using a YAML file. See `config.yaml` for an example configuration with detailed comments.

Key configuration sections:

- **input**: Paths to input data files
- **output**: Output directory and file naming
- **spatial**: Target resolution and resampling method
- **temporal**: Date range and output frequency
- **processing**: Chunk sizes and parallel processing options
- **calculation**: Parameters for the Thornthwaite method
- **seasons**: Definition of seasonal periods for Kc coefficients

## Methodology

### Thornthwaite Method

The Thornthwaite method calculates potential evapotranspiration (PET) based on mean monthly temperature and day length (which depends on latitude).

The formula is:

PET = 16 * (10 * T / I)^a * L

Where:
- T is the mean monthly temperature (°C)
- I is the annual heat index, calculated as the sum of 12 monthly index values
- a is an empirically derived exponent based on I
- L is a correction factor for day length and days in month

### Water Balance Calculation

The water balance is calculated as:

WB = P - ET

Where:
- WB is the water balance (mm)
- P is precipitation (mm)
- ET is evapotranspiration (mm), calculated as PET * Kc
- Kc is the crop coefficient based on land-use type and season

## Output

The application produces:

- Water balance grids as GeoTIFF files at the specified resolution
- Metadata files with processing parameters and statistics
- Detailed logs of processing steps

## Project Structure

```
├── main.py                  # Entry point
├── config.yaml              # Example configuration
├── pyproject.toml           # Project metadata and dependencies
├── README.md                # Project documentation
├── src/                     # Source code
│   ├── __init__.py          # Package initialization
│   ├── data_io.py           # Data loading/saving functions
│   ├── calculations.py      # Thornthwaite, water balance
│   ├── resampling.py        # Spatial resolution handling
│   └── utils.py             # Helper functions, logging
├── examples/                # Example scripts
│   ├── generate_test_data.py # Generate synthetic test data
│   ├── simple_example.py     # Simple usage example
│   └── visualize_results.py  # Visualize water balance results
├── tests/                   # Unit tests
│   └── test_utils.py         # Tests for utility functions
├── data/                    # Input data directory
├── results/                 # Output results directory
├── logs/                    # Log files directory
└── figures/                 # Output figures directory
```

## Performance Considerations

- The application uses dask for chunked and parallel processing
- Memory usage is controlled by chunk size configuration
- For large datasets, adjust the `chunk_size` and `parallel_workers` parameters in the configuration
- The most memory-intensive operation is the resampling between different resolution grids

## License

MIT