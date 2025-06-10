#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate synthetic test data for the Climatic Water Balance Calculator.

This script creates synthetic temperature, precipitation, and land-use data
for testing the application without requiring real data.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import zarr
import rioxarray
from rasterio.transform import from_origin

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def generate_temperature_data(output_path, start_date, end_date, resolution=250):
    """Generate synthetic temperature data.
    
    Args:
        output_path: Path to save the zarr dataset
        start_date: Start date for the time series
        end_date: End date for the time series
        resolution: Spatial resolution in meters
    """
    # Create time range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_times = len(dates)
    
    # Create spatial grid (100x100 cells at 250m resolution)
    n_y, n_x = 100, 100
    y = np.arange(0, n_y * resolution, resolution)
    x = np.arange(0, n_x * resolution, resolution)
    
    # Create synthetic temperature data with seasonal cycle
    # Base temperature pattern: higher in summer, lower in winter
    day_of_year = np.array([d.dayofyear for d in dates])
    seasonal_cycle = 15 * np.sin(2 * np.pi * (day_of_year - 15) / 365) + 10  # Mean around 10°C, amplitude 15°C
    
    # Add spatial variation (temperature decreases with latitude/y)
    y_gradient = np.linspace(0, -5, n_y)  # 5°C decrease from south to north
    spatial_pattern = np.tile(y_gradient[:, np.newaxis], (1, n_x))
    
    # Combine temporal and spatial patterns
    temperature = np.zeros((n_times, n_y, n_x))
    for t in range(n_times):
        temperature[t, :, :] = seasonal_cycle[t] + spatial_pattern
        
        # Add some random noise
        temperature[t, :, :] += np.random.normal(0, 1, (n_y, n_x))
    
    # Create xarray dataset
    ds = xr.Dataset(
        data_vars={
            "temperature": (["time", "y", "x"], temperature, {
                "units": "celsius",
                "long_name": "Air Temperature",
                "standard_name": "air_temperature"
            })
        },
        coords={
            "time": (["time"], dates),
            "y": (["y"], y),
            "x": (["x"], x)
        },
        attrs={
            "description": "Synthetic temperature data for testing",
            "resolution": f"{resolution}m",
            "crs": "EPSG:32632"
        }
    )
    
    # Set spatial attributes for rioxarray
    transform = from_origin(x[0], y[0] + n_y * resolution, resolution, resolution)
    ds.rio.write_crs("EPSG:32632", inplace=True)
    ds.rio.write_transform(transform, inplace=True)
    
    # Save to zarr
    ds.to_zarr(output_path, mode="w")
    print(f"Generated temperature data saved to {output_path}")


def generate_precipitation_data(output_path, start_date, end_date, resolution=250):
    """Generate synthetic precipitation data.
    
    Args:
        output_path: Path to save the zarr dataset
        start_date: Start date for the time series
        end_date: End date for the time series
        resolution: Spatial resolution in meters
    """
    # Create time range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_times = len(dates)
    
    # Create spatial grid (100x100 cells at 250m resolution)
    n_y, n_x = 100, 100
    y = np.arange(0, n_y * resolution, resolution)
    x = np.arange(0, n_x * resolution, resolution)
    
    # Create synthetic precipitation data with seasonal cycle
    # More rain in winter/spring, less in summer
    day_of_year = np.array([d.dayofyear for d in dates])
    seasonal_prob = 0.3 - 0.2 * np.sin(2 * np.pi * (day_of_year - 15) / 365)  # Probability between 0.1 and 0.5
    
    # Add spatial variation (more rain in mountains/higher y)
    y_gradient = np.linspace(0, 2, n_y)  # Increase in precipitation with latitude
    spatial_pattern = np.tile(y_gradient[:, np.newaxis], (1, n_x))
    
    # Combine temporal and spatial patterns
    precipitation = np.zeros((n_times, n_y, n_x))
    for t in range(n_times):
        # Generate rain events (0 or positive values)
        rain_mask = np.random.random((n_y, n_x)) < seasonal_prob[t]
        rain_amount = np.zeros((n_y, n_x))
        rain_amount[rain_mask] = np.random.exponential(5, size=np.sum(rain_mask))  # Mean of 5mm when it rains
        
        # Apply spatial pattern (multiply by factor)
        precipitation[t, :, :] = rain_amount * (1 + 0.5 * spatial_pattern)
    
    # Create xarray dataset
    ds = xr.Dataset(
        data_vars={
            "precipitation": (["time", "y", "x"], precipitation, {
                "units": "mm",
                "long_name": "Precipitation",
                "standard_name": "precipitation_amount"
            })
        },
        coords={
            "time": (["time"], dates),
            "y": (["y"], y),
            "x": (["x"], x)
        },
        attrs={
            "description": "Synthetic precipitation data for testing",
            "resolution": f"{resolution}m",
            "crs": "EPSG:32632"
        }
    )
    
    # Set spatial attributes for rioxarray
    transform = from_origin(x[0], y[0] + n_y * resolution, resolution, resolution)
    ds.rio.write_crs("EPSG:32632", inplace=True)
    ds.rio.write_transform(transform, inplace=True)
    
    # Save to zarr
    ds.to_zarr(output_path, mode="w")
    print(f"Generated precipitation data saved to {output_path}")


def generate_landuse_data(output_path, resolution=5):
    """Generate synthetic land-use data.
    
    Args:
        output_path: Path to save the GeoTIFF file
        resolution: Spatial resolution in meters
    """
    # Create high-resolution grid (5000x5000 cells at 5m resolution)
    # This matches the extent of the temperature and precipitation data (100x100 at 250m)
    n_y, n_x = 5000, 5000
    y = np.arange(0, n_y * resolution, resolution)
    x = np.arange(0, n_x * resolution, resolution)
    
    # Create synthetic land-use data
    # 1: Forest, 2: Cropland, 3: Grassland, 4: Urban, 5: Water
    
    # Start with all grassland
    landuse = np.ones((n_y, n_x), dtype=np.int16) * 3
    
    # Add forest patches
    for _ in range(50):
        center_y = np.random.randint(0, n_y)
        center_x = np.random.randint(0, n_x)
        radius = np.random.randint(50, 200)  # Forest patch size
        
        y_grid, x_grid = np.ogrid[-center_y:n_y-center_y, -center_x:n_x-center_x]
        mask = y_grid*y_grid + x_grid*x_grid <= radius*radius
        landuse[mask] = 1
    
    # Add cropland areas
    for _ in range(30):
        start_y = np.random.randint(0, n_y - 500)
        start_x = np.random.randint(0, n_x - 500)
        size_y = np.random.randint(300, 500)
        size_x = np.random.randint(300, 500)
        
        landuse[start_y:start_y+size_y, start_x:start_x+size_x] = 2
    
    # Add urban areas
    for _ in range(10):
        center_y = np.random.randint(0, n_y)
        center_x = np.random.randint(0, n_x)
        radius = np.random.randint(100, 300)  # Urban area size
        
        y_grid, x_grid = np.ogrid[-center_y:n_y-center_y, -center_x:n_x-center_x]
        mask = y_grid*y_grid + x_grid*x_grid <= radius*radius
        landuse[mask] = 4
    
    # Add water bodies
    for _ in range(5):
        center_y = np.random.randint(0, n_y)
        center_x = np.random.randint(0, n_x)
        radius = np.random.randint(50, 150)  # Water body size
        
        y_grid, x_grid = np.ogrid[-center_y:n_y-center_y, -center_x:n_x-center_x]
        mask = y_grid*y_grid + x_grid*x_grid <= radius*radius
        landuse[mask] = 5
    
    # Add a river
    river_x = np.random.randint(n_x // 4, 3 * n_x // 4)
    river_width = np.random.randint(10, 30)
    landuse[:, river_x-river_width//2:river_x+river_width//2] = 5
    
    # Create xarray dataset
    da = xr.DataArray(
        landuse,
        dims=["y", "x"],
        coords={
            "y": (["y"], y),
            "x": (["x"], x)
        },
        attrs={
            "long_name": "Land Use Classification",
            "units": "1",  # Categorical
            "description": "1: Forest, 2: Cropland, 3: Grassland, 4: Urban, 5: Water"
        }
    )
    
    # Set spatial attributes for rioxarray
    transform = from_origin(x[0], y[0] + n_y * resolution, resolution, resolution)
    da.rio.write_crs("EPSG:32632", inplace=True)
    da.rio.write_transform(transform, inplace=True)
    
    # Save to GeoTIFF
    da.rio.to_raster(output_path)
    print(f"Generated land-use data saved to {output_path}")


def generate_kc_coefficients(output_path):
    """Generate Kc coefficients Excel file.
    
    Args:
        output_path: Path to save the Excel file
    """
    # Create DataFrame with Kc coefficients
    kc_data = {
        "landuse_code": [1, 2, 3, 4, 5],
        "landuse_name": ["Forest", "Cropland", "Grassland", "Urban", "Water"],
        "winter": [0.8, 0.4, 0.6, 0.3, 1.0],
        "spring": [1.0, 0.7, 0.8, 0.3, 1.0],
        "summer": [1.2, 1.1, 0.9, 0.3, 1.0],
        "autumn": [0.9, 0.6, 0.7, 0.3, 1.0]
    }
    
    kc_df = pd.DataFrame(kc_data)
    
    # Save to Excel
    kc_df.to_excel(output_path, index=False)
    print(f"Generated Kc coefficients saved to {output_path}")


def main():
    """Generate all test data."""
    # Set paths
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    temp_path = data_dir / "temperature.zarr"
    precip_path = data_dir / "precipitation.zarr"
    landuse_path = data_dir / "landuse.tif"
    kc_path = data_dir / "kc_values.xlsx"
    
    # Set date range (one year)
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    
    # Generate data
    print("Generating test data...")
    generate_temperature_data(temp_path, start_date, end_date)
    generate_precipitation_data(precip_path, start_date, end_date)
    generate_landuse_data(landuse_path)
    generate_kc_coefficients(kc_path)
    
    print("\nTest data generation complete!")
    print("You can now run the application with:")
    print("python main.py --config config.yaml")


if __name__ == "__main__":
    main()