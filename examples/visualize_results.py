#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize water balance results.

This script creates maps and time series plots of the water balance results.
"""

import os
import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray
import xarray as xr

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize water balance results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results",
        help="Directory containing water balance results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../figures",
        help="Directory to save figures"
    )
    return parser.parse_args()


def find_result_files(results_dir):
    """Find water balance result files.
    
    Args:
        results_dir: Directory containing water balance results
        
    Returns:
        List of paths to water balance result files
    """
    results_dir = Path(results_dir)
    result_files = list(results_dir.glob("**/*.tif"))
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return []
    
    print(f"Found {len(result_files)} result files")
    return result_files


def create_map(file_path, output_dir):
    """Create a map of water balance for a single file.
    
    Args:
        file_path: Path to water balance result file
        output_dir: Directory to save figure
    """
    # Load data
    da = rioxarray.open_rasterio(file_path)
    
    # Extract filename for plot title
    filename = file_path.name
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot water balance
    # Use diverging colormap centered at 0
    vmax = max(abs(da.min().values), abs(da.max().values))
    vmin = -vmax
    
    im = da.plot(
        ax=ax,
        cmap="RdBu",
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={
            "label": "Water Balance (mm)",
            "orientation": "horizontal",
            "shrink": 0.8,
            "pad": 0.05
        }
    )
    
    # Add title
    ax.set_title(f"Water Balance: {filename}")
    
    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"map_{filename.replace('.tif', '.png')}"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Created map: {output_path}")


def create_time_series(result_files, output_dir):
    """Create time series plot of water balance statistics.
    
    Args:
        result_files: List of paths to water balance result files
        output_dir: Directory to save figure
    """
    # Check if we have time series data
    if len(result_files) <= 1:
        print("Not enough data for time series plot")
        return
    
    # Extract dates and statistics from files
    dates = []
    means = []
    mins = []
    maxs = []
    
    for file_path in result_files:
        # Try to extract date from filename
        filename = file_path.name
        date_str = None
        
        # Look for date pattern in filename
        if "_monthly_" in filename:
            # Extract date from monthly filename pattern
            parts = filename.split("_")
            for i, part in enumerate(parts):
                if part == "monthly" and i < len(parts) - 1:
                    date_str = parts[i + 1]
                    break
        
        if date_str is None:
            # Skip files where we can't determine the date
            continue
        
        try:
            # Parse date
            date = pd.to_datetime(date_str)
            
            # Load data
            da = rioxarray.open_rasterio(file_path)
            
            # Calculate statistics
            mean_val = float(da.mean().values)
            min_val = float(da.min().values)
            max_val = float(da.max().values)
            
            # Append to lists
            dates.append(date)
            means.append(mean_val)
            mins.append(min_val)
            maxs.append(max_val)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not dates:
        print("No valid dates found for time series plot")
        return
    
    # Sort by date
    sorted_indices = np.argsort(dates)
    dates = [dates[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    mins = [mins[i] for i in sorted_indices]
    maxs = [maxs[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot statistics
    ax.plot(dates, means, 'b-', label="Mean")
    ax.fill_between(dates, mins, maxs, color='b', alpha=0.2, label="Min-Max Range")
    
    # Add reference line at 0
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Add labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Water Balance (mm)")
    ax.set_title("Water Balance Time Series")
    ax.legend()
    
    # Format x-axis
    fig.autofmt_xdate()
    
    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "water_balance_time_series.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Created time series plot: {output_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Find result files
    result_files = find_result_files(args.results_dir)
    
    if not result_files:
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create maps for each result file
    for file_path in result_files:
        create_map(file_path, output_dir)
    
    # Create time series plot
    create_time_series(result_files, output_dir)
    
    print(f"\nVisualization complete! Figures saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())