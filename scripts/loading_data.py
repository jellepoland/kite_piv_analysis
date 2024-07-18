import os
import numpy as np
import xarray as xr
import logging
import re
import sys

# Go back to root folder
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)
output_path = sys.path[0] + "/data/processed_data.nc"

# Load the processed data (for future use)
loaded_dataset = load_processed_data(output_path)
logging.info(f"Loaded dataset: {loaded_dataset}")

# Example of accessing data
logging.info(f"Velocity u for first file: {loaded_dataset['vel_u'].isel(file=0)}")
