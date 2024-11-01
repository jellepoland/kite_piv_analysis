import os
import numpy as np
import xarray as xr
import logging
import re
import sys

def read_dat_file(file_path: str) -> xr.Dataset:
    # Extract the case name and file name from the file path
    case_name = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)

    # Read the header information
    with open(file_path, "r") as file:
        header = [file.readline().strip() for _ in range(4)]

    # Extract the variable names
    variables = re.findall(r'"([^"]*)"', header[1])

    # Extract the i and j values
    i_value = int(re.search(r"I=(\d+)", header[2]).group(1))
    j_value = int(re.search(r"J=(\d+)", header[2]).group(1))

    # Load the data
    data = np.genfromtxt(file_path, skip_header=4)

    # Reshape the data into matrix form (j by i)
    data_matrix = data.reshape((j_value, i_value, -1))

    # Create an xarray Dataset
    coords = {
        'y': np.arange(j_value),
        'x': np.arange(i_value),
    }
    data_vars = {var: (('y', 'x'), data_matrix[..., idx]) for idx, var in enumerate(variables)}
    dataset = xr.Dataset(data_vars, coords=coords)

    # Add metadata
    dataset.attrs['case_name'] = case_name
    dataset.attrs['file_name'] = file_name
    dataset.attrs['i_value'] = i_value
    dataset.attrs['j_value'] = j_value

    return dataset

def process_all_dat_files(directory: str) -> xr.Dataset:
    all_datasets = []
    logging.info(f"Processing all .dat files in directory: {directory}")
    for root, _, files in os.walk(directory):
        logging.info(f"Processing files in directory: {root}")
        for file in files:
            if file.endswith('1.dat'):
                logging.info(f"Processing file: {file}")
                file_path = os.path.join(root, file)
                dataset = read_dat_file(file_path)
                all_datasets.append(dataset)
    
    # Combine all datasets
    combined_dataset = xr.concat(all_datasets, dim='file')
    return combined_dataset

def save_processed_data(dataset: xr.Dataset, output_file: str):
    dataset.to_netcdf(output_file)

def load_processed_data(input_file: str) -> xr.Dataset:
    return xr.open_dataset(input_file)

if __name__ == "__main__":

    # Go back to root folder
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, root_path)
    # Set up logging
    logging.basicConfig(level=logging.INFO)


    # dataset = read_dat_file(
    #     sys.path[0] + "/data/aoa_13/flipped_aoa_13_vw_15_H_918_Z1_Y1_X2_v5/B0001.dat"
    # )

    # Process all .dat files
    input_directory =  sys.path[0]+ "/data/aoa_13/"
    combined_dataset = process_all_dat_files(input_directory)
    logging.info(f"Combined dataset: {combined_dataset}")

    # Save the processed data
    output_file = "processed_data/combined_piv_data.nc"
    save_processed_data(combined_dataset, output_file)
    logging.info(f"Saved processed data to {output_file}")

    # Load the processed data (for future use)
    loaded_dataset = load_processed_data(output_file)
    logging.info(f"Loaded dataset: {loaded_dataset}")

    # Example of accessing data
    logging.info(f"Velocity u for first file: {loaded_dataset['Velocity u [m/s]'].isel(file=0)}")