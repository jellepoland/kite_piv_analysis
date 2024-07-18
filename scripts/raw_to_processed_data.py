import os
import numpy as np
import xarray as xr
import pandas as pd
import logging
import re
import sys
import openpyxl

print(sys.executable)


def read_xlsx_file(file_path: str) -> pd.DataFrame:
    # Convert Excel to CSV manually or using another tool
    csv_path = file_path.replace(".xlsx", ".csv")  # Example: Replace extension
    df = pd.read_csv(csv_path)
    return df


def read_dat_file(
    file_path: str, labbook_df: pd.DataFrame, aoa_value: float
) -> xr.Dataset:
    # Extract the case name and file name from the file path
    case_name = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)

    # Read the header information
    with open(file_path, "r") as file:
        header = [file.readline().strip() for _ in range(4)]

    # Extract the variable names
    variables_raw = re.findall(r'"([^"]*)"', header[1])

    # Hardcoding the variable names, to match required format
    variables_edited = [
        "x",
        "y",
        "vel_u",
        "vel_v",
        "vel_w",
        "vel_mag",
        "du_dx",
        "du_dy",
        "dv_dx",
        "dv_dy",
        "dw_dx",
        "dw_dy",
        "vorticity_w_z",
        "vorticity_mag",
        "divergence_2d",
        "swirling_strength_2d",
        "is_valid",
    ]

    # Extract the i and j values
    i_value = int(re.search(r"I=(\d+)", header[2]).group(1))
    j_value = int(re.search(r"J=(\d+)", header[2]).group(1))

    # Load the data
    data = np.genfromtxt(file_path, skip_header=4)

    # Reshape the data into matrix form (j by i)
    data_matrix = data.reshape((j_value, i_value, -1))

    # Create an xarray Dataset
    coords = {
        "y_": np.arange(j_value),
        "x_": np.arange(i_value),
    }
    data_vars = {
        var: (("y_", "x_"), data_matrix[..., idx])
        for idx, var in enumerate(variables_edited)
    }
    dataset = xr.Dataset(data_vars, coords=coords)

    # Add metadata from the .dat file
    dataset.attrs["case_name"] = case_name
    dataset.attrs["file_name"] = file_name
    dataset.attrs["i_value"] = i_value
    dataset.attrs["j_value"] = j_value
    dataset.attrs["variables_raw"] = variables_raw

    # Find the corresponding row in the lab book, only look at first 34 characters
    row = labbook_df[labbook_df["Filename"].str[:34] == case_name[:34]]
    # Check if row was found
    row_dict = row.to_dict()
    date = str(row_dict["Date"])
    row_dict["Date"] = date.replace("/", "_")

    for key, values in row_dict.items():
        values_str = str(values)
        # Step 1: Remove the outer curly braces and split by colon to separate key-value pair
        key_value_str = values_str.strip("{}").split(":")
        # Step 2: Clean up key and value strings by removing extra spaces and quotes
        value = key_value_str[1].strip().strip("'")

        if key == "aoa":
            value = float(aoa_value)
        dataset.attrs[key] = value
        logging.info(f"Adding key: {key}, value: {value}")
        if key == "Z":
            break

    # Calculate additional velocity (resultant velocity)
    dataset["vel_resultant"] = np.sqrt(
        dataset["vel_u"] ** 2 + dataset["vel_v"] ** 2 + dataset["vel_w"] ** 2
    )

    # Calculate induction velocity (subtracting mean stream)
    mean_velocity = dataset[["vel_u", "vel_v", "vel_w"]].mean(dim=["x_", "y_"])
    for comp in ["u", "v", "w"]:
        dataset[f"vel_induction_{comp}"] = (
            dataset[f"vel_{comp}"] - mean_velocity[f"vel_{comp}"]
        )

    return dataset


def process_all_dat_files(
    dat_file_path: str, lab_book_path: str, aoa_value
) -> xr.Dataset:
    lab_book = read_xlsx_file(lab_book_path)
    all_datasets = []
    logging.info(f"Processing all .dat files in directory: {dat_file_path}")
    for root, _, files in os.walk(dat_file_path):
        for file in files:
            if file.endswith("1.dat"):
                file_path = os.path.join(root, file)
                dataset = read_dat_file(file_path, lab_book, aoa_value)
                all_datasets.append(dataset)

    # Combine all datasets
    combined_dataset = xr.concat(all_datasets, dim="file")
    return combined_dataset


def save_processed_data(dataset: xr.Dataset, output_path: str):
    dataset.to_netcdf(output_path)


def load_processed_data(input_file: str) -> xr.Dataset:
    return xr.open_dataset(input_file)


if __name__ == "__main__":
    # Go back to root folder
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root_path)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Process all .dat files
    input_directory = sys.path[0] + "/data/aoa_13_smaller/"
    lab_book_path = sys.path[0] + "/data/Labook_110424_1216.xlsx"
    aoa_value = 13.0
    combined_dataset = process_all_dat_files(input_directory, lab_book_path, aoa_value)
    # logging.info(f"Combined dataset: {combined_dataset}")

    # Save the processed data
    output_path = sys.path[0] + "/processed_data/combined_piv_data.nc"
    save_processed_data(combined_dataset, output_path)
    # logging.info(f"Saved processed data to {output_path}")

    # Load the processed data (for future use)
    loaded_dataset = load_processed_data(output_path)
    # logging.info(f"Loaded dataset: {loaded_dataset}")

    # Example of accessing data
    logging.info(f"Velocity u for first file: {loaded_dataset['vel_u'].isel(file=0)}")
    logging.info(f"Angle of attack: {loaded_dataset.attrs['aoa']}")
    logging.info(f"Date: {loaded_dataset.attrs['Date']}")
    logging.info(f"Case name: {loaded_dataset.attrs['case_name']}")
    logging.info(f"File name: {loaded_dataset.attrs['file_name']}")
    logging.info(f"Number of files: {len(loaded_dataset.file)}")
    logging.info(f"Z: {loaded_dataset.attrs['Z']}")
