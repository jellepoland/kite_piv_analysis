import os
import numpy as np
import xarray as xr
import pandas as pd
import logging
import re
import sys
import openpyxl

print(sys.executable)


def read_dat_file(file_path: str, labbook_path: str, aoa_value: float) -> xr.Dataset:
    # Extract the case name and file name from the file path
    case_name_davis = os.path.basename(os.path.dirname(file_path))
    file_name_davis = os.path.basename(file_path)

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
        "vorticity_jw_z",
        "vorticity_jmag",
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

    # Create coordinates
    coords = {
        "y_j": np.arange(j_value),
        "x_i": np.arange(i_value),
    }

    # Create data variables
    variables_needing_filtering = [
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
        "vorticity_jw_z",
        "vorticity_jmag",
        "divergence_2d",
        "swirling_strength_2d",
    ]
    data_vars = {}
    for idx, var in enumerate(variables_edited):
        var_data = data_matrix[..., idx]
        # if var in variables_needing_filtering:
        #     # Replace zero values with nan for specific variables
        #     # var_data = np.where(var_data == 0, np.nan, var_data)
        data_vars[var] = (["y_j", "x_i"], var_data)

    # Create the dataset
    dataset = xr.Dataset(data_vars, coords=coords)

    # Add metadata
    dataset.attrs["case_name_davis"] = case_name_davis
    dataset.attrs["file_name_davis"] = file_name_davis
    dataset.attrs["i_value"] = i_value
    dataset.attrs["j_value"] = j_value
    dataset.attrs["variables_raw"] = variables_raw
    dataset.attrs["aoa"] = aoa_value

    ### Loading labbook
    labbook_df = pd.read_csv(labbook_path)
    # Find the corresponding row in the lab book, only look at first 34 characters
    row = labbook_df[labbook_df["file_name_labbook"].str[:35] == case_name_davis[:35]]

    # Adding a boolean flag to indicate if this is the last_measurement
    if row.empty:
        dataset["is_last_measurement"] = False
    else:
        logging.info(
            f"labbook: {row['file_name_labbook']}, case_name_davis: {case_name_davis}"
        )
        dataset["is_last_measurement"] = True

    # Check if row was found
    row_dict = row.to_dict()
    date = str(row_dict["date"])
    row_dict["date"] = date.replace("/", "_")

    for key, values in row_dict.items():
        values_str = str(values)
        # Step 1: Remove the outer curly braces and split by colon to separate key-value pair
        key_jvalue_str = values_str.strip("{}").split(":")
        logging.debug(f"key: {key}")
        logging.debug(f"values: {values}")
        logging.debug(
            f"key_jvalue_str: {key_jvalue_str}, len(key_jvalue_str): {len(key_jvalue_str)}"
        )
        # Step 2: Clean up key and value strings by removing extra spaces and quotes
        if len(key_jvalue_str) > 1:
            value = str(key_jvalue_str[1].strip().strip("'"))
        else:
            value = ""

        keys_that_are_floats = [
            "vw",
            "vw_set",
            "dpa",
            "pressure",
            "temp",
            "density",
            "h_table",
            "y_traverse",
            "x_traverse",
            "x_plane_number",
            "y_plane_number",
            "z_plane_number",
        ]
        # Add the aoa value, which is the same for each run
        if key == "aoa":
            value = float(aoa_value)
        # Convert the values to floats if they are supposed to be floats
        elif key in keys_that_are_floats:
            value = float(value)

        # appending the data to the dataset
        dataset[key] = xr.DataArray(np.array([value]), dims=["file"])

        logging.debug(f"Adding key: {key}, value: {value}")

        # Stop after the z_plane_number key
        if key == "z_plane_number":
            break

    # Calculate additional velocity (resultant velocity
    dataset["vel_resultant"] = np.sqrt(
        dataset["vel_u"] ** 2 + dataset["vel_v"] ** 2 + dataset["vel_w"] ** 2
    )

    # Calculate induction velocity (subtracting mean stream)
    mean_velocity = dataset[["vel_u", "vel_v", "vel_w"]].mean(dim=["y_j", "x_i"])
    for comp in ["u", "v", "w"]:
        dataset[f"vel_induction_{comp}"] = (
            dataset[f"vel_{comp}"] - mean_velocity[f"vel_{comp}"]
        )

    # Calculate Ux_iUinf
    dataset["Ux_Uinf"] = dataset["vel_u"] / dataset["vw"]

    return dataset


def process_all_dat_files(
    dat_file_path: str, labbook_path: str, aoa_value
) -> xr.Dataset:
    all_datasets = []
    logging.info(f"Processing all .dat files in directory: {dat_file_path}")
    for root, _, files in os.walk(dat_file_path):
        for file in files:
            if file.endswith("1.dat"):
                file_path = os.path.join(root, file)
                dataset = read_dat_file(file_path, labbook_path, aoa_value)
                # ONLY taking the last_measurement_values, neglecting the rest for now
                if dataset.is_last_measurement.values:
                    all_datasets.append(dataset)

    # Combine all datasets
    combined_dataset = xr.concat(all_datasets, dim="file")
    return combined_dataset


if __name__ == "__main__":
    # Go back to root folder
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root_path)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    ### Cleaning up the labbook notes:
    # The data was move to the top, to easier read out
    # Header names were adjusted to:
    # header_names = [
    #     "vw",
    #     "vw_set",
    #     "dpa",
    #     "pressure",
    #     "temp",
    #     "density",
    #     "h_table",
    #     "y_traverse",
    #     "x_traverse",
    #     "x_plane_number",
    #     "y_plane_number",
    #     "z_plane_number",
    # ]
    # REDONE_               was added to when measurement was redone
    # YAW_MISLAGINMENT_     was added to when measurement was redone, due to yaw misalignment (new flipped.._v5 cases)
    # row 25                Comment to change Davis Filenames
    # row 118               Comment Z3 to Z2 was corrected
    # row 140               Added a line of X's to separate the different measurement sets

    # Process all .dat files
    input_directory = sys.path[0] + "/data/aoa_13_test/"
    lab_book_path = sys.path[0] + "/data/labbook_cleaned.csv"
    aoa_value = 13.0
    combined_dataset = process_all_dat_files(input_directory, lab_book_path, aoa_value)
    # logging.info(f"Combined dataset: {combined_dataset}")

    # Save the processed data
    processed_data_path = sys.path[0] + "/processed_data/combined_piv_data.nc"
    combined_dataset.to_netcdf(processed_data_path)
