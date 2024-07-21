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

    ### Loading in the .dat file
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
    logging.debug(f"Data shape: {data.shape},type: {type(data)}")

    # Reshape the data into matrix form (i by j)
    data_matrix = data.reshape((i_value, j_value, -1))
    logging.info(f"data_matrix.shape: {data_matrix.shape}")
    logging.info(f"data_matrix type: {type(data_matrix)}")
    # logging.info(f"data_matrix: {data_matrix[0].shape}")
    # logging.info(f"data_matrix: {data_matrix[0][0].shape}")
    logging.info(f"variables_edited: {variables_edited}")
    # logging.info(f"data_matrix: {data_matrix[0][0]}")
    # logging.info(f" np.arange(data_matrix.shape[2]): {np.arange(data_matrix.shape[2])}")
    logging.info(
        f"data_matrix x_range: {data_matrix[:,:,0].min()} to {data_matrix[:,:,0].max()}"
    )
    logging.info(
        f"data_matrix y_range: {data_matrix[:,:,1].min()} to {data_matrix[:,:,1].max()}"
    )
    # flipping the data_matrix if is a "flipped" case
    if "flipped" in str(case_name_davis):
        logging.info(
            f"----Flipping the data_matrix, case_name_davis: {case_name_davis}"
        )
        new_data_matrix = data_matrix.copy()
        new_data_matrix = np.flip(new_data_matrix, axis=1)
        # after flipping the data points, y containing values also need a sign change
        new_data_matrix[:, :, y_index] = -new_data_matrix[:, :, y_index]

    else:
        new_data_matrix = data_matrix.copy()

    logging.info(f"data_matrix.shape: {new_data_matrix.shape}")
    logging.info(
        f"data_matrix x_range: {new_data_matrix[:,:,0].min()} to {new_data_matrix[:,:,0].max()}"
    )
    logging.info(
        f"data_matrix y_range: {new_data_matrix[:,:,1].min()} to {new_data_matrix[:,:,1].max()}"
    )
    breakpoint()

    ### Loading labbook
    labbook_dict = {}
    labbook_df = pd.read_csv(labbook_path)
    # Find the corresponding row in the lab book, only look at first 34 characters

    # checking for if dealing with flipped or normal
    if "flipped" in case_name_davis:
        untill_index = 35
    elif "normal" in case_name_davis:
        untill_index = 34
    else:
        logging.error(f"Case name not recognized: {case_name_davis}")

    row = labbook_df[
        labbook_df["file_name_labbook"].str[:untill_index]
        == case_name_davis[:untill_index]
    ]

    # Adding a boolean flag to indicate if this is the last_measurement
    if row.empty:
        labbook_dict["is_last_measurement"] = False
    else:
        logging.info(
            f"labbook: {row['file_name_labbook']}, case_name_davis: {case_name_davis}"
        )
        labbook_dict["is_last_measurement"] = True

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

        if key == "vw" and value == "":
            logging.info(f" -------------")
            logging.info(f"case_name_davis: {case_name_davis}" f"Setting vw to 15.0")
            value = 15.0
        # Convert the values to floats if they are supposed to be floats
        elif key in keys_that_are_floats and not value == "":
            logging.debug(f"case_name_davis: {case_name_davis}")
            logging.debug(f"Converting {key} to float, value: {value}")
            value = float(value)

        # appending the data to the dict
        labbook_dict[key] = value

        logging.debug(f"Adding key: {key}, value: {value}")

        # Stop after the z_plane_number key
        if key == "z_plane_number":
            break

    ### Calculating additional matrix level variables
    # resultant 3D velocity
    vel_resultant = np.sqrt(
        data_matrix[:, :, variables_edited.index("vel_u")] ** 2
        + data_matrix[:, :, variables_edited.index("vel_v")] ** 2
        + data_matrix[:, :, variables_edited.index("vel_w")] ** 2
    )
    # Add the resultant velocity to the data_matrix
    data_matrix = np.concatenate((data_matrix, vel_resultant[..., np.newaxis]), axis=2)
    variables_edited.append("vel_resultant")
    # # induction velocity in x, y, z
    # mean_velocity = np.mean(
    #     data_matrix[
    #         :, :, variables_edited.index("vel_u") : variables_edited.index("vel_w") + 1
    #     ],
    #     axis=(0, 1),
    # )
    # for comp in ["u", "v", "w"]:
    #     variable_value = (
    #         data_matrix[:, :, variables_edited.index(f"vel_{comp}")]
    #         - mean_velocity[comp]
    #     )
    #     data_matrix = np.concatenate(
    #         (data_matrix, variable_value[..., np.newaxis]), axis=2
    #     )
    #     variables_edited.append(f"vel_induction_{comp}")

    # Normalized streamwise velocity
    logging.info(f"labbook_dict['vw']: {labbook_dict['vw']}")
    ux_uinf = data_matrix[:, :, variables_edited.index("vel_u")] / labbook_dict["vw"]
    data_matrix = np.concatenate((data_matrix, ux_uinf[..., np.newaxis]), axis=2)
    variables_edited.append("ux_uinf")

    ### Creating the dataset
    # Create a single 3D DataArray to hold all variables
    data_array = xr.DataArray(
        data_matrix,
        dims=["x_i", "y_j", "variable"],
        coords={
            "x_i": np.arange(i_value),
            "y_j": np.arange(j_value),
            "variable": variables_edited,
        },
    )

    # Create the dataset with this single 3D DataArray
    dataset = xr.Dataset({"data": data_array})

    # Add plane specific information
    for key, value in labbook_dict.items():
        dataset[key] = xr.DataArray(np.array([value]), dims=["file"])

    dataset["case_name_davis"] = xr.DataArray(
        np.array([case_name_davis]), dims=["file"]
    )
    dataset["file_name_davis"] = xr.DataArray(
        np.array([file_name_davis]), dims=["file"]
    )
    ## adding additional dict key
    dataset["file_name"] = xr.DataArray(np.array([case_name_davis]), dims=["file"])

    # Add dataset wide (for whole aoa_13 sweep)
    dataset.attrs["i_value"] = i_value
    dataset.attrs["j_value"] = j_value
    dataset.attrs["k_value"] = len(variables_edited)
    dataset.attrs["aoa"] = aoa_value
    dataset.attrs["variables_edited"] = variables_edited
    dataset.attrs["variables_raw"] = variables_raw

    return dataset


def process_all_dat_files(
    dat_file_path: str, labbook_path: str, aoa_value
) -> xr.Dataset:
    all_datasets = []
    all_datasets_std = []
    logging.debug(f"Processing all .dat files in directory: {dat_file_path}")
    for root, _, files in os.walk(dat_file_path):
        for file in files:
            if file.endswith("1.dat"):
                file_path = os.path.join(root, file)
                dataset = read_dat_file(file_path, labbook_path, aoa_value)
                logging.debug(f"Processed dataset: {dataset}")
                # ONLY taking the last_measurement_values, neglecting the rest for now
                if dataset.is_last_measurement.values:
                    all_datasets.append(dataset)
            if file.endswith("2.dat"):
                file_path = os.path.join(root, file)
                dataset = read_dat_file(file_path, labbook_path, aoa_value)
                logging.debug(f"Processed dataset: {dataset}")
                # ONLY taking the last_measurement_values, neglecting the rest for now
                if dataset.is_last_measurement.values:
                    all_datasets_std.append(dataset)

    # Combine all datasets
    return xr.concat(all_datasets, dim="file"), xr.concat(all_datasets_std, dim="file")


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
    combined_dataset, combined_dataset_std = process_all_dat_files(
        input_directory, lab_book_path, aoa_value
    )
    size = combined_dataset.sizes.get("file")
    datapoint_list = [combined_dataset.isel(file=i) for i in range(size)]
    logging.info(f" ")
    for datapoint in datapoint_list:
        logging.info(f"processsed file_names: {datapoint.file_name.values}")
    # logging.info(f"Combined dataset: {combined_dataset}")

    # Save the processed data (B0001.dat)
    file_name = "combined_piv_data"
    processed_data_path = sys.path[0] + f"/processed_data/{file_name}.nc"
    combined_dataset.to_netcdf(processed_data_path)

    # Save the STANDARD DEVIATION (B0002.dat) processed data
    file_name = "combined_piv_data_std"
    processed_data_path = sys.path[0] + f"/processed_data/{file_name}.nc"
    combined_dataset_std.to_netcdf(processed_data_path)
