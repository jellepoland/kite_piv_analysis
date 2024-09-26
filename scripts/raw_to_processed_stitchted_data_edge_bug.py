import os
import numpy as np
import xarray as xr
import pandas as pd
import logging
import re
import sys
import openpyxl
from copy import deepcopy
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.ticker as mticker
import pickle


def read_single_dat_file(
    file_path: str,
    labbook_path: str,
    aoa_value: float,
    variables_edited,
):
    # Extract the case name and file name from the file path
    case_name_davis = os.path.basename(os.path.dirname(file_path))
    file_name_davis = os.path.basename(file_path)
    logging.debug(f"case_name_davis: {case_name_davis}")

    ### Loading in the .dat file
    # Read the header information
    with open(file_path, "r") as file:
        header = [file.readline().strip() for _ in range(4)]

    # Extract the variable names
    variables_raw = re.findall(r'"([^"]*)"', header[1])

    # Extract the i and j values
    i_value = int(re.search(r"I=(\d+)", header[2]).group(1))
    j_value = int(re.search(r"J=(\d+)", header[2]).group(1))

    # Load the data
    data = np.genfromtxt(file_path, skip_header=4)
    # data_matrix = data.reshape((i_value, j_value, -1))

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
        logging.debug(
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
            logging.debug(f" -------------")
            logging.debug(f"case_name_davis: {case_name_davis}" f"Setting vw to 15.0")
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

    if "X1" in case_name_davis:
        x_traverse = 0
    elif "X2" in case_name_davis:
        x_traverse = -300
    elif "X3" in case_name_davis:
        x_traverse = -600

    if "Z1" in case_name_davis:
        y_traverse = 200
    elif "Z2" in case_name_davis:
        y_traverse = 100
    elif "Z3" in case_name_davis:
        y_traverse = 80

    # David Part
    x = data[:, 0] - x_traverse
    y = data[:, 1]

    # calculate resolutions mm per point
    x_range = np.max(x) - np.min(x)
    num_points_x = len(np.unique(x))
    num_points_x = 182
    x_resolution = x_range / (num_points_x)
    y_range = np.max(y) - np.min(y)
    num_points_y = len(np.unique(y))
    num_points_y = 207
    y_resolution = y_range / (num_points_y)
    logging.debug(f"{x_range}, {num_points_y / y_range}")
    logging.debug(f"x_resolution: {x_resolution}")
    logging.debug(f"y_resolution: {y_resolution}")

    # travering the y-values when needed
    if "flipped" in case_name_davis:
        y = -y
        y += y_traverse
        variables_needing_flipping = ["vel_v", "du_dy", "dv_dx", "dw_dy"]
    else:
        variables_needing_flipping = []

    # grab the global mesh grid
    x_meshgrid_global, y_meshgrid_global = define_global_mesh_grid()

    sign = 1
    output = [case_name_davis]
    for k in range(2, len(variables_edited)):
        if variables_edited[k] in variables_needing_flipping:
            sign = -1
            logging.debug(f"flipped: {variables_edited[k]}")
        else:
            logging.debug(f"Not flipped var: {variables_edited[k]}")
            sign = 1

        ##TODO: recalculating vorticity_jw_z
        if variables_edited[k] == "vorticity_jw_z":
            # w_z = dv_dx - du_dy
            vorticity_before = data[:, k]
            data[:, k] = data[:, 8] - data[:, 7]
            vorticity_after = data[:, k]
            logging.debug(f"difference: {np.max(vorticity_before - vorticity_after)}")

        data_k = griddata(
            np.array([x, y]).T,
            sign * np.array(data[:, k]),
            (x_meshgrid_global, y_meshgrid_global),
            method="linear",
        )
        # turning nans into 0s
        data_k = np.nan_to_num(data_k)

        logging.debug(f"mean of data_k {np.max(data_k)}, var: {variables_edited[k]}")
        # Correcting for density shifts
        if k < len(variables_edited) - 1:
            logging.debug(
                f"---density correction --- k: {k}, var: {variables_edited[k]}"
            )
            data_k = data_k * labbook_dict["density"] / 1.225
        logging.debug(
            f"--post-- mean of data_k {np.max(data_k)}, var: {variables_edited[k]}"
        )

        # appending to output
        output.append(data_k)

    # logging
    logging.debug(f"x-range: {np.min(x)}, {np.max(x)}")
    logging.debug(f"y-range: {np.min(y)}, {np.max(y)}")
    print(f" ")

    return output


def define_global_mesh_grid():
    # the x and y ranges are arbitrarly set, and resolution is calculated in the beginning
    x_global = np.arange(-210, 840, 2.4810164835164836)
    y_global = np.arange(-205, 405, 2.4810164835164836)
    x_meshgrid_global, y_meshgrid_global = np.meshgrid(x_global, y_global)
    return x_meshgrid_global, y_meshgrid_global


def get_y_grouped_filtered_data(output_list):
    # grouping the data on spanwise position (y-values)
    y1_data_list = ["Y1"]
    y2_data_list = ["Y2"]
    y3_data_list = ["Y3"]
    y4_data_list = ["Y4"]
    y5_data_list = ["Y5"]
    y6_data_list = ["Y6"]
    y7_data_list = ["Y7"]
    for output in output_list:
        case_name_davis = str(output[0])
        logging.info(f"--- case_name_davis: {case_name_davis}")
        if "Y1" in case_name_davis:
            y1_data_list.append(output[:])
        elif "Y2" in case_name_davis:
            y2_data_list.append(output[:])
        elif "Y3" in case_name_davis:
            y3_data_list.append(output[:])
        elif "Y4" in case_name_davis:
            y4_data_list.append(output[:])
        elif "Y5" in case_name_davis:
            y5_data_list.append(output[:])
        elif "Y6" in case_name_davis:
            y6_data_list.append(output[:])
        elif "Y7" in case_name_davis:
            y7_data_list.append(output[:])

    y_grouped_data = [
        y1_data_list,
        y2_data_list,
        y3_data_list,
        y4_data_list,
        y5_data_list,
        y6_data_list,
        y7_data_list,
    ]
    y_grouped_filtered_data = []
    for all_y_data in y_grouped_data:
        if len(all_y_data) > 1:
            y_grouped_filtered_data.append(all_y_data)
    return y_grouped_filtered_data


def process_planes(input_directory, lab_book_path, aoa_value, variables_edited):

    # loop through all the files in the input_directory, and read the 6 planes
    output_list = []
    output_list_dat2 = []

    for root, _, files in os.walk(input_directory):
        for file in files:
            # Only consider normal values, and not standard deviation
            if file.endswith("1.dat"):
                print(f"Processing file: {file}, dir: {root}")
                file_path = os.path.join(root, file)
                output = read_single_dat_file(
                    file_path,
                    lab_book_path,
                    aoa_value,
                    variables_edited,
                )
                output_list.append(output)
            if file.endswith("2.dat"):
                print(f"Processing file: {file}, dir: {root}")
                file_path = os.path.join(root, file)
                output = read_single_dat_file(
                    file_path,
                    lab_book_path,
                    aoa_value,
                    variables_edited,
                )
                output_list_dat2.append(output)
    return output_list, output_list_dat2


def stitch_planes(
    y_grouped_filtered_data,
    y_grouped_filtered_data_dat2,
    variables_edited,
    save_processed_folder,
):

    # averaging the data
    index_of_is_valid = variables_edited.index("is_valid") - 2
    print(f"index_of_is_valid:{index_of_is_valid}")
    print(f"len(y_grouped_filtered_data): {len(y_grouped_filtered_data)}")
    print(f"len(y_grouped_filtered_data_dat2): {len(y_grouped_filtered_data_dat2)}")
    for y_index, all_y_data in enumerate(y_grouped_filtered_data):

        y_num = all_y_data[0]
        all_6_plane_data = all_y_data[1:]
        k_num = len(all_6_plane_data[0][1:])
        variables_processed = variables_edited[2 : 2 + k_num]
        all_6_plane_data_mean_overlap_k_list = []

        # logging
        logging.info(f"all_y_data len: {len(all_y_data)}")
        logging.info(f"y_num: {y_num}")
        logging.info(f"all_6_plane_data len: {len(all_6_plane_data)}")
        logging.info(f"k_num: {k_num}")
        logging.info(f"variables_processed: {variables_processed}")

        # dealing with 2nd dat
        all_6_plane_data_dat2 = y_grouped_filtered_data_dat2[y_index][1:]
        logging.info(f"--> len(all_6_plane_data): {len(all_6_plane_data)}")
        logging.info(f"--> len(all_6_plane_data_dat2): {len(all_6_plane_data_dat2)}")

        # looping through each variable k (that was present in the .dat file)
        for k in range(k_num):
            sum_top = 0
            n_valid_points = 0
            print(f"\n --- processing variable: {variables_processed[k]}, k: {k} --- ")
            # loop through each plane of data
            for plane_idx, (plane_data, plane_data_dat2) in enumerate(
                zip(all_6_plane_data, all_6_plane_data_dat2)
            ):
                case_name = plane_data[0]
                plane_data_filtered = plane_data[1:]
                data_k = plane_data_filtered[k]

                # for row in data_k:
                #     print(f"row: {row}")

                # Use the is_valid from .dat file to mask the data
                #   The is_valid column tells you whether a particular data point is valid or not (likely represented by non-zero values).
                #   the astype int, converts the boolean mask (True/False) to an integer mask (1 for valid data, 0 for invalid data).
                mask = (plane_data_filtered[index_of_is_valid] != 0).astype(int)

                # Using the standard deviations present in data_k_dat2 to further mask the data
                plane_data_dat2_filtered = plane_data_dat2[1:]
                mask = mask * (plane_data_dat2_filtered[k] < 1).astype(int)
                # if variables_processed[k] == "vel_u":
                #     mask = mask * (data_k[k] < 10).astype(int)
                # elif variables_processed[k] == "vel_v":
                #     mask = mask * (data_k[k] < 1).astype(int)

                # if k < len(plane_data_dat2_filtered):
                #     print(f"--> plane_data_dat2: {plane_data_dat2_filtered[k]}")

                # Use the mask, to find only that data that is valid
                #   For valid data points (where mask == 1), multiply the data_k values by the mask.
                #   This ensures that only valid data contributes to the summation.
                #   sum_top will be the sum of all VALID data points across all planes.
                sum_top = sum_top + data_k * mask

                # Use the mask to sum the number of valid data points
                #   The mask (which contains 1s for valid points and 0s for invalid points) is summed,
                #   so sum_bottom will contain the total number of valid data points for each position
                #   the number of valid data points will be either 0,1,2,3,4 (max overlap)
                #   This tracks how many planes have valid data at each data point (used later for averaging).
                n_valid_points += mask

                # logging
                logging.info(f"plane_idx: {plane_idx}, k:{k}")
                # logging.info(f"case_name: {case_name}")
                # logging.info(f"plane_data shape: {np.array(plane_data_filtered).shape}")
                # logging.info(f"data_k shape: {data_k.shape}")

            # calculting mean, when is not the is_valid variable (or column)
            if k != index_of_is_valid:
                # print(f" --- handling stichting for k: {k} ---")
                # Avoid division by zero, by setting all zeros to 1
                #   this is not a problem as the sum top will be zero where sum_bottom is zero
                #   Replacing zeros in sum_bottom with nan values
                # n_valid_points = np.where(n_valid_points == 0, np.nan, n_valid_points)

                # Calculate the mean overlap / average value
                #   valid data divided by the number of valid data points
                plane_data_mean_overlap = sum_top / n_valid_points
                # plane_data_mean_overlap = sum_top

                # #### NEW CODE ####
                # from scipy.interpolate import NearestNDInterpolator
                # from scipy.spatial import cKDTree
                # import numpy as np

                # def nearest_neighbor_interpolation(
                #     x_mesh, y_mesh, z_values, valid_mask, num_neighbors=10
                # ):
                #     """
                #     Perform nearest neighbor interpolation on the grid.
                #     x_mesh, y_mesh: Meshgrid points (2D arrays)
                #     z_values: Values at the points (2D array)
                #     valid_mask: Boolean mask where values are valid (2D array)
                #     """
                #     # Flatten the meshgrid and values arrays
                #     x_flat = x_mesh.flatten()
                #     y_flat = y_mesh.flatten()
                #     z_flat = z_values.flatten()
                #     valid_mask_flat = valid_mask.flatten()

                #     # Get the valid points (x, y) and their corresponding values
                #     valid_points = np.column_stack(
                #         (x_flat[valid_mask_flat], y_flat[valid_mask_flat])
                #     )  # shape (n_valid_points, 2)
                #     valid_values = z_flat[valid_mask_flat]  # shape (n_valid_points,)

                #     # Build a k-d tree for fast neighbor lookup
                #     tree = cKDTree(valid_points)

                #     # Prepare for interpolating locally
                #     interpolated_values = (
                #         z_flat.copy()
                #     )  # Initialize with existing values

                #     # Loop over all points where valid_mask is False (i.e., where we need interpolation)
                #     for idx in np.where(~valid_mask_flat)[
                #         0
                #     ]:  # Iterate over indices where valid_mask_flat is False
                #         point = np.array([x_flat[idx], y_flat[idx]])

                #         # Find the `num_neighbors` nearest valid points to this point
                #         distances, neighbors_idx = tree.query(point, k=num_neighbors)

                #         # Get the values at these neighbors
                #         neighbor_points = valid_points[neighbors_idx]
                #         neighbor_values = valid_values[neighbors_idx]

                #         # Perform nearest neighbor interpolation using the `num_neighbors` closest points
                #         interpolator = NearestNDInterpolator(
                #             neighbor_points, neighbor_values
                #         )
                #         interpolated_values[idx] = interpolator(point)

                #     # Reshape interpolated values back to the original grid shape
                #     interpolated_values = interpolated_values.reshape(x_mesh.shape)

                #     return interpolated_values

                # # Initialize final mean overlap result
                # plane_data_mean_overlap = np.zeros_like(sum_top)

                # # Define mask for where n_valid_points is zero (i.e., no valid points)
                # mask_zero = n_valid_points == 0
                # plane_data_mean_overlap[mask_zero] = (
                #     np.nan
                # )  # Set to NaN or another flag for invalid data

                # # Define mask for where n_valid_points is 1 (i.e., only one valid point)
                # mask_one = n_valid_points == 1
                # plane_data_mean_overlap[mask_one] = sum_top[
                #     mask_one
                # ]  # Take the sum_top directly (since only one valid point)

                # # Define mask for where n_valid_points is > 1 (i.e., multiple valid points)
                # mask_more_than_one = (n_valid_points > 1).astype(int)

                # # Apply nearest neighbor interpolation for areas where n_valid_points > 1
                # if np.any(mask_more_than_one):
                #     x_meshgrid_global, y_meshgrid_global = define_global_mesh_grid()
                #     # Get the (x, y) points where the mask is True
                #     x_valid = x_meshgrid_global[mask_more_than_one]
                #     y_valid = y_meshgrid_global[mask_more_than_one]

                #     # Perform interpolation at these points only
                #     interpolated_values = nearest_neighbor_interpolation(
                #         x_meshgrid_global,
                #         y_meshgrid_global,
                #         sum_top,
                #         mask_more_than_one,
                #     )

                #     # Assign the interpolated values only to the masked locations
                #     plane_data_mean_overlap[mask_more_than_one] = interpolated_values[
                #         mask_more_than_one
                #     ]

                # ### ABOVE IS NEW CODE ###

            # if is_valid column, don't take mean, simply take the is_valid counter
            else:
                plane_data_mean_overlap = n_valid_points
            logging.debug(
                f"plane_data_mean_overlap: {plane_data_mean_overlap.shape}, max: {np.max(plane_data_mean_overlap)}, min: {np.min(plane_data_mean_overlap)}"
            )

            all_6_plane_data_mean_overlap_k_list.append(plane_data_mean_overlap)

        logging.debug(
            f"all_6_plane_data_mean_overlap_k_list: {np.array(all_6_plane_data_mean_overlap_k_list).shape}"
        )

        # Saving the data
        x_meshgrid_global, y_meshgrid_global = define_global_mesh_grid()

        save_path = save_processed_folder + f"{y_num}.csv"
        data_dict = {"x": x_meshgrid_global.flatten(), "y": y_meshgrid_global.flatten()}
        # Add each MEANED variable's data to the dictionary
        for data, var_name in zip(
            all_6_plane_data_mean_overlap_k_list, variables_processed
        ):
            data_dict[var_name] = data.flatten()

        # Create a DataFrame and save as CSV
        df = pd.DataFrame(data_dict)
        df.to_csv(save_path, index=False)

    return df


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

    input_directory = sys.path[0] + "/data/y2_testing_for_edge_bug/"
    lab_book_path = sys.path[0] + "/data/labbook_cleaned.csv"
    save_processed_folder = sys.path[0] + "/processed_data/y2_testing_for_edge_bug/"
    save_plots_folder = sys.path[0] + "/results/"
    aoa_value = 13

    # # Process the data
    # output_list, output_list_dat2 = process_planes(
    #     input_directory, lab_book_path, aoa_value, variables_edited
    # )
    # # Group the data on y-values
    # y_grouped_filtered_data = get_y_grouped_filtered_data(output_list)
    # y_grouped_filtered_data_dat2 = get_y_grouped_filtered_data(output_list_dat2)

    # # save the data using pickle
    # with open(save_processed_folder + "y_grouped_filtered_data.pkl", "wb") as f:
    #     pickle.dump(y_grouped_filtered_data, f)
    # with open(save_processed_folder + "y_grouped_filtered_data_dat2.pkl", "wb") as f:
    #     pickle.dump(y_grouped_filtered_data_dat2, f)

    # load the data using pickle
    with open(save_processed_folder + "y_grouped_filtered_data.pkl", "rb") as f:
        y_grouped_filtered_data = pickle.load(f)
    with open(save_processed_folder + "y_grouped_filtered_data_dat2.pkl", "rb") as f:
        y_grouped_filtered_data_dat2 = pickle.load(f)

    # Stich the data together
    df_stichted = stitch_planes(
        y_grouped_filtered_data,
        y_grouped_filtered_data_dat2,
        variables_edited,
        save_processed_folder,
    )
    # # # Load the processed data
    # loaded_data = pd.read_csv(save_processed_folder + f"Y2.csv")
    # # loaded_data = pd.read_csv(save_processed_folder + f"Y2.csv")
    # logging.info(f"loaded_data:{loaded_data}")
    loaded_data = df_stichted

    x_meshgrid_global, y_meshgrid_global = define_global_mesh_grid()

    vel_u_on_meshgrid_global = griddata(
        np.array([x_meshgrid_global.flatten(), y_meshgrid_global.flatten()]).T,
        np.array(loaded_data["vel_u"].values),
        (x_meshgrid_global, y_meshgrid_global),
        method="linear",
    )
    vel_v_on_meshgrid_global = griddata(
        np.array([x_meshgrid_global.flatten(), y_meshgrid_global.flatten()]).T,
        np.array(loaded_data["vel_v"].values),
        (x_meshgrid_global, y_meshgrid_global),
        method="linear",
    )

    from plotting_single_variable import saving_a_plot

    saving_a_plot(
        x_meshgrid_global=x_meshgrid_global,
        y_meshgrid_global=y_meshgrid_global,
        u_for_quiver=vel_u_on_meshgrid_global,
        v_for_quiver=vel_v_on_meshgrid_global,
        color_data=vel_u_on_meshgrid_global / 15.0,
        save_plots_folder=save_plots_folder,
        plot_type=".pdf",
        min_cbar_value=0.75,
        max_cbar_value=1.25,
        max_mask_value=5,  # 1.75,
        min_mask_value=0.05,  # 0.75,
        title="y2_testing_for_edge_bug",
        cmap="viridis",
    )
    plt.show()
