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


def read_dat_file(
    file_path: str,
    labbook_path: str,
    aoa_value: float,
    x_meshgrid_global,
    y_meshgrid_global,
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

        logging.info(f"Adding key: {key}, value: {value}")

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
    logging.info(f"{x_range}, {num_points_y / y_range}")
    logging.info(f"x_resolution: {x_resolution}")
    logging.info(f"y_resolution: {y_resolution}")

    # travering the y-values when needed
    if "flipped" in case_name_davis:
        y = -y
        y += y_traverse
        variables_needing_flipping = ["vel_v", "du_dy", "dv_dx", "dw_dy"]
    else:
        variables_needing_flipping = []

    sign = 1
    output = [case_name_davis]
    for k in range(2, len(variables_edited)):
        if variables_edited[k] in variables_needing_flipping:
            sign = -1
            logging.info(f"flipped: {variables_edited[k]}")
        else:
            logging.info(f"Not flipped var: {variables_edited[k]}")
            sign = 1

        ##TODO: recalculating vorticity_jw_z
        if variables_edited[k] == "vorticity_jw_z":
            # w_z = dv_dx - du_dy
            vorticity_before = data[:, k]
            data[:, k] = data[:, 8] - data[:, 7]
            vorticity_after = data[:, k]
            print(f"difference: {np.max(vorticity_before - vorticity_after)}")

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
    logging.info(f"x-range: {np.min(x)}, {np.max(x)}")
    logging.info(f"y-range: {np.min(y)}, {np.max(y)}")
    print(f" ")

    return output


# def save_meshgrid_data_csv(all_data, variable_names, save_path):
#     # Create X and Y coordinates based on the shape of the data
#     y, x = np.mgrid[0 : all_data[0].shape[0], 0 : all_data[0].shape[1]]

#     # Create a dictionary to hold all the data
#     data_dict = {"X": x.flatten(), "Y": y.flatten()}

#     # Add each variable's data to the dictionary
#     for data, var_name in zip(all_data, variable_names):
#         data_dict[var_name] = data.flatten()

#     # Create a DataFrame and save as CSV
#     df = pd.DataFrame(data_dict)
#     df.to_csv(save_path, index=False)

#     print(f"Data saved to {save_path}")


def process_all_dat_files(
    variables_edited,
    input_directory,
    lab_book_path,
    save_processed_folder,
    save_plots_folder,
    aoa_value,
    vw,
    plot_type=".png",
    min_cbar_value=0.75,
    max_cbar_value=1.25,
):

    logging.info(f"Processing all .dat files in directory: {input_directory}")

    # the x and y ranges are arbitrarly set, and resolution is calculated in the beginning
    x_global = np.arange(-210, 840, 2.4810164835164836)
    y_global = np.arange(-205, 405, 2.4810164835164836)
    x_meshgrid_global, y_meshgrid_global = np.meshgrid(x_global, y_global)

    output_list = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            # Only consider normal values, and not standard deviation
            if file.endswith("1.dat"):
                print(f"Processing file: {file}")
                file_path = os.path.join(root, file)
                output = read_dat_file(
                    file_path,
                    lab_book_path,
                    aoa_value,
                    x_meshgrid_global,
                    y_meshgrid_global,
                    variables_edited,
                )
                output_list.append(output)

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

    # averaging the data
    index_of_is_valid = variables_edited.index("is_valid") - 2
    print(f"index_of_is_valid:{index_of_is_valid}")
    for all_y_data in y_grouped_filtered_data:
        logging.info(f"all_y_data len: {len(all_y_data)}")
        y_num = all_y_data[0]
        logging.info(f"y_num: {y_num}")
        all_6_plane_data = all_y_data[1:]
        logging.info(f"all_6_plane_data len: {len(all_6_plane_data)}")

        k_num = len(all_6_plane_data[0][1:])
        variables_processed = variables_edited[2 : 2 + k_num]
        logging.info(f"k_num: {k_num}")
        logging.info(f"variables_processed: {variables_processed}")

        all_6_plane_data_mean_overlap_k_list = []
        sum_bottom_k_list = []
        for k in range(k_num):
            sum_top = 0
            sum_bottom = 0
            for plane_idx, plane_data in enumerate(all_6_plane_data):
                logging.info(f"plane_idx: {plane_idx}, k:{k}")
                case_name = plane_data[0]
                logging.info(f"case_name: {case_name}")
                plane_data = plane_data[1:]
                logging.info(f"plane_data shape: {np.array(plane_data).shape}")
                data_k = plane_data[k]
                logging.info(f"data_k shape: {data_k.shape}")
                # Use the is_valid from .dat file to mask the data
                # the astype int, ensure that it uses 0 and 1 as values
                mask = (plane_data[index_of_is_valid] != 0).astype(int)
                # Use the mask, to find only that data that is valid
                sum_top = sum_top + data_k * mask
                # Use the mask to sum the number of valid data points
                # each datapoint will thus be either 0,1,2,3,4 (max overlap)
                sum_bottom = sum_bottom + mask

                # new_data_k = data_k * mask
                # # find where the data overlaps
                # np.where(sum_bottom == 2, , 0)

            print(f"      --- variable: {variables_processed[k]} --- ")
            # calculting mean, when not the is_valid column
            if k != index_of_is_valid:
                print(f"-- handling stichting for k: {k} --")
                # Avoid division by zero, by setting all zeros to 1
                # this is not a problem as the sum top will be zero where sum_bottom is zero

                # TODO: this is an alternative approach
                # would allow you to smoothen things better
                # plane_data_mean_overlap = np.zeros(sum_bottom.shape)
                # for i in range(sum_bottom.shape[0]):
                #     for j in range(sum_bottom.shape[1]):
                #         if sum_bottom[i, j] == 0:
                #             plane_data_mean_overlap[i, j] = np.nan
                #         elif sum_bottom[i, j] == 1:
                #             plane_data_mean_overlap[i, j] = sum_top[i, j]
                #         elif sum_bottom[i, j] == 2:
                #             plane_data_mean_overlap[i, j] = sum_top[i, j] / 2
                #         elif sum_bottom[i, j] == 3:
                #             plane_data_mean_overlap[i, j] = sum_top[i, j] / 3
                #         elif sum_bottom[i, j] == 4:
                #             plane_data_mean_overlap[i, j] = sum_top[i, j] / 4
                #         else:
                #             raise ValueError(
                #                 f"to many datapoints here sum_bottom[i, j]: {sum_bottom[i, j]}"
                #             )

                # TODO: FAILED , to memory intensive attempt at radial basis function interpolation
                # def radialbasisfunction_interpolation(sum_top, sum_bottom):
                #     from scipy.interpolate import Rbf

                #     # Determine the dimensions of the data
                #     rows, cols = sum_bottom.shape

                #     # Create grid coordinates
                #     x, y = np.indices((rows, cols))

                #     # Flatten the arrays and select valid points
                #     x_flat = x.flatten()
                #     y_flat = y.flatten()
                #     sum_top_flat = sum_top.flatten()
                #     sum_bottom_flat = sum_bottom.flatten()

                #     # Identify valid points where sum_bottom is not zero
                #     valid_indices = sum_bottom_flat > 0
                #     x_valid = x_flat[valid_indices]
                #     y_valid = y_flat[valid_indices]
                #     z_valid = (
                #         sum_top_flat[valid_indices] / sum_bottom_flat[valid_indices]
                #     )

                #     # Create the RBF interpolator
                #     rbf = Rbf(x_valid, y_valid, z_valid, function="linear")

                #     # Generate the grid of coordinates for interpolation
                #     xi, yi = np.indices((rows, cols))

                #     # Interpolate the data over the full grid
                #     plane_data_mean_overlap = rbf(xi, yi)

                #     # plane_data_mean_overlap now contains the interpolated values

                #     # To handle the areas where sum_bottom was zero, you may want to keep these as NaN or some default value.
                #     plane_data_mean_overlap[sum_bottom == 0] = np.nan

                #     return plane_data_mean_overlap

                # plane_data_mean_overlap = radialbasisfunction_interpolation(
                #     sum_top, sum_bottom
                # )

                # TODO: attempt at nearest neighbour interpolation
                # def nearestneighbour_interpolation(sum_bottom, sum_top):
                #     # Determine the dimensions of the data
                #     rows, cols = sum_bottom.shape

                #     # Create grid coordinates
                #     x, y = np.indices((rows, cols))

                #     # Flatten the arrays and select valid points
                #     x_flat = x.flatten()
                #     y_flat = y.flatten()
                #     sum_top_flat = sum_top.flatten()
                #     sum_bottom_flat = sum_bottom.flatten()

                #     # Identify valid points where sum_bottom is not zero
                #     valid_indices = sum_bottom_flat > 0
                #     points = np.column_stack(
                #         (x_flat[valid_indices], y_flat[valid_indices])
                #     )
                #     values = (
                #         sum_top_flat[valid_indices] / sum_bottom_flat[valid_indices]
                #     )

                #     # Interpolation grid
                #     xi, yi = np.indices((rows, cols))

                #     # Perform Nearest Neighbour interpolation
                #     plane_data_mean_overlap = griddata(
                #         points, values, (xi, yi), method="nearest"
                #     )

                #     # Set areas where there was no valid data to NaN
                #     plane_data_mean_overlap[sum_bottom == 0] = np.nan
                #     return plane_data_mean_overlap

                # plane_data_mean_overlap = nearestneighbour_interpolation(
                #     sum_bottom, sum_top
                # )

                # print(f"shape bottom: {sum_bottom.shape}, shape top: {sum_top.shape}")
                sum_bottom = np.where(sum_bottom == 0, np.nan, sum_bottom)
                plane_data_mean_overlap = sum_top / sum_bottom
            # if is_valid column, don't take mean, simply take the is_valid counter
            else:
                plane_data_mean_overlap = sum_bottom
            logging.info(
                f"plane_data_mean_overlap: {plane_data_mean_overlap.shape}, max: {np.max(plane_data_mean_overlap)}, min: {np.min(plane_data_mean_overlap)}"
            )
            sum_bottom_k_list.append(sum_bottom)
            all_6_plane_data_mean_overlap_k_list.append(plane_data_mean_overlap)

        logging.info(
            f"all_6_plane_data_mean_overlap_k_list: {np.array(all_6_plane_data_mean_overlap_k_list).shape}"
        )

        # Saving the data
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

    lab_book_path = sys.path[0] + "/data/labbook_cleaned.csv"
    save_processed_folder = sys.path[0] + "/processed_data/"
    save_plots_folder = sys.path[0] + "/results/"

    # Process all .dat files
    process_all_dat_files(
        variables_edited,
        input_directory=sys.path[0] + "/data/y3/",
        lab_book_path=sys.path[0] + "/data/labbook_cleaned.csv",
        save_processed_folder=sys.path[0] + "/processed_data/",
        # save_processed_folder=sys.path[0] + "/processed_data/",
        save_plots_folder=sys.path[0] + "/results/aoa_13/all_planes/",
        aoa_value=13.0,
        vw=15.0,
        min_cbar_value=5,
        max_cbar_value=20,
        plot_type=".png",
    )

    # # Load the processed data
    # loaded_data = pd.read_csv(
    #     save_processed_folder + f"/y1_testing_for_edge_bug/Y1.csv"
    # )
    # logging.info(f"loaded_data:{loaded_data}")
    # x_global = np.arange(-210, 840, 2.4810164835164836)
    # y_global = np.arange(-205, 405, 2.4810164835164836)
    # x_meshgrid_global, y_meshgrid_global = np.meshgrid(x_global, y_global)

    # vel_u_on_meshgrid_global = griddata(
    #     np.array([x_meshgrid_global.flatten(), y_meshgrid_global.flatten()]).T,
    #     np.array(loaded_data["vel_u"].values),
    #     (x_meshgrid_global, y_meshgrid_global),
    #     method="linear",
    # )
    # vel_v_on_meshgrid_global = griddata(
    #     np.array([x_meshgrid_global.flatten(), y_meshgrid_global.flatten()]).T,
    #     np.array(loaded_data["vel_v"].values),
    #     (x_meshgrid_global, y_meshgrid_global),
    #     method="linear",
    # )

    # from plotting_single_variable import saving_a_plot

    # saving_a_plot(
    #     x_meshgrid_global=x_meshgrid_global,
    #     y_meshgrid_global=y_meshgrid_global,
    #     u_for_quiver=vel_u_on_meshgrid_global,
    #     v_for_quiver=vel_v_on_meshgrid_global,
    #     color_data=vel_u_on_meshgrid_global / 15.0,
    #     save_plots_folder=save_plots_folder,
    #     plot_type=".pdf",
    #     min_cbar_value=0.75,
    #     max_cbar_value=1.25,
    #     max_mask_value=1.75,
    #     min_mask_value=0.75,
    #     title="y1_testing_for_edge_bug",
    # )
