import os
import numpy as np
import xarray as xr
import pandas as pd
import logging
import re
import sys
import openpyxl
from copy import deepcopy
from plotting import plot_quiver
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
    # travering the y-values when needed
    if "flipped" in case_name_davis:
        y = -y
        y += y_traverse

    u_x_mesh_global = griddata(
        np.array([x, y]).T,
        np.array(data[:, 2]),
        (x_meshgrid_global, y_meshgrid_global),
        method="linear",
    )
    u_x_mesh_global = np.nan_to_num(u_x_mesh_global, nan=0)

    # logging
    logging.info(f"x-range: {np.min(x)}, {np.max(x)}")
    logging.info(f"y-range: {np.min(y)}, {np.max(y)}")
    print(f" ")
    output = [case_name_davis, u_x_mesh_global]
    return output


def process_all_dat_files(
    input_directory,
    lab_book_path,
    save_plots_folder,
    aoa_value,
    vw,
    plot_type=".png",
    min_cbar_value=0.75,
    max_cbar_value=1.25,
):

    logging.info(f"Processing all .dat files in directory: {input_directory}")

    x_global = np.arange(-300, 900, 1)
    y_global = np.arange(-300, 500, 1)
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
    for y_data in y_grouped_data:
        if len(y_data) > 1:
            y_grouped_filtered_data.append(y_data)

    # averaging the data
    for y_data in y_grouped_filtered_data:
        logging.debug(f"y_data: {y_data}")
        y_num = y_data[0]
        ux_list = y_data[1:]
        logging.info(f"y_num: {y_num}")

        sum_top = 0
        sum_bottom = 0
        for ux in ux_list:
            ux = ux[1]
            ux_uinf = ux / float(vw)
            mask = (ux != 0).astype(int)
            sum_top = sum_top + ux_uinf * mask
            sum_bottom = sum_bottom + mask

        sum_bottom = np.where(sum_bottom == 0, 1, sum_bottom)
        ux_mean_uinf = sum_top / sum_bottom

        # Mask the data to the color ranges
        ux_mean_uinf = np.ma.masked_where(ux_mean_uinf > max_cbar_value, ux_mean_uinf)
        ux_mean_uinf = np.ma.masked_where(ux_mean_uinf < min_cbar_value, ux_mean_uinf)
        # ux_mean_uinf = np.clip(ux_mean_uinf, min_cbar_value, max_cbar_value)
        fig, ax = plt.subplots()
        cax = plt.contourf(
            x_meshgrid_global,
            y_meshgrid_global,
            ux_mean_uinf,
            cmap="RdBu",
            levels=50,
            extend="both",
            vmin=min_cbar_value,
            vmax=max_cbar_value,
        )

        mid_cbar_value = np.mean([min_cbar_value, max_cbar_value])
        cbar = fig.colorbar(
            cax,
            ticks=[
                min_cbar_value,
                mid_cbar_value,
                max_cbar_value,
            ],
            format=mticker.FixedFormatter(
                [f"< {min_cbar_value}", f"{mid_cbar_value}", f"> {max_cbar_value}"]
            ),
            extend="both",
        )
        labels = cbar.ax.get_yticklabels()
        labels[0].set_verticalalignment("top")
        labels[-1].set_verticalalignment("bottom")
        cbar.set_label("Ux/Uinf", rotation=0)
        plt.savefig(save_plots_folder + y_num + plot_type)


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
    process_all_dat_files(
        input_directory=sys.path[0] + "/data/aoa_13/",
        lab_book_path=sys.path[0] + "/data/labbook_cleaned.csv",
        save_plots_folder=sys.path[0] + "/results/aoa_13/all_planes/",
        aoa_value=13.0,
        vw=15.0,
        min_cbar_value=0.75,
        max_cbar_value=1.25,
        plot_type=".png",
    )
