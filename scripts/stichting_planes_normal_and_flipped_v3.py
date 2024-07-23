import os
import numpy as np
import xarray as xr
import logging
from copy import deepcopy
import re
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from plotting import plot_quiver
import numpy as np
import xarray as xr


# defining new data matrix below a specified y max value
def filter_dataset_on_y_range(dataset, y_min, y_max, j_number):
    # # grabbing the y index
    # y_index = dataset.variables_edited.index("y")
    data_matrix = deepcopy(dataset["data"].values)

    new_data_matrix = np.empty((data_matrix.shape[0], j_number, data_matrix.shape[2]))
    # looping over each row top-to-bottom
    j_counter = 0
    for i_idx in range(data_matrix.shape[0]):
        for j_idx in range(data_matrix.shape[1]):
            if y_min < data_matrix[i_idx, j_idx, 1] < y_max:
                # new_data_matrix.append(data_matrix[i_idx, j_idx, :])
                new_data_matrix[i_idx, j_counter, :] = data_matrix[i_idx, j_idx, :]
    #             j_counter += 1
    # return np.array(new_data_matrix).reshape(
    #     (data_matrix.shape[0], j_counter, data_matrix.shape[2])
    # )
    return new_data_matrix


# def create_populated_dataset_both(
#     file_name_both,
#     i_both_value,
#     j_both_value,
#     k_both_value,
#     data_matrix_top,
#     data_matrix_overlap,
#     data_matrix_bottom,
#     j_both_start_overlap,
#     j_both_end_overlap,
#     y_both_min,
#     y_both_start_overlap,
#     y_both_end_overlap,
#     y_both_max,
# ):
#     # Create empty data array

#     data_matrix_both = np.full((i_both_value, j_both_value, k_both_value), np.nan)

#     # looping over each row top-to-bottom
#     for i in range(data_matrix_both.shape[0]):
#         for j in range(data_matrix_both.shape[1]):
#             if j < j_both_start_overlap:
#                 j_top = j
#                 data_matrix_both[i, j, :] = data_matrix_top[i, j_top, :]
#             elif j_both_start_overlap < j < j_both_end_overlap:
#                 j_overlap = j - j_both_start_overlap
#                 data_matrix_both[i, j, :] = data_matrix_overlap[i, j_overlap, :]
#             else:
#                 j_bottom = j - j_both_end_overlap
#                 data_matrix_both[i, j, :] = data_matrix_bottom[i, j_bottom, :]

#     # Create DataArray
#     data_array = xr.DataArray(
#         data_matrix_both,
#         dims=["x_i", "y_j", "variable"],
#         coords={
#             "x_i": np.arange(i_both_value),
#             "y_j": np.arange(j_both_value),
#             "variable": xr_dataset_normal.variables_edited,
#         },
#     )

#     # Create Dataset
#     new_dataset = xr.Dataset({"data": data_array})
#     new_dataset.attrs = xr_dataset_normal.attrs.copy()
#     new_dataset.attrs["i_value"] = i_both_value
#     new_dataset.attrs["j_value"] = j_both_value
#     new_dataset.attrs["k_value"] = k_both_value
#     new_dataset["file_name"] = xr.DataArray(np.array([file_name_both]), dims=["file"])

#     print(
#         f"Final y range: {np.nanmin(empty_data[:,:,1]), np.nanmax(empty_data[:,:,1])}"
#     )
#     return new_dataset


def stitching_normal_to_flipped(
    xr_dataset_normal, xr_dataset_flipped, file_name_both, y_traverse_step
):

    ### 1. defining the new mastergrid
    x_min = 0
    x_max = xr_dataset_normal.data.sel(variable="x").max().values
    y_min = 0
    y_start_overlap = y_traverse_step
    y_len_normal = (
        xr_dataset_normal.data.sel(variable="y").max().values
        - xr_dataset_normal.data.sel(variable="y").min().values
    )
    y_len_flipped = (
        xr_dataset_flipped.data.sel(variable="y").max().values
        - xr_dataset_flipped.data.sel(variable="y").min().values
    )
    y_end_overlap = y_len_normal
    y_max = y_traverse_step + y_len_flipped

    # logging
    # logging.info(f"y_len_normal: {y_len_normal}")
    # logging.info(f"y_len_flipped: {y_len_flipped}")
    # logging.info(f"y_traverse_step: {y_traverse_step}")
    # logging.info(f"x_min: {x_min}")
    # logging.info(f"x_max: {x_max}")
    # logging.info(f"y_both_min: {y_both_min}")
    # logging.info(f"y_both_start_overlap: {y_both_start_overlap}")
    # logging.info(f"y_both_end_overlap: {y_both_end_overlap}")
    # logging.info(f"y_both_max: {y_both_max}")

    ### 2. Correcting y-coordinates
    def correct_y_coordinates(dataset, y_offset):
        # grabbing the x index
        y_index = dataset.variables_edited.index("y")
        # grabbing the data matrix
        data_matrix = dataset["data"].values
        # creating a copy to make sure we don't modify the original
        new_data_matrix = data_matrix.copy()
        # adding the offset to the y values
        new_data_matrix[:, :, y_index] = new_data_matrix[:, :, y_index] + y_offset
        # creating a new DataArray
        new_data_array = xr.DataArray(
            new_data_matrix,
            dims=["x_i", "y_j", "variable"],
            coords={
                "x_i": np.arange(new_data_matrix.shape[0]),
                "y_j": np.arange(new_data_matrix.shape[1]),
                "variable": dataset.variables_edited,
            },
        )
        # channging the data in the dataset
        dataset["data"] = new_data_array
        return dataset

    # Correct flipped dataset
    y_offset = y_traverse_step
    xr_dataset_flipped = correct_y_coordinates(xr_dataset_flipped, y_offset)

    # logging
    logging.info(
        f"CORRECTED: Flipped y-coordinate range: {xr_dataset_flipped.data.sel(variable='y').min().values} to {xr_dataset_flipped.data.sel(variable='y').max().values}"
    )

    logging.info(
        f"CORRECTED: Normal y-coordinate range: {xr_dataset_normal.data.sel(variable='y').min().values} to {xr_dataset_normal.data.sel(variable='y').max().values}"
    )

    ### 2. Finidng the i values
    # calculation resolution as n_datapoints / (mm distance)
    y_res_flipped = (xr_dataset_flipped.j_value) / y_len_flipped
    y_res_normal = (xr_dataset_normal.j_value) / y_len_normal
    i_start_overlap = y_res_flipped * y_start_overlap
    i_end_overlap = y_res_flipped * y_end_overlap
    i_max = i_start_overlap + y_res_normal * y_len_normal

    # logging
    logging.info(f"y_res_flipped: {y_res_flipped}")
    logging.info(f"y_res_normal: {y_res_normal}")
    logging.info(f"i_start_overlap: {i_start_overlap}")
    logging.info(f"i_end_overlap: {i_end_overlap}")
    logging.info(f"i_max: {i_max}")

    ### 3. Populating data
    data_matrix_flipped = xr_dataset_flipped["data"].values
    data_matrix_normal = xr_dataset_normal["data"].values

    # creating new_data_matrix
    num_rows = int(i_max)
    num_cols = int(data_matrix_normal.shape[1])
    num_vars = int(data_matrix_normal.shape[2])
    new_data_matrix = np.full((num_rows, num_cols, num_vars), np.nan)

    # looping through each variable
    i_total = 0
    i_sub = 0
    for k in range(data_matrix_flipped.shape[2]):
        for i in range(data_matrix_flipped.shape[0]):
            # while in top-region
            while i_total <= i_start_overlap:
                # making sure the is reversed for the flipped data
                new_data_matrix[int(i_total), :, k] = data_matrix_flipped[
                    int(i_end_overlap - i), :, k
                ]
                i_total += 1
            # while in overlap region
            while i_start_overlap < i_total <= i_end_overlap:
                new_data_matrix[int(i_total), :, k] = data_matrix_normal[
                    int(i_sub), :, k
                ]
                i_total += 1
                i_sub += 1
            # while in bottom region
            while i_total < (i_max - 1):
                print(f"i: {i}, i_total: {i_total}, i_sub: {i_sub}")
                new_data_matrix[int(i_total), :, k] = data_matrix_normal[
                    int(i_sub), :, k
                ]
                i_total += 1
                i_sub += 1

    # logging
    logging.info(f"new_data_matrix shape: {new_data_matrix.shape}")
    logging.info(
        f"new_data_matrix y range: {np.min(new_data_matrix[:,:,1]), np.max(new_data_matrix[:,:,1])}"
    )
    logging.info(
        f"new_data_matrix x range: {np.nanmin(new_data_matrix[:,:,0]), np.nanmax(new_data_matrix[:,:,0])}"
    )

    # Create DataArray
    data_array = xr.DataArray(
        new_data_matrix,
        dims=["x_i", "y_j", "variable"],
        coords={
            "x_i": np.arange(num_rows),
            "y_j": np.arange(num_cols),
            "variable": xr_dataset_normal.variables_edited,
        },
    )

    # Create Dataset
    new_dataset = xr.Dataset({"data": data_array})
    new_dataset.attrs = xr_dataset_normal.attrs.copy()
    new_dataset.attrs["i_value"] = num_rows
    new_dataset.attrs["j_value"] = num_cols
    new_dataset.attrs["k_value"] = num_vars
    new_dataset["file_name"] = xr.DataArray(np.array([file_name_both]), dims=["file"])

    # print(
    #     f"Final y range: {np.nanmin(new_data_matrix[:,:,1]), np.nanmax(new_data_matrix[:,:,1])}"
    # )

    return new_dataset


def stitching_plotting_saving_y_normal_to_flipped(
    load_path_file,
    save_path_file,
    save_path_folder_plots,
    cmap,
    colorbar_label,
    u_inf,
    is_with_quiver,
    plot_name_end,
    is_show_plot,
    cbar_variable_name,
    max_cbar_value,
    min_cbar_value,
):
    loaded_dataset = xr.open_dataset(load_path_file)
    size = loaded_dataset.sizes.get("file")
    datapoint_list = [loaded_dataset.isel(file=i) for i in range(size)]
    flipped_data_point_list = []
    normal_data_point_list = []
    y_value_list = []
    for datapoint in datapoint_list:
        logging.debug(f"datapoint.file_name: {datapoint.file_name.values}")
        if "normal" in str(datapoint.file_name.values):
            normal_data_point_list.append(datapoint)
            y_value_list.append(int(datapoint["y_plane_number"].values))
            logging.info(f"normal")
        elif "flipped" in str(datapoint.file_name.values):
            flipped_data_point_list.append(datapoint)

        # print(f"file_name: {datapoint.variables_edited}")
        # print(f"datapoint: {datapoint.data.sel(variable='y').values}")

    # Grouping the datapoints
    datapoint_list_grouped = []
    for y_value in y_value_list:
        logging.debug(f"y: {y_value}")
        for normal, flipped in zip(normal_data_point_list, flipped_data_point_list):
            if int(normal["y_plane_number"].values) == y_value:
                normal_with_this_y = normal
            if int(flipped["y_plane_number"].values) == y_value:
                flipped_with_this_y = flipped
        datapoint_list_grouped.append([normal_with_this_y, flipped_with_this_y])

    for group in datapoint_list_grouped:
        logging.info(f"Group: {group[0].file_name.values}, {group[1].file_name.values}")

    all_xr_dataset_both = []

    # TODO: tweak these values
    additional_tweaked_y_traverse_step_z1 = 550  # 750
    additional_tweaked_y_traverse_step_z2 = 550
    additional_tweaked_y_traverse_step_z3 = 550

    # looping over each plane
    # TODO: remove data_point_list_grouped[0:1] to process all y-planes
    for datapoint_group, y_value in zip(datapoint_list_grouped, y_value_list):
        logging.info(f"y_value: {y_value}")
        xr_dataset_normal = datapoint_group[0]
        xr_dataset_flipped = datapoint_group[1]
        logging.info(f"normal: {xr_dataset_normal.file_name.values}")
        logging.info(f"flipped: {xr_dataset_flipped.file_name.values}")

        y_plane_number = y_value

        # Making sure we are dealing with the same y_plane_number
        y_plane_number_from_file_normal = xr_dataset_normal["y_plane_number"].values
        y_plane_number_from_file_flipped = xr_dataset_flipped["y_plane_number"].values
        if (
            y_plane_number_from_file_normal != y_plane_number_from_file_flipped
            or y_plane_number_from_file_normal != y_plane_number
        ):
            raise ValueError(
                f"y_plane_number from normal and flipped files are not the same: {y_plane_number_from_file_normal}, {y_plane_number_from_file_flipped}"
            )
        file_name = f"aoa_13_both_z1_y{y_plane_number}"
        h_table_normal = xr_dataset_normal["h_table"].values
        h_table_flipped = xr_dataset_flipped["h_table"].values
        delta_h_table = h_table_normal - h_table_flipped
        z_plane_number = xr_dataset_normal["z_plane_number"].values
        if z_plane_number == 1:
            y_traverse_step = delta_h_table - additional_tweaked_y_traverse_step_z1
        elif z_plane_number == 2:
            y_traverse_step = delta_h_table - additional_tweaked_y_traverse_step_z2
        elif z_plane_number == 3:
            y_traverse_step = delta_h_table - additional_tweaked_y_traverse_step_z3

        logging.info(f"h_table_normal: {h_table_normal}")
        logging.info(f"h_table_flipped: {h_table_flipped}")
        logging.info(f"delta_h_table: {delta_h_table}")
        logging.info(f"z_plane_number: {z_plane_number}")
        logging.info(
            f"additional_tweaked_y_traverse_step_z1: {additional_tweaked_y_traverse_step_z1}"
        )
        logging.info(f"y_traverse_step: {y_traverse_step}")

        xr_dataset_both = stitching_normal_to_flipped(
            xr_dataset_normal,
            xr_dataset_flipped,
            file_name_both=file_name,
            y_traverse_step=y_traverse_step,
        )

        # logging
        logging.info(f"-------------------")
        logging.debug(f"datapoint_group: {len(datapoint_group)}")
        logging.info(f"xr_dataset_normal: {xr_dataset_normal.file_name.values}")
        logging.info(f"xr_dataset_flipped: {xr_dataset_flipped.file_name.values}")
        logging.info(f"y_plane_number: {y_plane_number}")
        logging.info(f"file_name: {file_name}")
        logging.info(f"shape: {xr_dataset_both.data.shape}")
        logging.info(f"i_value: {xr_dataset_both.i_value}")
        logging.info(f"j_value: {xr_dataset_both.j_value}")
        # logging.info(f"k_variables: {xr_dataset_y3_x123.k_variables}")
        logging.info(f"file_name: {xr_dataset_both['file_name'].values}")
        logging.info(
            f'x_values, min, max: {xr_dataset_both.data.sel(variable="x").min().values}, {xr_dataset_both.data.sel(variable="x").max().values}'
        )
        logging.info(
            f'y_values, min, max: {xr_dataset_both.data.sel(variable="y").min().values}, {xr_dataset_both.data.sel(variable="y").max().values}'
        )

        ### Create plots
        plot_quiver(
            xr_dataset_both.data.sel(variable="x").values,
            xr_dataset_both.data.sel(variable="y").values,
            xr_dataset_both.data.sel(variable="vel_u").values,
            xr_dataset_both.data.sel(variable="vel_v").values,
            color_values=xr_dataset_both.data.sel(variable=cbar_variable_name).values,
            colorbar_label=colorbar_label,
            title=file_name + f"(from file: {save_path_file})",
            u_inf=u_inf,  # TODO: change this to vw
            save_path=save_path_folder_plots + file_name + plot_name_end,
            subsample=10,  # Adjust subsample factor as needed
            cmap=cmap,
            is_with_quiver=is_with_quiver,
            is_show_plot=is_show_plot,
            max_cbar_value=max_cbar_value,
            min_cbar_value=min_cbar_value,
        )

        # Append to list
        all_xr_dataset_both.append(xr_dataset_both)

    xr_all_stitched = xr.concat(all_xr_dataset_both, dim="file")

    # Save the stitched dataset
    xr_all_stitched.to_netcdf(save_path_file)


if __name__ == "__main__":
    # Go back to root folder
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root_path)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load the processed data
    load_path_file = sys.path[0] + "/processed_data/combined_piv_data_x123_planes.nc"
    save_path_file = sys.path[0] + "/processed_data/combined_piv_data_all_planes.nc"
    save_path_folder_plots = sys.path[0] + "/results/aoa_13/all_planes/"

    stitching_plotting_saving_y_normal_to_flipped(
        load_path_file,
        save_path_file,
        save_path_folder_plots,
        cmap="RdBu",
        colorbar_label=r"$\frac{U_x}{U_\infty}$",
        u_inf=15,
        is_with_quiver=False,
        plot_name_end=".png",
        is_show_plot=True,
        cbar_variable_name="ux_uinf",
        max_cbar_value=1.2,
        min_cbar_value=0.8,
    )
    # # STANDARD DEVIATION Load the processed data
    # load_path_file = (
    #     sys.path[0] + "/processed_data/combined_piv_data_x123_planes_std.nc"
    # )
    # save_path_file = sys.path[0] + "/processed_data/combined_piv_data_all_planes.nc"
    # save_path_folder_plots = sys.path[0] + "/results/aoa_13/all_planes/"
    # stitching_plotting_saving_y_normal_to_flipped(
    #     load_path_file,
    #     save_path_file,
    #     save_path_folder_plots,
    #     cmap="jet",
    #     colorbar_label=r"$std. vel_u$",
    #     u_inf=1,
    #     is_with_quiver=False,
    #     plot_name_end="_std.png",
    #     is_show_plot=False,
    #     cbar_variable_name="vel_u",
    #     max_cbar_value=8,
    #     min_cbar_value=0,
    # )
