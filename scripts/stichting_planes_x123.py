import os
import numpy as np
import xarray as xr
import logging
import re
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from plotting import plot_quiver


def create_populated_dataset_x123(
    file_name_x123,
    i_x123_value,
    j_x123_value,
    k_x123_value,
    xr_dataset_x1,
    xr_dataset_x2,
    xr_dataset_x3,
    i_x123_start_overlap_1,
    i_x123_end_overlap_1,
    i_x123_start_overlap_2,
    i_x123_end_overlap_2,
    i_x1_start_overlap_1,
    i_x1_end_overlap_1,
    i_x2_start_overlap_1,
    i_x2_end_overlap_1,
    i_x2_start_overlap_2,
    i_x2_end_overlap_2,
    i_x3_start_overlap_2,
    i_x3_end_overlap_2,
):
    # Create empty data array
    empty_data = np.full((i_x123_value, j_x123_value, k_x123_value), np.nan)
    logging.debug(f"empty_data.shape: {empty_data.shape}")

    # Populate non-overlapping regions directly
    # x1 region
    x1_slice = slice(0, i_x1_start_overlap_1)
    x123_slice = slice(0, i_x123_start_overlap_1)
    logging.debug(f"--- x1 region non-overlapping")
    logging.debug(f"x1_slice: {x1_slice}")
    logging.debug(f"x123_slice: {x123_slice}")
    logging.debug(f"x1 data shape: {xr_dataset_x1.data.values[x1_slice, :, :].shape}")
    logging.debug(f"empty_data slice shape: {empty_data[x123_slice, :, :].shape}")
    empty_data[x123_slice, :, :] = xr_dataset_x1.data.values[x1_slice, :, :]

    # overlap_1 region
    x1_slice = slice(i_x1_start_overlap_1, i_x1_end_overlap_1)
    x123_slice = slice(i_x123_start_overlap_1, i_x123_end_overlap_1)
    logging.debug(f"--- overlap_1 region")
    logging.debug(f"x1_slice: {x1_slice}")
    logging.debug(f"x123_slice: {x123_slice}")
    logging.debug(f"x1 data shape: {xr_dataset_x1.data.values[x1_slice, :, :].shape}")
    logging.debug(f"empty_data slice shape: {empty_data[x123_slice, :, :].shape}")
    empty_data[x123_slice, :, :] = xr_dataset_x1.data.values[x1_slice, :, :]

    # x2 region (between overlaps)
    x2_slice = slice(i_x2_end_overlap_1, i_x2_start_overlap_2)
    x123_slice = slice(i_x123_end_overlap_1, i_x123_start_overlap_2)
    logging.debug(f"--- x2 region between overlaps")
    logging.debug(f"x2_slice: {x2_slice}")
    logging.debug(f"x123_slice: {x123_slice}")
    logging.debug(f"x2 data shape: {xr_dataset_x2.data.values[x2_slice, :, :].shape}")
    logging.debug(f"empty_data slice shape: {empty_data[x123_slice, :, :].shape}")
    empty_data[x123_slice, :, :] = xr_dataset_x2.data.values[x2_slice, :, :]

    # overlap_2 region
    x2_slice = slice(i_x2_start_overlap_2, i_x2_end_overlap_2)
    x123_slice = slice(i_x123_start_overlap_2, i_x123_end_overlap_2)
    logging.debug(f"--- overlap_2 region")
    logging.debug(f"x2_slice: {x2_slice}")
    logging.debug(f"x123_slice: {x123_slice}")
    logging.debug(f"x2 data shape: {xr_dataset_x2.data.values[x2_slice, :, :].shape}")
    logging.debug(f"empty_data slice shape: {empty_data[x123_slice, :, :].shape}")
    empty_data[x123_slice, :, :] = xr_dataset_x2.data.values[x2_slice, :, :]

    # x3 region
    x3_slice = slice(i_x3_end_overlap_2, None)
    x123_slice = slice(i_x123_end_overlap_2, None)
    logging.debug(f"--- x3 region")
    logging.debug(f"x3_slice: {x3_slice}")
    logging.debug(f"x123_slice: {x123_slice}")
    logging.debug(f"x3 data shape: {xr_dataset_x3.data.values[x3_slice, :, :].shape}")
    logging.debug(f"empty_data slice shape: {empty_data[x123_slice, :, :].shape}")
    empty_data[x123_slice, :, :] = xr_dataset_x3.data.values[x3_slice, :, :]

    variables_edited = ["x", "y", "vel_u", "vel_v"]
    for k, variable in enumerate(variables_edited):
        print(f" ")
        logging.info(f"Variable: {variable}")
        for data_i in [xr_dataset_x1, xr_dataset_x2, xr_dataset_x3]:
            mid_index = 100
            logging.info(f"{data_i.case_name_davis.values}")
            logging.info(f"NEW (first row): {empty_data[0, :5, k]}")
            logging.info(f"origial (first row): {data_i.data.values[0, :5, k]}")
            logging.info(f"NEW (mid row): {empty_data[mid_index, :5, k]}")
            logging.info(f"original (mid row): {data_i.data.values[mid_index, :5, k]}")
            logging.info(f"NEW (last row): {empty_data[-1, :5, k]}")
            logging.info(f"original (last row): {data_i.data.values[-1, :5, k]}")

    breakpoint()
    ### Create DataArray
    logging.debug(f"empty_data.shape: {empty_data.shape}")
    logging.debug(f"xr_dataset_x1.variables_edited: {xr_dataset_x1.variables_edited}")
    data_array = xr.DataArray(
        empty_data,
        dims=["x_i", "y_j", "variable"],
        coords={
            "x_i": np.arange(i_x123_value),
            "y_j": np.arange(j_x123_value),
            "variable": xr_dataset_x1.variables_edited,
        },
    )
    logging.debug(f"data_array.shape: {data_array.shape}")
    # Create Dataset
    new_dataset = xr.Dataset({"data": data_array})

    logging.debug(f"new_dataset.data.shape: {new_dataset.data.shape}")

    ### Attributes
    # copying attributes
    new_dataset.attrs = xr_dataset_x1.attrs.copy()
    # editing the ones that have changed
    new_dataset.attrs["i_value"] = i_x123_value
    new_dataset.attrs["j_value"] = j_x123_value
    new_dataset.attrs["k_value"] = k_x123_value

    # TODO: think about how you will handle this
    # can't just average the wind speed values for example...
    ### Stitched Plane Specific Information
    # adding a file_name
    new_dataset["file_name"] = xr.DataArray(np.array([file_name_x123]), dims=["file"])
    y_plane_number = xr_dataset_x1["y_plane_number"].values
    new_dataset["y_plane_number"] = xr.DataArray(
        np.array([y_plane_number]), dims=["file"]
    )
    h_table = xr_dataset_x1["h_table"].values
    new_dataset["h_table"] = xr.DataArray(np.array([h_table]), dims=["file"])
    z_plane_number = xr_dataset_x1["z_plane_number"].values
    new_dataset["z_plane_number"] = xr.DataArray(
        np.array([z_plane_number]), dims=["file"]
    )

    return new_dataset


def stitching_x123_planes(
    xr_dataset_x1, xr_dataset_x2, xr_dataset_x3, file_name_x123, x_traverse_step
):

    ### 1. defining the mastergrid
    x_min = 0
    x_start_overlap_1 = x_traverse_step
    x_len_x1 = (
        xr_dataset_x1.data.sel(variable="x").max().values
        - xr_dataset_x1.data.sel(variable="x").min().values
    )
    x_end_overlap_1 = x_len_x1
    x_start_overlap_2 = 2 * x_traverse_step
    x_len_x2 = (
        xr_dataset_x2.data.sel(variable="x").max().values
        - xr_dataset_x2.data.sel(variable="x").min().values
    )
    x_end_overlap_2 = x_start_overlap_2 + x_len_x2
    x_len_x3 = (
        xr_dataset_x3.data.sel(variable="x").max().values
        - xr_dataset_x3.data.sel(variable="x").min().values
    )

    x_max = 2 * x_traverse_step + x_len_x3

    y_min = 0
    # assuming that x1,x2,x3 planes have the same y_min and y_max
    y_max = xr_dataset_x1.data.sel(variable="y").max().values

    # logging
    logging.debug(f"x_len_x1: {x_len_x1}")
    logging.debug(f"x_len_x2: {x_len_x2}")
    logging.debug(f"x_len_x3: {x_len_x3}")
    logging.debug(f"x_traverse_step: {x_traverse_step}")
    logging.debug(f"x_min: {x_min}")
    logging.debug(f"x_start_overlap_1: {x_start_overlap_1}")
    logging.debug(f"x_end_overlap_1: {x_end_overlap_1}")
    logging.debug(f"x_start_overlap_2: {x_start_overlap_2}")
    logging.debug(f"x_end_overlap_2: {x_end_overlap_2}")
    logging.debug(f"x_max: {x_max}")
    logging.debug(f"y_min: {y_min}")
    logging.debug(f"y_max: {y_max}")

    ### 2. Calculating resolution
    # assuming that x1,x2,x3 planes have the same resolution
    # calculation resolution as n_datapoints / (mm distance)
    x_res_x1 = (xr_dataset_x1.i_value) / x_len_x1
    x_res_x2 = (xr_dataset_x2.i_value) / x_len_x2
    x_res_x3 = (xr_dataset_x3.i_value) / x_len_x3
    i_x123_start = 0
    i_x123_start_overlap_1 = int(x_traverse_step * x_res_x1)
    i_x123_end_overlap_1 = int(x_len_x1 * x_res_x1)
    x_delta_overlap_1 = i_x123_end_overlap_1 - i_x123_start_overlap_1
    i_x123_start_overlap_2 = int(i_x123_start_overlap_1 + x_traverse_step * x_res_x2)
    i_x123_end_overlap_2 = int(
        i_x123_start_overlap_2 + ((x_len_x2 * x_res_x2) - i_x123_start_overlap_1)
    )
    x_delta_overlap_2 = i_x123_end_overlap_2 - i_x123_start_overlap_2
    i_x123_end = int(i_x123_start_overlap_2 + x_len_x3 * x_res_x3)
    # Defining i values for the 3 planes
    i_x1_start_overlap_1 = int(i_x123_start_overlap_1)
    i_x1_end_overlap_1 = int(i_x123_end_overlap_1)
    i_x2_start_overlap_1 = int(i_x123_start_overlap_1 - i_x123_start_overlap_1)
    i_x2_end_overlap_1 = int(i_x123_end_overlap_1 - i_x123_start_overlap_1)
    i_x2_start_overlap_2 = int(i_x123_start_overlap_2 - i_x123_start_overlap_1)
    i_x2_end_overlap_2 = int(i_x123_end_overlap_2 - i_x123_start_overlap_1)
    i_x3_start_overlap_2 = int(i_x123_start_overlap_2 - i_x123_start_overlap_2)
    i_x3_end_overlap_2 = int(i_x123_end_overlap_2 - i_x123_start_overlap_2)

    # logging
    logging.debug(f"--- i_x1, j_x1: {xr_dataset_x1.i_value}, {xr_dataset_x1.j_value}")
    logging.debug(f"x_res_x1: {x_res_x1}")
    logging.debug(f"x_res_x2: {x_res_x2}")
    logging.debug(f"x_res_x3: {x_res_x3}")
    logging.debug(f"i_x123_start: {i_x123_start}")
    logging.debug(f"i_x123_start_overlap_1: {i_x123_start_overlap_1}")
    logging.debug(f"i_x123_end_overlap_1: {i_x123_end_overlap_1}")
    logging.debug(f"x_delta_overlap_1: {x_delta_overlap_1}")
    logging.debug(f"i_x123_start_overlap_2: {i_x123_start_overlap_2}")
    logging.debug(f"i_x123_end_overlap_2: {i_x123_end_overlap_2}")
    logging.debug(f"x_delta_overlap_2: {x_delta_overlap_2}")
    logging.debug(f"i_x123_end: {i_x123_end}")
    logging.debug(f"i_x1_start_overlap_1: {i_x1_start_overlap_1}")
    logging.debug(f"i_x1_end_overlap_1: {i_x1_end_overlap_1}")
    logging.debug(f"i_x2_start_overlap_1: {i_x2_start_overlap_1}")
    logging.debug(f"i_x2_end_overlap_1: {i_x2_end_overlap_1}")
    logging.debug(f"i_x2_start_overlap_2: {i_x2_start_overlap_2}")
    logging.debug(f"i_x2_end_overlap_2: {i_x2_end_overlap_2}")
    logging.debug(f"i_x3_start_overlap_2: {i_x3_start_overlap_2}")
    logging.debug(f"i_x3_end_overlap_2: {i_x3_end_overlap_2}")

    ### 3. Correcting x-coordinates
    def correct_x_coordinates(dataset, x_offset):
        # grabbing the x index
        x_index = dataset.variables_edited.index("x")
        # grabbing the data matrix
        data_matrix = dataset["data"].values
        # creating a copy to make sure we don't modify the original
        new_data_matrix = data_matrix.copy()
        new_data_matrix[:, :, x_index] = data_matrix[:, :, x_index] + x_offset
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

    # Correct x1 dataset (no offset needed)
    # logging.debug(f'dataset["data"].values: {xr_dataset_x1.data.values}')
    # xr_dataset_x1_corrected_x = correct_x_coordinates(xr_dataset_x1, 1e5)
    # logging.debug(f'dataset["data"].values: {xr_dataset_x1_corrected_x.data.values}')

    # Correct x1 dataset (no offset needed)
    x1_offset = -xr_dataset_x1.data.sel(variable="x").min().values
    xr_dataset_x1_corrected_x = correct_x_coordinates(xr_dataset_x1, x1_offset)

    # Correct x2 dataset
    x2_offset = -xr_dataset_x2.data.sel(variable="x").min().values + x_traverse_step
    xr_dataset_x2_corrected_x = correct_x_coordinates(xr_dataset_x2, x2_offset)

    # Correct x3 dataset
    x3_offset = -xr_dataset_x3.data.sel(variable="x").min().values + 2 * x_traverse_step
    xr_dataset_x3_corrected_x = correct_x_coordinates(xr_dataset_x3, x3_offset)

    # Log the corrections
    logging.debug(
        f"X1 x-coordinate range: {xr_dataset_x1_corrected_x.data.sel(variable='x').min().values} to {xr_dataset_x1_corrected_x.data.sel(variable='x').max().values}"
    )
    logging.debug(
        f"X2 x-coordinate range: {xr_dataset_x2_corrected_x.data.sel(variable='x').min().values} to {xr_dataset_x2_corrected_x.data.sel(variable='x').max().values}"
    )
    logging.debug(
        f"X3 x-coordinate range: {xr_dataset_x3_corrected_x.data.sel(variable='x').min().values} to {xr_dataset_x3_corrected_x.data.sel(variable='x').max().values}"
    )

    ### 3. Populating data
    xr_dataset_x123 = create_populated_dataset_x123(
        file_name_x123=file_name_x123,
        i_x123_value=i_x123_end,
        j_x123_value=xr_dataset_x1.j_value,
        k_x123_value=len(xr_dataset_x1.variables_edited),
        xr_dataset_x1=xr_dataset_x1_corrected_x,
        xr_dataset_x2=xr_dataset_x2_corrected_x,
        xr_dataset_x3=xr_dataset_x3_corrected_x,
        i_x123_start_overlap_1=i_x123_start_overlap_1,
        i_x123_end_overlap_1=i_x123_end_overlap_1,
        i_x123_start_overlap_2=i_x123_start_overlap_2,
        i_x123_end_overlap_2=i_x123_end_overlap_2,
        i_x1_start_overlap_1=i_x1_start_overlap_1,
        i_x1_end_overlap_1=i_x1_end_overlap_1,
        i_x2_start_overlap_1=i_x2_start_overlap_1,
        i_x2_end_overlap_1=i_x2_end_overlap_1,
        i_x2_start_overlap_2=i_x2_start_overlap_2,
        i_x2_end_overlap_2=i_x2_end_overlap_2,
        i_x3_start_overlap_2=i_x3_start_overlap_2,
        i_x3_end_overlap_2=i_x3_end_overlap_2,
    )

    return xr_dataset_x123


def stitching_plotting_saving_x123_planes(
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
    datapoint_list_grouped = [
        datapoint_list[i : i + 3] for i in range(0, len(datapoint_list), 3)
    ]
    logging.debug(f"datapoint_list.shape: {len(datapoint_list)}")
    logging.debug(f"datapoint_list_grouped.shape: {len(datapoint_list_grouped)}")

    all_x123_stitched = []
    # looping over each y-plane
    for datapoint_group in datapoint_list_grouped:
        xr_dataset_x1 = datapoint_group[0]
        xr_dataset_x2 = datapoint_group[1]
        xr_dataset_x3 = datapoint_group[2]
        y_plane_number = int(xr_dataset_x1["y_plane_number"].values)
        z_plane_number = int(xr_dataset_x1["z_plane_number"].values)
        x_traverse_step = int(xr_dataset_x2["y_traverse"].values)

        if int(x_traverse_step) != 300:
            raise ValueError(
                f"x_traverse_step is not 300, but {x_traverse_step}, check labbook"
            )

        if "flipped" in str(datapoint_group[0]["case_name_davis"].values):
            orientation = "flipped"
        elif "normal" in str(datapoint_group[0]["case_name_davis"].values):
            orientation = "normal"

        file_name = f"aoa_13_{orientation}_z{z_plane_number}_y{y_plane_number}_x123"

        xr_dataset_x123 = stitching_x123_planes(
            xr_dataset_x1,
            xr_dataset_x2,
            xr_dataset_x3,
            file_name_x123=file_name,
            x_traverse_step=x_traverse_step,
        )

        # logging
        logging.info(f"-------------------")
        logging.debug(f"datapoint_group: {len(datapoint_group)}")
        logging.info(f"datapoint_group: {datapoint_group[0].file_name.values}")
        logging.info(f"datapoint_group: {datapoint_group[1].file_name.values}")
        logging.info(f"datapoint_group: {datapoint_group[2].file_name.values}")
        logging.info(f"y_plane_number: {y_plane_number}")
        logging.info(f"file_name: {file_name}")
        logging.info(f"shape: {xr_dataset_x123.data.shape}")
        logging.info(f"i_value: {xr_dataset_x123.i_value}")
        logging.info(f"j_value: {xr_dataset_x123.j_value}")
        # logging.info(f"k_variables: {xr_dataset_y3_x123.k_variables}")
        logging.info(f"file_name: {xr_dataset_x123['file_name'].values}")
        logging.info(
            f'x_values, min, max: {xr_dataset_x123.data.sel(variable="x").min().values}, {xr_dataset_x123.data.sel(variable="x").max().values}'
        )

        ### Create plots
        plot_quiver(
            xr_dataset_x123.data.sel(variable="x").values,
            xr_dataset_x123.data.sel(variable="y").values,
            xr_dataset_x123.data.sel(variable="vel_u").values,
            xr_dataset_x123.data.sel(variable="vel_v").values,
            color_values=xr_dataset_x123.data.sel(variable=cbar_variable_name).values,
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
        all_x123_stitched.append(xr_dataset_x123)

    xr_all_x123_stitched = xr.concat(all_x123_stitched, dim="file")

    # Save the stitched dataset
    xr_all_x123_stitched.to_netcdf(save_path_file)


if __name__ == "__main__":
    # Go back to root folder
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root_path)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load the processed data
    load_path_file = sys.path[0] + "/processed_data/combined_piv_data.nc"
    save_path_file = sys.path[0] + "/processed_data/combined_piv_data_x123_planes.nc"
    save_path_folder_plots = sys.path[0] + "/results/aoa_13/x123_planes/"
    stitching_plotting_saving_x123_planes(
        load_path_file,
        save_path_file,
        save_path_folder_plots,
        cmap="RdBu",
        colorbar_label=r"$\frac{U_x}{U_\infty}$",
        u_inf=15,
        is_with_quiver=True,
        plot_name_end=".png",
        is_show_plot=False,
        cbar_variable_name="ux_uinf",
        max_cbar_value=1.2,
        min_cbar_value=0.8,
    )
    # STANDARD DEVIATION Load the processed data
    load_path_file = sys.path[0] + "/processed_data/combined_piv_data_std.nc"
    save_path_file = (
        sys.path[0] + "/processed_data/combined_piv_data_x123_planes_std.nc"
    )
    save_path_folder_plots = sys.path[0] + "/results/aoa_13/x123_planes/"
    stitching_plotting_saving_x123_planes(
        load_path_file,
        save_path_file,
        save_path_folder_plots,
        cmap="jet",
        colorbar_label=r"$std. vel_u$",
        u_inf=1,
        is_with_quiver=False,
        plot_name_end="_std.png",
        is_show_plot=False,
        cbar_variable_name="vel_u",
        max_cbar_value=2,
        min_cbar_value=0,
    )
