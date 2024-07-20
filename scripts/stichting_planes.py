import os
import numpy as np
import xarray as xr
import logging
import re
import sys
from datetime import datetime
import matplotlib.pyplot as plt


def create_empty_dataset_like(template_dataset, shape):
    empty_data = np.empty(shape)
    data_array = xr.DataArray(
        empty_data,
        dims=["x_i", "y_j", "variable"],
        coords={
            "x_i": np.arange(shape[0]),
            "y_j": np.arange(shape[1]),
            "variable": template_dataset.data.variable,
        },
    )
    new_dataset = xr.Dataset({"data": data_array})

    # Copy attributes and other coordinates
    new_dataset.attrs = template_dataset.attrs.copy()
    for coord in template_dataset.coords:
        if coord not in ["x_i", "y_j", "variable"]:
            new_dataset[coord] = template_dataset[coord]

    return new_dataset


# Usage:
new_dataset = create_empty_dataset_like(dataset, (100, 100, len(variables_edited)))


def stitching_x123_planes(data_x1, data_x2, data_x3, x_traverse_step=300):

    ### 1. defining the mastergrid
    x_min = 0
    x_start_overlap_1 = x_traverse_step
    x_end_overlap_1 = data_x1.x.max().values - data_x1.x.min().values
    x_start_overlap_2 = 2 * x_traverse_step
    x_len_x2 = data_x2.x.max().values - data_x2.x.min().values
    x_end_overlap_2 = x_start_overlap_2 + x_len_x2
    # finding the maximum x value of the X3 plane
    x_len_x3 = data_x3.x.max().values - data_x3.x.min().values
    logging.info(f"x_len_x3: {x_len_x3}")
    x_max = 2 * x_traverse_step + x_len_x3
    logging.info(f"x_max: {x_max}")
    y_min = 0
    # assuming that x1,x2,x3 planes have the same y_min and y_max
    y_max = data_x1.y.max().values

    ### 2. Calculating resolution
    # assuming that x1,x2,x3 planes have the same resolution
    # calculation resolution as n_datapoints / (mm distance)
    x_res = (data_x1.i_value) / (data_x1.x.max().values - data_x1.x.min().values)
    i_start_overlap_1 = x_traverse_step * x_res
    i_start_overlap_2 = 2 * x_traverse_step * x_res

    n_x_points = int(x_max * x_res)
    n_y_points = data_x1.j_value
    # y_res = (data_x1.j_value) / (data_x1.y.max().values - data_x1.y.min().values)
    # # calc. n_datapoints untill x_traverse_step
    # n_data_points_in_x1 = x_traverse_step * x_res
    ## calc. n_datapoints for the 3 planes
    # n_data_points_in_x3 = x_max * x_res * data_x1.j_value

    ### 3. Populating data
    x_mastergrid = np.linspace(x_min, x_max, n_x_points)
    y_mastergrid = np.linspace(y_min, y_max, n_y_points)
    i_value = n_x_points
    j_value = n_y_points

    ## Create empty xr dataset
    data = np.zeros((i_value, j_value, len(data_x1.variables_edited)))
    coords = {"x_i": x_mastergrid, "y_j": y_mastergrid}
    data_matrix = data.reshape((i_value, j_value, -1))
    variables_edited = data_x1.variables_edited
    logging.info(f"variables_edited: {variables_edited}")
    data_vars = {}
    for idx, var in enumerate(variables_edited):
        var_data = [data_matrix[..., idx]]
        data_vars[var] = (["x_i", "y_j"], var_data)

    dataset = xr.Dataset(data_vars, coords=coords)

    # populating data
    # Looping over y values
    for j in range(j_value):
        # looping over the whole first row, and then the second row, etc.
        for i in range(i_value):

            current_x_location = x_mastergrid[i]
            logging.info(f"current_x_location: {current_x_location}")
            logging.info(f"i,j: {i},{j}")
            logging.info(f"data_x1.isel(x_i=i, y_j=j): {data_x1.isel(x_i=i, y_j=j)}")

            # check if the x value is in the x1, non-overlapping part
            if current_x_location < x_start_overlap_1:
                # if within just add the data_x1 values
                for var in data_x1.variables_edited:
                    dataset[var].values[i, j] = data_x1[var].isel(x_i=i, y_j=j).values
                breakpoint()
            # if within overlap_1 region
            elif x_start_overlap_1 <= current_x_location <= x_end_overlap_1:
                # if within overlap region to a smoothing thing
                # TODO: for now just add the data_x2
                i_local_x1 = i
                i_local_x2 = i - i_start_overlap_1
                data[i, j, :] = data_x2.isel(x_i=i_local_x2, y_j=j)
            # if in x2, between overlap_1 and overlap_2
            elif x_end_overlap_1 < current_x_location < x_start_overlap_2:
                i_local_x2 = i - i_start_overlap_1
                data[i, j, :] = data_x2.isel(x_i=i_local_x2, y_j=j)
            # if within overlap_2 region
            elif x_start_overlap_2 <= current_x_location <= x_end_overlap_2:
                # if within overlap region to a smoothing thing
                # TODO: for now just add the data_x2
                i_local_x2 = i - i_start_overlap_1
                i_local_x3 = i - i_start_overlap_2
                data[i, j, :] = data_x2.isel(x_i=i_local_x2, y_j=j)
            # if within the x3 region
            elif x_end_overlap_2 < current_x_location:
                # if within just add the data_x3 values
                i_local_x3 = i - i_start_overlap_2
                data[i, j, :] = data_x3.isel(x_i=i_local_x3, y_j=j)
            else:
                raise ValueError("Something went wrong, outside of the x_range")

    return dataset


if __name__ == "__main__":
    # Go back to root folder
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root_path)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load the processed data
    processed_data_path = sys.path[0] + "/processed_data/combined_piv_data.nc"
    loaded_dataset = xr.open_dataset(processed_data_path)
    size = loaded_dataset.sizes.get("file")
    # logging.info(f"Data attrs: {loaded_dataset.attrs}")
    datapoint_list = [loaded_dataset.isel(file=i) for i in range(size)]

    data_x1 = datapoint_list[0]
    data_x2 = datapoint_list[1]
    data_x3 = datapoint_list[2]

    mastergrid = stitching_x123_planes(data_x1, data_x2, data_x3)

    # for i, datapoint in enumerate(datapoint_list):
    #     case_name_davis = datapoint.case_name_davis.values
    #     logging.info(f"datapoint.data_vars: {datapoint.data_vars}")
    #     logging.info(f"case_name: {case_name_davis}")
    #     # logging.info(f"FileName: {datapoint.file_name_labbook.values}")
