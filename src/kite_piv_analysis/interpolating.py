import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from kite_piv_analysis.defining_bound_volume import boundary_rectangle
from kite_piv_analysis.utils import reading_optimal_bound_placement
from kite_piv_analysis import calculating_airfoil_centre


def inverse_distance_weighting(points, values, grid_points, power=2):
    """
    Performs inverse distance weighting (IDW) interpolation for specified grid points.
    """
    interpolated_values = []
    for gp in grid_points:
        # Calculate distances from the grid point to all known points
        distances = np.linalg.norm(points - gp, axis=1)
        weights = 1 / (
            distances**power + 1e-6
        )  # Add small value to avoid division by zero
        weights /= weights.sum()  # Normalize weights
        interpolated_value = np.dot(weights, values)
        interpolated_values.append(interpolated_value)

    return np.array(interpolated_values)


def distance_weighting(points, values, grid_points, power=2):
    """
    Custom distance weighting interpolation where farther points have higher weight.
    This assigns weights proportional to distance instead of the inverse of distance.
    """
    interpolated_values = []
    for gp in grid_points:
        # Calculate distances from the grid point to all known points
        distances = np.linalg.norm(points - gp, axis=1)

        # Weight proportional to distance (adding small constant to avoid zero weight at distance 0)
        weights = distances**power + 1e-6  # Increase weight with distance
        weights /= weights.sum()  # Normalize weights

        # Calculate weighted sum for interpolated value
        interpolated_value = np.dot(weights, values)
        interpolated_values.append(interpolated_value)

    return np.array(interpolated_values)


def interpolate_missing_data(
    df,
    interpolation_zone_i,
    columns=[
        "u",
        "v",
        "w",
        "V",
    ],
):

    x_min, x_max, y_min, y_max = interpolation_zone_i["bounds"]
    increase_weight_points_close = interpolation_zone_i["increase_weight_points_close"]
    increase_weight_points_far = interpolation_zone_i["increase_weight_points_far"]
    method = interpolation_zone_i["method"]
    in_zone = (
        (df["x"] >= x_min)
        & (df["x"] <= x_max)
        & (df["y"] >= y_min)
        & (df["y"] <= y_max)
    )

    # Track originally invalid velocity points (before interpolation).
    was_masked = in_zone & (df["u"].isna() | df["v"].isna())

    # Filter data within the expanded bounding box and non-NaN values for interpolation
    subset = df[in_zone]

    # Dictionary to store interpolated values for each column
    interpolated_values = {}
    nan_targets = {}

    for col in columns:
        # Get coordinates with available data for interpolation for this specific column.
        valid_data = subset.dropna(subset=[col])
        points = valid_data[["x", "y"]].values  # Points where data is known

        # Values for known data points in the current column
        values = valid_data[col].values

        # Identify cells to interpolate (i.e., cells with NaN values within bounds)
        nan_cells = df[in_zone & df[col].isna()]
        nan_points = nan_cells[["x", "y"]].values  # Points needing interpolation
        nan_targets[col] = nan_cells.index

        if len(nan_points) == 0:
            continue
        if len(points) < 3:
            # Not enough support points for robust interpolation.
            continue

        if increase_weight_points_close:
            # Custom inverse distance weighting for interpolated values
            interpolated_values[col] = inverse_distance_weighting(
                points, values, nan_points
            )
        elif increase_weight_points_far:
            # Custom distance weighting for interpolated values
            interpolated_values[col] = distance_weighting(points, values, nan_points)
        else:
            # Use griddata for standard interpolation
            interpolated_values[col] = griddata(
                points, values, nan_points, method=method
            )

    # Fill the interpolated values into the dataframe
    for col in columns:
        if col not in interpolated_values:
            continue
        idx = nan_targets[col]
        if len(idx) == 0:
            continue
        df.loc[idx, col] = interpolated_values[col]

    # Recompute derivative-derived quantities from completed velocity fields.
    # Overwrite:
    # 1) originally velocity-masked points, and
    # 2) in-zone points that still miss any derived field while u/v are available.
    derived_cols = ["dudx", "dudy", "dvdx", "dvdy", "dwdx", "dwdy", "vort_z"]
    present_derived_cols = [c for c in derived_cols if c in df.columns]
    if present_derived_cols:
        derived_missing = in_zone & df[present_derived_cols].isna().any(axis=1)
    else:
        derived_missing = in_zone & False
    uv_available = in_zone & df["u"].notna() & df["v"].notna()
    recompute_mask = (was_masked | (derived_missing & uv_available)).astype(bool)

    if recompute_mask.any():
        n_rows = len(np.unique(df["y"]))
        n_cols = len(np.unique(df["x"]))

        x_grid = df["x"].values.reshape(n_rows, n_cols)
        y_grid = df["y"].values.reshape(n_rows, n_cols)
        u_grid = df["u"].values.reshape(n_rows, n_cols)
        v_grid = df["v"].values.reshape(n_rows, n_cols)
        w_grid = df["w"].values.reshape(n_rows, n_cols)

        # Robust spacing estimate for near-uniform structured grids.
        dx_samples = np.diff(x_grid, axis=1)
        dy_samples = np.diff(y_grid, axis=0)
        dx = float(np.nanmedian(dx_samples[np.isfinite(dx_samples)]))
        dy = float(np.nanmedian(dy_samples[np.isfinite(dy_samples)]))
        if not np.isfinite(dx) or abs(dx) < 1e-12:
            dx = float(np.nanmedian(np.diff(np.sort(df["x"].unique()))))
        if not np.isfinite(dy) or abs(dy) < 1e-12:
            dy = float(np.nanmedian(np.diff(np.sort(df["y"].unique()))))

        # np.gradient returns [d/dy, d/dx] for a 2D array with shape (ny, nx).
        dudy_grid, dudx_grid = np.gradient(u_grid, dy, dx)
        dvdy_grid, dvdx_grid = np.gradient(v_grid, dy, dx)
        dwdy_grid, dwdx_grid = np.gradient(w_grid, dy, dx)
        vort_z_grid = dvdx_grid - dudy_grid

        mask_flat = recompute_mask.values
        df.loc[mask_flat, "dudx"] = dudx_grid.reshape(-1)[mask_flat]
        df.loc[mask_flat, "dudy"] = dudy_grid.reshape(-1)[mask_flat]
        df.loc[mask_flat, "dvdx"] = dvdx_grid.reshape(-1)[mask_flat]
        df.loc[mask_flat, "dvdy"] = dvdy_grid.reshape(-1)[mask_flat]
        df.loc[mask_flat, "dwdx"] = dwdx_grid.reshape(-1)[mask_flat]
        df.loc[mask_flat, "dwdy"] = dwdy_grid.reshape(-1)[mask_flat]
        df.loc[mask_flat, "vort_z"] = vort_z_grid.reshape(-1)[mask_flat]

    # plotting a rectangle around the interpolation zone
    d1centre = x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2
    drot = 0
    dLx = x_max - x_min
    dLy = y_max - y_min
    iP = 25
    d2curve_rectangle_interpolated_zone = boundary_rectangle(
        d1centre, drot, dLx, dLy, iP
    )
    return df, d2curve_rectangle_interpolated_zone


def find_areas_needing_interpolation(
    df: pd.DataFrame,
    alpha: int,
    y_num: int,
    rectangle_size: float,
    iP: int = 49,
    dLx: float = None,
    dLy: float = None,
    drot: float = None,
    n_lim: float = None,
):
    # Get the airfoil center and optimal bound placement
    airfoil_center = calculating_airfoil_centre.main(alpha, y_num)

    if dLx is None or dLy is None:
        dLx, dLy, N_datapoints = reading_optimal_bound_placement(
            alpha, y_num, is_with_N_datapoints=True
        )
    else:
        dLx_optimal, dLy_optimal, N_datapoints = reading_optimal_bound_placement(
            alpha, y_num, is_with_N_datapoints=True
        )

    if drot is None:
        drot = 0

    # Generate the boundary rectangle
    d2curve_rectangle = boundary_rectangle(airfoil_center, 0, dLx, dLy, iP)

    interpolation_zones = []

    # print(f" inside the find area function")
    if n_lim is None:
        n_lim = N_datapoints

    for x, y in d2curve_rectangle:
        # Define the bounds of the rectangle
        half_size = rectangle_size / 2
        mask = (
            (df["x"] >= x - half_size)
            & (df["x"] <= x + half_size)
            & (df["y"] >= y - half_size)
            & (df["y"] <= y + half_size)
        )

        # counting the number of data points within the rectangle, that are not NaN or zero
        n_datapoints_counted = len(df[mask].dropna())

        # print(f"len(df[mask])", n_datapoints_counted)

        # Check if there are fewer than N_datapoints within the rectangle
        # print(f"n_datapoints_counted: {n_datapoints_counted}, n_lim: {n_lim}")
        if n_datapoints_counted < n_lim:
            interpolation_zones.append(
                {
                    "bounds": [
                        x - half_size,
                        x + half_size,
                        y - half_size,
                        y + half_size,
                    ],
                    "increase_weight_points_close": False,
                    "increase_weight_points_far": True,
                    "method": "linear",
                }
            )
    return interpolation_zones

    # if plot_params["alpha"] == 6:
    #     plot_params["interpolation_zones"] = (
    #         # {
    #         #     "bounds": [0.43, 0.5, 0.14, 0.19],
    #         #     "increase_weight_points_close": False,
    #         #     "increase_weight_points_far": True,
    #         #     "method": "linear",
    #         # },
    #         {
    #             "bounds": [0.43, 0.55, 0.11, 0.21],
    #             "increase_weight_points_close": False,
    #             "increase_weight_points_far": True,
    #             "method": "linear",
    #         },
    #         {
    #             "bounds": [0.43, 0.55, -0.1, 0.04],
    #             "increase_weight_points_close": False,
    #             "increase_weight_points_far": True,
    #             "method": "linear",
    #         },
    #         {
    #             "bounds": [0.22, 0.3, -0.15, -0.07],
    #             "increase_weight_points_close": False,
    #             "increase_weight_points_far": True,
    #             "method": "linear",
    #         },
    #     )
    # else:
    #     plot_params["interpolation_zones"] = (
    #         # {
    #         #     "bounds": [0.43, 0.5, 0.14, 0.19],
    #         #     "increase_weight_points_close": False,
    #         #     "increase_weight_points_far": True,
    #         #     "method": "linear",
    #         # },
    #         {
    #             "bounds": [0.47, 0.55, 0.01, 0.1],
    #             "increase_weight_points_close": False,
    #             "increase_weight_points_far": True,
    #             "method": "linear",
    #         },
    #         {
    #             "bounds": [0.47, 0.55, -0.05, 0.01],
    #             "increase_weight_points_close": False,
    #             "increase_weight_points_far": True,
    #             "method": "linear",
    #         },
    #         {
    #             "bounds": [0.35, 0.55, -0.1, -0.05],
    #             "increase_weight_points_close": False,
    #             "increase_weight_points_far": True,
    #             "method": "linear",
    #         },
    #         {
    #             "bounds": [0.11, 0.2, -0.12, -0.07],
    #             "increase_weight_points_close": False,
    #             "increase_weight_points_far": True,
    #             "method": "linear",
    #         },
    #     )
