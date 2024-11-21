import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from pathlib import Path
import pandas as pd
import os
from typing import TypedDict, Optional
from utils import project_dir
from io import StringIO
from defining_bound_volume import boundary_ellipse, boundary_rectangle
import force_from_noca
from calculating_circulation import calculate_circulation

from plotting import *


def scaling_velocity(data_array, headers, vel_scaling=15):
    """Scale velocity components in the data array by the given factor, ignoring x, y, z columns."""
    # Find the indices of velocity-related columns (anything except 'x', 'y', 'z')
    velocity_indices = [
        i for i, header in enumerate(headers) if header not in ["x", "y", "z"]
    ]
    # Scale the velocity components by the given factor
    data_array[:, velocity_indices] *= vel_scaling
    return data_array


# def scaling_velocity(df, velocity_columns, vel_scaling=15):
#     """Scale velocity components in the DataFrame."""
#     for col in velocity_columns:
#         if col not in ["x", "y", "z"]:
#             df[col] *= vel_scaling
#     return df


def transform_raw_csv_to_processed_df(alpha=6) -> pd.DataFrame:
    file_path = (
        Path(project_dir)
        / "data"
        / "CFD_slices"
        / "spanwise_slices"
        / f"alpha_{alpha}_CFD_spanwise_slice_25cm_1.csv"
    )
    # Load the raw data
    df = pd.read_csv(file_path)

    # Transform the headers
    df.columns = ["z", "y", "x", "u", "v", "w"]

    # filter data
    y_range = (-1.6, 0.5)
    x_range = (-0.2, 2.4)
    # y_range = (-0.25, 0.25)
    # x_range = (0, 0.4)
    y_mask = (df["y"] >= y_range[0]) & (df["y"] <= y_range[1])
    x_mask = (df["x"] >= x_range[0]) & (df["x"] <= x_range[1])
    # add a mask for when u,v,w are all 0
    # vel_mask = df["u"] == 0  # | (df["v"] == 0) | (df["w"] == 0)
    mask = x_mask & y_mask
    df = df[mask]

    # Store the x,y values where u, v, w are all zero
    vel_mask = (df["u"] == 0) & (df["v"] == 0) & (df["w"] == 0)
    zero_vel_df = df[vel_mask]

    # scale velocity
    headers = ["z", "y", "x", "u", "v", "w"]
    scaled_data = scaling_velocity(df.values, headers, vel_scaling=15)
    final_df = pd.DataFrame(scaled_data, columns=headers)

    # scale dimensions
    final_df["x"] *= 2.584 / 6.5
    final_df["y"] *= 2.584 / 6.5
    zero_vel_df["x"] *= 2.584 / 6.5
    zero_vel_df["y"] *= 2.584 / 6.5

    # remove columns
    # final_df.drop(columns=["z", "u", "v"], inplace=True)
    # print(f"final header: {final_df.columns}")

    # Print max and min values
    print(f"max u: {df['u'].max()}")
    print(f"min u: {df['u'].min()}")
    print(f"max v: {df['v'].max()}")
    print(f"min v: {df['v'].min()}")
    print(f"max w: {df['w'].max()}")
    print(f"min w: {df['w'].min()}")

    # Saving df
    save_path = (
        Path(project_dir)
        / "processed_data"
        / "CFD"
        / "spanwise_slices"
        / f"alpha_{alpha}_CFD_spanwise_slice_50cm_1.csv"
    )
    final_df.to_csv(save_path, index=False)
    return final_df, zero_vel_df


# def preprocess_data(df, plot_params):
#     """
#     Create a subsampled grid of (x, y) pairs and extract matching data
#     without using a costly merge operation.
#     """
#     # Extract unique x and y values
#     x_unique = np.sort(df["x"].unique())
#     y_unique = np.sort(df["y"].unique())

#     # Subsample x and y values
#     subsample_factor = 10  # Default to 10
#     x_subsampled = x_unique[::subsample_factor]
#     y_subsampled = y_unique[::subsample_factor]

#     # Create a subsampled grid of (x, y) pairs
#     x_grid, y_grid = np.meshgrid(x_subsampled, y_subsampled)

#     # Use broadcasting to find the rows in the DataFrame that match the subsampled grid
#     mask_x = df["x"].isin(x_subsampled)
#     mask_y = df["y"].isin(y_subsampled)
#     df_subsampled = df[mask_x & mask_y]

#     # Optional: Ensure grid completeness by filling missing values
#     # Pivot the data into a grid-like structure to fill gaps
#     color_column = plot_params["color_data_col_name"]
#     pivot_table = df_subsampled.pivot(index="y", columns="x", values=color_column)
#     pivot_table = pivot_table.reindex(
#         index=y_subsampled, columns=x_subsampled, fill_value=0
#     )

#     # Flatten back into a DataFrame for compatibility with plotting
#     df_processed = pivot_table.stack().reset_index()
#     df_processed.columns = ["y", "x", color_column]

#     return df_processed, x_grid, y_grid


def main(plot_params: dict) -> None:
    """Create a single plot with the specified parameters."""
    fig, ax = plt.subplots()

    # Transform raw data to processed data
    df, zero_vel_df = transform_raw_csv_to_processed_df(plot_params["alpha"])

    # Extract unique x, y, and w values
    x_unique = df["x"].values
    y_unique = df["y"].values
    color_values = df[plot_params["color_data_col_name"]].values

    # Create a regular grid based on unique x and y values
    x_grid = np.linspace(
        x_unique.min(),
        x_unique.max(),
        int(len(x_unique) / plot_params["subsample_color"]),
    )
    y_grid = np.linspace(
        y_unique.min(),
        y_unique.max(),
        int(len(y_unique) / plot_params["subsample_color"]),
    )

    # Create a meshgrid
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Interpolate the w values onto the grid using griddata
    color_data = griddata(
        (x_unique, y_unique), color_values, (X_grid, Y_grid), method="linear"
    )

    if plot_params["min_cbar_value"] is None or plot_params["max_cbar_value"] is None:
        mean_val = np.nanmean(color_data)
        std_val = np.nanstd(color_data)
        plot_params["min_cbar_value"] = (
            mean_val - plot_params["cbar_value_factor_of_std"] * std_val
        )
        plot_params["max_cbar_value"] = (
            mean_val + plot_params["cbar_value_factor_of_std"] * std_val
        )
    # Plot the contour
    plot_params["cax"] = ax.contourf(
        X_grid,
        Y_grid,
        color_data,
        levels=plot_params["countour_levels"],
        cmap=plot_params["cmap"],
        vmin=plot_params["min_cbar_value"],
        vmax=plot_params["max_cbar_value"],
    )

    # Plotting zeros
    ax.scatter(zero_vel_df["x"], zero_vel_df["y"], c="black", s=0.3)

    add_colorbar(fig, ax, plot_params)
    save_path = (
        Path(plot_params["save_dir"])
        / f"alpha_{plot_params['alpha']}_Y{plot_params['y_num']}_{plot_params['color_data_col_name']}.pdf"
    )
    ax.set_aspect("equal")
    ax.set_xlim(plot_params["xlim"])
    ax.set_ylim(plot_params["ylim"])
    fig.savefig(save_path)
    plt.close()


def plot_contour_with_mask(plot_params):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Transform raw data to processed data
    df, zero_vel_df = transform_raw_csv_to_processed_df(plot_params["alpha"])

    # Extract unique x, y, and w values
    x_unique = df["x"].values
    y_unique = df["y"].values
    color_values = df[plot_params["color_data_col_name"]].values

    # Create a regular grid based on unique x and y values
    x_grid = np.linspace(
        x_unique.min(),
        x_unique.max(),
        int(len(x_unique) / plot_params["subsample_color"]),
    )
    y_grid = np.linspace(
        y_unique.min(),
        y_unique.max(),
        int(len(y_unique) / plot_params["subsample_color"]),
    )

    # Create a meshgrid
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Interpolate the w values onto the grid using griddata
    color_data = griddata(
        (x_unique, y_unique), color_values, (X_grid, Y_grid), method="linear"
    )

    if plot_params["min_cbar_value"] is None or plot_params["max_cbar_value"] is None:
        mean_val = np.nanmean(color_data)
        std_val = np.nanstd(color_data)
        plot_params["min_cbar_value"] = (
            mean_val - plot_params["cbar_value_factor_of_std"] * std_val
        )
        plot_params["max_cbar_value"] = (
            mean_val + plot_params["cbar_value_factor_of_std"] * std_val
        )

    # First plot: Original data
    ax1 = axes[0]
    plot_params["cax"] = ax1.contourf(
        X_grid,
        Y_grid,
        color_data,
        levels=plot_params["countour_levels"],
        cmap=plot_params["cmap"],
        vmin=plot_params["min_cbar_value"],
        vmax=plot_params["max_cbar_value"],
    )

    # Plotting zeros on the first plot
    ax1.scatter(zero_vel_df["x"], zero_vel_df["y"], c="black", s=0.3)

    # Second plot: Masked data (abs(color_values) > 4)
    mask = np.abs(color_values) <= 4  # Mask values greater than 4
    x_unique_masked = x_unique[mask]
    y_unique_masked = y_unique[mask]
    color_values_masked = color_values[mask]

    # Interpolate the masked w values onto the grid using griddata
    color_data_masked = griddata(
        (x_unique_masked, y_unique_masked),
        color_values_masked,
        (X_grid, Y_grid),
        method="linear",
    )

    ax2 = axes[1]
    plot_params["cax"] = ax2.contourf(
        X_grid,
        Y_grid,
        color_data_masked,
        levels=plot_params["countour_levels"],
        cmap=plot_params["cmap"],
        vmin=plot_params["min_cbar_value"],
        vmax=plot_params["max_cbar_value"],
    )

    # Plotting zeros on the second plot
    ax2.scatter(zero_vel_df["x"], zero_vel_df["y"], c="black", s=0.3)

    # Adjusting axes and adding colorbars
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(plot_params["xlim"])
        ax.set_ylim(plot_params["ylim"])

    add_colorbar(fig, axes[0], plot_params)

    # Save the figure
    save_path = (
        Path(plot_params["save_dir"])
        / f"alpha_{plot_params['alpha']}_Y{plot_params['y_num']}_{plot_params['color_data_col_name']}_with_mask.pdf"
    )
    fig.savefig(save_path)
    plt.close()


def plot_contour_with_colored_data(plot_params):
    fig, ax = plt.subplots(figsize=(5, 5))

    # Transform raw data to processed data
    df, zero_vel_df = transform_raw_csv_to_processed_df(plot_params["alpha"])

    # Extract unique x, y, and w values
    x_unique = df["x"].values
    y_unique = df["y"].values
    color_values = df[plot_params["color_data_col_name"]].values

    # Mask the values where abs(color_values) > 4 and color them pink
    mask_pink = np.abs(color_values) > 3
    color_values_pink = np.copy(color_values)
    color_values_pink[mask_pink] = (
        np.nan
    )  # Set masked values to NaN to avoid interpolation interference

    # Create a regular grid based on unique x and y values
    x_grid = np.linspace(
        x_unique.min(),
        x_unique.max(),
        int(len(x_unique) / plot_params["subsample_color"]),
    )
    y_grid = np.linspace(
        y_unique.min(),
        y_unique.max(),
        int(len(y_unique) / plot_params["subsample_color"]),
    )

    # Create a meshgrid
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Interpolate the w values onto the grid using griddata
    color_data = griddata(
        (x_unique, y_unique), color_values_pink, (X_grid, Y_grid), method="linear"
    )

    if plot_params["min_cbar_value"] is None or plot_params["max_cbar_value"] is None:
        mean_val = np.nanmean(color_data)
        std_val = np.nanstd(color_data)
        plot_params["min_cbar_value"] = (
            mean_val - plot_params["cbar_value_factor_of_std"] * std_val
        )
        plot_params["max_cbar_value"] = (
            mean_val + plot_params["cbar_value_factor_of_std"] * std_val
        )

    # # Plot the contour
    plot_params["cax"] = ax.contourf(
        X_grid,
        Y_grid,
        color_data,
        levels=plot_params["countour_levels"],
        cmap=plot_params["cmap"],
        vmin=plot_params["min_cbar_value"],
        vmax=plot_params["max_cbar_value"],
    )

    # Plot the points where abs(color_values) > 4 in pink
    ax.scatter(
        x_unique[mask_pink],
        y_unique[mask_pink],
        c="yellow",
        s=1,  # Adjust point size as needed
        label="abs(w) > 3",
    )
    ax.grid(False)
    # ax.tick_params(
    #     axis="both", which="both", bottom=False, top=False, left=False, right=False
    # )

    # Add colorbar and labels
    add_colorbar(fig, ax, plot_params)

    # Plotting zeros
    ax.scatter(zero_vel_df["x"], zero_vel_df["y"], c="black", s=0.3)

    # Adjust plot settings
    ax.set_aspect("equal")
    ax.set_xlim(plot_params["xlim"])
    ax.set_ylim(plot_params["ylim"])

    # Save the plot
    save_path = (
        Path(plot_params["save_dir"])
        / f"alpha_{plot_params['alpha']}_Y{plot_params['y_num']}_{plot_params['color_data_col_name']}_colored.pdf"
    )
    fig.savefig(save_path)
    plt.close()


def plot_contour_with_colored_data(plot_params):
    # Create a figure with two side-by-side subplots
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
        gridspec_kw={
            "hspace": 0.01,  # A bit more vertical space for labels
            "wspace": 0.07,
        },
    )
    # Create two separate plot_params for each subplot
    plot_params_1 = plot_params.copy()
    plot_params_2 = plot_params.copy()

    # Set different alpha values for each subplot
    plot_params_1["alpha"] = 6
    plot_params_2["alpha"] = 16

    for idx, (ax, curr_plot_params) in enumerate(
        zip(axes, [plot_params_1, plot_params_2])
    ):
        # Transform raw data to processed data
        df, zero_vel_df = transform_raw_csv_to_processed_df(curr_plot_params["alpha"])

        # Extract unique x, y, and w values
        x_unique = df["x"].values
        y_unique = df["y"].values
        color_values = df[curr_plot_params["color_data_col_name"]].values

        # Mask the values where abs(color_values) > 4 and color them pink
        mask_pink = np.abs(color_values) > 3
        color_values_pink = np.copy(color_values)
        color_values_pink[mask_pink] = (
            np.nan
        )  # Set masked values to NaN to avoid interpolation interference

        # Create a regular grid based on unique x and y values
        x_grid = np.linspace(
            x_unique.min(),
            x_unique.max(),
            int(len(x_unique) / curr_plot_params["subsample_color"]),
        )
        y_grid = np.linspace(
            y_unique.min(),
            y_unique.max(),
            int(len(y_unique) / curr_plot_params["subsample_color"]),
        )

        # Create a meshgrid
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        # Interpolate the w values onto the grid using griddata
        color_data = griddata(
            (x_unique, y_unique), color_values_pink, (X_grid, Y_grid), method="linear"
        )

        if (
            curr_plot_params["min_cbar_value"] is None
            or curr_plot_params["max_cbar_value"] is None
        ):
            mean_val = np.nanmean(color_data)
            std_val = np.nanstd(color_data)
            curr_plot_params["min_cbar_value"] = (
                mean_val - curr_plot_params["cbar_value_factor_of_std"] * std_val
            )
            curr_plot_params["max_cbar_value"] = (
                mean_val + curr_plot_params["cbar_value_factor_of_std"] * std_val
            )

        # Plot the contour
        curr_plot_params["cax"] = ax.contourf(
            X_grid,
            Y_grid,
            color_data,
            levels=curr_plot_params["countour_levels"],
            cmap=curr_plot_params["cmap"],
            vmin=curr_plot_params["min_cbar_value"],
            vmax=curr_plot_params["max_cbar_value"],
        )

        # Plot the points where abs(color_values) > 4 in pink
        ax.scatter(
            x_unique[mask_pink],
            y_unique[mask_pink],
            c="lightgreen",
            s=1,  # Adjust point size as needed
            label="abs(w) > 3",
        )
        ax.grid(False)

        # Plotting zeros
        ax.scatter(zero_vel_df["x"], zero_vel_df["y"], c="black", s=0.3)

        # Adjust plot settings
        ax.set_aspect("equal")
        ax.set_xlim(curr_plot_params["xlim"])
        ax.set_ylim(curr_plot_params["ylim"])

        # Add title to distinguish the subplots
        ax.set_title(r"$\alpha$ = {} ".format(curr_plot_params["alpha"]))

        # Apply label logic
        if curr_plot_params["alpha"] == 6:
            # For alpha 6: bottom x-label, no y-label
            ax.xaxis.set_label_position("bottom")
            ax.xaxis.tick_bottom()
            ax.set_xlabel("x [m]")

            ax.set_ylabel(None)
            ax.tick_params(labelleft=False, labelright=False)
        else:  # alpha 16
            # For alpha 16: bottom x-label, left y-label
            ax.xaxis.set_label_position("bottom")
            ax.xaxis.tick_bottom()
            ax.set_xlabel("x [m]")

            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_ylabel("y [m]")

    # Add a single colorbar for the entire figure
    plot_params["cax"] = curr_plot_params["cax"]
    add_vertical_colorbar_for_row(fig, axes[:], plot_params)

    # Save the plot
    save_path = (
        Path(project_dir)
        / "results"
        / "paper_plots"
        / "spanwise_CFD_alpha_comparison.pdf"
    )
    fig.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    from plot_styling import set_plot_style

    set_plot_style()

    plot_params: PlotParams = {
        # Basic configuration
        "is_CFD": False,
        "spanwise_CFD": True,
        "y_num": 1,
        "alpha": 16,
        "d_alpha_rod": 7.25,
        "is_with_mask": False,
        "is_with_interpolation": False,
        "is_with_bound": False,
        "is_with_airfoil": False,
        "is_with_overlay": False,
        "is_CFD_PIV_comparison": False,
        # Plot settings
        "xlim": (0.0, 0.8),  # (-0.05, 0.75),  #
        "ylim": (-0.6, 0.1),  # (-0.65, 0.35),  #
        # Color and contour settings
        "color_data_col_name": "w",
        "min_cbar_value": -3,
        "max_cbar_value": 3,
        "is_with_cbar": True,
        "cbar_value_factor_of_std": 2.0,
        "subsample_color": 40,
        "countour_levels": 100,
        "cmap": "coolwarm",
        # Quiver settings
        "is_with_quiver": True,
        "subsample_quiver": 5,
        "u_inf": 15.0,
        # Saving directory
        "save_dir": Path(project_dir) / "results" / "spanwise_CFD_slices",
        # Data loading path
    }

    # main(plot_params)
    # plot_contour_with_mask(plot_params)
    plot_contour_with_colored_data(plot_params)
    print(
        f'spanwise CFD plot with color = {plot_params["color_data_col_name"]} | Y{plot_params["y_num"]} | α = {plot_params["alpha"]}°'
    )
