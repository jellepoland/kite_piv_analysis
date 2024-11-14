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


def main(plot_params: dict) -> None:
    """Create a 4x3 comparison of CFD and PIV data, with PIV masked/unmasked."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(
        rf'Y{plot_params["y_num"]} | α = {plot_params["alpha"]}° | V_inf = {plot_params["u_inf"]}m/s'
    )

    is_with_circulation_analysis = plot_params["is_with_circulation_analysis"]
    is_with_bound = plot_params["is_with_bound"]
    is_with_interpolation = plot_params["is_with_interpolation"]
    data_labels = ["u", "v", "w", "V"]

    for i, label in enumerate(data_labels):

        # Update color data label
        plot_params["color_data_col_name"] = label

        ### PIV raw
        plot_params["is_with_interpolation"] = False
        plot_params["is_with_bound"] = False
        plot_params["is_with_circulation_analysis"] = False
        plot_params["is_with_mask"] = False
        df_piv, x_mesh_piv, y_mesh_piv, plot_params = load_data(
            plot_params | {"is_CFD": False}
        )
        plot_params = plotting_on_ax(
            fig, axes[i, 0], df_piv, x_mesh_piv, y_mesh_piv, plot_params
        )
        axes[i, 0].set_title(f"PIV Raw")
        if plot_params["is_with_cbar"]:
            add_colorbar(fig, axes[i, 0], plot_params)

        ### PIV Mask
        plot_params["is_with_interpolation"] = False
        plot_params["is_with_bound"] = False
        plot_params["is_with_circulation_analysis"] = False
        plot_params["is_with_mask"] = True
        plot_params["column_to_mask"] = "w"
        plot_params["mask_lower_bound"] = -2.5
        plot_params["mask_upper_bound"] = 2.5
        df_piv, x_mesh_piv, y_mesh_piv, plot_params = load_data(
            plot_params | {"is_CFD": False}
        )
        plot_params = plotting_on_ax(
            fig, axes[i, 1], df_piv, x_mesh_piv, y_mesh_piv, plot_params
        )
        axes[i, 1].set_title(
            f'PIV Masked for {plot_params["column_to_mask"]} in bounds {plot_params["mask_lower_bound"]} to {plot_params["mask_upper_bound"]}'
        )
        if plot_params["is_with_cbar"]:
            add_colorbar(fig, axes[i, 1], plot_params)

        ### PIV Mask Reinterpolated
        if is_with_bound:
            plot_params["is_with_bound"] = True
        if is_with_circulation_analysis:
            plot_params["is_with_circulation_analysis"] = True
        if is_with_interpolation:
            plot_params["is_with_interpolation"] = True

        plot_params["interpolation_method"] = "nearest"
        df_piv, x_mesh_piv, y_mesh_piv, plot_params = load_data(
            plot_params | {"is_CFD": False}
        )
        plot_params = plotting_on_ax(
            fig, axes[i, 2], df_piv, x_mesh_piv, y_mesh_piv, plot_params
        )
        axes[i, 2].set_title(f"PIV Masked Re-interpolated zones")
        if plot_params["is_with_cbar"]:
            add_colorbar(fig, axes[i, 2], plot_params)

        ### Reset things
        plot_params["min_cbar_value"] = None
        plot_params["max_cbar_value"] = None

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle space
    save_path = (
        Path(plot_params["save_dir"])
        / f"alpha_{plot_params['alpha']}_Y{plot_params['y_num']}.pdf"
    )
    fig.savefig(save_path)
    plt.close()


if __name__ == "__main__":

    plot_params: PlotParams = {
        # Basic configuration
        "is_CFD": False,
        "spanwise_CFD": False,
        "y_num": 1,
        "alpha": 6,
        "project_dir": project_dir,
        "plot_type": ".pdf",
        "title": None,
        "is_CFD_PIV_comparison": False,
        "color_data_col_name": "V",
        "is_CFD_PIV_comparison_multicomponent_masked": True,
        "run_for_all_planes": False,
        # plot ranges
        "xlim": (-0.2, 0.8),
        "ylim": (-0.2, 0.4),
        # Color and contour settings
        "is_with_cbar": True,
        "cbar_value_factor_of_std": 2.0,
        "min_cbar_value": None,
        "max_cbar_value": None,
        "subsample_color": 1,
        "countour_levels": 100,
        "cmap": "viridis",
        # Quiver settings
        "is_with_quiver": True,
        "subsample_quiver": 5,
        "u_inf": 15.0,
        # PIV specific settings
        "d_alpha_rod": 7.25,
        # Overlay settings
        "is_with_overlay": False,
        "overlay_alpha": 0.4,
        # Airfoil settings
        "is_with_airfoil": True,
        "airfoil_transparency": 1.0,
        # Raw image settings
        "subsample_factor_raw_images": 1,
        "intensity_lower_bound": 10000,
        # Boundary settings
        "is_with_bound": True,
        "d1centre": np.array([0.24, 0.13]),
        "drot": 0.0,
        "dLx": 0.56,
        "dLy": 0.4,
        "iP": 65,
        # insert
        # "d1centre": np.array([0.27, 0.13]),
        # "drot": 0,
        # "dLx": 0.8,
        # "dLy": 0.4,
        # "iP": 35,
        ##
        "ellipse_color": "red",
        "rectangle_color": "red",
        "bound_linewidth": 1.0,
        "bound_alpha": 1.0,
        # Circulation analysis
        "is_with_circulation_analysis": False,
        "rho": 1.225,
        "mu": 1.7894e-5,
        "is_with_maximim_vorticity_location_correction": True,
        "chord": 0.37,
        # Mask settings
        "is_with_mask": False,
        "column_to_mask": "w",
        "mask_lower_bound": -3,
        "mask_upper_bound": 3,
        ## Interpolation settings
        "is_with_interpolation": True,
        ## save_path
        "save_dir": Path(project_dir) / "results" / "reflection_and_reinterpolation",
    }

    if plot_params["alpha"] == 6:
        plot_params["interpolation_zones"] = (
            # {
            #     "bounds": [0.43, 0.5, 0.14, 0.19],
            #     "increase_weight_points_close": False,
            #     "increase_weight_points_far": True,
            #     "method": "linear",
            # },
            {
                "bounds": [0.43, 0.55, 0.11, 0.21],
                "increase_weight_points_close": False,
                "increase_weight_points_far": True,
                "method": "linear",
            },
            {
                "bounds": [0.43, 0.55, -0.1, 0.04],
                "increase_weight_points_close": False,
                "increase_weight_points_far": True,
                "method": "linear",
            },
            {
                "bounds": [0.22, 0.3, -0.15, -0.07],
                "increase_weight_points_close": False,
                "increase_weight_points_far": True,
                "method": "linear",
            },
        )
    else:
        plot_params["interpolation_zones"] = (
            # {
            #     "bounds": [0.43, 0.5, 0.14, 0.19],
            #     "increase_weight_points_close": False,
            #     "increase_weight_points_far": True,
            #     "method": "linear",
            # },
            {
                "bounds": [0.47, 0.55, 0.01, 0.1],
                "increase_weight_points_close": False,
                "increase_weight_points_far": True,
                "method": "linear",
            },
            {
                "bounds": [0.47, 0.55, -0.05, 0.01],
                "increase_weight_points_close": False,
                "increase_weight_points_far": True,
                "method": "linear",
            },
            {
                "bounds": [0.35, 0.55, -0.1, -0.05],
                "increase_weight_points_close": False,
                "increase_weight_points_far": True,
                "method": "linear",
            },
            {
                "bounds": [0.11, 0.2, -0.12, -0.07],
                "increase_weight_points_close": False,
                "increase_weight_points_far": True,
                "method": "linear",
            },
        )

    main(plot_params)
    if plot_params["is_CFD_PIV_comparison"]:
        type_label = "CFD_PIV"
    else:
        type_label = "CFD" if plot_params["is_CFD"] else "PIV"

    print(
        f'{type_label} plot with color = {plot_params["color_data_col_name"]} | Y{plot_params["y_num"]} | α = {plot_params["alpha"]}°'
    )
