import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from defining_bound_volume import boundary_ellipse, boundary_rectangle
import force_from_noca
from utils import project_dir
from plot_styling import set_plot_style, plot_on_ax
import matplotlib.pyplot as plt
import calculating_airfoil_centre
from plotting import *


@dataclass
class NOCAParameters:
    is_ellipse: bool = field(default_factory=lambda: True)
    d1centre: np.ndarray = field(default_factory=lambda: np.array([0.27, 0.13]))
    drot: float = field(default_factory=lambda: 0)
    dLx: float = field(default_factory=lambda: 0.7)
    dLy: float = field(default_factory=lambda: 0.33)
    iP: int = field(default_factory=lambda: 93)


# def get_sweep_values(parameter_name: str, alpha: float, y_num: int) -> List:

#     if alpha == 6 and y_num == 1:
#         dLx = np.linspace(0.05, 1.0, 25)

#     sweep_values = {
#         "is_ellipse": [False, True],
#         "dLx": dLX,
#         "dLy": np.linspace(0.05, 0.6, 30),
#         "iP": np.linspace(5, 50, 25),
#         "area": np.linspace(0.2, 0.4, 5),
#     }
#     return sweep_values[parameter_name]


def get_sweep_values(
    parameter_name: str,
    alpha: float,
    y_num: int,
    airfoil_center: tuple,
    xlim: tuple,
    ylim: tuple,
) -> List:
    """
    Returns sweep values for the specified parameter, dynamically adjusting ranges for dLx and dLy
    based on airfoil center and specified limits.
    """
    if parameter_name == "dLx":
        # Compute the maximum range for dLx based on xlim and airfoil_center[0]
        max_positive_dLx = xlim[1] - airfoil_center[0]
        max_negative_dLx = airfoil_center[0] - xlim[0]
        max_dLx = min(
            max_positive_dLx, max_negative_dLx
        )  # Take the minimum of both bounds
        # print(
        #     f"max_positive_dLx: {max_positive_dLx}, max_negative_dLx: {max_negative_dLx}, max_dLx: {max_dLx}"
        # )
        dLx = np.linspace(0.35, 1.8 * max_dLx, 50)  # Adjust the range dynamically
        return dLx

    elif parameter_name == "dLy":
        # Compute the maximum range for dLy based on ylim and airfoil_center[1]
        max_positive_dLy = ylim[1] - airfoil_center[1]
        max_negative_dLy = airfoil_center[1] - ylim[0]
        max_dLy = min(
            max_positive_dLy, max_negative_dLy
        )  # Take the minimum of both bounds
        # print(
        #     f"max_positive_dLy: {max_positive_dLy}, max_negative_dLy: {max_negative_dLy}, max_dLy: {max_dLy}"
        # )
        # print(f"max_dLy + airfoil_center: {max_dLy + airfoil_center[1]}")
        dLy = np.linspace(0.1, 1.8 * max_dLy, 50)  # Adjust the range dynamically
        return dLy

    elif parameter_name == "is_ellipse":
        return [False, True]

    elif parameter_name == "iP":
        # make sure to not include 9+6n numbers as they cause problems in making the rectangle
        # return [int(i) for i in np.linspace(20, 151, 50) if ((int(i) - 9) % 6 != 0)]
        return [
            20,
            25,
            35,
            40,
            45,
            55,
            65,
            70,
            75,
            80,
            85,
            95,
            105,
        ]

    elif parameter_name == "area":
        return np.linspace(0.2, 0.4, 5)

    elif parameter_name == "drot":
        return np.linspace(-10, 10, 20)

    raise ValueError(f"Unknown parameter name: {parameter_name}")


def reading_optimal_bound_placement(
    alpha: float,
    y_num: int,
    is_with_N_datapoints: bool = False,
) -> Tuple[float, float]:
    """
    Read the optimal bound placement from a CSV file.

    Args:
        alpha (float): Angle of attack.
        y_num (int): Y value index.

    Returns:
        Tuple[float, float]: Optimal dLx and dLy values.
    """
    # Reading in the airfoil centers
    df_optimal_bound_placement = pd.read_csv(
        Path(project_dir) / "processed_data" / "optimal_bound_placement.csv"
    )

    mask = (df_optimal_bound_placement["alpha"] == alpha) & (
        df_optimal_bound_placement["Y"] == y_num
    )
    try:
        dLx, dLy = df_optimal_bound_placement.loc[mask, ["dLx", "dLy"]].values[0]
    except IndexError:
        # If the specific (alpha, y_num) combination is not found, return NaNs
        dLx, dLy = np.nan, np.nan

    # to enable the interpolator to work within the convergence study
    if is_with_N_datapoints:
        N_datapoints = df_optimal_bound_placement.loc[mask, "N_datapoints"].values[0]
        return dLx, dLy, N_datapoints

    else:
        return dLx, dLy


def parameter_sweep_noca(
    is_CFD: bool,
    alpha: float,
    y_num: int,
    parameter_name: str,
    is_ellipse: bool = True,
    base_params: NOCAParameters = NOCAParameters(),
    mu: float = 1.7894e-5,
    is_with_maximim_vorticity_location_correction: bool = True,
    rho: float = 1.225,
    U_inf: float = 15,
    c: float = 0.37,
    is_small_piv: bool = False,
    max_number_of_interpolation_zones: int = 5,
) -> pd.DataFrame:
    """
    Perform parameter sweep for NOCA analysis.

    Args:
        parameter_name: Name of parameter to sweep ('is_ellipse', 'd1centre', 'dLx', 'dLy', 'iP', 'area')
        parameter_values: List or array of values to sweep through
        base_params: Base parameters to use (only the swept parameter will be changed)
        mu: Dynamic viscosity
        is_with_maximim_vorticity_location_correction: Boolean for vorticity correction
        rho: Fluid density
        U_inf: Freestream velocity
        c: Chord length

    Returns:
        DataFrame containing sweep results
    """
    # Loading the data
    plot_params = {
        "y_num": y_num,
        "alpha": alpha,
        "is_CFD": is_CFD,
        "d_alpha_rod": 7.25,
        "column_to_mask": "w",
        "mask_lower_bound": -3,
        "mask_upper_bound": 3,
        "csv_file_path_std": None,
        "spanwise_CFD": False,
        # interpolation settings
        "rectangle_size": 0.05,
    }
    df, x_meshgrid, y_meshgrid, plot_params = load_data(plot_params)

    # Apply mask if is PIV
    if not plot_params["is_CFD"]:
        df = apply_mask(df, plot_params)

    results = []

    # Reading in the airfoil centers
    airfoil_center = calculating_airfoil_centre.main(alpha, y_num)

    # Get sweep values for the specified parameter
    parameter_values = get_sweep_values(
        parameter_name, alpha, y_num, airfoil_center, xlim=(-0.2, 0.8), ylim=(-0.2, 0.4)
    )

    # Checking the parameter value range, if it is out of bounds, stop the loop
    xlim = (-0.2, 0.8)
    ylim = (-0.2, 0.4)

    if parameter_name == "dLx":
        max_dLy = ylim[1] - airfoil_center[1]
    elif parameter_name == "dLy":
        max_dLx = xlim[1] - airfoil_center[0]

    # if parameter_name == "dLx":
    #     x_values_plus = airfoil_center[0] + parameter_values
    #     x_values_minus = airfoil_center[0] - parameter_values

    #     if np.any(x_values_plus > xlim[1]):
    #         raise ValueError(
    #             f"Change x_values_plus: {x_values_plus}, it is out of bounds"
    #         )
    #     if np.any(x_values_minus < xlim[0]):
    #         raise ValueError(
    #             f"Change x_values_minus: {x_values_minus}, it is out of bounds"
    #         )
    # elif parameter_name == "dLy":
    #     y_values_plus = airfoil_center[1] + parameter_values
    #     y_values_minus = airfoil_center[1] - parameter_values

    #     if np.any(y_values_plus > ylim[1]):
    #         raise ValueError(
    #             f"Change y_values_plus: {y_values_plus}, it is out of bounds"
    #         )
    #     if np.any(y_values_minus < ylim[0]):
    #         raise ValueError(
    #             f"Change y_values_minus: {y_values_minus}, it is out of bounds"
    #         )

    dLx, dLy = reading_optimal_bound_placement(alpha, y_num)

    # When PIV, the cells have to be interpolated, therefore we are only checking convergence around a small area, surrounding the optimal bound
    if is_small_piv and not is_CFD:
        if parameter_name == "dLx":
            parameter_values = np.linspace(dLx - 0.05 * dLx, dLx + 0.05 * dLx, 10)
        elif parameter_name == "dLy":
            parameter_values = np.linspace(dLy - 0.05 * dLy, dLy + 0.05 * dLy, 10)

    for value in parameter_values:
        # Create copy of base parameters
        current_params = NOCAParameters(**vars(base_params))

        # Set the airfoil center
        current_params.d1centre = airfoil_center

        # Update the parameter being swept
        if is_ellipse:
            current_params.is_ellipse = True
        else:
            current_params.is_ellipse = False

        # elif parameter_name == "d1centre":
        #     current_params.d1centre = np.array(value)
        if parameter_name == "dLx":
            current_params.dLx = value
            current_params.dLy = dLy  # 1.7 * max_dLy
        elif parameter_name == "dLy":
            current_params.dLy = value
            current_params.dLx = dLx  # 1.7 * max_dLx
        elif parameter_name == "iP":
            current_params.iP = int(value)
        elif parameter_name == "drot":
            current_params.drot = np.deg2rad(value)
        elif parameter_name == "area":
            # For area sweeps, adjust dLx and dLy to maintain aspect ratio
            aspect_ratio = current_params.dLx / current_params.dLy
            current_params.dLy = np.sqrt(value / (np.pi * aspect_ratio))
            current_params.dLx = current_params.dLy * aspect_ratio
        else:
            raise ValueError(f"Unknown parameter name: {parameter_name}")

        # Generate boundary curve
        if current_params.is_ellipse:
            d2curve = boundary_ellipse(
                current_params.d1centre,
                current_params.drot,
                current_params.dLx,
                current_params.dLy,
                current_params.iP,
            )
        else:
            d2curve = boundary_rectangle(
                current_params.d1centre,
                current_params.drot,
                current_params.dLx,
                current_params.dLy,
                current_params.iP,
            )

        # Interpolate the data if PIV
        is_skip_this_parameter = False
        if not plot_params["is_CFD"]:
            plot_params["interpolation_zones"] = find_areas_needing_interpolation(
                df,
                plot_params["alpha"],
                plot_params["y_num"],
                plot_params["rectangle_size"],
                dLx=current_params.dLx,
                dLy=current_params.dLy,
            )

            # to make sure that we are not interpolating too many areas, we set the is_skip_parameter

            if (
                len(plot_params["interpolation_zones"])
                < max_number_of_interpolation_zones
            ):

                for interpolation_zone_i in plot_params["interpolation_zones"]:
                    df, d2curve_rectangle_interpolated_zone = interpolate_missing_data(
                        df,
                        interpolation_zone_i,
                    )
            else:
                print(
                    f'---> SKIPPED n_zones: {len(plot_params["interpolation_zones"])}, dLx: {current_params.dLx:.2f}, dLy: {current_params.dLy:.2f}'
                )
                is_skip_this_parameter = True
        if is_skip_this_parameter:
            continue

        # Run NOCA analysis
        F_x, F_y, C_l, C_d = force_from_noca.main(
            df,
            d2curve,
            mu=mu,
            is_with_maximim_vorticity_location_correction=is_with_maximim_vorticity_location_correction,
            rho=rho,
            U_inf=U_inf,
            c=c,
        )

        # Calculate area
        if current_params.is_ellipse:
            area = np.pi * current_params.dLx * current_params.dLy / 4
        else:
            area = current_params.dLx * current_params.dLy

        # Store results
        results.append(
            {
                "parameter_value": value,
                "F_x": F_x,
                "F_y": F_y,
                "C_l": C_l,
                "C_d": C_d,
                "area": area,
                "is_ellipse": current_params.is_ellipse,
                "d1centre_x": current_params.d1centre[0],
                "d1centre_y": current_params.d1centre[1],
                "dLx": current_params.dLx,
                "dLy": current_params.dLy,
                "iP": int(current_params.iP),
                "drot": np.rad2deg(current_params.drot),
            }
        )

    return results


def plot_noca_coefficients(
    results_df: pd.DataFrame,
    parameter_name: str,
    save_path: Optional[Union[str, Path]] = None,
    colors: tuple = ("blue", "red"),
    markers: tuple = ("o", "s"),
    markersize: int = 1,
) -> tuple:
    """
    Create a double plot showing C_l and C_d versus the swept parameter.

    Args:
        results_df: DataFrame containing the sweep results
        parameter_name: Name of the parameter that was swept
        figure_size: Size of the figure (width, height)
        save_path: Optional path to save the figure
        colors: Tuple of colors for C_l and C_d plots
        markers: Tuple of markers for C_l and C_d plots
        markersize: Size of markers

    Returns:
        tuple: (fig, (ax1, ax2)) - Figure and axes objects
    """
    set_plot_style()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.3)  # Add space between subplots

    # Get x values (parameter values)
    x = results_df["parameter_value"]

    # Format x-label based on parameter name
    x_label_map = {
        "is_ellipse": "Shape Type (False=Rectangle, True=Ellipse)",
        "d1centre": "Center Position [m]",
        "dLx": "X Length [m]",
        "dLy": "Y Length [m]",
        "iP": "Number of Points",
        "area": "Area [m²]",
    }
    x_label = x_label_map.get(parameter_name, parameter_name)

    # Special handling for d1centre parameter
    if parameter_name == "d1centre":
        # Create more meaningful x values for center position
        x = [
            f"({row.d1centre_x:.2f}, {row.d1centre_y:.2f})"
            for _, row in results_df.iterrows()
        ]

    # Plot C_l
    plot_on_ax(
        ax=ax1,
        x=x,
        y=results_df["C_l"],
        label="$C_l$",
        color=colors[0],
        marker=markers[0],
        markersize=markersize,
        x_label=x_label,
        y_label="$C_l$ [-]",
        is_with_x_label=False,  # Only show x-label on bottom plot
    )

    # Plot C_d
    plot_on_ax(
        ax=ax2,
        x=x,
        y=results_df["C_d"],
        label="$C_d$",
        color=colors[1],
        marker=markers[1],
        markersize=markersize,
        x_label=x_label,
        y_label="$C_d$ [-]",
    )

    # Add title
    title_map = {
        "is_ellipse": "Shape Type Effect on Force Coefficients",
        "d1centre": "Center Position Effect on Force Coefficients",
        "dLx": "X Length Effect on Force Coefficients",
        "dLy": "Y Length Effect on Force Coefficients",
        "iP": "Number of Points Effect on Force Coefficients",
        "area": "Area Effect on Force Coefficients",
    }
    fig.suptitle(
        title_map.get(parameter_name, f"{parameter_name} Effect on Force Coefficients")
    )

    # Save figure if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig, (ax1, ax2)


# def plot_noca_coefficients_grid(
#     results_dict: dict,
#     save_path: Optional[Union[str, Path]] = None,
#     colors: tuple = ("blue", "red"),
#     markers: tuple = ("o", "s"),
#     markersize: int = 6,
# ):
#     """
#     Create a grid plot with 4 rows and 2 columns, where:
#     - The first column shows $C_l$ for different parameters.
#     - The second column shows $C_d$ for different parameters.

#     Args:
#         results_dict: Dictionary containing results DataFrames for parameters.
#                       Keys should be parameter names ('dLx', 'dLy', 'iP', 'drot').
#         save_path: Optional path to save the figure.
#         colors: Tuple of colors for $C_l$ and $C_d$ plots.
#         markers: Tuple of markers for $C_l$ and $C_d$ plots.
#         markersize: Size of markers.

#     Returns:
#         tuple: (fig, axes) - Figure and axes objects.
#     """
#     set_plot_style()

#     # Create figure and axes
#     fig, axes = plt.subplots(4, 2, figsize=(10, 20))
#     fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing

#     # Define parameter names and labels
#     parameter_names = ["dLx", "dLy", "iP", "drot"]
#     x_label_map = {
#         "dLx": "X Length [m]",
#         "dLy": "Y Length [m]",
#         "iP": "Number of Points",
#         "drot": "Rotation Angle [deg]",
#     }

#     # Iterate over rows (parameters) and plot $C_l$ and $C_d$
#     for i, param in enumerate(parameter_names):
#         results_df = results_dict.get(param)
#         if results_df is None:
#             raise ValueError(
#                 f"Missing results for parameter '{param}' in results_dict."
#             )

#         # Extract x-values (parameter values)
#         x = results_df["parameter_value"]
#         x_label = x_label_map.get(param, param)

#         # Plot $C_l$ in the first column
#         plot_on_ax(
#             ax=axes[i, 0],
#             x=x,
#             y=results_df["C_l"],
#             label="$C_l$",
#             color=colors[0],
#             marker=markers[0],
#             markersize=markersize,
#             x_label=x_label,
#             y_label="$C_l$ [-]",
#             is_with_grid=False,
#         )

#         # Plot $C_d$ in the second column
#         plot_on_ax(
#             ax=axes[i, 1],
#             x=x,
#             y=results_df["C_d"],
#             label="$C_d$",
#             color=colors[1],
#             marker=markers[1],
#             markersize=markersize,
#             x_label=x_label,
#             y_label="$C_d$ [-]",
#             is_with_grid=False,
#         )

#         # Add titles for rows
#         # axes[i, 0].set_title(f"{param}: $C_l$")
#         # axes[i, 1].set_title(f"{param}: $C_d$")

#     # Save figure if path provided
#     if save_path is not None:
#         save_path = Path(save_path)
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(save_path, bbox_inches="tight", dpi=300)
#         print(f"Figure saved to {save_path}")

#     plt.show()
#     return fig, axes


# def plot_noca_coefficients_grid(
#     is_CFD: bool,
#     alpha: int,
#     y_num: int,
#     save_path: Optional[Union[str, Path]] = None,
#     parameter_names: List = ["drot", "iP", "dLx", "dLy"],
#     colors: tuple = ("blue", "red"),
#     markers: tuple = ("o", "s"),
#     markersize: int = 6,
# ):
#     """
#     Create a grid plot with 4 rows and 4 columns:
#     - Columns 1 and 2 show $C_l$ for different parameters (Ellipse and Non-Ellipse).
#     - Columns 3 and 4 show $C_d$ for different parameters (Ellipse and Non-Ellipse).

#     Args:
#         is_CFD: Boolean indicating if results are CFD-based.
#         alpha: Angle of attack.
#         y_num: Y value index.
#         save_path: Optional path to save the figure.
#         colors: Tuple of colors for $C_l$ and $C_d$ plots.
#         markers: Tuple of markers for $C_l$ and $C_d$ plots.
#         markersize: Size of markers.

#     Returns:
#         tuple: (fig, axes) - Figure and axes objects.
#     """
#     set_plot_style()

#     # Parameter names

#     # Collect results for all parameters
#     results_dict = {}
#     for param in parameter_names:
#         results_dict[param] = {
#             "Ellipse": parameter_sweep_noca(
#                 is_CFD, alpha, y_num, param, is_ellipse=True
#             ),
#             "Non-Ellipse": parameter_sweep_noca(
#                 is_CFD, alpha, y_num, param, is_ellipse=False
#             ),
#         }

#     # Create figure and axes
#     fig, axes = plt.subplots(
#         len(parameter_names), 4, figsize=(20, int(len(parameter_names) * 5))
#     )
#     fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing
#     if is_CFD:
#         type_data = "CFD"
#     else:
#         type_data = "PIV"
#     # plt.title(f"{type_data} | Alpha={alpha} | Y={y_num}")

#     # Define labels
#     x_label_map = {
#         "dLx": "X Length [m]",
#         "dLy": "Y Length [m]",
#         "iP": "Number of Points",
#         "drot": "Rotation Angle [deg]",
#     }

#     # Iterate over rows (parameters) and plot
#     for row_idx, param in enumerate(parameter_names):
#         for col_idx, (key, result) in enumerate(
#             results_dict[param].items()
#         ):  # Ellipse, Non-Ellipse
#             results_df = pd.DataFrame(result)
#             x = results_df["parameter_value"]
#             x_label = x_label_map.get(param, param)

#             # Plot $C_l$ in columns 1 (Ellipse) and 2 (Non-Ellipse)
#             plot_on_ax(
#                 ax=axes[row_idx, col_idx * 2],
#                 x=x,
#                 y=results_df["C_l"],
#                 label=f"$C_l$ ({key})",
#                 color=colors[0],
#                 # marker=markers[0],
#                 markersize=markersize,
#                 x_label=x_label,
#                 y_label="$C_l$ [-]",
#                 is_with_grid=False,
#             )

#             # Plot $C_d$ in columns 3 (Ellipse) and 4 (Non-Ellipse)
#             plot_on_ax(
#                 ax=axes[row_idx, col_idx * 2 + 1],
#                 x=x,
#                 y=results_df["C_d"],
#                 label=f"$C_d$ ({key})",
#                 color=colors[1],
#                 # marker=markers[1],
#                 markersize=markersize,
#                 x_label=x_label,
#                 y_label="$C_d$ [-]",
#                 is_with_grid=False,
#             )

#     # Save figure if path provided
#     if save_path is not None:
#         save_path = Path(save_path)
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(save_path)  # , bbox_inches="tight", dpi=300)
#         print(f"Figure saved to {save_path}")

#     # plt.show()
#     return fig, axes


def plot_noca_coefficients_grid(
    is_CFD: bool,
    alpha: int,
    y_num: int,
    save_path: Optional[Union[str, Path]] = None,
    parameter_names: List = ["drot", "iP", "dLx", "dLy"],
    colors: tuple = ("blue", "red"),
    markers: tuple = ("o", "s"),
    markersize: int = 6,
):
    """
    Create a grid plot with 4 rows and 4 columns:
    - Columns 1 and 2 show $C_l$ for different parameters (Ellipse and Non-Ellipse).
    - Columns 3 and 4 show $C_d$ for different parameters (Ellipse and Non-Ellipse).

    Args:
        is_CFD: Boolean indicating if results are CFD-based.
        alpha: Angle of attack.
        y_num: Y value index.
        save_path: Optional path to save the figure.
        colors: Tuple of colors for $C_l$ and $C_d$ plots.
        markers: Tuple of markers for $C_l$ and $C_d$ plots.
        markersize: Size of markers.

    Returns:
        tuple: (fig, axes) - Figure and axes objects.
    """
    set_plot_style()

    # Collect results for all parameters
    results_dict = {}
    for param in parameter_names:
        results_dict[param] = {
            "Ellipse": parameter_sweep_noca(
                is_CFD, alpha, y_num, param, is_ellipse=True
            ),
            "Non-Ellipse": parameter_sweep_noca(
                is_CFD, alpha, y_num, param, is_ellipse=False
            ),
        }

    # Create figure and axes
    fig, axes = plt.subplots(
        len(parameter_names), 4, figsize=(20, int(len(parameter_names) * 3.5))
    )
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing
    if is_CFD:
        type_data = "CFD"
    else:
        type_data = "PIV"
    # plt.title(f"{type_data} | Alpha={alpha} | Y={y_num}")

    # Define labels
    x_label_map = {
        "dLx": "X Length [m]",
        "dLy": "Y Length [m]",
        "iP": "Number of Points",
        "drot": "Rotation Angle [deg]",
    }

    # Iterate over rows (parameters) and plot
    for row_idx, param in enumerate(parameter_names):
        for col_idx, (key, result) in enumerate(
            results_dict[param].items()
        ):  # Ellipse, Non-Ellipse
            results_df = pd.DataFrame(result)
            x = results_df["parameter_value"]
            x_label = x_label_map.get(param, param)

            # Plot $C_l$ in columns 1 (Ellipse) and 2 (Non-Ellipse)
            plot_on_ax(
                ax=axes[row_idx, col_idx * 2],
                x=x,
                y=results_df["C_l"],
                label=f"$C_l$ ({key})",
                color=colors[0],
                # marker=markers[0] if key == "Ellipse" else markers[1],
                # markersize=markersize,
                x_label=x_label,
                y_label="$C_l$ [-]",
                is_with_grid=False,
            )
            if key == "Ellipse":
                axes[row_idx, col_idx * 2].set_title(f"Ellipse {param}")
            else:
                axes[row_idx, col_idx * 2].set_title(f"Rectangle {param}")

            # Plot $C_d$ in columns 3 (Ellipse) and 4 (Non-Ellipse)
            plot_on_ax(
                ax=axes[row_idx, col_idx * 2 + 1],
                x=x,
                y=results_df["C_d"],
                label=f"$C_d$ ({key})",
                color=colors[1],
                # marker=markers[0] if key == "Ellipse" else markers[1],
                # markersize=markersize,
                x_label=x_label,
                y_label="$C_d$ [-]",
                is_with_grid=False,
            )
            if key == "Ellipse":
                axes[row_idx, col_idx * 2 + 1].set_title(f"Ellipse {param}")
            else:
                axes[row_idx, col_idx * 2 + 1].set_title(f"Rectangle {param}")

    # Save figure if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)  # , bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    # plt.show()
    return fig, axes


def plot_noca_coefficients_grid_CFD_PIV(
    alpha: int,
    y_num: int,
    save_path: Optional[Union[str, Path]],
    parameter_names: List,
    colors: tuple = ("blue", "red"),
    markers: tuple = ("o", "s"),
    markersize: int = 6,
):
    """
    Create a grid plot with 4 rows and 4 columns:
    - Columns 1 and 2 show $C_l$ for different parameters (Ellipse and Non-Ellipse).
    - Columns 3 and 4 show $C_d$ for different parameters (Ellipse and Non-Ellipse).
    Each plot contains lines for both PIV and CFD data.

    Args:
        alpha: Angle of attack.
        y_num: Y value index.
        save_path: Optional path to save the figure.
        colors: Tuple of colors for $C_l$ and $C_d$ plots.
        markers: Tuple of markers for $C_l$ and $C_d$ plots.
        markersize: Size of markers.

    Returns:
        tuple: (fig, axes) - Figure and axes objects.
    """
    set_plot_style()

    # Collect results for all parameters and both data types (CFD and PIV)
    results_dict = {data_type: {} for data_type in ["CFD", "PIV"]}
    for data_type in results_dict:
        is_CFD = data_type == "CFD"
        for param in parameter_names:
            results_dict[data_type][param] = {
                "Ellipse": parameter_sweep_noca(
                    is_CFD, alpha, y_num, param, is_ellipse=True
                ),
                "Non-Ellipse": parameter_sweep_noca(
                    is_CFD, alpha, y_num, param, is_ellipse=False
                ),
            }

    # Create figure and axes
    fig, axes = plt.subplots(
        len(parameter_names), 4, figsize=(20, int(len(parameter_names) * 4.3))
    )
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing

    dLx, dLy = reading_optimal_bound_placement(alpha, y_num)
    fig.suptitle(f"Current settings: dLx: {dLx}, dLy: {dLy}")
    # Define labels
    x_label_map = {
        "dLx": "X [m]",
        "dLy": "Y [m]",
        "iP": "Number of Points",
        "drot": "Rotation Angle [deg]",
    }

    # Iterate over rows (parameters) and plot
    for row_idx, param in enumerate(parameter_names):
        for col_idx, (key, _) in enumerate(
            results_dict["CFD"][param].items()
        ):  # Ellipse, Non-Ellipse
            x_label = x_label_map.get(param, param)

            # Iterate over data types (PIV and CFD)
            for data_type_idx, data_type in enumerate(["CFD", "PIV"]):
                results_df = pd.DataFrame(results_dict[data_type][param][key])
                x = results_df["parameter_value"]

                # Define line styles and markers
                linestyle = "-" if data_type == "CFD" else "--"
                marker = markers[data_type_idx]

                # Plot $C_l$ in columns 1 (Ellipse) and 2 (Non-Ellipse)
                plot_on_ax(
                    ax=axes[row_idx, col_idx * 2],
                    x=x,
                    y=results_df["C_l"],
                    label=f"$C_l$ ({data_type}, {key})",
                    color=colors[0],
                    # marker=marker,
                    # markersize=markersize,
                    linestyle=linestyle,
                    x_label=x_label,
                    y_label=("$C_l$ [-]"),  # Only set once
                    is_with_grid=True,
                )
                title_key = "Ellipse" if key == "Ellipse" else "Rectangle"
                axes[row_idx, col_idx * 2].set_title(f"{title_key} {param}")

                # Plot $C_d$ in columns 3 (Ellipse) and 4 (Non-Ellipse)
                plot_on_ax(
                    ax=axes[row_idx, col_idx * 2 + 1],
                    x=x,
                    y=results_df["C_d"],
                    label=f"$C_d$ ({data_type}, {key})",
                    color=colors[1],
                    # marker=marker,
                    # markersize=markersize,
                    linestyle=linestyle,
                    x_label=x_label,
                    y_label=("$C_d$ [-]"),  # Only set once
                    is_with_grid=True,
                )

    # Save figure if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)  # , bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    # Return figure and axes
    return fig, axes


def plot_noca_coefficients_2x2(
    alpha: int,
    y_num: int,
    save_path: Optional[Union[str, Path]] = None,
    colors: tuple = ("blue", "red"),
    markers: tuple = ("o", "s"),
    markersize: int = 6,
    data_types: List = ["CFD", "PIV"],
    is_small_piv: bool = False,
):
    """
    Create a 2x2 grid plot showing CL and CD for dLx and dLy parameters.

    Args:
        alpha: Angle of attack.
        y_num: Y value index.
        save_path: Optional path to save the figure.
        colors: Tuple of colors for CL and CD plots.
        markers: Tuple of markers for CFD and PIV data.
        markersize: Size of markers.

    Returns:
        tuple: (fig, axes) - Figure and axes objects.
    """
    set_plot_style()

    # Parameters to plot
    parameter_names = ["dLx", "dLy"]

    # Collect results for all parameters and both data types (CFD and PIV)

    results_dict = {data_type: {} for data_type in data_types}
    for data_type in results_dict:
        is_CFD = data_type == "CFD"
        for param in parameter_names:
            print(f"\n alpha: {alpha} | Y{y_num} | {data_type} | {param}")
            results_dict[data_type][param] = {
                "Ellipse": parameter_sweep_noca(
                    is_CFD,
                    alpha,
                    y_num,
                    param,
                    is_ellipse=True,
                    is_small_piv=is_small_piv,
                ),
                "Rectangle": parameter_sweep_noca(
                    is_CFD,
                    alpha,
                    y_num,
                    param,
                    is_ellipse=False,
                    is_small_piv=is_small_piv,
                ),
            }

    # Create figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    dLx, dLy = reading_optimal_bound_placement(alpha, y_num)
    fig.suptitle(f"Current settings: dLx: {dLx}, dLy: {dLy}")

    param_to_bound_map = {
        "dLx": dLx,
        "dLy": dLy,
    }
    # Define labels
    x_label_map = {
        "dLx": "X [m]",
        "dLy": "Y [m]",
    }

    # Track global min and max for y-axis ranges
    cl_min, cl_max = float("inf"), float("-inf")
    cd_min, cd_max = float("inf"), float("-inf")

    # First pass: calculate global y-axis ranges
    for param in parameter_names:
        for key in ["Ellipse", "Rectangle"]:
            for data_type in data_types:
                results_df = pd.DataFrame(results_dict[data_type][param][key])
                cl_min = min(cl_min, results_df["C_l"].min())
                cl_max = max(cl_max, results_df["C_l"].max())
                cd_min = min(cd_min, results_df["C_d"].min())
                cd_max = max(cd_max, results_df["C_d"].max())

    cd_min = -0.2
    cd_max = 0.4
    cl_min = 0.3
    cl_max = 0.9

    # Plotting
    for idx, param in enumerate(parameter_names):
        x_label = x_label_map.get(param, param)

        # Plotting CL (top row)
        cl_ax = axes[0, idx]

        # plot an area surrounding the 5% left and %5 right of the optimal bound
        cl_ax.axvspan(
            param_to_bound_map[param] - 0.05 * param_to_bound_map[param],
            param_to_bound_map[param] + 0.05 * param_to_bound_map[param],
            color="gray",
            alpha=0.2,
        )

        # Iterate over data types and shapes
        for data_type_idx, data_type in enumerate(data_types):
            for key_idx, key in enumerate(["Ellipse", "Rectangle"]):
                results_df = pd.DataFrame(results_dict[data_type][param][key])
                x = results_df["parameter_value"]

                linestyle = "-" if data_type == "CFD" else "--"
                line_label = f"$C_l$ ({data_type}, {key})"

                # if CL plot blue
                if key == "Ellipse":
                    color = colors[0]
                else:
                    color = colors[1]

                plot_on_ax(
                    ax=cl_ax,
                    x=x,
                    y=results_df["C_l"],
                    label=line_label,
                    color=color,
                    # marker=markers[data_type_idx],
                    linestyle=linestyle,
                    x_label=x_label,
                    y_label="$C_l$ [-]",
                    is_with_grid=False,
                )

        cl_ax.set_title(f"{param} Effect on $C_l$")
        cl_ax.set_ylim(cl_min, cl_max)

        # Plotting CD (bottom row)
        cd_ax = axes[1, idx]

        # plot an area surrounding the 5% left and %5 right of the optimal bound
        cd_ax.axvspan(
            param_to_bound_map[param] - 0.05 * param_to_bound_map[param],
            param_to_bound_map[param] + 0.05 * param_to_bound_map[param],
            color="gray",
            alpha=0.2,
        )

        for data_type_idx, data_type in enumerate(data_types):
            for key_idx, key in enumerate(["Ellipse", "Rectangle"]):
                results_df = pd.DataFrame(results_dict[data_type][param][key])
                x = results_df["parameter_value"]

                linestyle = "-" if data_type == "CFD" else "--"
                line_label = f"$C_d$ ({data_type}, {key})"

                if key == "Ellipse":
                    color = colors[0]
                else:
                    color = colors[1]

                plot_on_ax(
                    ax=cd_ax,
                    x=x,
                    y=results_df["C_d"],
                    label=line_label,
                    color=color,
                    # marker=markers[data_type_idx],
                    linestyle=linestyle,
                    x_label=x_label,
                    y_label="$C_d$ [-]",
                    is_with_grid=False,
                )

        cd_ax.set_title(f"{param} Effect on $C_d$")
        cd_ax.set_ylim(cd_min, cd_max)
        if idx == 0:
            cl_ax.legend(loc="lower center")
            cd_ax.legend(loc="lower center")

    # Adjust layout and save
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Figure saved to {save_path}")

    return fig, axes


if __name__ == "__main__":

    # Settings
    is_CFD = True
    alpha = 6
    y_num = 2
    parameter_name = "dLx"

    # dict_results_df_dLx = parameter_sweep_noca(is_CFD, alpha, y_num, parameter_name)

    # Plotting the results
    if is_CFD:
        type_data = "CFD"
    else:
        type_data = "PIV"

    # plot_noca_coefficients(
    #     pd.DataFrame(results_df),
    #     parameter_name,
    #     save_path,
    #     # colors: tuple = ("blue", "red"),
    #     # markers: tuple = ("o", "s"),
    #     # markersize: int = 6,
    # )

    # # For a single parameter sweep
    # dict_results_df_iP = parameter_sweep_noca(is_CFD, alpha, y_num, "iP")
    # dict_results_df_dLx = parameter_sweep_noca(is_CFD, alpha, y_num, "dLx")
    # dict_results_df_dLy = parameter_sweep_noca(is_CFD, alpha, y_num, "dLy")
    # dict_results_df_drot = parameter_sweep_noca(is_CFD, alpha, y_num, "drot")

    # # Plotting
    # plot_noca_coefficients_grid(
    #     {
    #         "dLx": pd.DataFrame(dict_results_df_dLx),
    #         "dLy": pd.DataFrame(dict_results_df_dLy),
    #         "iP": pd.DataFrame(dict_results_df_iP),
    #         "drot": pd.DataFrame(dict_results_df_drot),
    #     },
    #     Path(project_dir)
    #     / "results"
    #     / "convergence_study"
    #     / f"alpha_{alpha}"
    #     / "CFD"
    #     / f"Y_{y_num}_all.pdf",
    # )

    # # result

    # # Or run all sweeps at once
    # all_results = run_parameter_sweeps(df_1D, Path("./noca_results"))

    ### Running for relevant files
    # Settings
    # is_CFD = False
    # alpha = 6

    # # Plotting the results
    # if is_CFD:
    #     type_data = "CFD"
    # else:
    #     type_data = "PIV"

    # for y_num in [1]:  # [1, 2, 3, 4, 5]:
    #     save_path = (
    #         Path(project_dir)
    #         / "results"
    #         / "convergence_study"
    #         / f"alpha_{alpha}"
    #         / f"{type_data}"
    #         / f"Y_{y_num}_dLx_dLy.pdf"
    #     )
    #     plot_noca_coefficients_grid(
    #         is_CFD,
    #         alpha,
    #         y_num,
    #         save_path,
    #         parameter_names=["dLx", "dLy"],
    #         # colors: tuple = ("blue", "red"),
    #         # markers: tuple = ("o", "s"),
    #         # markersize: int = 6,
    #     )

    # # Settings
    # alpha = 16

    # # Plotting the results
    # if is_CFD:
    #     type_data = "CFD"
    # else:
    #     type_data = "PIV"

    # for y_num in [1]:
    #     save_path = (
    #         Path(project_dir)
    #         / "results"
    #         / "convergence_study"
    #         / f"alpha_{alpha}"
    #         / f"{type_data}"
    #         / f"Y_{y_num}_dLx_dLy.pdf"
    #     )
    #     plot_noca_coefficients_grid(
    #         is_CFD,
    #         alpha,
    #         y_num,
    #         save_path,
    #         parameter_names=["dLx", "dLy"],
    #         # colors: tuple = ("blue", "red"),
    #         # markers: tuple = ("o", "s"),
    #         # markersize: int = 6,
    #     )

    ## Running for relevant files
    # alpha = 6

    # for y_num in [1, 2, 3, 4, 5]:
    #     save_path = (
    #         Path(project_dir)
    #         / "results"
    #         / "convergence_study"
    #         / f"alpha_{alpha}"
    #         / f"CFD_PIV_Y_{y_num}_dLx_dLy_small.pdf"
    #     )
    #     plot_noca_coefficients_grid_CFD_PIV(
    #         alpha,
    #         y_num,
    #         save_path,
    #         parameter_names=["dLx", "dLy"],
    #         # colors: tuple = ("blue", "red"),
    #         # markers: tuple = ("o", "s"),
    #         # markersize: int = 6,
    #     )

    # alpha = 16

    # for y_num in [1]:
    #     save_path = (
    #         Path(project_dir)
    #         / "results"
    #         / "convergence_study"
    #         / f"alpha_{alpha}"
    #         / f"CFD_PIV_Y_{y_num}_dLx_dLy_small.pdf"
    #     )
    #     plot_noca_coefficients_grid_CFD_PIV(
    #         alpha,
    #         y_num,
    #         save_path,
    #         parameter_names=["dLx", "dLy"],
    #         # colors: tuple = ("blue", "red"),
    #         # markers: tuple = ("o", "s"),
    #         # markersize: int = 6,
    #     )

    ## Running the 2x2
    alpha = 6
    for y_num in [1, 2, 3, 4, 5]:
        save_path = (
            Path(project_dir)
            / "results"
            / "convergence_study"
            / f"alpha_{alpha}"
            / f"CFD_PIV_Y_{y_num}_dLx_dLy_2x2_all.pdf"
        )
        plot_noca_coefficients_2x2(
            alpha,
            y_num,
            save_path,
            data_types=["CFD", "PIV"],
            # is_small_piv=True,
        )

    alpha = 16
    y_num = 1
    save_path = (
        Path(project_dir)
        / "results"
        / "convergence_study"
        / f"alpha_{alpha}"
        / f"CFD_PIV_Y_{y_num}_dLx_dLy_2x2_all.pdf"
    )
    plot_noca_coefficients_2x2(
        alpha,
        y_num,
        save_path,
        data_types=["CFD", "PIV"],
        # is_small_piv=True,
    )
