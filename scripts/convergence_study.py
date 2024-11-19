import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from defining_bound_volume import boundary_ellipse, boundary_rectangle
import force_from_noca
from utils import project_dir, csv_reader
from plot_styling import set_plot_style, plot_on_ax
import matplotlib.pyplot as plt
from calculating_airfoil_centre import reading_center_from_csv


@dataclass
class NOCAParameters:
    is_ellipse: bool = field(default_factory=lambda: True)
    d1centre: np.ndarray = field(default_factory=lambda: np.array([0.27, 0.13]))
    drot: float = field(default_factory=lambda: 0)
    dLx: float = field(default_factory=lambda: 0.4)
    dLy: float = field(default_factory=lambda: 0.225)
    iP: int = field(default_factory=lambda: 45)


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
        print(
            f"max_positive_dLx: {max_positive_dLx}, max_negative_dLx: {max_negative_dLx}, max_dLx: {max_dLx}"
        )
        dLx = np.linspace(0.25 * max_dLx, max_dLx, 50)  # Adjust the range dynamically
        return dLx

    elif parameter_name == "dLy":
        # Compute the maximum range for dLy based on ylim and airfoil_center[1]
        max_positive_dLy = ylim[1] - airfoil_center[1]
        max_negative_dLy = airfoil_center[1] - ylim[0]
        max_dLy = min(
            max_positive_dLy, max_negative_dLy
        )  # Take the minimum of both bounds
        print(
            f"max_positive_dLy: {max_positive_dLy}, max_negative_dLy: {max_negative_dLy}, max_dLy: {max_dLy}"
        )
        print(f"max_dLy + airfoil_center: {max_dLy + airfoil_center[1]}")
        dLy = np.linspace(0.25 * max_dLy, max_dLy, 50)  # Adjust the range dynamically
        return dLy

    elif parameter_name == "is_ellipse":
        return [False, True]

    elif parameter_name == "iP":
        # make sure to not include 9+6n numbers as they cause problems in making the rectangle
        # return [int(i) for i in np.linspace(20, 151, 50) if ((int(i) - 9) % 6 != 0)]
        return [
            20,
            25,
            30,
            35,
            40,
            45,
            55,
            65,
            70,
            75,
            80,
            85,
            90,
            95,
            105,
        ]

    elif parameter_name == "area":
        return np.linspace(0.2, 0.4, 5)

    elif parameter_name == "drot":
        return np.linspace(-10, 10, 20)

    raise ValueError(f"Unknown parameter name: {parameter_name}")


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
) -> pd.DataFrame:
    """
    Perform parameter sweep for NOCA analysis.

    Args:
        df_1D: DataFrame containing the flow field data
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
    df_1D = csv_reader(is_CFD, alpha, y_num)

    results = []

    # Reading in the airfoil centers
    airfoil_center = reading_center_from_csv(alpha, y_num)

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
            current_params.dLy = max_dLy
        elif parameter_name == "dLy":
            current_params.dLy = value
            current_params.dLx = max_dLx
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
        # Run NOCA analysis
        F_x, F_y, C_l, C_d = force_from_noca.main(
            df_1D,
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


def plot_noca_coefficients_grid(
    is_CFD: bool,
    alpha: int,
    y_num: int,
    save_path: Optional[Union[str, Path]] = None,
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

    # Parameter names
    parameter_names = ["drot", "iP", "dLx", "dLy"]

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
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5, wspace=0.4)  # Adjust spacing
    if is_CFD:
        type_data = "CFD"
    else:
        type_data = "PIV"
    plt.title(f"{type_data} | Alpha={alpha} | Y={y_num}")

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
                marker=markers[0],
                markersize=markersize,
                x_label=x_label,
                y_label="$C_l$ [-]",
                is_with_grid=False,
            )

            # Plot $C_d$ in columns 3 (Ellipse) and 4 (Non-Ellipse)
            plot_on_ax(
                ax=axes[row_idx, col_idx * 2 + 1],
                x=x,
                y=results_df["C_d"],
                label=f"$C_d$ ({key})",
                color=colors[1],
                marker=markers[1],
                markersize=markersize,
                x_label=x_label,
                y_label="$C_d$ [-]",
                is_with_grid=False,
            )

    # Save figure if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)  # , bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    plt.show()
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
    save_path = (
        Path(project_dir)
        / "results"
        / "convergence_study"
        / f"alpha_{alpha}"
        / f"{type_data}"
        / f"Y_{y_num}_{parameter_name}.pdf"
    )

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

    plot_noca_coefficients_grid(
        is_CFD,
        alpha,
        y_num,
        save_path,
        # colors: tuple = ("blue", "red"),
        # markers: tuple = ("o", "s"),
        # markersize: int = 6,
    )
