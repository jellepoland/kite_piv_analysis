import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from defining_bound_volume import boundary_ellipse, boundary_rectangle
import force_from_noca
from utils import project_dir
from typing import List, Dict, Any, Optional
from plot_styling import set_plot_style, plot_on_ax
import matplotlib.pyplot as plt
import calculating_airfoil_centre
from plotting import *
from utils import reading_optimal_bound_placement
import calculating_circulation


@dataclass
class NOCAParameters:
    is_ellipse: bool = field(default_factory=lambda: True)
    d1centre: np.ndarray = field(default_factory=lambda: np.array([0.27, 0.13]))
    drot: float = field(default_factory=lambda: 0)
    dLx: float = field(default_factory=lambda: 0.7)
    dLy: float = field(default_factory=lambda: 0.33)
    iP: int = field(default_factory=lambda: 360)


def get_sweep_values(
    parameter_name: str,
    alpha: float,
    y_num: int,
    airfoil_center: tuple,
    xlim: tuple,
    ylim: tuple,
    fast_factor: float = 1,
) -> List:
    """
    Returns sweep values for the specified parameter, dynamically adjusting ranges for dLx and dLy
    based on airfoil center and specified limits.
    """
    optimal_dLx, optimal_dLy, optimal_iP = reading_optimal_bound_placement(
        alpha, y_num, is_with_N_datapoints=True
    )
    if parameter_name == "dLx":
        # Compute the maximum range for dLx based on xlim and airfoil_center[0]
        max_positive_dLx = xlim[1] - airfoil_center[0]
        max_negative_dLx = airfoil_center[0] - xlim[0]
        max_dLx = min(max_positive_dLx, max_negative_dLx)
        dLx = np.linspace(
            0.35, 2.0 * max_dLx, int(50 / fast_factor)
        )  # Adjust the range dynamically
        # dLx = np.linspace(0.35, 2.0 * max_dLx, 5)  # Adjust the range dynamically
        return np.append(optimal_dLx, dLx)
    elif parameter_name == "dLy":
        # Compute the maximum range for dLy based on ylim and airfoil_center[1]
        max_positive_dLy = ylim[1] - airfoil_center[1]
        max_negative_dLy = airfoil_center[1] - ylim[0]
        max_dLy = min(max_positive_dLy, max_negative_dLy)
        dLy = np.linspace(
            0.1, 2.0 * max_dLy, int(50 / fast_factor)
        )  # Adjust the range dynamically
        # dLy = np.linspace(0.1, 2.0 * max_dLy, 5)  # Adjust the range dynamically

        return np.append(optimal_dLy, dLy)

    elif parameter_name == "iP":
        iP = 4 * np.arange(1, 100, int(2 * fast_factor))
        # iP = 4 * np.arange(1, 100, 25)
        return np.append(optimal_iP, iP)

    elif parameter_name == "drot":
        return np.linspace(-10, 10, 3)

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
    U_inf: float = 15,
    is_small_piv: bool = False,
    max_perc_interpolated_zones: float = 1.2,
    fast_factor: float = 1,
    n_datapoints: int = 104304,
    parameter_values=None,
    dLx=None,
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
    x_airfoil, y_airfoil, chord = calculating_airfoil_centre.main(
        alpha, y_num, is_with_chord=True
    )
    airfoil_center = (x_airfoil, y_airfoil)

    # Get sweep values for the specified parameter
    if parameter_values is None:
        parameter_values = get_sweep_values(
            parameter_name,
            alpha,
            y_num,
            airfoil_center,
            xlim=(-0.2, 0.8),
            ylim=(-0.2, 0.4),
            fast_factor=fast_factor,
        )

    if dLx is None:
        dLx, dLy, iP = reading_optimal_bound_placement(
            alpha, y_num, is_with_N_datapoints=True
        )
    else:
        _, dLy, iP = reading_optimal_bound_placement(
            alpha, y_num, is_with_N_datapoints=True
        )

    # When PIV, the cells have to be interpolated, therefore we are only checking convergence around a small area, surrounding the optimal bound
    if is_small_piv and not is_CFD:
        if parameter_name == "dLx":
            parameter_values = np.linspace(dLx - 0.05 * dLx, dLx + 0.05 * dLx, 10)
        elif parameter_name == "dLy":
            parameter_values = np.linspace(dLy - 0.05 * dLy, dLy + 0.05 * dLy, 10)

    for value in parameter_values:
        # Create copy of base parameters
        current_params = NOCAParameters(**vars(base_params))
        current_params.dLx = dLx
        current_params.dLy = dLy
        current_params.iP = int(iP)
        current_params.d1centre = airfoil_center
        current_params.is_ellipse = is_ellipse

        if parameter_name == "dLx":
            current_params.dLx = value
        elif parameter_name == "dLy":
            current_params.dLy = value
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

        # print(f"{parameter_name} = {value:.3f}")

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
        perc_of_interpolated_points = 0
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

            # Determine number of poitns that are interpolated
            n_points_nan = 0
            for interpolation_zone_i in plot_params["interpolation_zones"]:
                x_min, x_max, y_min, y_max = interpolation_zone_i["bounds"]
                subset = df[
                    (df["x"] >= x_min)
                    & (df["x"] <= x_max)
                    & (df["y"] >= y_min)
                    & (df["y"] <= y_max)
                ]
                dropped_by_nan = subset[subset["u"].isna()]
                n_points_nan += len(dropped_by_nan)

            perc_of_interpolated_points = 100 * (n_points_nan / n_datapoints)
            # print(f"perc_of_interpolated_points: {perc_of_interpolated_points:.2f}%")

            # to make sure that we are not interpolating too many areas, we set the is_skip_parameter
            if perc_of_interpolated_points < max_perc_interpolated_zones:

                for interpolation_zone_i in plot_params["interpolation_zones"]:
                    df, d2curve_rectangle_interpolated_zone = interpolate_missing_data(
                        df,
                        interpolation_zone_i,
                    )
            else:
                print(
                    f"---> SKIPPED perc_of_interpolated_points: {perc_of_interpolated_points:.2f}%, dLx: {current_params.dLx:.2f}, dLy: {current_params.dLy:.2f}"
                )
                is_skip_this_parameter = True
        if is_skip_this_parameter:
            continue

        # Run NOCA analysis
        if alpha == 6:
            rho = 1.20
        else:
            rho = 1.18
        F_x, F_y, C_l, C_d = force_from_noca.main(
            df,
            d2curve,
            mu=mu,
            is_with_maximim_vorticity_location_correction=is_with_maximim_vorticity_location_correction,
            rho=rho,
            U_inf=U_inf,
            c=chord,
        )

        # Calculating Circulation
        Gamma = calculating_circulation.calculate_circulation(df, d2curve)

        F_kutta = rho * U_inf * Gamma / (0.5 * rho * U_inf**2 * chord)

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
                "perc_of_interpolated_points": perc_of_interpolated_points,
                "Gamma": Gamma,
                "F_kutta": F_kutta,
            }
        )

    return results


def storing_and_collecting_results(
    alpha,
    y_num,
    parameter_names,
    data_types=["CFD", "PIV"],
    is_small_piv=False,
    fast_factor: float = 1,
    # project_dir=None,
):
    """
    Collect parameter sweep results for different data types, save as CSV, and return as a nested dictionary.

    Args:
        alpha: Angle of attack.
        y_num: Y value index.
        parameter_names: List of parameters to sweep.
        data_types: Types of data to collect.
        is_small_piv: Flag for small PIV dataset.
        fast_factor: Speed-up factor for parameter sweep.
        project_dir: Base directory for saving the files.

    Returns:
        dict: Nested dictionary of collected results.
    """
    if project_dir is None:
        raise ValueError("project_dir must be specified")

    # Define the save folder
    save_folder = Path(project_dir) / "processed_data" / "convergence_study"
    save_folder.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    # Initialize the results dictionary
    results_dict = {data_type: {} for data_type in data_types}

    # Iterate over data types and parameters
    for data_type in data_types:
        is_CFD = data_type == "CFD"
        for param in parameter_names:
            results_dict[data_type][param] = {}

            # Collect results for Ellipse and Rectangle
            for is_ellipse, shape in [(True, "Ellipse"), (False, "Rectangle")]:
                results = parameter_sweep_noca(
                    is_CFD,
                    alpha,
                    y_num,
                    param,
                    is_ellipse=is_ellipse,
                    is_small_piv=is_small_piv,
                    fast_factor=fast_factor,
                )
                results_dict[data_type][param][shape] = results

                # Save the results to CSV
                if results:
                    df = pd.DataFrame(results)
                    save_path = (
                        save_folder
                        / f"alpha_{alpha}_Y{y_num}_{data_type}_{shape}_{param}.csv"
                    )
                    df.to_csv(save_path, index=False)
                    print(f"Saved: {save_path}")

    return results_dict


def storing_PIV_percentage_sweep(
    alpha: float,
    y_num: int,
    n_points: int = 4,
    data_type: str = "PIV",
    fast_factor: float = 1,
    percentage_range: int = 20,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect comprehensive parameter sweep results for Ellipse and Rectangle shapes.

    Args:
        alpha: Angle of attack.
        y_num: Y value index.
        data_type: Type of data to collect.
        is_small_piv: Flag for small PIV dataset.
        fast_factor: Speed-up factor for parameter sweep.
        percentage_range: Percentage range for parameter values.
        project_dir: Base directory for saving the files.

    Returns:
        Dictionary of collected results for each shape.
    """
    if project_dir is None:
        raise ValueError("project_dir must be specified")

    # Define the save folder
    save_folder = (
        Path(project_dir) / "processed_data" / "convergence_study" / "PIV_sweep"
    )
    save_folder.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    # Initialize the results dictionary
    results_dict = {}

    # Get base values for dLx, dLy, and iP
    dLx_base, dLy_base, iP = reading_optimal_bound_placement(
        alpha, y_num, is_with_N_datapoints=True
    )

    # Prepare parameter values for dLx and dLy
    dLx_values = np.linspace(
        dLx_base - dLx_base * 0.5 * (percentage_range / 100),
        dLx_base + dLx_base * 0.5 * (percentage_range / 100),
        n_points,
    )
    dLy_values = np.linspace(
        dLy_base - dLy_base * 0.5 * (percentage_range / 100),
        dLy_base + dLy_base * 0.5 * (percentage_range / 100),
        n_points,
    )
    # Determine if CFD or not (default to False)
    is_CFD = False

    # Main loop through shapes
    for is_ellipse, shape in [(True, "Ellipse"), (False, "Rectangle")]:
        # Collect all results for this shape
        all_results = []

        # Comprehensive sweep through dLx and dLy combinations
        for dLx_i in dLx_values:
            results = parameter_sweep_noca(
                is_CFD,
                alpha,
                y_num,
                is_ellipse=is_ellipse,
                fast_factor=fast_factor,
                parameter_name="dLy",
                parameter_values=dLy_values,  # Pass dLx value
                dLx=dLx_i,
            )
            # Append results for this specific dLx, dLy combination
            all_results.extend(results)

        # Store results for this shape
        results_dict[shape] = all_results

        # Save results to CSV if not empty
        if all_results:
            df = pd.DataFrame(all_results)
            save_path = save_folder / f"alpha_{alpha}_Y{y_num}_{data_type}_{shape}.csv"
            df.to_csv(save_path, index=False)
            print(f"Saved: {save_path}")

    return results_dict


def load_saved_results(
    alpha,
    y_num,
    parameter_names,
    data_types=["CFD", "PIV"],
    # project_dir=None,
):
    """
    Load parameter sweep results from CSV files into a nested dictionary.

    Args:
        alpha: Angle of attack.
        y_num: Y value index.
        parameter_names: List of parameters to load.
        data_types: Types of data to collect.
        project_dir: Base directory for loading the files.

    Returns:
        dict: Nested dictionary of loaded results.
    """
    if project_dir is None:
        raise ValueError("project_dir must be specified")

    # Define the folder path
    folder_path = Path(project_dir) / "processed_data" / "convergence_study"

    # Initialize the results dictionary
    results_dict = {data_type: {} for data_type in data_types}

    # Iterate over data types and parameters
    for data_type in data_types:
        for param in parameter_names:
            results_dict[data_type][param] = {}

            # Load results for Ellipse and Rectangle
            for shape in ["Ellipse", "Rectangle"]:
                file_path = (
                    folder_path
                    / f"alpha_{alpha}_Y{y_num}_{data_type}_{shape}_{param}.csv"
                )
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    results_dict[data_type][param][shape] = df
                else:
                    print(f"File not found: {file_path}")

    return results_dict


def plot_noca_coefficients_grid(
    alpha: int,
    y_num: int,
    save_path: Optional[Union[str, Path]] = None,
    parameter_names: list = ["iP", "dLx", "dLy"],
    colors: tuple = ("blue", "red"),
    data_types: List = ["CFD", "PIV"],
    is_small_piv: bool = False,
    is_save: bool = True,
    fast_factor: float = 1,
):
    """
    Create a 2x2 grid plot showing CL and CD for different parameters.

    Args:
        alpha: Angle of attack.
        y_num: Y value index.
        save_path: Optional path to save the figure.
        colors: Tuple of colors for CL and CD plots.
        data_types: Types of data to plot.
        is_small_piv: Flag for small PIV dataset.

    Returns:
        tuple: (fig, axes) - Figure and axes objects.
    """
    set_plot_style()

    # results_dict = _collect_parameter_results(
    #     alpha, y_num, parameter_names, data_types, is_small_piv, fast_factor
    # )
    # results_dict = load_results_to_dict(alpha, y_num, parameter_names)
    results_dict = load_saved_results(alpha, y_num, parameter_names)

    # Create figure and axes
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Get optimal bound placements

    dLx, dLy, iP = reading_optimal_bound_placement(
        alpha, y_num, is_with_N_datapoints=True
    )
    # fig.suptitle(f"Settings= iP: {iP}, dLx: {dLx}, dLy: {dLy}")

    # Mapping of parameters to their bounds and labels
    param_config = {
        "iP": {"bound": iP, "label": r"$N_{\textrm{b}}$"},
        "dLx": {"bound": dLx, "label": r"$W_{\textrm{b}}$ [m]"},
        "dLy": {"bound": dLy, "label": r"$H_{\textrm{b}}$ [m]"},
    }

    # Plot parameters
    plot_parameters = {
        "C_l": {
            "row": 0,
            "ylim": (0.2, 1.2),
            "ylabel": "$C_l$ [-]",
            "title_template": "{param} Effect on $C_l$",
        },
        "C_d": {
            "row": 1,
            "ylim": (-0.2, 0.4),
            "ylabel": "$C_d$ [-]",
            "title_template": "{param} Effect on $C_d$",
        },
    }

    # Main plotting loop
    for param in parameter_names:
        for coef_type, plot_config in plot_parameters.items():
            ax = axes[plot_config["row"], parameter_names.index(param)]

            # Plotting current value and highlighting optimal regime
            bound = param_config[param]["bound"]
            ax.axvline(x=bound, color="black", linestyle="-", alpha=0.5)
            if param == "dLx" or param == "dLy":
                ax.axvspan(bound * 0.9, bound * 1.1, color="gray", alpha=0.2)

            # Plot each data type and shape
            for data_type in data_types:
                for key in ["Ellipse", "Rectangle"]:

                    dict_local = results_dict[data_type][param]
                    if key not in dict_local.keys():
                        continue
                    # Select data and plot
                    results_df = pd.DataFrame(dict_local[key])
                    if len(results_df) == 0:
                        continue

                    # sort on param
                    results_df = results_df.sort_values(by=param)

                    # Determine color and linestyle
                    color = colors[0] if data_type == "CFD" else colors[1]
                    linestyle = "-" if key == "Ellipse" else "--"
                    if coef_type == "C_l" and param == "iP":
                        label = f"{data_type} {key}"
                    else:
                        label = None

                    # determine if y-label/x-label should be present
                    if param == "iP":
                        is_ylabel = True
                    else:
                        is_ylabel = False

                    if coef_type == "C_d":
                        is_xlabel = True
                    else:
                        is_xlabel = False

                    plot_on_ax(
                        ax=ax,
                        x=results_df[param],
                        y=results_df[f"{coef_type}"],
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        x_label=param_config[param]["label"],
                        y_label=plot_config["ylabel"],
                        is_with_grid=False,
                        is_with_x_label=is_xlabel,
                        is_with_y_label=is_ylabel,
                        is_with_x_tick_label=is_xlabel,
                        is_with_y_tick_label=is_ylabel,
                    )

                    # Adding markers with coloring based on perc_of_interpolated_points
                    if data_type == "PIV" and param != "iP":
                        sc = ax.scatter(
                            results_df[param],
                            results_df[f"{coef_type}"],
                            c=results_df["perc_of_interpolated_points"],
                            cmap="viridis",
                            vmin=0,
                            vmax=0.5,
                        )
                        if key == "Ellipse" and param == "dLy":
                            cbar = ax.figure.colorbar(
                                sc, ax=ax, label=f"\% of interpolated data points"
                            )

            # Set plot details
            # ax.set_title(plot_config["title_template"].format(param=param))
            ax.set_ylim(plot_config["ylim"])

    ## Adding legend
    # Initialize an empty list to collect all handles and labels
    handles, labels = [], []

    # Loop through each axis and collect handles and labels
    for ax in fig.axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)

    plt.tight_layout()

    fig.legend(
        handles,
        labels,
        loc="lower center",  # Position the legend at the bottom center
        ncol=4,
        bbox_to_anchor=(
            0.5,
            0.001,
        ),  # Fine-tune position (centered, slightly below the figure)
    )
    # Add a combined legend at the bottom of the figure
    fig.subplots_adjust(
        bottom=0.1
    )  # Increase the bottom margin (0.3 is an example, adjust as needed)

    # Save figure if path provided
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
    y_num = 1
    parameter_names = ["iP", "dLx", "dLy"]
    fast_factor = 1

    for alpha in [6]:
        for y_num in [1, 2, 3, 4, 5]:
            storing_and_collecting_results(
                alpha, y_num, parameter_names, fast_factor=fast_factor
            )
            storing_PIV_percentage_sweep(alpha, y_num, n_points=10)

            save_path = (
                Path(project_dir)
                / "results"
                / "convergence_study"
                / f"alpha_{alpha}"
                / f"n_point_CFD_PIV_Y_{y_num}_2x3.pdf"
            )
            plot_noca_coefficients_grid(
                alpha,
                y_num,
                save_path,
                data_types=["CFD", "PIV"],
                fast_factor=fast_factor,
            )

    for alpha in [16]:
        for y_num in [1]:
            storing_and_collecting_results(
                alpha, y_num, parameter_names, fast_factor=fast_factor
            )
            storing_PIV_percentage_sweep(alpha, y_num, n_points=10)

            save_path = (
                Path(project_dir)
                / "results"
                / "convergence_study"
                / f"alpha_{alpha}"
                / f"n_point_CFD_PIV_Y_{y_num}_2x3.pdf"
            )
            plot_noca_coefficients_grid(
                alpha,
                y_num,
                save_path,
                data_types=["CFD", "PIV"],
                fast_factor=fast_factor,
            )
