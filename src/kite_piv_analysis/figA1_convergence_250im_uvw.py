import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path
import numpy as np
from kite_piv_analysis.utils import project_dir
from kite_piv_analysis.plot_styling import set_plot_style, plot_on_ax


def read_single_dat_file_into_df(file_path):
    """Read a single .dat file and return a pandas DataFrame."""
    # Read header information
    variables = None
    data_lines = []

    with open(file_path, "r") as f:
        for line in f:
            if "VARIABLES" in line:
                # Extract variable names
                variables = [
                    var.strip(' "') for var in line.split("=")[1].strip().split(",")
                ]
            elif line[0].isdigit() or line[0] == "-":
                # Data line
                data_lines.append([float(val) for val in line.strip().split()])

    # Create DataFrame
    df = pd.DataFrame(data_lines, columns=variables)
    return df


# def calculate_values(point_coords):
#     """Calculate values at a specific point for all files in the folder."""
#     # loop through all dat files in folder
#     values_at_point_list = []
#     folder_path = Path(
#         project_dir,
#         "data",
#         "raw_dat_files",
#         "convergence",
#         "flipped_aoa_23_vw_15_H_918_Z1_Y4_X2",
#     )

#     # Make sure folder exists
#     if not folder_path.exists():
#         raise FileNotFoundError(f"Folder not found: {folder_path}")

#     V_value_list = []
#     # Get all .dat files in the folder
#     for file in folder_path.glob("*.dat"):
#         # Read the file
#         df = read_single_dat_file_into_df(file)

#         # Find the closest point to the specified coordinates
#         # Assuming point_coords is a tuple of (x, y)
#         x_coord, y_coord = point_coords
#         df["distance"] = np.sqrt(
#             (df["x [mm]"] - x_coord) ** 2 + (df["y [mm]"] - y_coord) ** 2
#         )
#         closest_point_idx = df["distance"].idxmin()
#         values_at_point = df.iloc[closest_point_idx].drop("distance")

#         values_at_point_list.append(values_at_point)

#         ## find all V values
#         V_values = df["Velocity |V| [m/s]"].values

#     return values_at_point_list, V_values


# def plot_convergence(variable, values_at_point_list):
#     """Plot convergence of a specific variable."""
#     VARIABLES = [
#         "x [mm]",
#         "y [mm]",
#         "Velocity u [m/s]",
#         "Velocity v [m/s]",
#         "Velocity w [m/s]",
#         "Velocity |V| [m/s]",
#         "du/dx [1/s]",
#         "du/dy [1/s]",
#         "dv/dx [1/s]",
#         "dv/dy [1/s]",
#         "dw/dx [1/s]",
#         "dw/dy [1/s]",
#         "Vorticity w_z (dv/dx - du/dy) [1/s]",
#         "|Vorticity| [1/s]",
#         "Divergence 2D (du/dx + dv/dy) [1/s]",
#         "Swirling strength 2D (L_ci) [1/s^2]",
#         "isValid",
#     ]
#     if variable == "V":
#         variable_to_be_plotted = f"Velocity |{variable}| [m/s]"
#     else:
#         variable_to_be_plotted = f"Velocity {variable} [m/s]"

#     # randomly shuffle the values at point list
#     random.shuffle(values_at_point_list)

#     # extract the variable to be plotted
#     variable_values = [point[variable_to_be_plotted] for point in values_at_point_list]

#     # Create x-axis as number of samples
#     x_axis = range(1, len(variable_values) + 1)

#     # plot convergence
#     set_plot_style()
#     fig, ax = plt.subplots(figsize=(10, 5))
#     plot_on_ax(
#         ax,
#         x_axis,
#         variable_values,
#         label="Local value",
#         linestyle="-",
#         color="b",
#         is_with_grid=True,
#     )
#     # ax.plot(x_axis, variable_values, "b-", linewidth=1)
#     ax.set_xlabel("Number of Samples")
#     ax.set_ylabel(f"{variable} [m/s]")
#     # ax.set_title(f"Convergence of {variable_to_be_plotted}")

#     # Calculate and plot running mean
#     running_mean = np.cumsum(variable_values) / np.arange(1, len(variable_values) + 1)
#     ax.plot(x_axis, running_mean, "r-", linewidth=2, label="Running Mean")

#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(
#         Path(project_dir, "results", "paper_plots", f"convergence_250im_{variable}.pdf")
#     )


def calculate_values(point_coords):
    """
    Calculate values at a specific point for all files in the folder and average V values.

    Returns:
        tuple: (values_at_point_list, averaged_V_values)
            - values_at_point_list: List of Series containing values at specified point
            - averaged_V_values: Array of averaged |V| values for entire field
    """
    # loop through all dat files in folder
    values_at_point_list = []
    folder_path = Path(
        project_dir,
        "data",
        "raw_dat_files_convergence",
        "flipped_aoa_23_vw_15_H_918_Z1_Y4_X2",
    )

    # Make sure folder exists
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Initialize list to store V values from all files
    all_V_values = []
    all_is_valid = []
    all_w = []
    first_file = True

    # Get all .dat files in the folder
    for file in folder_path.glob("*.dat"):
        # Read the file
        df = read_single_dat_file_into_df(file)

        # Store the shape of the velocity field from first file
        if first_file:
            field_shape = df["Velocity |V| [m/s]"].values.shape
            first_file = False

        # Find the closest point to the specified coordinates
        x_coord, y_coord = point_coords
        df["distance"] = np.sqrt(
            (df["x [mm]"] - x_coord) ** 2 + (df["y [mm]"] - y_coord) ** 2
        )
        closest_point_idx = df["distance"].idxmin()
        values_at_point = df.iloc[closest_point_idx].drop("distance")

        values_at_point_list.append(values_at_point)

        # Store V values from this file
        all_V_values.append(df["Velocity |V| [m/s]"].values)

        # Store isValid and w arrays for later filtering (preserve ordering)
        if "isValid" in df.columns:
            all_is_valid.append(df["isValid"].values)
        else:
            # if not present, create an all-ones mask
            all_is_valid.append(np.ones_like(df["Velocity |V| [m/s]"].values))

        if "Velocity w [m/s]" in df.columns:
            all_w.append(df["Velocity w [m/s]"].values)
        else:
            all_w.append(np.zeros_like(df["Velocity |V| [m/s]"].values))

    # Convert list of V values to numpy array and calculate mean
    all_V_values = np.array(all_V_values)
    averaged_V_values = np.mean(all_V_values, axis=0)

    # Ensure the averaged values maintain the same shape as original data
    assert averaged_V_values.shape == field_shape, "Shape mismatch in averaged values"

    return values_at_point_list, averaged_V_values, all_V_values, all_is_valid, all_w


def plot_variable_on_ax(
    ax,
    variable_column,
    variable_label,
    values_at_point_list,
    is_xlabel=False,
    is_legend=False,
):
    # Right plot: convergence plot
    if variable_column == "V":
        variable_to_be_plotted = f"Velocity |{variable_column}| [m/s]"
    else:
        variable_to_be_plotted = f"Velocity {variable_column} [m/s]"

    # Randomly shuffle the values for convergence analysis
    shuffled_values = values_at_point_list.copy()
    random.shuffle(shuffled_values)

    # Extract the variable to be plotted
    variable_values = [point.loc[variable_to_be_plotted] for point in shuffled_values]
    x_axis = range(1, len(variable_values) + 1)

    # Create convergence plot
    plot_on_ax(
        ax,
        x_axis,
        variable_values,
        label="Local value",
        linestyle="-",
        color="b",
        is_with_grid=False,
        is_with_x_label=is_xlabel,
        # is_with_x_ticks=is_xlabel,
        is_with_x_tick_label=is_xlabel,
        x_label="Number of samples",
        y_label=f"{variable_label} " + r"(ms$^{-1}$)",
        is_with_legend=False,
    )

    # Calculate and plot running mean
    running_mean = np.cumsum(variable_values) / np.arange(1, len(variable_values) + 1)
    ax.plot(x_axis, running_mean, "r-", linewidth=2, label="Running Mean")
    if is_legend:
        ax.legend()


def _build_shuffled_data(values_at_point_list):
    """Build a single shuffled ordering and extract raw + filtered values for u, v, w."""
    n_samples = len(values_at_point_list)
    indices = list(range(n_samples))
    random.seed(42)
    random.shuffle(indices)

    variables = [
        ("u", r"$u_{x}$"),
        ("v", r"$u_{y}$"),
        ("w", r"$u_{z}$"),
    ]

    rows_data = []
    for var_col, var_label in variables:
        colname = (
            f"Velocity |{var_col}| [m/s]"
            if var_col == "V"
            else f"Velocity {var_col} [m/s]"
        )

        unfiltered = [values_at_point_list[i].loc[colname] for i in indices]

        filtered = []
        for i, val in zip(indices, unfiltered):
            series = values_at_point_list[i]
            is_valid = series.get("isValid", 1)
            w_val = series.get("Velocity w [m/s]", 0.0)
            if (is_valid != 0) and (abs(w_val) <= 3.0):
                filtered.append(val)
            else:
                filtered.append(np.nan)

        rows_data.append((var_col, var_label, unfiltered, filtered))

    return n_samples, rows_data


def _running_mean_nan(values):
    """Compute a running mean that ignores NaN entries."""
    result = []
    cumsum = 0.0
    count = 0
    for v in values:
        if np.isnan(v):
            result.append(np.nan if count == 0 else cumsum / count)
        else:
            cumsum += v
            count += 1
            result.append(cumsum / count)
    return result


def plot_convergence_single_col(values_at_point_list):
    """
    Single-column (3×1) convergence plot showing filtered data only.
    Saved as fig14_convergence_250im_uvw.pdf
    """
    set_plot_style()
    n_samples, rows_data = _build_shuffled_data(values_at_point_list)
    x_axis = range(1, n_samples + 1)

    fig, axes = plt.subplots(3, 1, figsize=(5, 4))

    for row, (_var_col, var_label, _unfiltered, filtered) in enumerate(rows_data):
        ax = axes[row]
        plot_on_ax(
            ax,
            x_axis,
            filtered,
            label="Local value",
            linestyle="-",
            color="b",
            is_with_grid=False,
            is_with_x_label=(row == 2),
            is_with_x_tick_label=(row == 2),
            x_label="Number of samples",
            y_label=f"{var_label} " + r"(ms$^{-1}$)",
            is_with_legend=False,
        )
        running_mean_f = _running_mean_nan(filtered)
        ax.plot(x_axis, running_mean_f, "r-", linewidth=2, label="Running Mean")
        if row == 2:
            ax.legend(ncol=2, fontsize=13)

    plt.tight_layout(pad=0.02)
    # save_path = Path(
    #     project_dir,
    #     "results",
    #     "paper_plots_21_10_2025",
    #     "fig14_convergence_250im_uvw.pdf",
    # )
    save_path = Path(
        project_dir,
        "results",
        "paper_plots_24_03_2026",
        "figA1.pdf",
    )
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.close()
    print(f"Convergence plot (1-col) saved to {save_path}")


def plot_convergence_two_col(values_at_point_list):
    """
    Two-column (3×2) convergence plot: left = raw, right = filtered.
    Saved as fig14_convergence_250im_uvw_2col.pdf
    """
    set_plot_style()
    n_samples, rows_data = _build_shuffled_data(values_at_point_list)
    x_axis = range(1, n_samples + 1)

    fig, axes = plt.subplots(3, 2, figsize=(10, 7))

    for row, (_var_col, var_label, unfiltered, filtered) in enumerate(rows_data):
        # Left: raw
        ax_left = axes[row, 0]
        plot_on_ax(
            ax_left,
            x_axis,
            unfiltered,
            label="Local value",
            linestyle="-",
            color="b",
            is_with_grid=False,
            is_with_x_label=(row == 2),
            is_with_x_tick_label=(row == 2),
            x_label="Number of samples",
            y_label=f"{var_label} (ms$^{{-1}}$)",
        )
        running_mean_un = np.cumsum(np.nan_to_num(unfiltered, nan=0.0)) / np.arange(
            1, n_samples + 1
        )
        ax_left.plot(x_axis, running_mean_un, "r-", linewidth=2)
        if row == 0:
            ax_left.set_title("Raw data")

        # Right: filtered
        ax_right = axes[row, 1]
        plot_on_ax(
            ax_right,
            x_axis,
            filtered,
            label="Local value (filtered)",
            linestyle="-",
            color="b",
            is_with_grid=False,
            is_with_x_label=(row == 2),
            is_with_x_tick_label=(row == 2),
            x_label="Number of samples",
            y_label=f"{var_label} (ms$^{{-1}}$)",
        )
        running_mean_f = _running_mean_nan(filtered)
        ax_right.plot(x_axis, running_mean_f, "r-", linewidth=2)
        if row == 0:
            ax_right.set_title("Filtered: isValid and |w|<=3 m/s")

    plt.tight_layout(pad=0.02)
    save_path = Path(
        project_dir,
        "results",
        "paper_plots_21_10_2025",
        "fig14_convergence_250im_uvw_2col.pdf",
    )
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.close()
    print(f"Convergence plot (2-col) saved to {save_path}")


# Example usage
def main():
    # Example coordinates for the point of interest
    point_coords = (-0.75368, -118.177)  # in free-stream, looks good

    # Calculate values for Y4
    values, V_values, all_V_values, all_is_valid, all_w = calculate_values(point_coords)

    # Plot 1: single-column (filtered only) → fig14_convergence_250im_uvw.pdf
    plot_convergence_single_col(values)

    # Plot 2: two-column (raw vs filtered) → fig14_convergence_250im_uvw_2col.pdf
    plot_convergence_two_col(values)


if __name__ == "__main__":
    main()
