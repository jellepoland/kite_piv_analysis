import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from pathlib import Path
import pandas as pd


def plot_airfoil(
    y_num: int,
    alpha: int,
    project_dir: Path,
    ax,
):
    """
    Plots the airfoil onto the existing contour plot, using pre-defined x, y positions for different Y-values.

    Parameters:
    y_num (int): The Y-section number for which airfoil needs to be plotted.
    project_dir (Path): The project directory where the airfoil data is located.
    ax (matplotlib axes): The axes to plot on.
    x_meshgrid_global (ndarray): The global x-meshgrid.
    y_meshgrid_global (ndarray): The global y-meshgrid.
    """
    # Set predefined x and y positions for airfoil placement
    if y_num == 1:
        x_pos = 0.015
        y_pos = 0.13
    elif y_num == 2:
        x_pos = 0.015
        y_pos = 0.13
    elif y_num == 3:
        x_pos = 0.015
        y_pos = 0.13
    elif y_num == 4:
        x_pos = 0.015
        y_pos = 0.13
    elif y_num == 5:
        x_pos = 0.015
        y_pos = 0.13
    elif y_num == 6:
        x_pos = 0.015
        y_pos = 0.13
    elif y_num == 7:
        x_pos = 0.015
        y_pos = 0.13
    else:
        raise ValueError(f"Y{y_num} is not defined with an x and y position")

    # Path to the airfoil .dat file
    airfoil_file = Path(project_dir) / "data" / "airfoils" / f"y{y_num}.dat"

    # Read the .dat file manually handling headers and whitespaces
    airfoil_data = pd.read_csv(airfoil_file, header=None, skiprows=1, sep="\s+")

    # Manually set the correct column names
    airfoil_data.columns = ["x [m]", "y [m]", "x/c [-]", "y/c [-]"]

    # Extract x and y columns
    airfoil_x = airfoil_data["x [m]"].values
    airfoil_y = airfoil_data["y [m]"].values

    # Adjust the airfoil to predefined location
    airfoil_x_shifted = airfoil_x + x_pos
    airfoil_y_shifted = airfoil_y + y_pos

    # Rotate airfoil
    alpha_rad = -np.radians(alpha)
    airfoil_x_rotated = airfoil_x_shifted * np.cos(
        alpha_rad
    ) - airfoil_y_shifted * np.sin(alpha_rad)
    airfoil_y_rotated = airfoil_x_shifted * np.sin(
        alpha_rad
    ) + airfoil_y_shifted * np.cos(alpha_rad)

    # Plot the airfoil as a black enclosed area
    ax.plot(airfoil_x_rotated, airfoil_y_rotated, color="black")

    # Optionally, close the airfoil by connecting last point to the first
    ax.fill(airfoil_x_rotated, airfoil_y_rotated, "black", alpha=0.3)


def saving_a_plot(
    y_num: int,
    alpha: float,
    project_dir: Path,
    plot_type: str,
    title: str,
    color_data_col_name: str,
    cbar_value_factor_of_std: float,
    min_cbar_value,
    max_cbar_value,
    subsample_color: int,
    countour_levels: int,
    is_with_quiver: bool,
    subsample_quiver: int,
    u_inf: int,
    cmap: str,
    d_alpha_rods: float,
):

    # importing data
    aoa_rod = round(alpha + d_alpha_rods, 0)
    csv_file_path = (
        Path(project_dir)
        / "data"
        / "stichted_planes_erik"
        / f"aoa_{int(aoa_rod)}"
        / f"aoa_{int(aoa_rod)}_Y{int(y_num)}_stitched.csv"
    )
    df = pd.read_csv(csv_file_path)

    # Convert x, y coordinates to meshgrid
    x_unique = df["x"].unique()
    y_unique = df["y"].unique()
    x_meshgrid_global, y_meshgrid_global = np.meshgrid(x_unique, y_unique)

    # Set default color_data as V if not provided
    color_data = df[color_data_col_name].values.reshape(len(y_unique), len(x_unique))

    # If no cbar ranges are given, calculate it based on standard deviation
    if min_cbar_value is None or max_cbar_value is None:
        mean_val = np.nanmean(color_data)
        std_val = np.nanstd(color_data)
        min_cbar_value = mean_val - cbar_value_factor_of_std * std_val
        max_cbar_value = mean_val + cbar_value_factor_of_std * std_val

    fig, ax = plt.subplots()
    # subsampling the color
    x_mesh_sub = x_meshgrid_global[::subsample_color, ::subsample_color]
    y_mesh_sub = y_meshgrid_global[::subsample_color, ::subsample_color]
    color_data_sub = color_data[::subsample_color, ::subsample_color]

    cax = plt.contourf(
        x_mesh_sub,
        y_mesh_sub,
        color_data_sub,
        cmap=cmap,
        levels=countour_levels,
        extend="both",
        vmin=min_cbar_value,
        vmax=max_cbar_value,
    )

    mid_cbar_value = np.mean([min_cbar_value, max_cbar_value])
    cbar = fig.colorbar(
        cax,
        ticks=[
            min_cbar_value,
            mid_cbar_value,
            max_cbar_value,
        ],
        format=mticker.FixedFormatter(
            [
                f"< {min_cbar_value:.2f}",
                f"{mid_cbar_value:.2f}",
                f"> {max_cbar_value:.2f}",
            ]
        ),
        extend="both",
    )
    labels = cbar.ax.get_yticklabels()
    labels[0].set_verticalalignment("top")
    labels[-1].set_verticalalignment("bottom")
    cbar.set_label(color_data_col_name, rotation=0)

    if is_with_quiver:
        # Get u and v values and reshape
        u_values = df["u"].values.reshape(len(y_unique), len(x_unique))
        v_values = df["v"].values.reshape(len(y_unique), len(x_unique))

        # Subsample the data
        x_sub = x_meshgrid_global[::subsample_quiver, ::subsample_quiver]
        y_sub = y_meshgrid_global[::subsample_quiver, ::subsample_quiver]
        u_sub = u_values[::subsample_quiver, ::subsample_quiver]
        v_sub = v_values[::subsample_quiver, ::subsample_quiver]

        # Remove NaN values
        valid_mask = ~(np.isnan(u_sub) | np.isnan(v_sub))

        # # Calculate velocity magnitude for scaling
        # vel_mag = np.sqrt(u_sub[valid_mask] ** 2 + v_sub[valid_mask] ** 2)

        ax.quiver(
            x_sub[valid_mask],
            y_sub[valid_mask],
            u_sub[valid_mask] / u_inf,  # Normalize by u_inf
            v_sub[valid_mask] / u_inf,  # Normalize by u_inf
            color="k",
            angles="xy",
            scale_units="xy",
        )

    ## Plotting the airfoil
    plot_airfoil(y_num, alpha, project_dir, ax)

    ## Saving the plot
    if title is None:
        title = rf"Y{y_num} | $\alpha$ = {alpha} | {int(u_inf)}m/s"

    plt.title(title)

    # defining saving folder
    save_plots_folder = Path(project_dir) / "results" / f"alpha_{int(alpha)}"
    save_plots_folder.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_plots_folder / f"Y{y_num}{plot_type}")
    plt.close()


def main(
    y_num: int,
    alpha: float,
    project_dir: Path,
    plot_type=".pdf",
    title=None,
    color_data_col_name: str = "V",
    cbar_value_factor_of_std: float = 1,
    min_cbar_value=None,
    max_cbar_value=None,
    subsample_color: int = 1,
    countour_levels=100,
    is_with_quiver=False,
    subsample_quiver=10,
    u_inf=15,
    cmap: str = "RdBu",
    d_alpha_rods: float = 7.25,
):
    saving_a_plot(
        y_num,
        alpha,
        project_dir,
        plot_type,
        title,
        color_data_col_name,
        cbar_value_factor_of_std,
        min_cbar_value,
        max_cbar_value,
        subsample_color,
        countour_levels,
        is_with_quiver,
        subsample_quiver,
        u_inf,
        cmap,
        d_alpha_rods,
    )


if __name__ == "__main__":
    project_dir = "/home/jellepoland/ownCloud/phd/code/kite_piv_analysis"
    main(y_num=7, alpha=6, project_dir=project_dir)
