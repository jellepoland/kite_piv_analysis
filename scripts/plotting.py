import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from pathlib import Path
import pandas as pd
import os
from io import StringIO


def overlay_raw_image(
    y_num: int,
    alpha: int,
    project_dir: Path,
    ax,
    d_alpha_rod: float,
    overlay_alpha: float = 0.1,
    subsample_factor: int = 500,
    intensity_lower_bound: int = 10000,
):
    # Defining the folder
    raw_image_dir = Path(project_dir) / "data" / "raw_images"
    # Find the folder with the specific conditions
    matched_folder = None
    for folder_name in os.listdir(raw_image_dir):
        print(f"folder_name: {folder_name}")
        if (
            folder_name.startswith(f"normal_aoa_{int(alpha+d_alpha_rod)}")
            and f"Y{y_num}" in folder_name
        ):
            matched_folder = folder_name
            break

    # If no folder is found, raise an error
    if matched_folder is None:
        raise FileNotFoundError(
            f"No matching folder found for alpha={alpha} and Y{y_num}"
        )
    else:
        print(f'Found folder "{matched_folder}"')

    # Construct the path to the B0001.dat file
    dat_file_path = Path(raw_image_dir) / matched_folder / "B0001.dat"

    def read_correct_camera_data(dat_file_path):
        with open(dat_file_path, "r") as f:
            content = f.read().strip().split("\n\n")  # Split content by double newline

        df_list = []
        for content_i in content:
            df = pd.read_csv(
                StringIO(content_i),
                sep="\s+",
                skiprows=5,  # Skip header rows
                names=["x [mm]", "y [mm]", "Intensity [counts]", "isValid"],
                on_bad_lines="skip",
            )
            df_list.append(df)

        df = pd.concat(df_list)
        return df

    # Read the B0001.dat file
    df = read_correct_camera_data(dat_file_path)

    # Drop rows with invalid data
    df = df[df["isValid"] == 1]

    # Drop rows with intensity values below a threshold
    df = df[df["Intensity [counts]"] > intensity_lower_bound]

    # Use .loc[] to modify the "x [mm]" column safely
    df["x [mm]"] = pd.to_numeric(df["x [mm]"], errors="coerce")
    df["y [mm]"] = pd.to_numeric(df["y [mm]"], errors="coerce")
    df["Intensity [counts]"] = pd.to_numeric(df["Intensity [counts]"], errors="coerce")

    # Subsample the data
    df_subsampled = df.iloc[::subsample_factor, :]

    # change x-locations
    df_subsampled.loc[:, "x [mm]"] += 300

    # change y-locations
    # Assuming 'y [mm]' is the column name for y-values
    ymin = df_subsampled["y [mm]"].min()
    ymax = df_subsampled["y [mm]"].max()
    ymid = (ymin + ymax) / 2
    print(f"ymin: {ymin}, ymax: {ymax}, ymid: {ymid}")
    df_subsampled.loc[:, "y [mm]"] -= ymid

    # Plot the intensity values as an overlay
    sc = ax.scatter(
        df_subsampled["x [mm]"] / 1e3,
        df_subsampled["y [mm]"] / 1e3,
        c=df_subsampled["Intensity [counts]"],
        cmap="gray",
        alpha=overlay_alpha,
        s=0.1,  # Adjust point size if needed
    )

    # # Add a colorbar for intensity values
    plt.colorbar(sc, ax=ax, label="Intensity [counts]")


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
        x_pos = 0  # 0.016
        y_pos = 0.14
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
    # alpha += 7.25
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
    ax.fill(airfoil_x_rotated, airfoil_y_rotated, "black", alpha=1)


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
    d_alpha_rod: float,
):

    # importing data
    aoa_rod = round(alpha + d_alpha_rod, 0)
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

    ## Overlaying with raw_image
    overlay_raw_image(
        y_num=1,
        alpha=6,
        project_dir=Path(project_dir),
        ax=ax,
        d_alpha_rod=d_alpha_rod,
        overlay_alpha=0.4,
        subsample_factor=200,
    )

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
    d_alpha_rod: float = 7.25,
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
        d_alpha_rod,
    )


if __name__ == "__main__":
    project_dir = "/home/jellepoland/ownCloud/phd/code/kite_piv_analysis"
    main(y_num=1, alpha=6, project_dir=project_dir)
