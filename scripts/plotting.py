import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from pathlib import Path
import pandas as pd
import os
from utils import project_dir
from io import StringIO
from defining_bound_volume import boundary_ellipse, boundary_rectangle


def displacing_subsampling_plotting(
    ax,
    df: pd.DataFrame,
    intensity_lower_bound: float,
    subsample_factor: int,
    delta_x: float,
    delta_y: float,
    cmap: str,
    label: str,
    is_flipped: bool,
    overlay_alpha: float,
):

    # Drop rows with invalid data
    df = df[df["isValid"] == 1]

    # Drop rows with intensity values below a threshold
    df = df[df["Intensity [counts]"] > intensity_lower_bound]

    # Use .loc[] to modify the "x [mm]" column safely
    df["x [mm]"] = pd.to_numeric(df["x [mm]"], errors="coerce")
    df["y [mm]"] = pd.to_numeric(df["y [mm]"], errors="coerce")
    df["Intensity [counts]"] = pd.to_numeric(df["Intensity [counts]"], errors="coerce")

    # flipping the data if needed
    if is_flipped:
        df["y [mm]"] = -df["y [mm]"]

    # Subsample the data
    df_subsampled = df.iloc[::subsample_factor, :]

    # change x and y -locations
    df_subsampled.loc[:, "x [mm]"] += delta_x
    df_subsampled.loc[:, "y [mm]"] += delta_y

    # First scatter plot
    sc = ax.scatter(
        df_subsampled["x [mm]"] / 1e3,
        df_subsampled["y [mm]"] / 1e3,
        c=df_subsampled["Intensity [counts]"],
        cmap=cmap,
        alpha=overlay_alpha,
        s=0.1,
        label=label,
    )
    # plt.colorbar(sc, ax=ax, label="Intensity [counts]")


def overlay_raw_image(
    y_num: int,
    alpha: int,
    project_dir: Path,
    ax,
    d_alpha_rod: float,
    overlay_alpha: float,
    subsample_factor_raw_images: int,
    intensity_lower_bound: int,
):
    # Defining the folder
    folder_dir = (
        "/run/media/jellepoland/HSL-Drive-001/Jelle_Poland_KiteOJF_20240420/PIV_raw"
    )
    raw_image_dir = (
        Path(folder_dir) / "raw_images" / f"aoa_{int(alpha+d_alpha_rod)}" / f"Y{y_num}"
    )

    # raw_image_dir = (
    #     Path(project_dir)
    #     / "data"
    #     / "raw_images"
    #     / f"aoa_{int(alpha+d_alpha_rod)}"
    #     / f"Y{y_num}"
    # )
    for folder_name in os.listdir(raw_image_dir):

        # reading X value from folder name, last character
        x_num = int(folder_name[-1])

        # Reading config type
        if "flipped" in folder_name:
            is_flipped = True
            cmap = "viridis"
            label = "flipped"
            overlay_alpha = 0.3
        else:
            is_flipped = False
            cmap = "gray"
            label = "normal"
            overlay_alpha = 0.1

        # reading the csv file with the translation values as df
        df = pd.read_csv(
            Path(project_dir) / "data" / "planes_location.csv",
            index_col=0,
        )

        # filter on alpha, y_num, x_num, config
        df = df[df["alpha"] == alpha]
        df = df[df["Y"] == y_num]
        df = df[df["X"] == x_num]
        df = df[df["config"] == label]

        # Get out the x and y position
        delta_x = df["delta_x"].values[0]
        delta_y = df["delta_y"].values[0]

        # reading out the subsampled .csv file
        dat_file_path = Path(raw_image_dir) / folder_name / "B0001_subsampled.csv"
        df_raw_image = pd.read_csv(dat_file_path)

        displacing_subsampling_plotting(
            ax,
            df_raw_image,
            intensity_lower_bound,
            subsample_factor_raw_images,
            delta_x=delta_x,
            delta_y=delta_y,
            cmap=cmap,
            label=label,
            is_flipped=is_flipped,
            overlay_alpha=overlay_alpha,
        )

    # ## Normal
    # df_normal = pd.read_csv(dat_file_path_list[1])
    # df_subsampled_normal, sc1 = displacing_subsampling_plotting(
    #     ax,
    #     df_normal,
    #     intensity_lower_bound,
    #     subsample_factor_raw_images,
    #     delta_x=300,
    #     delta_y=0,
    #     cmap="gray",
    #     label="normal",
    #     is_flipped=False,
    #     overlay_alpha=overlay_alpha,
    # )
    # ## Flipped
    # df_flipped = pd.read_csv(dat_file_path_list[0])
    # df_subsampled_flipped, sc2 = displacing_subsampling_plotting(
    #     ax,
    #     df_flipped,
    #     intensity_lower_bound,
    #     subsample_factor_raw_images,
    #     delta_x=320,
    #     delta_y=201,
    #     cmap="viridis",
    #     label="flipped",
    #     is_flipped=True,
    #     overlay_alpha=overlay_alpha / 2,
    # )


def plot_airfoil(
    y_num: int,
    alpha: int,
    project_dir: Path,
    ax,
    airfoil_transparency: float,
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

    # reading the csv file with the translation values as df
    df = pd.read_csv(
        Path(project_dir) / "data" / "airfoils" / "airfoil_translation_values.csv",
        index_col=0,
    )
    # filter on alpha
    df = df[df["alpha"] == alpha]
    # filter on y_num
    df = df[df["Y"] == y_num]
    # Get out the x and y position
    x_pos = df["x"].values[0]
    y_pos = df["y"].values[0]

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
    ax.plot(
        airfoil_x_rotated,
        airfoil_y_rotated,
        color="black",
        linewidth=0.4,
    )

    # Optionally, close the airfoil by connecting last point to the first
    ax.fill(airfoil_x_rotated, airfoil_y_rotated, "black", alpha=airfoil_transparency)


def saving_a_plot(
    is_CFD: Path,
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
    is_with_overlay: bool,
    overlay_alpha: float,
    is_with_airfoil: bool,
    airfoil_transparency: float,
    subsample_factor_raw_images: int,
    intensity_lower_bound: int,
    is_with_bound: bool,
):
    # importing data
    if is_CFD:
        csv_file_path = (
            Path(project_dir)
            / "processed_data"
            / "CFD"
            / f"alpha_{int(alpha)}"
            / f"Y{y_num}_paraview_corrected.csv"
        )
    else:
        aoa_rod = round(alpha + d_alpha_rod, 0)
        csv_file_path = (
            Path(project_dir)
            / "processed_data"
            / "stichted_planes_erik"
            / f"aoa_{int(aoa_rod)}"
            / f"aoa_{int(aoa_rod)}_Y{int(y_num)}_stitched.csv"
        )

    fig, ax = plt.subplots()
    plt.gca().set_aspect("equal", adjustable="box")
    df = pd.read_csv(csv_file_path)

    # Apply mask
    # df = masking_the_data(df)
    # df = df[np.abs(df["v"]) > 5]
    # df = df[np.abs(df["w"]) > 5]
    ##TODO: standard-deviation w mask

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

    ### Airfoil
    if is_with_airfoil:
        plot_airfoil(y_num, alpha, project_dir, ax, airfoil_transparency)

    ### Overlay
    if is_with_overlay:
        ## Overlaying with raw_image
        overlay_raw_image(
            y_num=y_num,
            alpha=alpha,
            project_dir=Path(project_dir),
            ax=ax,
            d_alpha_rod=d_alpha_rod,
            overlay_alpha=overlay_alpha,
            subsample_factor_raw_images=subsample_factor_raw_images,
            intensity_lower_bound=intensity_lower_bound,
        )

    ### Bound Volume
    if is_with_bound:
        d1centre = np.array([0.3, 0.15])
        drot = 0
        dLx = 0.7
        dLy = 0.4
        iP = 27  # 9, 15, 21,  119
        d2curve = boundary_ellipse(d1centre, drot, dLx, dLy, iP)
        ax.plot(
            d2curve[:, 0],  # x-coordinates of the boundary
            d2curve[:, 1],  # y-coordinates of the boundary
            color="black",  # Boundary color (e.g., red)
            linestyle="--",  # Dashed line for visibility
            linewidth=1,  # Line width for boundary
            alpha=0.5,
        )
        print(f"ELLIPSE volume plotted, len(d2curve): {len(d2curve)}")
        d2curve = boundary_rectangle(d1centre, drot, dLx, dLy, iP)
        ax.plot(
            d2curve[:, 0],  # x-coordinates of the boundary
            d2curve[:, 1],  # y-coordinates of the boundary
            color="black",  # Boundary color (e.g., red)
            linestyle="-",  # Dashed line for visibility
            linewidth=1,  # Line width for boundary
            alpha=0.5,
            # marker="o",
        )
        print(f"RECTANGLE volume plotted, len(d2curve): {len(d2curve)}")
    ## Saving the plot
    if title is None:
        title = rf"Y{y_num} | $\alpha$ = {alpha} | {int(u_inf)}m/s"

    plt.title(title)

    # defining saving folder
    if is_CFD:
        save_plots_folder = (
            Path(project_dir) / "results" / f"alpha_{int(alpha)}" / "CFD"
        )
    else:
        save_plots_folder = (
            Path(project_dir) / "results" / f"alpha_{int(alpha)}" / "PIV"
        )
    save_plots_folder.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_plots_folder / f"Y{y_num}{plot_type}")
    plt.close()


def main(
    is_CFD: bool,
    y_num: int,
    alpha: float,
    project_dir: Path,
    plot_type=".pdf",
    title=None,
    color_data_col_name: str = "V",
    cbar_value_factor_of_std: float = 2,
    min_cbar_value=None,
    max_cbar_value=None,
    subsample_color: int = 1,
    countour_levels=100,
    is_with_quiver=False,
    subsample_quiver=10,
    u_inf=15,
    cmap: str = "RdBu",
    d_alpha_rod: float = 7.25,
    is_with_overlay: bool = True,
    overlay_alpha: float = 0.4,
    is_with_airfoil: bool = True,
    airfoil_transparency: float = 0.3,
    subsample_factor_raw_images: int = 1,
    intensity_lower_bound: int = 10000,
    is_with_bound: bool = True,
):

    print(f"\n--> Plotting for Y{y_num} at alpha = {alpha} degrees")

    saving_a_plot(
        is_CFD,
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
        is_with_overlay,
        overlay_alpha,
        is_with_airfoil,
        airfoil_transparency,
        subsample_factor_raw_images,
        intensity_lower_bound,
        is_with_bound,
    )


if __name__ == "__main__":
    main(
        is_CFD=True,
        y_num=1,
        alpha=6,
        project_dir=project_dir,
        is_with_airfoil=True,
        airfoil_transparency=1.0,
        is_with_overlay=False,
        intensity_lower_bound=10000,
    )
