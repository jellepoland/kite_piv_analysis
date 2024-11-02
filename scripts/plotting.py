import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


class PlotParams(TypedDict):
    # Basic configuration
    is_CFD: bool
    y_num: int
    alpha: float
    project_dir: Path
    plot_type: str
    title: Optional[str]
    is_CFD_PIV_comparison: bool
    color_data_col_name: str

    # Color and contour settings
    is_with_cbar: bool
    cbar_value_factor_of_std: float
    min_cbar_value: Optional[float]
    max_cbar_value: Optional[float]
    subsample_color: int
    countour_levels: int
    cmap: str

    # Quiver settings
    is_with_quiver: bool
    subsample_quiver: int
    u_inf: float

    # PIV specific settings
    d_alpha_rod: float

    # Overlay settings
    is_with_overlay: bool
    overlay_alpha: float

    # Airfoil settings
    is_with_airfoil: bool
    airfoil_transparency: float

    # Raw image settings
    subsample_factor_raw_images: int
    intensity_lower_bound: int

    # Boundary settings
    is_with_bound: bool
    d1centre: np.ndarray
    drot: float
    dLx: float
    dLy: float
    iP: int
    ellipse_color: str
    rectangle_color: str
    bound_linewidth: float
    bound_alpha: float

    # Circulation analysis
    is_with_circulation_analysis: bool
    rho: float
    mu: float
    is_with_maximim_vorticity_location_correction: bool

    # Mask settings
    is_with_mask: bool
    column_to_mask: str
    mask_lower_bound: float
    mask_upper_bound: float


def load_data(plot_params: dict) -> tuple:
    """Load and process data from either CFD or PIV sources."""

    y_num = plot_params["y_num"]
    alpha = plot_params["alpha"]
    is_CFD = plot_params["is_CFD"]
    d_alpha_rod = plot_params["d_alpha_rod"]

    if is_CFD:
        csv_file_path = (
            Path(project_dir)
            / "processed_data"
            / "CFD"
            / f"alpha_{int(alpha)}"
            / f"Y{y_num}_paraview_corrected.csv"
        )
        plot_params["csv_file_path_std"] = None
    else:
        aoa_rod = round(alpha + d_alpha_rod, 0)
        csv_file_path = (
            Path(project_dir)
            / "processed_data"
            / "stichted_planes_erik"
            / f"aoa_{int(aoa_rod)}"
            / f"aoa_{int(aoa_rod)}_Y{int(y_num)}_stitched.csv"
        )
        plot_params["csv_file_path_std"] = (
            Path(project_dir)
            / "processed_data"
            / "stichted_planes_erik"
            / f"aoa_{int(aoa_rod)}"
            / f"aoa_{int(aoa_rod)}_Y{int(y_num)}_stitched_std.csv"
        )

    df = pd.read_csv(csv_file_path)

    x_unique = df["x"].unique()
    y_unique = df["y"].unique()
    x_meshgrid, y_meshgrid = np.meshgrid(x_unique, y_unique)

    return df, x_meshgrid, y_meshgrid, plot_params


def apply_mask(
    df: pd.DataFrame,
    plot_params: dict,
    columns_to_mask: list = [
        "u",
        "v",
        "w",
        "V",
        "dudx",
        "dudy",
        "dvdx",
        "dvdy",
        "dwdx",
        "dwdy",
        "vort_z",
    ],
) -> pd.DataFrame:
    """
    Apply a mask to a DataFrame, setting values to NaN in specified columns for rows
    where the target column's values are outside the specified range.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame to be masked.
    column : str
        The name of the column on which to apply the mask.
    lower : float
        The lower bound of the value range.
    upper : float
        The upper bound of the value range.
    columns_to_mask : list
        List of column names to set to NaN in rows where the specified column's
        values fall outside the bounds.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame with specified columns set to NaN for rows where the target
        column's values are outside the specified bounds.
    """
    column = plot_params["column_to_mask"]
    lower = plot_params["mask_lower_bound"]
    upper = plot_params["mask_upper_bound"]
    csv_file_path_std = plot_params["csv_file_path_std"]

    # Create a copy of the DataFrame
    df_to_return = df.copy()

    # if dealing with standard deviation masking, read the std file
    if "std" in column:
        if csv_file_path_std is None:
            raise ValueError(
                "Standard deviation masking requires a CSV file path, only available for PIV"
            )
        df_masked = pd.read_csv(csv_file_path_std)
        column = column.split("_")[0]
    else:
        df_masked = df.copy()

    # Identify rows where the target column's values are outside the specified range
    mask = (df_masked[column] < lower) | (df_masked[column] > upper)

    # Apply NaN to the specified columns in rows where the mask is True
    df_to_return.loc[mask, columns_to_mask] = np.nan

    return df_to_return


def plot_color_contour(ax, df, x_meshgrid, y_meshgrid, plot_params):
    ## Getting the color data
    x_unique = df["x"].unique()
    y_unique = df["y"].unique()
    color_data = df[plot_params["color_data_col_name"]].values.reshape(
        len(y_unique), len(x_unique)
    )
    # Subsample and plot contours
    x_mesh_sub = x_meshgrid[
        :: plot_params["subsample_color"], :: plot_params["subsample_color"]
    ]
    y_mesh_sub = y_meshgrid[
        :: plot_params["subsample_color"], :: plot_params["subsample_color"]
    ]
    color_data_sub = color_data[
        :: plot_params["subsample_color"], :: plot_params["subsample_color"]
    ]

    if plot_params["min_cbar_value"] is None or plot_params["max_cbar_value"] is None:
        mean_val = np.nanmean(color_data)
        std_val = np.nanstd(color_data)
        plot_params["min_cbar_value"] = (
            mean_val - plot_params["cbar_value_factor_of_std"] * std_val
        )
        plot_params["max_cbar_value"] = (
            mean_val + plot_params["cbar_value_factor_of_std"] * std_val
        )

    # ### USING PCOLORMESH
    # cax = ax.pcolormesh(
    #     x_mesh_sub,
    #     y_mesh_sub,
    #     color_data_sub,
    #     # shading="auto",
    #     cmap=plot_params["cmap"],
    #     vmin=plot_params["min_cbar_value"],
    #     vmax=plot_params["max_cbar_value"],
    # )  # 'shading' set to 'auto' to avoid warning

    #### USING CONTOURF
    cax = ax.contourf(
        x_mesh_sub,
        y_mesh_sub,
        color_data_sub,
        levels=plot_params["countour_levels"],
        cmap=plot_params["cmap"],
        vmin=plot_params["min_cbar_value"],
        vmax=plot_params["max_cbar_value"],
    )

    plot_params["cax"] = cax

    return plot_params


def add_quiver(ax, df, x_meshgrid, y_meshgrid, plot_params):
    # Define x_unique and y_unique
    x_unique = df["x"].unique()
    y_unique = df["y"].unique()

    # Get u and v values and reshape
    u_values = df["u"].values.reshape(len(y_unique), len(x_unique))
    v_values = df["v"].values.reshape(len(y_unique), len(x_unique))

    # Subsample the data
    subsample_quiver = plot_params["subsample_quiver"]
    x_sub = x_meshgrid[::subsample_quiver, ::subsample_quiver]
    y_sub = y_meshgrid[::subsample_quiver, ::subsample_quiver]
    u_sub = u_values[::subsample_quiver, ::subsample_quiver]
    v_sub = v_values[::subsample_quiver, ::subsample_quiver]

    # Remove NaN values
    valid_mask = ~(np.isnan(u_sub) | np.isnan(v_sub))

    # # Calculate velocity magnitude for scaling
    # vel_mag = np.sqrt(u_sub[valid_mask] ** 2 + v_sub[valid_mask] ** 2)

    ax.quiver(
        x_sub[valid_mask],
        y_sub[valid_mask],
        u_sub[valid_mask] / plot_params["u_inf"],  # Normalize by u_inf
        v_sub[valid_mask] / plot_params["u_inf"],  # Normalize by u_inf
        color="k",
        angles="xy",
        scale_units="xy",
    )


def plot_airfoil(
    ax,
    plot_params,
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

    y_num = plot_params["y_num"]
    alpha = plot_params["alpha"]
    airfoil_transparency = plot_params["airfoil_transparency"]

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


def overlay_raw_image(
    ax,
    plot_params,
):

    y_num = plot_params["y_num"]
    alpha = plot_params["alpha"]
    d_alpha_rod = plot_params["d_alpha_rod"]
    overlay_alpha = plot_params["overlay_alpha"]
    subsample_factor_raw_images = plot_params["subsample_factor_raw_images"]
    intensity_lower_bound = plot_params["intensity_lower_bound"]

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


def add_boundaries(ax, plot_params):
    d1centre = plot_params["d1centre"]
    drot = plot_params["drot"]
    dLx = plot_params["dLx"]
    dLy = plot_params["dLy"]
    iP = plot_params["iP"]
    ellipse_color = plot_params["ellipse_color"]
    rectangle_color = plot_params["rectangle_color"]
    bound_linewidth = plot_params["bound_linewidth"]
    bound_alpha = plot_params["bound_alpha"]

    d2curve_ellipse = boundary_ellipse(d1centre, drot, dLx, dLy, iP)
    ax.plot(
        d2curve_ellipse[:, 0],  # x-coordinates of the boundary
        d2curve_ellipse[:, 1],  # y-coordinates of the boundary
        color=ellipse_color,  # Boundary color (e.g., red)
        linestyle="--",  # Dashed line for visibility
        linewidth=bound_linewidth,  # Line width for boundary
        alpha=bound_alpha,
    )
    d2curve_rectangle = boundary_rectangle(d1centre, drot, dLx, dLy, iP)
    ax.plot(
        d2curve_rectangle[:, 0],  # x-coordinates of the boundary
        d2curve_rectangle[:, 1],  # y-coordinates of the boundary
        color=rectangle_color,  # Boundary color (e.g., red)
        linestyle="-",  # Dashed line for visibility
        linewidth=bound_linewidth,  # Line width for boundary
        alpha=bound_alpha,
        # marker="o",
    )

    return d2curve_ellipse, d2curve_rectangle


def add_circulation_analysis(
    fig, ax, df, plot_params, d2curve_ellipse, d2curve_rectangle
):

    u_inf = plot_params["u_inf"]
    rho = plot_params["rho"]
    df_1D = df
    d1centre = plot_params["d1centre"]
    drot = plot_params["drot"]
    dLx = plot_params["dLx"]
    dLy = plot_params["dLy"]
    iP = plot_params["iP"]
    mu = plot_params["mu"]
    is_with_maximim_vorticity_location_correction = plot_params[
        "is_with_maximim_vorticity_location_correction"
    ]

    ellipse_gamma = calculate_circulation(df, d2curve_ellipse)
    rectangle_gamma = calculate_circulation(df, d2curve_rectangle)
    ellipse_kutta_force = ellipse_gamma * u_inf * rho
    rectangle_kutta_force = rectangle_gamma * u_inf * rho

    is_ellipse = True
    force_normal_ellipse_array, force_tangential_ellipse_array = force_from_noca.main(
        df_1D,
        is_ellipse,
        d1centre,
        drot,
        dLx,
        dLy,
        iP,
        mu,
        is_with_maximim_vorticity_location_correction,
    )
    is_ellipse = False
    force_normal_rectangle_array, force_tangential_rectangle_array = (
        force_from_noca.main(
            df_1D,
            is_ellipse,
            d1centre,
            drot,
            dLx,
            dLy,
            iP,
            mu,
            is_with_maximim_vorticity_location_correction,
        )
    )
    force_normal_ellipse = force_normal_ellipse_array[0]
    force_tangential_ellipse = force_tangential_ellipse_array[0]
    force_normal_rectangle = force_normal_rectangle_array[0]
    force_tangential_rectangle = force_tangential_rectangle_array[0]

    # Adding text below the plot using LaTeX formatting
    text = (
        f"Ellipse:    $\\Gamma$: {ellipse_gamma:.2f} m2/s --> $F_{{\\rho \\Gamma u}}$: {ellipse_kutta_force:.2f} N"
        f"  |  NOCA --> $F_n$: {force_normal_ellipse:.2f} N, $F_t$: {force_tangential_ellipse:.2f} N\n"
        f"Rectangle:  $\\Gamma$: {rectangle_gamma:.2f} m2/s --> $F_{{\\rho \\Gamma u}}$: {rectangle_kutta_force:.2f} N"
        f"  |  NOCA --> $F_n$: {force_normal_rectangle:.2f} N, $F_t$: {force_tangential_rectangle:.2f} N"
    )

    ax.text(
        0.5,
        0.1,
        text,
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=6,
        transform=ax.transAxes,
        wrap=True,
    )


def plotting_on_ax(
    fig,
    ax,
    df: pd.DataFrame,
    x_meshgrid: np.ndarray,
    y_meshgrid: np.ndarray,
    plot_params: dict,
) -> None:

    ax.set_aspect("equal", adjustable="box")

    if plot_params.get("is_with_mask", False) and not plot_params["is_CFD"]:
        df = apply_mask(df, plot_params)

    plot_params = plot_color_contour(ax, df, x_meshgrid, y_meshgrid, plot_params)

    # # Add optional elements
    # if plot_params.get("is_with_cbar", False):
    #     add_colorbar(fig, plot_params)

    if plot_params.get("is_with_quiver", False):
        add_quiver(ax, df, x_meshgrid, y_meshgrid, plot_params)

    if plot_params.get("is_with_airfoil", False):
        plot_airfoil(ax, plot_params)

    if plot_params.get("is_with_overlay", False):
        overlay_raw_image(ax, plot_params)

    if plot_params.get("is_with_bound", False):
        d2curve_ellipse, d2curve_rectangle = add_boundaries(ax, plot_params)

        if plot_params.get("is_with_circulation_analysis", False):
            add_circulation_analysis(
                fig, ax, df, plot_params, d2curve_ellipse, d2curve_rectangle
            )

    ax.set_xlabel("x [m]")
    ax.set_xlim(-0.2, 0.8)
    if plot_params["is_CFD"] or not plot_params["is_CFD_PIV_comparison"]:
        ax.set_ylabel("y [m]")
    ax.set_ylim(-0.2, 0.4)

    return plot_params


def add_colorbar(fig, ax, plot_params):

    cax = plot_params["cax"]
    vmin = plot_params["min_cbar_value"]
    vmax = plot_params["max_cbar_value"]
    ### USING PCOLORMESH
    # Create a horizontal color bar for the figure
    # cbar = fig.colorbar(
    #     cax,
    #     ax=ax,
    #     orientation="horizontal",
    #     fraction=0.02,
    #     pad=0.1,
    # )
    # cbar.set_label(plot_params["color_data_col_name"])

    ### USING CONTOURF
    divider = make_axes_locatable(ax)
    divider_ax = divider.append_axes("top", size="5%", pad=0.3)
    cbar = plt.colorbar(
        ScalarMappable(norm=cax.norm, cmap=cax.cmap),
        cax=divider_ax,
        ticks=np.linspace(int(vmin), int(vmax), 10),
        orientation="horizontal",
    )
    # adjust tick labels
    cbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Adjust label alignment
    # labels = cbar.ax.get_yticklabels()
    # Set the ticks to appear on top of the color bar
    cbar.ax.xaxis.set_ticks_position("top")  # Move ticks to the top
    cbar.ax.xaxis.set_label_position("top")  # Move label to the top
    # labels[0].set_verticalalignment("top")
    # labels[-1].set_verticalalignment("bottom")
    cbar.set_label(plot_params["color_data_col_name"], rotation=0, fontsize=8)


def save_plot(
    fig: plt.Figure,
    plot_params: dict,
) -> None:
    """Save the plot to the appropriate directory."""
    alpha = plot_params["alpha"]
    y_num = plot_params["y_num"]
    is_CFD = plot_params["is_CFD"]
    plot_type = plot_params["plot_type"]
    color_data_col_name = plot_params["color_data_col_name"]

    if plot_params["is_CFD_PIV_comparison"]:
        save_plots_folder = (
            Path(project_dir) / "results" / f"alpha_{int(alpha)}" / "CFD_PIV"
        )
    else:
        if is_CFD:
            save_plots_folder = (
                Path(project_dir) / "results" / f"alpha_{int(alpha)}" / "CFD"
            )
        else:
            save_plots_folder = (
                Path(project_dir) / "results" / f"alpha_{int(alpha)}" / "PIV"
            )

    save_plots_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_plots_folder / f"Y{y_num}_{color_data_col_name}{plot_type}")
    plt.close()


def plotting_single(plot_params: dict) -> None:
    """Create a single plot with the specified parameters."""
    fig, ax = plt.subplots()
    plt.title(
        f'Y_{plot_params["y_num"]} | α = {plot_params["alpha"]}° | {plot_params["u_inf"]}m/s'
    )

    # Load, plot and save
    df, x_meshgrid, y_meshgrid, plot_params = load_data(plot_params)
    plot_params = plotting_on_ax(fig, ax, df, x_meshgrid, y_meshgrid, plot_params)

    if plot_params["is_with_cbar"]:
        add_colorbar(fig, ax, plot_params)

    save_plot(fig, plot_params)


def plotting_CFD_PIV_comparison(plot_params: dict) -> None:
    """Create a side-by-side comparison of CFD and PIV data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.title(
        rf'Y{plot_params["y_num"]} | α = {plot_params["alpha"]}° | V_inf = {plot_params["u_inf"]}m/s'
    )

    # Load and plot CFD data
    print(f"Plotting CFD")
    df_cfd, x_mesh_cfd, y_mesh_cfd, plot_params = load_data(
        plot_params | {"is_CFD": True}
    )
    plot_params = plotting_on_ax(fig, ax1, df_cfd, x_mesh_cfd, y_mesh_cfd, plot_params)
    ax1.set_title("CFD")
    if plot_params["is_with_cbar"]:
        add_colorbar(fig, ax1, plot_params)

    # Load and plot PIV data
    print(f"Plotting PIV")
    df_piv, x_mesh_piv, y_mesh_piv, plot_params = load_data(
        plot_params | {"is_CFD": False}
    )
    plot_params = plotting_on_ax(fig, ax2, df_piv, x_mesh_piv, y_mesh_piv, plot_params)
    ax2.set_title("PIV")
    if plot_params["is_with_cbar"]:
        add_colorbar(fig, ax2, plot_params)

    save_plot(fig, plot_params)


def main(plot_params: dict) -> None:
    if plot_params["is_CFD_PIV_comparison"]:
        plotting_CFD_PIV_comparison(plot_params)
    else:
        plotting_single(plot_params)


if __name__ == "__main__":

    plot_params: PlotParams = {
        # Basic configuration
        "is_CFD": False,
        "y_num": 1,
        "alpha": 6,
        "project_dir": project_dir,
        "plot_type": ".pdf",
        "title": None,
        "is_CFD_PIV_comparison": False,
        "color_data_col_name": "u",
        # Color and contour settings
        "is_with_cbar": True,
        "cbar_value_factor_of_std": 2.0,
        "min_cbar_value": None,
        "max_cbar_value": None,
        "subsample_color": 1,
        "countour_levels": 100,
        "cmap": "viridis",
        # Quiver settings
        "is_with_quiver": False,
        "subsample_quiver": 10,
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
        "d1centre": np.array([0.27, 0.13]),
        "drot": 0.0,
        "dLx": 0.8,
        "dLy": 0.4,
        "iP": 27,
        "ellipse_color": "black",
        "rectangle_color": "black",
        "bound_linewidth": 1.0,
        "bound_alpha": 0.5,
        # Circulation analysis
        "is_with_circulation_analysis": False,
        "rho": 1.225,
        "mu": 1.7894e-5,
        "is_with_maximim_vorticity_location_correction": True,
        # Mask settings
        "is_with_mask": True,
        "column_to_mask": "V_std",
        "mask_lower_bound": -5,
        "mask_upper_bound": 5,
    }
    main(plot_params)
    if plot_params["is_CFD_PIV_comparison"]:
        type_label = "CFD_PIV"
    else:
        type_label = "CFD" if plot_params["is_CFD"] else "PIV"

    print(
        f'{type_label} plot with color = {plot_params["color_data_col_name"]} | Y{plot_params["y_num"]} | α = {plot_params["alpha"]}°'
    )
