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
import extract_spanwise_contour
import transforming_paraview_output
from plotting import *


def transform_raw_csv_to_processed_df(alpha=6, x_cm=25) -> pd.DataFrame:

    file_path = (
        Path(project_dir)
        / "data"
        / "CFD_slices"
        / "spanwise_slices"
        / f"alpha_{alpha}_CFD_spanwise_slice_{int(x_cm)}cm_1.csv"
    )

    # Load the raw data
    df = pd.read_csv(file_path)

    ### new
    # Transform the headers
    header_mapping = {
        "Points:0": "z",
        "Points:1": "y",
        "Points:2": "x",
        "Time": "time",
        "ReThetat": "ReTheta",
        "U:0": "u",
        "U:1": "v",
        "U:2": "w",
        "gammaInt": "gamma_int",
        "grad(U):0": "dudx",
        "grad(U):1": "dudy",
        "grad(U):2": "dudz",
        "grad(U):3": "dvdx",
        "grad(U):4": "dvdy",
        "grad(U):5": "dvdz",
        "grad(U):6": "dwdx",
        "grad(U):7": "dwdy",
        "grad(U):8": "dwdz",
        "vorticity:2": "vort_z",
        "k": "tke",
        "nut": "nu_t",
        "omega": "omega",
        "p": "p",
        "vorticity:0": "vort_x",
        "vorticity:1": "vort_y",
        # "vorticity:2": "vorticity_z",
        "wallShearStress:0": "tau_w_x",
        "wallShearStress:1": "tau_w_y",
        "wallShearStress:2": "tau_w_z",
        "yPlus": "y_plus",
    }

    df = df.rename(columns=header_mapping)
    df["V"] = np.sqrt(df["u"] ** 2 + df["v"] ** 2 + df["w"] ** 2)

    df["vort_mag"] = np.sqrt(df["vort_x"] ** 2 + df["vort_y"] ** 2 + df["vort_z"] ** 2)

    variable_list = [
        "x",
        "y",
        "u",
        "v",
        "w",
        "V",
        "dudx",
        "dudy",
        "dudz",
        "dvdx",
        "dvdy",
        "dvdz",
        "dwdx",
        "dwdy",
        "dwdz",
        "vort_x",
        "vort_y",
        "vort_z",
        "p",
        "tau_w_x",
        "tau_w_y",
        # "is_valid",
        "vort_mag",
    ]
    df = df[variable_list]
    headers = variable_list

    ### old
    # headers = ["z", "y", "x", "u", "v", "w"]
    # df.columns = ["z", "y", "x", "u", "v", "w"]

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
    # df["shear_mag"] = df["tau_w_x"] ** 2 + df["tau_w_y"] ** 2
    vel_mask = (
        # (df["u"] == 0)
        # & (df["v"] == 0)
        # & (df["w"] == 0)
        # &
        (np.abs(df["tau_w_x"]) > 1e-16)
        & (np.abs(df["tau_w_y"]) > 1e-16)
    )
    zero_vel_df = df[vel_mask]
    # df = df[~vel_mask]

    # # masking for vorticity
    # vort_mask = (
    #     (np.abs(df["vort_x"]) > 1e-1)
    #     & (np.abs(df["vort_y"]) > 1e-1)
    #     & (np.abs(df["vort_z"]) > 1e-1)
    #     & (np.abs(df["vort_x"]) < 2e2)
    #     & (np.abs(df["vort_y"]) < 2e2)
    #     & (np.abs(df["vort_z"]) < 2e2)
    # )
    # df = df[vort_mask]

    df["vort_mag"] = np.sqrt(df["vort_x"] ** 2 + df["vort_y"] ** 2 + df["vort_z"] ** 2)

    # scale velocity
    scaled_data = transforming_paraview_output.scaling_CFD(df.values, headers)
    final_df = pd.DataFrame(scaled_data, columns=headers)

    # scale dimensions
    final_df["x"] *= 2.584 / 6.5
    final_df["y"] *= 2.584 / 6.5
    zero_vel_df["x"] *= 2.584 / 6.5
    zero_vel_df["y"] *= 2.584 / 6.5

    # Print max and min values
    print(f"max u: {df['u'].max()}")
    print(f"min u: {df['u'].min()}")
    print(f"max v: {df['v'].max()}")
    print(f"min v: {df['v'].min()}")
    print(f"max w: {df['w'].max()}")
    print(f"min w: {df['w'].min()}")
    print(f'vort_x min: {df["vort_x"].min():.1f}, min {df["vort_x"].max():.1f}')
    print(f'vort_y min: {df["vort_y"].min():.1f}, min {df["vort_y"].max():.1f}')
    print(f'vort_z min: {df["vort_z"].min():.1f}, min {df["vort_z"].max():.1f}')

    return final_df, zero_vel_df


def compute_lambda2(df: pd.DataFrame) -> pd.DataFrame:
    # Define the components of the strain-rate tensor (S) and the rotation-rate tensor (Omega)
    dudx = df["dudx"]
    dvdy = df["dvdy"]
    dwdz = df["dwdz"]
    dudy = df["dudy"]
    dvdx = df["dvdx"]
    dudz = df["dudz"]
    dvdz = df["dvdz"]
    dwdy = df["dwdy"]
    dwdx = df["dwdx"]

    # Strain-rate tensor components
    Sxx = dudx
    Syy = dvdy
    Szz = dwdz
    Sxy = 0.5 * (dudy + dvdx)
    Sxz = 0.5 * (dudz + dwdx)
    Syz = 0.5 * (dvdz + dwdy)

    # Rotation-rate tensor components
    Oxy = 0.5 * (dudy - dvdx)
    Oxz = 0.5 * (dudz - dwdx)
    Oyz = 0.5 * (dvdz - dwdy)

    # Compute the combined tensor S^2 + Omega^2 for each point in the domain
    Axx = Sxx**2 + Oxy**2 + Oxz**2
    Ayy = Syy**2 + Oxy**2 + Oyz**2
    Azz = Szz**2 + Oxz**2 + Oyz**2
    Axy = Sxy**2 + Oxy * Sxy
    Axz = Sxz**2 + Oxz * Sxz
    Ayz = Syz**2 + Oyz * Syz

    # Construct the tensor for each point and calculate eigenvalues
    Lambda2 = []

    for i in range(len(Sxx)):
        # Construct the 3x3 tensor A
        A = np.array(
            [
                [Axx[i], Axy[i], Axz[i]],
                [Axy[i], Ayy[i], Ayz[i]],
                [Axz[i], Ayz[i], Azz[i]],
            ]
        )

        # Compute eigenvalues and sort them in ascending order
        eigenvalues = np.linalg.eigvalsh(A)
        eigenvalues.sort()

        # Lambda-2 is the second-largest eigenvalue
        Lambda2.append(eigenvalues[1])

    # Store Lambda-2 in the dataframe
    print(
        f"lambda2 mean,min,max: {np.mean(Lambda2):.2f}, {np.min(Lambda2):.2f}, {np.max(Lambda2):.2f}"
    )
    df["lambda2"] = Lambda2
    return df


def plot_contour_with_colored_data_two_rows_three_cols(plot_params):

    # x_cm values to plot
    n_rows = 5
    # x_cm_values = [10, 25, 40, 50, 60, 75]
    # x_cm_values = [5, 50, 100]
    # x_cm_values = [10, 20, 30, 40, 50]
    x_cm_values = [10, 15, 20, 25, 30]
    # x_cm_values = [10, 20]
    # x_cm_values = [10, 25]
    n_cols = len(x_cm_values)

    # Create a figure with two rows, three columns
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 5 * (n_rows - 1)),
        gridspec_kw={
            "hspace": 0.07,  # Vertical space between rows
            "wspace": -0.11,  # Horizontal space between columns
        },
    )

    ax_row_list = []
    for i in range(n_rows):
        ax_row_list.append(axes[i])

    # Data types to plot (first row vertical velocity, second row pressure)
    # Transform raw data to processed data
    alpha = 6
    df_list, zero_vel_df_list = [], []
    for x_cm in x_cm_values:
        df, zero_vel_df = transform_raw_csv_to_processed_df(alpha, x_cm)
        # df['curl'] = df['dudy'] / df['dudx']
        df_list.append(df)
        zero_vel_df_list.append(zero_vel_df)

    for row in range(n_rows):
        for col in range(n_cols):

            df = df_list[col]
            zero_vel_df = zero_vel_df_list[col]

            # Current subplot and plot parameters
            ax = axes[row, col]
            curr_plot_params = plot_params.copy()

            # Set x_cm value
            x_cm = x_cm_values[col]

            # if row == 0:
            #     color_name = "vort_x"
            #     curr_plot_params["color_data_col_name"] = color_name
            #     curr_plot_params["min_cbar_value"] = -100
            #     curr_plot_params["max_cbar_value"] = 100
            # if row == 0:
            #     color_name = "vort_y"
            #     curr_plot_params["color_data_col_name"] = color_name
            #     curr_plot_params["min_cbar_value"] = -150
            #     curr_plot_params["max_cbar_value"] = 150

            # elif row == 2:
            #     color_name = "vort_z"
            #     curr_plot_params["color_data_col_name"] = color_name
            #     curr_plot_params["min_cbar_value"] = 0
            #     curr_plot_params["max_cbar_value"] = 300
            ### if row == 0:
            ###     color_name = "V"
            ###     curr_plot_params["color_data_col_name"] = color_name
            ###     curr_plot_params["min_cbar_value"] = 0
            ###     curr_plot_params["max_cbar_value"] = 5
            if row == 0:
                color_name = "u"
                curr_plot_params["color_data_col_name"] = color_name
                curr_plot_params["min_cbar_value"] = -4
                curr_plot_params["max_cbar_value"] = 4
            elif row == 1:
                color_name = "v"
                curr_plot_params["color_data_col_name"] = color_name
                curr_plot_params["min_cbar_value"] = -2
                curr_plot_params["max_cbar_value"] = 2
            elif row == 2:
                color_name = "w"
                curr_plot_params["color_data_col_name"] = color_name
                curr_plot_params["min_cbar_value"] = -2
                curr_plot_params["max_cbar_value"] = 2
            elif row == 3:
                dudx = df["dudx"]
                dvdy = df["dvdy"]
                dwdz = df["dwdz"]
                dudy = df["dudy"]
                dvdx = df["dvdx"]
                dudz = df["dudz"]
                dvdz = df["dvdz"]
                dwdy = df["dwdy"]
                dwdx = df["dwdx"]
                Sxx = dudx
                Syy = dvdy
                Szz = dwdz
                Sxy = 0.5 * (dudy + dvdx)
                Sxz = 0.5 * (dudz + dwdx)
                Syz = 0.5 * (dvdz + dwdy)

                # ||S||^2: Frobenius norm of the rate-of-strain tensor
                S_squared = Sxx**2 + Syy**2 + Szz**2 + 2 * (Sxy**2 + Sxz**2 + Syz**2)

                # Compute the rate-of-rotation tensor Omega_ij
                Oxy = 0.5 * (dudy - dvdx)
                Oxz = 0.5 * (dudz - dwdx)
                Oyz = 0.5 * (dvdz - dwdy)

                # ||Omega||^2: Frobenius norm of the rate-of-rotation tensor
                Omega_squared = 2 * (Oxy**2 + Oxz**2 + Oyz**2)

                # Compute Q-criterion
                Q = 0.5 * (Omega_squared - S_squared)
                df["Q"] = Q
                # print(f"Q_mean: {Q.mean()}")
                # print(f"Q_max: {Q.max()}")
                # print(f"Q_min: {Q.min()}")
                color_name = "Q"
                curr_plot_params["color_data_col_name"] = color_name
                curr_plot_params["min_cbar_value"] = -2000  # -2000
                curr_plot_params["max_cbar_value"] = 2000  # 2000
            elif row == 4:
                df = compute_lambda2(df)
                color_name = "lambda2"
                curr_plot_params["color_data_col_name"] = color_name
                curr_plot_params["min_cbar_value"] = -500  # -2000
                curr_plot_params["max_cbar_value"] = 20000  # 2000

            # Extract unique x, y, and color values
            x_unique = df["x"].values
            y_unique = df["y"].values
            color_values = df[color_name].values

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

            # Interpolate the values onto the grid using griddata
            color_data = griddata(
                (x_unique, y_unique),
                color_values,
                (X_grid, Y_grid),
                method="linear",
            )
            u_data = griddata(
                (x_unique, y_unique),
                df["u"].values,
                (X_grid, Y_grid),
                method="linear",
            )
            v_data = griddata(
                (x_unique, y_unique),
                df["v"].values,
                (X_grid, Y_grid),
                method="linear",
            )
            w_data = griddata(
                (x_unique, y_unique),
                df["w"].values,
                (X_grid, Y_grid),
                method="linear",
            )

            # if row == 0:
            #     step = int(10 * (25 / curr_plot_params["subsample_color"]))
            #     ax.quiver(
            #         X_grid[::step, ::step],
            #         Y_grid[::step, ::step],
            #         w_data[::step, ::step],
            #         v_data[::step, ::step],
            #         color="black",
            #         scale=50,
            #         alpha=0.4,
            #     )

            # if "Q" in color_name or "vort" in color_name or "lambda" in color_name:
            curr_plot_params["cax"] = ax.pcolormesh(
                X_grid,
                Y_grid,
                color_data,
                cmap=curr_plot_params["cmap"],
                shading="auto",
                vmin=curr_plot_params["min_cbar_value"],
                vmax=curr_plot_params["max_cbar_value"],
            )
            # else:
            #     #### Plot the contour
            #     curr_plot_params["cax"] = ax.contourf(
            #         X_grid,
            #         Y_grid,
            #         color_data,
            #         levels=curr_plot_params["countour_levels"],
            #         cmap=curr_plot_params["cmap"],
            #         vmin=curr_plot_params["min_cbar_value"],
            #         vmax=curr_plot_params["max_cbar_value"],
            #     )
            # curr_plot_params["cax"] = ax.streamplot(
            #     X_grid,
            #     Y_grid,
            #     w_data,
            #     v_data,
            #     color=color_data,
            #     cmap=curr_plot_params["cmap"],
            #     linewidth=0.5,
            #     density=1,
            # )

            ## masking zero based on limits
            # x_min = curr_plot_params["xlim"][0] * 0.9
            # x_max = curr_plot_params["xlim"][1] * 1.1
            # y_min = curr_plot_params["ylim"][0] * 0.9
            # y_max = curr_plot_params["ylim"][1] * 1.1
            # mask_x = (zero_vel_df["x"] >= x_min) & (zero_vel_df["x"] <= x_max)
            # mask_y = (zero_vel_df["y"] >= y_min) & (zero_vel_df["y"] <= y_max)
            # mask = mask_x & mask_y
            # zero_vel_df = zero_vel_df[mask]
            ## plotting zero
            # ax.scatter(zero_vel_df["x"], zero_vel_df["y"], c="black", s=0.3)
            ax = extract_spanwise_contour.main(
                ax,
                csv_path=Path(project_dir)
                / "processed_data"
                / "CFD"
                / "spanwise_slices"
                / f"alpha_{alpha}_CFD_{int(x_cm)}cm_outline_wing.csv",
            )

            # Adjust plot settings
            ax.set_aspect("equal")
            ax.set_xlim(curr_plot_params["xlim"])
            ax.set_ylim(curr_plot_params["ylim"])
            ax.grid(False)

            # Tick and label management
            if col == n_cols - 1:
                ax.set_ylabel("z [m]")
                ax.yaxis.set_label_position("right")
                ax.tick_params(labelleft=False, labelright=True, labelbottom=False)
            else:
                ax.tick_params(labelleft=False, labelbottom=False)

            if row == (n_rows - 1):
                ax.set_xlabel("y [m]")
                ax.tick_params(labelbottom=True)
            if row == 0:
                ax.set_title(f"x/c = {x_cm/100:.2f}")
            elif row == n_rows - 1:
                ax.tick_params(labelbottom=True)

            # Add colorbar for each row
            if col == 0:
                if color_name == "u":
                    label = r"$u$ [m/s]"
                elif color_name == "v":
                    label = r"$v$ [m/s]"
                elif color_name == "w":
                    label = r"$w$ [m/s]"
                elif color_name == "Q":
                    label = r"$Q$ [1/s]"
                elif color_name == "lambda2":
                    label = r"$\lambda_{\mathrm{2}}$ [1/s]"
                add_vertical_colorbar_for_row(
                    fig,
                    ax_row_list[row],
                    curr_plot_params,
                    label=label,
                    labelpad=25,
                    x_offset=0.03,
                )

    # Save the plot
    save_path = (
        Path(project_dir)
        / "results"
        / "paper_plots"
        / f"spanwise_CFD_comparison_v_and_p_x_{x_cm_values[0]}_to_{x_cm_values[-1]}.png"
    )
    fig.savefig(save_path, dpi=500)
    plt.close()


def main():
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
        # "xlim": (0.0, 0.8),  # (-0.05, 0.75),  #
        # "ylim": (-0.6, 0.1),  # (-0.65, 0.35),  #
        # "xlim": (0.0, 0.5),  # (-0.05, 0.75),  #
        # "ylim": (-0.2, 0.05),  # (-0.65, 0.35),  #
        "xlim": (0.22, 0.38),
        "ylim": (-0.1, 0.02),
        # Color and contour settings
        "min_cbar_value": -1,
        "max_cbar_value": 1,
        "is_with_cbar": True,
        "cbar_value_factor_of_std": 2.0,
        "subsample_color": 25,  # 25,
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
    # plot_contour_with_colored_data(plot_params, mask_bound=30)
    # plot_contour_with_colored_data_two_rows(plot_params, x_cm=25)
    plot_contour_with_colored_data_two_rows_three_cols(plot_params)


if __name__ == "__main__":
    main()
