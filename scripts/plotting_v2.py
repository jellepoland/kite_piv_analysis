import os
import numpy as np
import xarray as xr
import logging
import re
import sys
from copy import deepcopy
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.ticker as mticker
from scipy.interpolate import griddata


def load_processed_data(input_file: str) -> xr.Dataset:
    return xr.open_dataset(input_file)


def plot_quiver(
    x,
    y,
    u,
    v,
    color_values,
    colorbar_label,
    title,
    u_inf,
    scale=None,
    color_size=5,
    color_alpha=0.9,
    width_arrow=0.02,
    scale_arrow=9,
    cmap="RdBu",
    save_path=None,
    subsample=5,  # New parameter to control subsampling
    is_with_quiver=True,
    is_show_plot=True,
    cbar_label_spacing=20,
    cbar_fontsize=16,
    background_alpha=0.9,
    max_cbar_value=1.2,
    min_cbar_value=0.8,
):
    """
    Create a quiver plot of vector field with color based on velocity magnitude.

    Parameters:
    x, y : 2D arrays
        The x and y coordinates of the arrow locations
    u, v : 2D arrays
        The x and y components of the arrows
    title : str
        The title of the plot
    scale : float, optional
        The scaling factor for arrow size. If None, autoscaling is used.
    width : float, optional
        Width of the arrows
    cmap : str, optional
        Colormap to use for coloring the arrows
    colorbar_label : str, optional
        Label for the colorbar
    save_path : str, optional
        If provided, saves the figure to this path
    subsample : int, optional
        Factor by which to subsample the quiver plot
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Mask zero values in color_values
    masked_color_values = np.ma.masked_where(color_values == 0.0, color_values)

    # # filter color_values_sub, to only take values within a specific range
    # if is_with_cbar_masking:
    #     masked_color_values = np.ma.masked_where(
    #         masked_color_values > max_cbar_value, masked_color_values
    #     )
    #     masked_color_values = np.ma.masked_where(
    #         masked_color_values < min_cbar_value, masked_color_values
    #     )
    ########
    # old color approach
    #####
    # # Create a custom colormap with black for masked values
    # base_cmap = get_cmap(cmap)
    # cmap_colors = base_cmap(np.arange(base_cmap.N))
    # cmap_colors = np.vstack(
    #     (np.array([0, 0, 0, 1]), cmap_colors)
    # )  # Add black as the first color
    # custom_cmap = ListedColormap(cmap_colors)
    # ## using quiver
    # # Create quiver plot
    # quiv = ax.quiver(
    #     x_sub_colors,
    #     y_sub_colors,
    #     u_sub_colors,
    #     v_sub_colors,
    #     color_values_sub,
    #     width=1.5 * width * subsample_colors,
    #     cmap=custom_cmap,
    #     alpha=background_alpha,
    # )
    # # cbar
    # cbar = plt.colorbar(quiv)
    # cbar.set_label(
    #     colorbar_label,
    #     rotation=0,
    #     labelpad=cbar_label_spacing,
    #     fontsize=cbar_fontsize,
    # )
    #########

    #### new approach
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    ### cbar
    cax = plt.scatter(
        x,
        y,
        c=masked_color_values,
        cmap=cmap,
        vmin=min_cbar_value,
        vmax=max_cbar_value,
        s=color_size,
        alpha=color_alpha,
    )
    # Smooth background using contourf
    # cax = ax.contourf(
    #     x,
    #     y,
    #     masked_color_values,
    #     levels=1000,
    #     cmap=cmap,
    #     vmin=min_cbar_value,
    #     vmax=max_cbar_value,
    # )
    # # Smooth background using pcolormesh
    # cax = ax.pcolormesh(
    #     x,
    #     y,
    #     masked_color_values,
    #     cmap=cmap,
    #     shading="auto",
    #     vmin=min_cbar_value,
    #     vmax=max_cbar_value,
    # )
    mid_cbar_value = np.mean([min_cbar_value, max_cbar_value])
    cbar = fig.colorbar(
        cax,
        ticks=[
            min_cbar_value,
            mid_cbar_value,
            max_cbar_value,
        ],
        format=mticker.FixedFormatter(
            [f"< {min_cbar_value}", f"{mid_cbar_value}", f"> {max_cbar_value}"]
        ),
        extend="both",
    )
    labels = cbar.ax.get_yticklabels()
    labels[0].set_verticalalignment("top")
    labels[-1].set_verticalalignment("bottom")
    cbar.set_label(
        colorbar_label,
        rotation=0,
        labelpad=cbar_label_spacing,
        fontsize=cbar_fontsize,
    )

    # ## using pcolormesh
    # # Plot the background color based on color_values using pcolormesh
    # c = ax.pcolormesh(x, y, masked_color_values, cmap=custom_cmap)  # , shading="auto")
    # cbar = fig.colorbar(c, ax=ax)
    # cbar.set_label(colorbar_label)

    # ## using imshow
    # # Plot the background color using imshow for a smoother appearance
    # im = ax.imshow(
    #     masked_color_values,
    #     cmap=custom_cmap,
    #     extent=[y.min(), y.max(), x.min(), x.max()],
    #     origin="lower",
    #     aspect="auto",
    # )
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.set_label(colorbar_label)

    # ## using contourf
    # # Plot the background color using contourf for a smoother appearance
    # c = ax.contourf(
    #     x,
    #     y,
    #     masked_color_values,
    #     levels=20,
    #     cmap=custom_cmap,
    #     alpha=1.0,
    #     antialiased=True,
    # )
    # cbar = fig.colorbar(c, ax=ax)
    # cbar.set_label(colorbar_label)

    if is_with_quiver:
        ### QUIVER PLOT ###
        # Create a custom colormap with black for masked values
        base_cmap = get_cmap(cmap)
        cmap_colors = base_cmap(np.arange(base_cmap.N))
        cmap_colors = np.vstack(
            (np.array([0, 0, 0, 0]), cmap_colors)
        )  # Add transparent as the first color
        custom_cmap = ListedColormap(cmap_colors)

        # Normalize color_values to [0, 1] range
        norm = Normalize(vmin=np.min(color_values), vmax=np.max(color_values))
        sm = ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])

        # Calculate the magnitude of velocity
        vel_mag = np.sqrt(u**2 + v**2)

        # Subsample the data
        x_sub = x[::subsample, ::subsample]
        y_sub = y[::subsample, ::subsample]
        u_sub = u[::subsample, ::subsample]
        v_sub = v[::subsample, ::subsample]
        vel_mag_sub = vel_mag[::subsample, ::subsample]
        color_values_sub = color_values[::subsample, ::subsample]

        # Plot the quiver arrows on top in black, with transparency for zero values
        for i in range(x_sub.shape[0]):
            for j in range(y_sub.shape[1]):
                scale_param = subsample * (vel_mag_sub[i, j] / u_inf)
                if scale_param != 0:
                    # if color_values_sub[i, j] != 0:
                    ax.quiver(
                        x_sub[i, j],
                        y_sub[i, j],
                        u_sub[i, j],
                        v_sub[i, j],
                        color="k",
                        scale=scale_arrow * (1 / scale_param),
                        scale_units="xy",
                        angles="xy",
                        width=width_arrow * (1 / scale_param),
                    )

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Add timestamp to the plot
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.text(
        0.95,
        0.01,
        f"Created: {timestamp}",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.5,
    )

    # Make layout tight
    plt.tight_layout()

    # Save if path is provided
    if save_path:
        # create directory if it does not exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    # Show plot
    if is_show_plot:
        plt.show()


def saving_a_plot(
    x_meshgrid_global,
    y_meshgrid_global,
    u_for_quiver,
    v_for_quiver,
    color_data,
    save_plots_folder,
    plot_type,
    min_cbar_value,
    max_cbar_value,
    max_mask_value,
    min_mask_value,
    title,
    countour_levels=40,
    is_with_quiver=True,
    subsample=12,
    u_inf=15.0,
    width_arrow=0.02,
    scale_arrow=9,
    cmap="RdBu",
):

    # Mask the data to the color ranges
    color_data = np.ma.masked_where(color_data > max_mask_value, color_data)
    color_data = np.ma.masked_where(color_data < min_mask_value, color_data)
    # color_data = np.ma.filled(color_data, np.nan)
    # ux_mean_uinf = np.clip(ux_mean_uinf, min_cbar_value, max_cbar_value)
    fig, ax = plt.subplots()

    subsample_color = 1
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
    # cax = plt.contourf(
    #     x_meshgrid_global,
    #     y_meshgrid_global,
    #     color_data,
    #     cmap=cmap,
    #     levels=countour_levels,
    #     extend="both",
    #     vmin=min_cbar_value,
    #     vmax=max_cbar_value,
    # )

    mid_cbar_value = np.mean([min_cbar_value, max_cbar_value])
    cbar = fig.colorbar(
        cax,
        ticks=[
            min_cbar_value,
            mid_cbar_value,
            max_cbar_value,
        ],
        format=mticker.FixedFormatter(
            [f"< {min_cbar_value}", f"{mid_cbar_value}", f"> {max_cbar_value}"]
        ),
        extend="both",
    )
    labels = cbar.ax.get_yticklabels()
    labels[0].set_verticalalignment("top")
    labels[-1].set_verticalalignment("bottom")
    cbar.set_label("Ux/Uinf", rotation=0)
    plt.title(title)

    if is_with_quiver:

        x = x_meshgrid_global  # .flatten()
        y = y_meshgrid_global  # .flatten()
        u = u_for_quiver
        v = v_for_quiver
        color_data = color_data

        ### QUIVER PLOT ###
        # Create a custom colormap with black for masked values
        base_cmap = get_cmap(cmap)
        cmap_colors = base_cmap(np.arange(base_cmap.N))
        cmap_colors = np.vstack(
            (np.array([0, 0, 0, 0]), cmap_colors)
        )  # Add transparent as the first color
        custom_cmap = ListedColormap(cmap_colors)

        # Normalize color_values to [0, 1] range
        norm = Normalize(vmin=np.min(color_data), vmax=np.max(color_data))
        sm = ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])

        # Calculate the magnitude of velocity
        vel_mag = np.sqrt(u**2 + v**2)

        # Subsample the data
        x_sub = x[::subsample, ::subsample]
        y_sub = y[::subsample, ::subsample]
        u_sub = u[::subsample, ::subsample]
        v_sub = v[::subsample, ::subsample]
        vel_mag_sub = vel_mag[::subsample, ::subsample]
        # color_values_sub = color_data[::subsample, ::subsample]

        ax.quiver(
            x_sub,
            y_sub,
            u_sub,
            v_sub,
            color="k",
            angles="xy",
            scale=0.9,
            scale_units="xy",
            width=0.0025,
        )

    plt.savefig(save_plots_folder + title + plot_type)


if __name__ == "__main__":
    # Go back to root folder
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root_path)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load the processed data
    y_num = "3"
    file_name = f"Y{y_num}"
    processed_data_path = sys.path[0] + f"/processed_data/{file_name}.csv"
    loaded_data = pd.read_csv(processed_data_path)
    logging.info(f"loaded_data:{loaded_data}")
    x_global = np.arange(-210, 840, 2.4810164835164836)
    y_global = np.arange(-205, 405, 2.4810164835164836)
    x_meshgrid_global, y_meshgrid_global = np.meshgrid(x_global, y_global)
    # x_global = loaded_data["x"].values
    # y_global = loaded_data["y"].values
    # vel_u = loaded_data["vel_u"].values
    # x_meshgrid_global, y_meshgrid_global = np.meshgrid(
    #     loaded_data["x"].values, loaded_data["y"].values
    # )
    vel_u_on_meshgrid_global = griddata(
        np.array([x_meshgrid_global.flatten(), y_meshgrid_global.flatten()]).T,
        np.array(loaded_data["vel_u"].values),
        (x_meshgrid_global, y_meshgrid_global),
        method="linear",
    )
    vel_v_on_meshgrid_global = griddata(
        np.array([x_meshgrid_global.flatten(), y_meshgrid_global.flatten()]).T,
        np.array(loaded_data["vel_v"].values),
        (x_meshgrid_global, y_meshgrid_global),
        method="linear",
    )

    saving_a_plot(
        x_meshgrid_global=x_meshgrid_global,
        y_meshgrid_global=y_meshgrid_global,
        u_for_quiver=vel_u_on_meshgrid_global,
        v_for_quiver=vel_v_on_meshgrid_global,
        color_data=vel_u_on_meshgrid_global / 15.0,
        save_plots_folder=sys.path[0] + "/results/aoa_13/",
        plot_type=".pdf",
        min_cbar_value=0.75,
        max_cbar_value=1.25,
        max_mask_value=1.25,
        min_mask_value=0.75,
        title=file_name,
    )

    # plot_quiver(
    #     loaded_data["x"].values,
    #     loaded_data["y"].values,
    #     loaded_data["vel_u"].values,
    #     loaded_data["vel_v"].values,
    #     color_values=loaded_data["vel_u"].values / 15.0,
    #     u_inf=15.0,
    #     colorbar_label=r"$Ux/Uing$",
    #     title=file_name,
    #     save_path=sys.path[0] + f"/results/aoa_13/all_plabes/{file_name}.png",
    #     subsample=10,  # Adjust subsample factor as needed
    #     is_show_plot=True,
    # )

    # loaded_dataset = pd.opencsv():
    # size = loaded_dataset.sizes.get("file")
    # # logging.info(f"Data attrs: {loaded_dataset.attrs}")
    # datapoint_list = [loaded_dataset.isel(file=i) for i in range(size)]

    # for datapoint in datapoint_list[0:1]:
    #     case_name_davis = datapoint.case_name_davis.values
    #     # logging.info(f"datapoint.data_vars: {datapoint.data_vars}")
    #     # logging.info(f"data, {datapoint.variables_edited}")
    #     # logging.info(f"case_name: {case_name_davis}")
    #     # logging.info(f"FileName: {datapoint.file_name_labbook.values}")
    #     # logging.info(f" ----")
    #     # logging.info(datapoint.data.sel(variable="x").values)
    #     # logging.info(datapoint.data_vars)
    #     # logging.info(f" ----")
    #     logging.info(f"file_name: {datapoint['file_name'].values}")
    #     # j_slice = slice(0, 10)

    #     data_matrix = datapoint.data.values
    #     # ## setting the ymin to 0
    #     # y_min = -datapoint.data.sel(variable="y").min().values
    #     # logging.info(f"y_min: {y_min}")
    #     # data_matrix[:, :, 1] = data_matrix[:, :, 1] + y_min
    #     logging.info(
    #         f"Data matrix ymin: {np.nanmin(data_matrix[:, :, 1])}, ymax: {np.nanmax(data_matrix[:, :, 1])}"
    #     )

    #     plot_quiver(
    #         datapoint.data.sel(variable="x").values,
    #         datapoint.data.sel(variable="y").values,
    #         datapoint.data.sel(variable="vel_u").values,
    #         datapoint.data.sel(variable="vel_v").values,
    #         color_values=datapoint.data.sel(variable="ux_uinf").values,
    #         u_inf=datapoint["vw"].values,
    #         colorbar_label=r"$\frac{U_x}{U_\infty}$",
    #         title=case_name_davis,
    #         save_path=sys.path[0]
    #         + f"/results/aoa_13/seperate_planes/{case_name_davis}.png",
    #         subsample=10,  # Adjust subsample factor as needed
    #         is_show_plot=False,
    #     )
