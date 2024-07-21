import os
import numpy as np
import xarray as xr
import logging
import re
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable, get_cmap


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
    u_inf=15,
    scale=None,
    width=0.005,
    width_arrow=0.02,
    scale_arrow=9,
    cmap="RdBu",
    save_path=None,
    subsample=5,  # New parameter to control subsampling
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
    masked_color_values = np.ma.masked_where(color_values == 0, color_values)

    # new_color_values = np.ones(color_values.shape)
    # for i in range(color_values.shape[0]):
    #     for j in range(color_values.shape[1]):
    #         if color_values[i, j] != 0:
    #             new_color_values[i, j] = color_values[i, j]

    # masked_color_values = new_color_values

    # Create a custom colormap with black for masked values
    base_cmap = get_cmap(cmap)
    cmap_colors = base_cmap(np.arange(base_cmap.N))
    cmap_colors = np.vstack(
        (np.array([0, 0, 0, 1]), cmap_colors)
    )  # Add black as the first color
    custom_cmap = ListedColormap(cmap_colors)

    ## subsample the data for colors
    subsample_colors = 1
    x_sub_colors = x[::subsample_colors, ::subsample_colors]
    y_sub_colors = y[::subsample_colors, ::subsample_colors]
    u_sub_colors = u[::subsample_colors, ::subsample_colors]
    v_sub_colors = v[::subsample_colors, ::subsample_colors]
    color_values_sub = masked_color_values[::subsample_colors, ::subsample_colors]

    ## using quiver
    # Create quiver plot
    quiv = ax.quiver(
        x_sub_colors,
        y_sub_colors,
        u_sub_colors,
        v_sub_colors,
        color_values_sub,
        width=1.5 * width * subsample_colors,
        cmap=custom_cmap,
        alpha=0.9,
    )

    # Add colorbar
    cbar = plt.colorbar(quiv)
    cbar.set_label(colorbar_label)

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
    plt.show()


if __name__ == "__main__":
    # Go back to root folder
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root_path)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load the processed data
    processed_data_path = sys.path[0] + "/processed_data/combined_piv_data.nc"
    loaded_dataset = xr.open_dataset(processed_data_path)
    size = loaded_dataset.sizes.get("file")
    # logging.info(f"Data attrs: {loaded_dataset.attrs}")
    datapoint_list = [loaded_dataset.isel(file=i) for i in range(size)]

    for datapoint in datapoint_list[0:1]:
        case_name_davis = datapoint.case_name_davis.values
        # logging.info(f"datapoint.data_vars: {datapoint.data_vars}")
        # logging.info(f"data, {datapoint.variables_edited}")
        # logging.info(f"case_name: {case_name_davis}")
        # logging.info(f"FileName: {datapoint.file_name_labbook.values}")
        # logging.info(f" ----")
        # logging.info(datapoint.data.sel(variable="x").values)
        logging.info(datapoint.data_vars)
        # logging.info(f" ----")
        logging.info(f"file_name: {datapoint['file_name'].values}")

        plot_quiver(
            datapoint.data.sel(variable="x").values,
            datapoint.data.sel(variable="y").values,
            datapoint.data.sel(variable="vel_u").values,
            datapoint.data.sel(variable="vel_v").values,
            color_values=datapoint.data.sel(variable="ux_uinf").values,
            colorbar_label=r"$\frac{U_x}{U_\infty}$",
            title="Vector Field Example",
            save_path=sys.path[0] + f"/results/aoa_13/{case_name_davis}.png",
            subsample=10,  # Adjust subsample factor as needed
        )
