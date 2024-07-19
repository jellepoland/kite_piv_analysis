import os
import numpy as np
import xarray as xr
import logging
import re
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap


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
    scale=None,
    width=0.005,
    cmap="RdBu",
    save_path=None,
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
    """

    # Mask zero values in color_values
    masked_color_values = np.ma.masked_where(color_values == 0, color_values)

    # Create a custom colormap with black for masked values
    base_cmap = get_cmap(cmap)
    cmap_colors = base_cmap(np.arange(base_cmap.N))
    cmap_colors = np.vstack(
        (np.array([0, 0, 0, 1]), cmap_colors)
    )  # Add black as the first color
    custom_cmap = ListedColormap(cmap_colors)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create quiver plot
    quiv = ax.quiver(
        x,
        y,
        u,
        v,
        masked_color_values,
        scale=scale,
        scale_units="xy",
        angles="xy",
        width=width,
        cmap=custom_cmap,
    )

    # Add colorbar
    cbar = fig.colorbar(quiv)
    cbar.set_label(colorbar_label)

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
    logging.info(f"Data attrs: {loaded_dataset.attrs}")
    datapoint_list = [loaded_dataset.isel(file=i) for i in range(size)]

    for datapoint in datapoint_list[0:1]:
        case_name_davis = datapoint.case_name_davis
        # logging.info(f"datapoint.data_vars: {datapoint.data_vars}")
        logging.info(f"case_name: {case_name_davis}")
        logging.info(f"FileName: {datapoint.file_name_labbook.values}")

        # TODO:
        # color should be Ux / Uinf
        # quiver should be based on umag 2D array
        # cmap red to blue with white in middle around 1

        # Example usage:
        x, y = datapoint.x.values, datapoint.y.values
        u = datapoint.vel_u.values
        v = datapoint.vel_v.values
        plot_quiver(
            x,
            y,
            u,
            v,
            color_values=datapoint.Ux_Uinf.values,
            colorbar_label=r"$\frac{U_x}{U_\infty}$",
            title="Vector Field Example",
            save_path=f"results/aoa_13/{case_name_davis}.png",
        )
