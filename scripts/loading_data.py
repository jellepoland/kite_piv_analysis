import os
import numpy as np
import xarray as xr
import logging
import re
import sys
import matplotlib.pyplot as plt


def load_processed_data(input_file: str) -> xr.Dataset:
    return xr.open_dataset(input_file)


def plot_quiver(
    x,
    y,
    u,
    v,
    title,
    scale=None,
    width=0.005,
    cmap="viridis",
    colorbar_label="Velocity Magnitude",
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

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate velocity magnitude for coloring
    velocity_mag = np.sqrt(u**2 + v**2)

    # Create quiver plot
    quiv = ax.quiver(
        x,
        y,
        u,
        v,
        velocity_mag,
        scale=scale,
        scale_units="xy",
        angles="xy",
        width=width,
        cmap=cmap,
    )

    # Add colorbar
    cbar = fig.colorbar(quiv)
    cbar.set_label(colorbar_label)

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Add timestamp to the plot
    from datetime import datetime

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
        case_name_davis = datapoint.case_name_davis.values
        logging.info(f"datapoint.data_vars: {datapoint.data_vars}")
        logging.info(f"case_name: {case_name_davis}")
        logging.info(f"FileName: {datapoint.file_name_labbook.values}")

        # Example usage:
        x, y = datapoint.x.values, datapoint.y.values
        u = datapoint.vel_u.values
        v = datapoint.vel_v.values
        plot_quiver(
            x,
            y,
            u,
            v,
            "Vector Field Example",
            save_path=f"results/aoa_13/{case_name_davis}.png",
        )
