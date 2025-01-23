import matplotlib.pyplot as plt
import pandas as pd
import extract_spanwise_contour
from plot_styling import set_plot_style, plot_on_ax
import numpy as np
from utils import project_dir
from pathlib import Path


def plot_spanwise_contour(curr_plot_params):
    """
    Plots a spanwise contour based on the given parameters and calls the `main` function to draw the contour.

    Parameters:
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on.
        curr_plot_params (dict): Dictionary containing plot parameters and CSV path info.

    Returns:
        ax: The axis with the plotted contour.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(4, 4.5))
    # Extract parameters
    alpha = curr_plot_params.get("alpha", 0)
    xlim = curr_plot_params.get("xlim", None)
    ylim = curr_plot_params.get("ylim", None)
    cmap = curr_plot_params.get("cmap", "viridis")

    csv_path = (
        Path(project_dir)
        / "processed_data"
        / "CFD"
        / "spanwise_slices"
        / f"alpha_{curr_plot_params['alpha']}_CFD_25cm_outline_wing.csv"
    )
    ax = extract_spanwise_contour.main(ax, csv_path)

    # Apply plot parameters
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.set_aspect("equal", adjustable="box")
    # ax.set_title(f"Spanwise Contour at \u03B1 = {alpha}", fontsize=10)
    ax.set_xlabel("$y$ [m]")
    ax.set_ylabel("$z$ [m]")

    # Define custom green color (e.g., RGB values or hex code)
    custom_green = "#00FF00"  # Example hex color for a green shade

    # Define the y-positions, heights, and labels
    y_positions = [0, 0.203, 0.287, 0.301, 0.399, 0.562, 0.632]
    line_height = 0.59  # Total height of each vertical line
    line_centers = [
        0,
        -0.01,
        -0.05,
        -0.05,
        -0.1,
        -0.2,
        -0.3,
    ]  # Center of the lines in the y-axis
    labels = [f"{i+1}" for i in range(len(y_positions))]

    # Plot vertical lines with specified height, center, and color
    for y, label, line_center in zip(y_positions, labels, line_centers):

        x_coord = [y, y]
        y_coord = [line_center - line_height / 2, line_center + line_height / 2]

        plot_on_ax(
            ax,
            x_coord,
            y_coord,
            linewidth=2,
            color=custom_green,
            x_label="$y$ [m]",
            y_label="$z$ [m]",
            is_with_grid=False,
        )
        # Add labels at the top of the lines
        if y == 0.287:
            delta_x = -0.01
        elif y == 0.301:
            delta_x = 0.005
        else:
            delta_x = 0
        ax.text(
            y + delta_x,  # x-coordinate of the label
            line_center
            + line_height / 2
            + 0.02,  # y-coordinate slightly above the line
            label,  # Label text
            color="black",  # Label color
            fontsize=12,  # Font size
            ha="center",  # Horizontal alignment
        )

    plt.tight_layout()

    # Save the plot if save_dir exists in plot_params
    save_dir = curr_plot_params.get("save_dir", None)
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"plane_location.pdf"
        plt.savefig(save_path)  # , dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return ax


# Example usage
if __name__ == "__main__":
    plot_params = {
        "alpha": 6,
        "xlim": (-0.1, 0.75),
        "ylim": (-0.65, 0.4),
        "save_dir": Path(project_dir, "results", "paper_plots"),
    }
    plot_spanwise_contour(plot_params)
    # plt.show()
