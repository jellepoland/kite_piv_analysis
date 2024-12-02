import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from plot_styling import set_plot_style, plot_on_ax
from utils import project_dir

if __name__ == "__main__":
    set_plot_style()

    # Loading
    df_y_locations = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / "circulation_plot"
        / "y_locations_with_gamma.csv",
        index_col=False,
    )

    df_VSM = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / "circulation_plot"
        / "VSM_circulation_distribution.csv",
        index_col=False,
    )

    # plotting the gamma
    # Create the figure and axis
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot the Gamma values for CFD (blue) and PIV (red)
    # Extract y-numbers (Y1, Y2, ...) and the corresponding Gamma values
    y_numbers = df_y_locations["PIV_mm"] / 1000
    cfd_ellipse_gamma = df_y_locations["Gamma_CFD_Ellipse"]
    cfd_rectangle_gamma = df_y_locations["Gamma_CFD_Rectangle"]
    piv_ellipse_gamma = df_y_locations["Gamma_PIV_Ellipse"]
    piv_rectangle_gamma = df_y_locations["Gamma_PIV_Rectangle"]

    # # Plotting CFD in blue with different markers
    # ax.plot(y_numbers, cfd_ellipse_gamma, "bo-", label="CFD Ellipse")  # Blue circle
    # ax.plot(
    #     y_numbers, cfd_rectangle_gamma, "bs--", label="CFD Rectangle"
    # )  # Blue square

    # # Plotting PIV in red with different markers
    # ax.plot(y_numbers, piv_ellipse_gamma, "rp-", label="PIV Ellipse")  # Red circle
    # ax.plot(y_numbers, piv_rectangle_gamma, "r*--", label="PIV Rectangle")  # Red square

    # # Plotting VSM
    # ax.plot(
    #     df_VSM["y"] / 6.5,
    #     df_VSM["Gamma"],
    #     color="black",
    #     linestyle="dashed",
    #     label="VSM",
    # )

    # Plotting CFD in blue with different markers
    plot_on_ax(
        ax,
        y_numbers,
        cfd_ellipse_gamma,
        label="CFD Ellipse",
        color="blue",
        marker="o",
        linestyle="-",
    )
    plot_on_ax(
        ax,
        y_numbers,
        cfd_rectangle_gamma,
        label="CFD Rectangle",
        color="blue",
        marker="s",
        linestyle="--",
    )

    # Plotting PIV in red with different markers
    plot_on_ax(
        ax,
        y_numbers,
        piv_ellipse_gamma,
        label="PIV Ellipse",
        color="red",
        marker="p",
        linestyle="-",
    )
    plot_on_ax(
        ax,
        y_numbers,
        piv_rectangle_gamma,
        label="PIV Rectangle",
        color="red",
        marker="*",
        linestyle="--",
    )

    # Plotting VSM
    plot_on_ax(
        ax,
        df_VSM["y"] / 6.5,
        df_VSM["Gamma"],
        label="VSM",
        color="black",
        linestyle="--",
    )

    # Set the labels and title
    ax.set_xlabel("y [m]")  # x-axis label
    ax.set_ylabel(rf"$\Gamma$")  # y-axis label

    ax.set_xlim(0, 0.7)
    # ax.set_ylim(0.25, 3.5)

    # Add a legend
    ax.legend()

    # Display the plot
    # plt.show()

    plt.savefig(
        Path(project_dir) / "results" / "paper_plots" / "gamma_distribution.pdf"
    )
