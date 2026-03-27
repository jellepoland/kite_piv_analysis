import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from kite_piv_analysis.utils import project_dir
from kite_piv_analysis.calculating_circulation import calculate_circulation
from kite_piv_analysis.utils import reading_optimal_bound_placement
from kite_piv_analysis import calculating_airfoil_centre
from kite_piv_analysis.defining_bound_volume import boundary_ellipse, boundary_rectangle
from kite_piv_analysis.plot_styling import set_plot_style, plot_on_ax
from kite_piv_analysis import force_from_noca
from kite_piv_analysis.plotting import (
    load_data,
    find_areas_needing_interpolation,
    interpolate_missing_data,
)


def run_VSM():

    from VSM.core.BodyAerodynamics import BodyAerodynamics
    from VSM.core.Solver import Solver

    vsm_input_path = Path(project_dir) / "data" / "vsm_input"
    geom_scaled_path = Path(vsm_input_path) / "wing_geometry_scaled.yaml"
    body_aero = BodyAerodynamics.instantiate(
        n_panels=150,
        file_path=geom_scaled_path,
        spanwise_panel_distribution="uniform",
    )
    Umag = 14.0
    alpha = 7
    body_aero.va_initialize(Umag, alpha)
    solver_base_version = Solver()
    results_CFD_CAD = solver_base_version.solve(body_aero)
    print(f"Rey: {results_CFD_CAD['Rey']:.2e}")

    # save results
    y_locations = [panel.aerodynamic_center[1] for panel in body_aero.panels]
    gamma_distribution = results_CFD_CAD["gamma_distribution"]
    df = pd.DataFrame(
        {
            "y": y_locations,
            "gamma_polar": gamma_distribution,
        }
    )
    df.to_csv(
        Path(project_dir)
        / "processed_data"
        / "gamma_distribution"
        / "VSM_gamma_distribution.csv",
        index=False,
    )
    return df


def get_VSM_gamma_distribution():

    run_VSM()

    df = pd.read_csv(
        Path(project_dir)
        / "processed_data"
        / "gamma_distribution"
        / "VSM_gamma_distribution.csv",
        index_col=False,
    )
    VSM_gamma_distribution = df["gamma_polar"].values
    CAD_y_coordinates = df["y"].values

    return VSM_gamma_distribution, CAD_y_coordinates


def plot_gamma_distribution(save_path):
    ## loading data
    df_y_locations = pd.read_csv(
        Path(project_dir) / "data" / "gamma_distribution" / "y_locations.csv",
        index_col=False,
    )
    y_numbers = df_y_locations["PIV_mm"] / 1000

    csv_path = Path(
        project_dir,
        "processed_data",
        "quantitative_chordwise_analysis_alpha_6_with_std.csv",
    )

    # only generate data if it does not exist yet, as it takes long
    if not csv_path.exists():
        from kite_piv_analysis.calculating_noca_and_kutta import (
            save_results_single_alpha,
        )

        print(f"CSV not found: {csv_path.name} — generating data (this takes ~45 min)")
        alpha = 6
        y_num_list = [1, 2, 3, 4, 5, 6, 7]
        try:
            save_results_single_alpha(alpha, y_num_list)
        except Exception as e:
            print(f"Failed to create results: {e}")
        else:
            print(f"Saved successfully to {csv_path}")

    df = pd.read_csv(csv_path)
    cfd_gamma_ellipse = df["ellipse_cfd_gamma"]
    cfd_gamma_rectangle = df["rectangle_cfd_gamma"]
    piv_gamma_ellipse = df["ellipse_piv_gamma"]
    piv_gamma_rectangle = df["rectangle_piv_gamma"]

    print(f"running VSM")
    VSM_gamma_distribution, CAD_y_coordinates = get_VSM_gamma_distribution()

    ## plotting
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))  # 9,5.5

    ### bounds
    factor_ci = 1.64  # 1.96 is 95%
    ## shape induced standard deviation
    n_shape_samples = 100
    shape_std_ellipse = df["ellipse_piv_gamma_std"] / np.sqrt(n_shape_samples)
    low_bound_shape_ellipse = piv_gamma_ellipse - factor_ci * shape_std_ellipse
    up_bound_shape_ellipse = piv_gamma_ellipse + factor_ci * shape_std_ellipse
    shape_std_rectangle = df["rectangle_piv_gamma_std"]
    low_bound_shape_rectangle = piv_gamma_rectangle - factor_ci * shape_std_rectangle
    up_bound_shape_rectangle = piv_gamma_rectangle + factor_ci * shape_std_rectangle

    ## Plotting the lines on top
    # filter valuese for y > 0
    mask = CAD_y_coordinates > -1e-2
    y_cad = CAD_y_coordinates[mask]
    VSM_gamma_distribution = VSM_gamma_distribution[mask]

    plot_on_ax(
        ax,
        y_cad,
        VSM_gamma_distribution,
        label="VSM",
        x_label=r"$y$ (m)",
        y_label=r"$\Gamma$ (m$^2$/s)",  # r"$\Gamma [$m^2$/s]",
        # title="VSM gamma distribution",
    )
    plot_on_ax(
        ax,
        y_numbers,
        cfd_gamma_ellipse,
        label="CFD Ellipse",
        color="blue",
        marker="o",
        linestyle="-",
    )
    plot_on_ax(
        ax,
        y_numbers,
        cfd_gamma_rectangle,
        label="CFD Rectangle",
        color="blue",
        marker="s",
        linestyle="--",
    )
    plot_on_ax(
        ax,
        y_numbers[:6],
        piv_gamma_ellipse[:6],
        label="PIV Ellipse",
        color="red",
        marker="p",
        linestyle="-",
    )
    plot_on_ax(
        ax,
        y_numbers[:6],
        piv_gamma_rectangle[:6],
        label="PIV Rectangle",
        color="red",
        marker="*",
        linestyle="--",
    )

    ## velocity standard deviation
    n_vel_samples = 250
    vel_std_ellipse = (
        piv_gamma_ellipse - df["ellipse_piv_gamma_lower_bound"]
    ).abs() / (1.96 * np.sqrt(n_vel_samples))
    low_bound_vel_ellipse = piv_gamma_ellipse - factor_ci * vel_std_ellipse
    up_bound_vel_ellipse = piv_gamma_ellipse + factor_ci * vel_std_ellipse
    vel_std_rectangle = (
        piv_gamma_rectangle - df["rectangle_piv_gamma_lower_bound"]
    ).abs() / (1.96 * np.sqrt(n_vel_samples))
    low_bound_vel_rectangle = piv_gamma_rectangle - factor_ci * vel_std_rectangle
    up_bound_vel_rectangle = piv_gamma_rectangle + factor_ci * vel_std_rectangle

    ###

    ## printing the percentage of the std due ot the velocity
    for i in range(6):
        # print(
        #     f"vel/shape y={y_numbers[i]:.3f}: ellipse: {vel_std_ellipse[i] / shape_std_ellipse[i]:.2f} rectangle: {vel_std_rectangle[i] / shape_std_rectangle[i]:.2f}"
        # )
        print(
            f"shape/vel y={y_numbers[i]:.3f}: ellipse: {shape_std_ellipse[i] / vel_std_ellipse[i]:.2f} rectangle: {shape_std_rectangle[i] / vel_std_rectangle[i]:.2f}"
        )

    ### Computing the combined uncertainty
    combined_std_ellipse = np.sqrt(shape_std_ellipse**2 + vel_std_ellipse**2)
    low_bound_combined_ellipse = piv_gamma_ellipse - factor_ci * combined_std_ellipse
    up_bound_combined_ellipse = piv_gamma_ellipse + factor_ci * combined_std_ellipse
    combined_std_rectangle = np.sqrt(shape_std_rectangle**2 + vel_std_rectangle**2)
    low_bound_combined_rectangle = (
        piv_gamma_rectangle - factor_ci * combined_std_rectangle
    )
    up_bound_combined_rectangle = (
        piv_gamma_rectangle + factor_ci * combined_std_rectangle
    )

    # Extract numerical arrays for ellipse
    x_ellipse = y_numbers[:6]
    y_ellipse = piv_gamma_ellipse[:6]
    y_ellipse_low = low_bound_combined_ellipse[:6]
    y_ellipse_high = up_bound_combined_ellipse[:6]

    # Compute the symmetric error above/below the mean
    # (if your bounds are asymmetric, you can pass an array [lower_error, upper_error] to yerr)
    err_ellipse_lower = y_ellipse - y_ellipse_low
    err_ellipse_upper = y_ellipse_high - y_ellipse
    err_ellipse = np.array([err_ellipse_lower, err_ellipse_upper])

    # Extract numerical arrays for rectangle
    x_rectangle = y_numbers[:6]
    y_rectangle = piv_gamma_rectangle[:6]
    y_rectangle_low = low_bound_combined_rectangle[:6]
    y_rectangle_high = up_bound_combined_rectangle[:6]

    err_rectangle_lower = y_rectangle - y_rectangle_low
    err_rectangle_upper = y_rectangle_high - y_rectangle
    err_rectangle = np.array([err_rectangle_lower, err_rectangle_upper])

    # Create a figure and axis
    # fig, ax = plt.subplots(figsize=(8, 5))

    # Plot ellipse points with error bars
    # fmt='o' makes circle markers; capsize adds little caps on error bars
    ax.errorbar(
        x_ellipse,
        y_ellipse,
        yerr=err_ellipse,
        fmt="p",
        capsize=3,
        color="red",
        ecolor="red",
        label=r"90\% CI $\sigma_{\Gamma}$ Ellipse",
    )

    # Plot rectangle points with error bars
    ax.errorbar(
        x_rectangle[:6],
        y_rectangle[:6],
        yerr=err_rectangle,
        fmt="*",
        capsize=3,
        color="red",
        ecolor="red",
        label=r"90\% CI $\sigma_{\Gamma}$ Rectangle",
    )

    ax.fill_between(
        y_numbers[:6],
        low_bound_vel_ellipse[:6],
        up_bound_vel_ellipse[:6],
        color="red",
        alpha=0.3,
        label=r"CI $\sigma_{\Gamma,\textrm{v}}$ Ellipse",
    )
    ax.fill_between(
        y_numbers[:6],
        low_bound_vel_rectangle[:6],
        up_bound_vel_rectangle[:6],
        color="red",
        alpha=0.15,
        label=r"CI $\sigma_{\Gamma,\textrm{v}}$ Rectangle",
    )

    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 2.5)
    ax.set_xlabel(r"$y$ (m)")
    ax.set_ylabel(r"$\Gamma$ (m$^2$s$^{-1}$)")
    plt.legend(ncol=3)
    plt.tight_layout(pad=0.02)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.01)

    print(f"\n--->saved figure to {save_path}")


def main():
    save_path = (
        Path(project_dir)
        / "results"
        / "paper_plots_21_10_2025"
        / "fig08_gamma_distribution_std_preprint.pdf"
    )
    plot_gamma_distribution(save_path)


if __name__ == "__main__":
    main()
