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

    vsm_input_path = (
        Path(project_dir) / "data" / "vsm_input" / "2D_airfoils_polars_plots_BEST"
    )
    geom_scaled_path = Path(vsm_input_path) / "aero_geometry_CFD_CAD_derived.yaml"
    body_aero = BodyAerodynamics.instantiate(
        n_panels=150,
        file_path=geom_scaled_path,
        spanwise_panel_distribution="uniform",
    )
    Umag = 2.15
    alpha = 7
    body_aero.va_initialize(Umag, alpha)
    solver_base_version = Solver()
    results_CFD_CAD = solver_base_version.solve(body_aero)
    print(f"Rey: {results_CFD_CAD['Rey']:.2e}")

    # save results
    y_locations = [panel.aerodynamic_center[1] for panel in body_aero.panels]
    # scale y location
    y_locations = np.array(y_locations) / 6.5
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
    """Load VSM gamma distribution. Only runs VSM if cached CSV doesn't exist."""

    vsm_csv_path = (
        Path(project_dir)
        / "processed_data"
        / "gamma_distribution"
        / "VSM_gamma_distribution.csv"
    )

    # Only run VSM if the CSV doesn't exist
    if not vsm_csv_path.exists():
        print("VSM CSV not found, running VSM solver...")
        run_VSM()
    else:
        print(f"Loading cached VSM data from {vsm_csv_path.name}")

    df = pd.read_csv(vsm_csv_path, index_col=False)
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

    # Plot CFD/PIV only up to Y6 (exclude Y7).
    plot_count = min(
        6,
        len(y_numbers),
        len(cfd_gamma_ellipse),
        len(cfd_gamma_rectangle),
        len(piv_gamma_ellipse),
        len(piv_gamma_rectangle),
    )
    y_plot = np.asarray(y_numbers.iloc[:plot_count], dtype=float)
    cfd_ellipse_plot = np.asarray(cfd_gamma_ellipse.iloc[:plot_count], dtype=float)
    cfd_rectangle_plot = np.asarray(cfd_gamma_rectangle.iloc[:plot_count], dtype=float)
    piv_ellipse_plot = np.asarray(piv_gamma_ellipse.iloc[:plot_count], dtype=float)
    piv_rectangle_plot = np.asarray(piv_gamma_rectangle.iloc[:plot_count], dtype=float)

    print(f"running VSM")
    VSM_gamma_distribution, CAD_y_coordinates = get_VSM_gamma_distribution()

    ## plotting
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    ## Plotting VSM reference line
    # filter values for y > 0
    mask = CAD_y_coordinates > -1e-6
    y_cad = CAD_y_coordinates[mask]
    VSM_gamma_distribution = VSM_gamma_distribution[mask]

    plot_on_ax(
        ax,
        y_cad,
        VSM_gamma_distribution,
        label="VSM",
        x_label=r"$y$ (m)",
        y_label=r"$\Gamma$ (m$^2$/s)",
        # marker="x",
    )

    # =========================================================================
    # CFD: Show mean of ellipse/rectangle with contour shape uncertainty
    # =========================================================================
    cfd_gamma_mean = (cfd_ellipse_plot + cfd_rectangle_plot) / 2
    cfd_gamma_min = np.minimum(cfd_ellipse_plot, cfd_rectangle_plot)
    cfd_gamma_max = np.maximum(cfd_ellipse_plot, cfd_rectangle_plot)

    # Plot CFD mean line
    plot_on_ax(
        ax,
        y_plot,
        cfd_gamma_mean,
        label="CFD",
        color="blue",
        marker="*",
        linestyle="-",
    )

    # CFD contour shape uncertainty (shaded region)
    ax.fill_between(
        y_plot,
        cfd_gamma_min,
        cfd_gamma_max,
        color="blue",
        alpha=0.2,
        label=r"CFD contour shape range",
    )

    # =========================================================================
    # PIV: Show mean of ellipse/rectangle with contour shape uncertainty
    # =========================================================================
    piv_gamma_mean = (piv_ellipse_plot + piv_rectangle_plot) / 2
    piv_gamma_min = np.minimum(piv_ellipse_plot, piv_rectangle_plot)
    piv_gamma_max = np.maximum(piv_ellipse_plot, piv_rectangle_plot)

    # Plot PIV mean line
    plot_on_ax(
        ax,
        y_plot,
        piv_gamma_mean,
        label="PIV",
        color="red",
        marker=".",
        linestyle="-",
    )

    # PIV contour shape uncertainty (shaded region), same style as CFD
    ax.fill_between(
        y_plot,
        piv_gamma_min,
        piv_gamma_max,
        color="red",
        alpha=0.2,
        label=r"PIV contour shape range",
    )

    # Print diagnostic information
    print("\n" + "=" * 70)
    print("PIV CIRCULATION: Contour shape sensitivity analysis")
    print("=" * 70)
    print(
        f"{'Y-plane':<10} {'Ellipse':<12} {'Rectangle':<12} {'Mean':<12} {'Range':<12} {'Rel. range':<12}"
    )
    print("-" * 70)
    for i in range(plot_count):
        ell = piv_ellipse_plot[i]
        rect = piv_rectangle_plot[i]
        mean = piv_gamma_mean[i]
        range_val = piv_gamma_max[i] - piv_gamma_min[i]
        rel_range = range_val / mean * 100 if mean > 0 else 0
        print(
            f"Y{i+1:<9} {ell:<12.3f} {rect:<12.3f} {mean:<12.3f} {range_val:<12.3f} {rel_range:<10.1f}%"
        )

    # Also print comparison with CFD
    print("\n" + "=" * 70)
    print("PIV vs CFD comparison (using mean values)")
    print("=" * 70)
    print(f"{'Y-plane':<10} {'PIV mean':<12} {'CFD mean':<12} {'Deviation':<12}")
    print("-" * 70)
    for i in range(plot_count):
        piv_mean = piv_gamma_mean[i]
        cfd_mean = cfd_gamma_mean[i]
        deviation = (piv_mean - cfd_mean) / cfd_mean * 100 if cfd_mean > 0 else 0
        print(f"Y{i+1:<9} {piv_mean:<12.3f} {cfd_mean:<12.3f} {deviation:+10.1f}%")

    ax.set_xlim(-0.05, 0.65)
    ax.set_ylim(0, 2.5)
    ax.set_xlabel(r"$y$ (m)")
    ax.set_ylabel(r"$\Gamma$ (m$^2$s$^{-1}$)")
    # Fixed legend order for readability.
    handles, labels = ax.get_legend_handles_labels()
    desired_order = [
        "CFD",
        "PIV",
        "VSM",
        "CFD contour shape range",
        "PIV contour shape range",
    ]
    ordered_handles = []
    ordered_labels = []
    for name in desired_order:
        if name in labels:
            idx = labels.index(name)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])
    ax.legend(ordered_handles, ordered_labels, ncol=2, loc="best")
    plt.tight_layout(pad=0.02)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.01)

    print(f"\n--->saved figure to {save_path}")


def main():
    # save_path = (
    #     Path(project_dir)
    #     / "results"
    #     / "paper_plots_24_03_2026"
    #     / "fig08_gamma_distribution_std.pdf"
    # )
    save_path = Path(project_dir) / "results" / "paper_plots_24_03_2026" / "fig08.pdf"
    plot_gamma_distribution(save_path)


if __name__ == "__main__":
    main()
