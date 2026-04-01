import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from kite_piv_analysis.utils import project_dir
from kite_piv_analysis.calculating_circulation import calculate_circulation
from kite_piv_analysis.utils import reading_optimal_bound_placement
from kite_piv_analysis import calculating_airfoil_centre
from kite_piv_analysis.defining_bound_volume import boundary_ellipse, boundary_rectangle
from kite_piv_analysis.plot_styling import PALETTE, set_plot_style, plot_on_ax
from kite_piv_analysis import force_from_noca
from kite_piv_analysis.plotting import (
    load_data,
    find_areas_needing_interpolation,
    interpolate_missing_data,
)

MEASUREMENT_PLANE_COLOR = "#00FF00"
MAX_PIV_MISSING_BEFORE_REQUIRED_PCT = 20.0
NOCA_RAW_CACHE_PATH = (
    Path(project_dir) / "processed_data" / "noca_term_breakdown_raw.csv"
)


def _load_allowed_piv_shape_keys(
    alpha: int,
    max_piv_missing_before_required_pct: float = MAX_PIV_MISSING_BEFORE_REQUIRED_PCT,
) -> set[tuple[int, str]] | None:
    """
    Return allowed (y_num, shape) keys for PIV rows based on missing-before filter.

    Returns None when raw cache is unavailable or lacks required columns, in which
    case no extra plot-time filtering should be applied.
    """
    if not NOCA_RAW_CACHE_PATH.exists():
        return None

    df_raw = pd.read_csv(NOCA_RAW_CACHE_PATH)
    required_cols = {
        "alpha",
        "y_num",
        "data_type",
        "shape",
        "perc_missing_before_required",
    }
    if not required_cols.issubset(df_raw.columns):
        return None

    data_type = df_raw["data_type"].astype(str).str.upper()
    perc_missing = pd.to_numeric(
        df_raw["perc_missing_before_required"], errors="coerce"
    )
    keep = (
        (df_raw["alpha"] == int(alpha))
        & (data_type == "PIV")
        & np.isfinite(perc_missing)
        & (perc_missing < float(max_piv_missing_before_required_pct))
    )

    df_keep = df_raw.loc[keep, ["y_num", "shape"]].copy()
    if df_keep.empty:
        return set()

    out: set[tuple[int, str]] = set()
    for _, row in df_keep.iterrows():
        y_num = pd.to_numeric(row["y_num"], errors="coerce")
        if not np.isfinite(y_num):
            continue
        shape = str(row["shape"]).strip().lower()
        if shape not in {"ellipse", "rectangle"}:
            continue
        out.add((int(y_num), shape))
    return out


def _apply_default_piv_missing_filter_to_gamma_df(
    df_gamma: pd.DataFrame,
    alpha: int = 6,
    max_piv_missing_before_required_pct: float = MAX_PIV_MISSING_BEFORE_REQUIRED_PCT,
) -> pd.DataFrame:
    """
    Set PIV gamma values to NaN where the default missing-before criterion fails.
    """
    allowed_keys = _load_allowed_piv_shape_keys(
        alpha=alpha,
        max_piv_missing_before_required_pct=max_piv_missing_before_required_pct,
    )
    if allowed_keys is None:
        return df_gamma
    if "y_num" not in df_gamma.columns:
        return df_gamma

    out = df_gamma.copy()
    y_vals = pd.to_numeric(out["y_num"], errors="coerce")
    for shape, col in [
        ("ellipse", "ellipse_piv_gamma"),
        ("rectangle", "rectangle_piv_gamma"),
    ]:
        if col not in out.columns:
            continue
        keep_mask = y_vals.apply(
            lambda y: bool(np.isfinite(y)) and (int(y), shape) in allowed_keys
        )
        out.loc[~keep_mask, col] = np.nan
    return out


def _load_shape_outline(
    alpha: int = 6, cm_offset: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    outline_path = (
        Path(project_dir)
        / "data"
        / "CFD_slices"
        / "spanwise_slices_outline_wing"
        / f"alpha_{alpha}_CFD_{cm_offset}cm_outline_wing.csv"
    )
    if not outline_path.exists():
        outline_path = (
            Path(project_dir)
            / "processed_data"
            / "CFD"
            / "spanwise_slices"
            / f"alpha_{alpha}_CFD_{cm_offset}cm_outline_wing.csv"
        )
    if not outline_path.exists():
        return np.array([]), np.array([])

    df_outline = pd.read_csv(outline_path)
    if "x" not in df_outline.columns or "y" not in df_outline.columns:
        return np.array([]), np.array([])
    return (
        df_outline["x"].to_numpy(dtype=float),
        df_outline["y"].to_numpy(dtype=float),
    )


def _add_shape_overlay(
    ax: plt.Axes, x_shape: np.ndarray, z_shape: np.ndarray, z_lim: tuple[float, float]
) -> plt.Axes:
    ax_right = ax.twinx()
    if len(x_shape) > 0:
        ax_right.fill(
            x_shape,
            z_shape,
            facecolor="black",
            edgecolor="black",
            alpha=0.2,
            linewidth=1.1,
            label="Shape outline",
            zorder=0.0,
        )
    ax_right.set_ylim(*z_lim)
    ax_right.set_ylabel(r"$z$ (m)")
    ax_right.tick_params(axis="y", labelright=True, right=True)
    ax_right.grid(False)
    return ax_right


def _shape_vertical_center_at_x(
    x0: float, x_shape: np.ndarray, z_shape: np.ndarray
) -> float:
    if len(x_shape) < 2 or len(z_shape) < 2:
        return np.nan

    intersections: list[float] = []
    for idx in range(len(x_shape) - 1):
        x1 = float(x_shape[idx])
        x2 = float(x_shape[idx + 1])
        z1 = float(z_shape[idx])
        z2 = float(z_shape[idx + 1])
        if not (
            np.isfinite(x1) and np.isfinite(x2) and np.isfinite(z1) and np.isfinite(z2)
        ):
            continue

        if abs(x2 - x1) < 1e-12:
            if abs(x0 - x1) < 1e-9:
                intersections.extend([z1, z2])
            continue

        t = (x0 - x1) / (x2 - x1)
        if 0.0 <= t <= 1.0:
            intersections.append(z1 + t * (z2 - z1))

    if not intersections:
        return float(z_shape[int(np.argmin(np.abs(x_shape - x0)))])

    intersections = sorted(intersections)
    unique_intersections: list[float] = []
    for value in intersections:
        if not unique_intersections or abs(value - unique_intersections[-1]) > 1e-6:
            unique_intersections.append(value)
    return 0.5 * (unique_intersections[0] + unique_intersections[-1])


def _add_shape_centered_marker_lines(
    ax_right: plt.Axes,
    x_positions: np.ndarray,
    x_shape: np.ndarray,
    z_shape: np.ndarray,
    line_height_m: float = 0.09,
    line_width: float = 2.0,
    color: str = MEASUREMENT_PLANE_COLOR,
    alpha: float = 1.0,
    legend_label: str = "Measurement planes",
) -> None:
    if len(x_positions) == 0 or len(x_shape) == 0 or len(z_shape) == 0:
        return

    half_height = 0.5 * float(line_height_m)
    for idx, x0 in enumerate(np.asarray(x_positions, dtype=float)):
        if not np.isfinite(x0):
            continue
        z_center = _shape_vertical_center_at_x(float(x0), x_shape, z_shape)
        if not np.isfinite(z_center):
            continue
        ax_right.plot(
            [x0, x0],
            [z_center - half_height, z_center + half_height],
            color=color,
            linewidth=line_width,
            alpha=alpha,
            solid_capstyle="round",
            zorder=1.0,
            label=(legend_label if idx == 0 else "_nolegend_"),
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
    y_measurement_planes = np.asarray(y_numbers, dtype=float)

    x_shape, z_shape = _load_shape_outline(alpha=6, cm_offset=25)
    if len(x_shape) > 0 and len(z_shape) > 0:
        z_min = float(np.nanmin(z_shape))
        z_max = float(np.nanmax(z_shape))
    else:
        z_min, z_max = -0.6, 0.1
    z_span = max(z_max - z_min, 1e-6)
    z_pad = 0.11 * z_span
    z_lim = (z_min - z_pad, z_max + z_pad)

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
    df = _apply_default_piv_missing_filter_to_gamma_df(
        df,
        alpha=6,
        max_piv_missing_before_required_pct=MAX_PIV_MISSING_BEFORE_REQUIRED_PCT,
    )
    cfd_gamma_ellipse = df["ellipse_cfd_gamma"]
    cfd_gamma_rectangle = df["rectangle_cfd_gamma"]
    piv_gamma_ellipse = df["ellipse_piv_gamma"]
    piv_gamma_rectangle = df["rectangle_piv_gamma"]

    # CFD: show all available planes; PIV: exclude Y7 (cap at 6).
    cfd_count = min(
        6,
        len(cfd_gamma_ellipse),
        len(cfd_gamma_rectangle),
    )
    piv_count = min(
        6,
        len(y_numbers),
        len(piv_gamma_ellipse),
        len(piv_gamma_rectangle),
    )
    y_cfd_plot = np.asarray(y_numbers.iloc[:cfd_count], dtype=float)
    cfd_ellipse_plot = np.asarray(cfd_gamma_ellipse.iloc[:cfd_count], dtype=float)
    cfd_rectangle_plot = np.asarray(cfd_gamma_rectangle.iloc[:cfd_count], dtype=float)
    y_piv_plot = np.asarray(y_numbers.iloc[:piv_count], dtype=float)
    piv_ellipse_plot = np.asarray(piv_gamma_ellipse.iloc[:piv_count], dtype=float)
    piv_rectangle_plot = np.asarray(piv_gamma_rectangle.iloc[:piv_count], dtype=float)

    print(f"running VSM")
    VSM_gamma_distribution, CAD_y_coordinates = get_VSM_gamma_distribution()

    ## plotting
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.7, 4.4))

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
    # CFD: Show mean of ellipse/rectangle with shape uncertainty
    # =========================================================================
    cfd_gamma_mean = (cfd_ellipse_plot + cfd_rectangle_plot) / 2
    cfd_gamma_min = np.minimum(cfd_ellipse_plot, cfd_rectangle_plot)
    cfd_gamma_max = np.maximum(cfd_ellipse_plot, cfd_rectangle_plot)

    # Plot CFD mean line
    plot_on_ax(
        ax,
        y_cfd_plot,
        cfd_gamma_mean,
        label="CFD",
        color="blue",
        marker="*",
        linestyle="-",
    )

    # CFD shape uncertainty (shaded region)
    ax.fill_between(
        y_cfd_plot,
        cfd_gamma_min,
        cfd_gamma_max,
        color="blue",
        alpha=0.2,
        label=r"CFD shape range",
    )

    # =========================================================================
    # PIV: Show mean of ellipse/rectangle with shape uncertainty
    # =========================================================================
    piv_gamma_mean = (piv_ellipse_plot + piv_rectangle_plot) / 2
    piv_gamma_min = np.minimum(piv_ellipse_plot, piv_rectangle_plot)
    piv_gamma_max = np.maximum(piv_ellipse_plot, piv_rectangle_plot)

    # Plot PIV mean line
    plot_on_ax(
        ax,
        y_piv_plot,
        piv_gamma_mean,
        label="PIV",
        color="red",
        marker=".",
        linestyle="--",
    )

    # PIV shape uncertainty (shaded region), same style as CFD
    ax.fill_between(
        y_piv_plot,
        piv_gamma_min,
        piv_gamma_max,
        color="red",
        alpha=0.2,
        label=r"PIV shape range",
    )

    # Print diagnostic information
    print("\n" + "=" * 70)
    print("PIV CIRCULATION: Contour shape sensitivity analysis")
    print("=" * 70)
    print(
        f"{'Y-plane':<10} {'Ellipse':<12} {'Rectangle':<12} {'Mean':<12} {'Range':<12} {'Rel. range':<12}"
    )
    print("-" * 70)
    for i in range(piv_count):
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
    for i in range(piv_count):
        piv_mean = piv_gamma_mean[i]
        cfd_mean = cfd_gamma_mean[i]
        deviation = (piv_mean - cfd_mean) / cfd_mean * 100 if cfd_mean > 0 else 0
        print(f"Y{i+1:<9} {piv_mean:<12.3f} {cfd_mean:<12.3f} {deviation:+10.1f}%")

    ax.set_xlim(-0.02, 0.65)
    ax.set_ylim(0, 3.0)
    ax.set_xlabel(r"$y$ (m)")
    ax.set_ylabel(r"$\Gamma$ (m$^2$s$^{-1}$)")
    ax.grid(False)

    ax_right = _add_shape_overlay(ax, x_shape=x_shape, z_shape=z_shape, z_lim=z_lim)
    _add_shape_centered_marker_lines(
        ax_right,
        x_positions=y_measurement_planes,
        x_shape=x_shape,
        z_shape=z_shape,
        color=MEASUREMENT_PLANE_COLOR,
        legend_label="Measurement planes",
    )
    # Keep data curves above overlay.
    ax_right.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    ax_right.set_ylim(*z_lim)

    # Enforce equal x/z data step size while preserving the prescribed limits.
    x_min, x_max = ax.get_xlim()
    x_range = float(x_max - x_min)
    z_range = float(z_lim[1] - z_lim[0])
    if x_range > 1e-12 and z_range > 1e-12:
        overlay_box_aspect = z_range / x_range
        ax.set_box_aspect(overlay_box_aspect)
        ax_right.set_box_aspect(overlay_box_aspect)

    # Fixed legend order for readability.
    handles, labels = ax.get_legend_handles_labels()
    handles_overlay, labels_overlay = ax_right.get_legend_handles_labels()
    handle_map = {}
    for handle, label in list(zip(handles, labels)) + list(
        zip(handles_overlay, labels_overlay)
    ):
        if label == "_nolegend_" or label in handle_map:
            continue
        handle_map[label] = handle

    if "Shape outline" not in handle_map:
        handle_map["Shape outline"] = Patch(
            facecolor="black",
            edgecolor="black",
            alpha=0.2,
        )
    if "Measurement planes" not in handle_map:
        handle_map["Measurement planes"] = Line2D(
            [0], [0], color=MEASUREMENT_PLANE_COLOR, alpha=0.8, linewidth=2.0
        )

    desired_order = [
        "CFD shape range",
        "PIV shape range",
        "Shape outline",
        "Measurement planes",
        "CFD",
        "PIV",
        "VSM",
    ]
    ordered_handles = []
    ordered_labels = []
    for name in desired_order:
        if name in handle_map:
            ordered_handles.append(handle_map[name])
            ordered_labels.append(name)
    ax.legend(
        ordered_handles,
        ordered_labels,
        ncol=2,
        loc="lower left",
        framealpha=1.0,
        fancybox=False,
        facecolor="white",
    )
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
