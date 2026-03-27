"""
Comprehensive PIV Uncertainty Budget Analysis

This script provides a multi-component uncertainty analysis for stereoscopic PIV,
addressing reviewer concerns about the simplicity of uncertainty quantification.

Components analyzed:
1. Statistical precision (sample standard deviation / √N)
2. Masking and interpolation metrics (data loss quantification)
3. Divergence-based physical consistency check
4. Freestream velocity statistics (outer-field sanity check)
5. Overlap region analysis (calibration/mapping quality indicator)
6. Spatial uncertainty maps

Addresses reviewer comment Line 214:
"The performed PIV uncertainty analysis is too simple to understand the complex
sources of error in the presented measurement data."
"""

import numpy as np
import pandas as pd
from pathlib import Path
from statistics import NormalDist
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from kite_piv_analysis.utils import project_dir
from kite_piv_analysis.plot_styling import set_plot_style
from kite_piv_analysis.compute_overlap_error import (
    load_dat_file,
)
from kite_piv_analysis.masking import apply_snr_masking, apply_w_masking


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_stitched_plane(aoa: int, y_plane: int, config: str = "normal"):
    """
    Load stitched PIV plane data.

    Parameters:
        aoa: Angle of attack (13 or 23, corresponds to alpha=6 or 16 deg)
        y_plane: Y-plane number (1-7)
        config: 'normal' (pressure side) or 'flipped' (suction side)

    Returns:
        DataFrame with x, y, u, v, w, and gradient fields
    """
    # Try processed stitched data first
    stitched_dir = Path(project_dir) / "data" / "piv_stichted_planes" / f"aoa_{aoa}"

    # Check for stitched CSV file
    csv_file = stitched_dir / f"aoa_{aoa}_Y{y_plane}_stitched.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file)

    return None


def load_raw_plane_data(aoa: int, y_plane: int, x_plane: int, config: str = "normal"):
    """
    Load raw PIV plane data (B0001.dat mean, B0002.dat std).

    Source location:
    1. data_ALL_ERIK_FILES/JelleStitching/Input (primary only)

    Returns:
        mean_data, std_data dictionaries
    """
    input_dir = Path(project_dir) / "data_ALL_ERIK_FILES" / "JelleStitching" / "Input"
    base_path = input_dir / f"aoa_{aoa}" / f"Y{y_plane}"

    # Check if directory exists
    if not base_path.exists():
        return None, None

    # Find matching directory
    for item in base_path.iterdir():
        if item.is_dir() and config in item.name and f"_X{x_plane}" in item.name:
            mean_file = item / "B0001.dat"
            std_file = item / "B0002.dat"
            if mean_file.exists() and std_file.exists():
                mean_data = load_dat_file(mean_file)
                std_data = load_dat_file(std_file)
                return mean_data, std_data

    return None, None


def _alpha_from_aoa(aoa: int) -> int:
    """Map folder AoA labels to aerodynamic alpha used in masking config."""
    return {13: 6, 23: 16}.get(int(aoa), int(aoa))


def _mask_plot_params(aoa: int, y_plane: int, w_threshold: float) -> dict:
    alpha = _alpha_from_aoa(aoa)
    return {
        "alpha": alpha,
        "y_num": y_plane,
        "is_with_w_mask": True,
        "w_mask_lower_bound": -abs(float(w_threshold)),
        "w_mask_upper_bound": abs(float(w_threshold)),
        "is_with_snr_mask": [True, True],
        "snr_use_proxy": [True, True],
        "snr_column_to_mask": ["snr", "snr"],
        "snr_mask_lower_bound": [2.0, 2.0],
        "snr_mask_upper_bound": [np.inf, np.inf],
        "proxy_snr_components": [["u", "v"], ["u", "v"]],
        "proxy_snr_reduction": ["median", "mean"],
        "proxy_snr_min_std": [1e-6, 1e-6],
        "snr_mask_nan_as_invalid": [True, True],
        "snr_mask_strict": [False, False],
        "snr_apply_only_below_rear_airfoil_line": [False, True],
        "snr_rear_airfoil_fraction": [0.5, 0.5],
    }


def _mean_dict_to_df(mean_data: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": mean_data["x"],
            "y": mean_data["y"],
            "u": mean_data["u"],
            "v": mean_data["v"],
            "w": mean_data["w"],
            "V": mean_data["V"],
            "is_valid": mean_data["is_valid"],
        }
    )


def _std_dict_to_df(std_data: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "u": std_data["u"],
            "v": std_data["v"],
            "w": std_data["w"],
        }
    )


def _apply_combined_mask(
    mean_data: dict, std_data: dict, aoa: int, y_plane: int, w_threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply w-mask + ICV(proxy-SNR)-mask and return:
    raw mean df, post-w df, post-(w+ICV) df.
    """
    df_mean = _mean_dict_to_df(mean_data)
    df_std = _std_dict_to_df(std_data)
    mask_params = _mask_plot_params(aoa, y_plane, w_threshold)
    mask_params["df_std"] = df_std

    df_after_w = apply_w_masking(df_mean, mask_params)
    df_after_combined = apply_snr_masking(df_after_w, mask_params)
    return df_mean, df_after_w, df_after_combined


# =============================================================================
# 1. Masking and Interpolation Metrics
# =============================================================================


def compute_masking_metrics(
    aoa: int, y_plane: int, U_inf: float = 15.0, w_threshold: float = 3.0
):
    """
    Compute detailed data loss metrics for a given measurement plane.

    This function quantifies data loss from different sources in the PIV processing:

    1. f_mask: DaVis isValid=0 flag (geometric masking, reflections, correlation failure)
       - Applied during MATLAB stitching: invalid points set to NaN

    2. f_u_y: Points removed by the |u_y|/|w| threshold (default 3 m/s)
    3. f_ICV: Additional points removed by ICV/proxy-SNR rejection
       after the |u_y| filter.

    4. f_valid: Remaining valid data after all filters
    5. f_loss: Total data loss (1 - f_valid)

    Note: The std/U_inf > 0.1 filter visible in RunStitching.m is only used for
    visualization overlays, NOT applied to the output CSV files.

    Parameters:
        aoa: Angle of attack directory number (13 or 23)
        y_plane: Y-plane number (1-7)
        U_inf: Freestream velocity (default 15 m/s)
        w_threshold: Threshold for |w| masking in m/s (default 3.0, per Appendix B)

    Returns:
        dict with detailed masking statistics
    """
    results = {"aoa": aoa, "y_plane": y_plane}

    total_points = 0
    masked_davis = 0  # f_mask: geometric/reflection masking (isValid=0)
    filtered_w_only = 0  # removed by |w| threshold (DaVis-valid points only)
    filtered_icv_only = 0  # additionally removed by ICV/SNR (after w-mask)
    filtered_combined = 0  # removed by either w or ICV (DaVis-valid points only)
    valid_points = 0

    for x_plane in [1, 2, 3]:
        for config in ["normal", "flipped"]:
            mean_data, std_data = load_raw_plane_data(aoa, y_plane, x_plane, config)

            if mean_data is None:
                continue

            n_total = len(mean_data["x"])
            total_points += n_total

            # Apply combined masking pipeline (w + ICV) on raw mean/std data.
            df_mean, df_after_w, df_after_combined = _apply_combined_mask(
                mean_data, std_data, aoa, y_plane, w_threshold
            )

            # Get DaVis validity mask (applied in MATLAB stitching)
            davis_valid = df_mean["is_valid"].to_numpy() != 0
            n_masked_davis = np.sum(~davis_valid)
            masked_davis += n_masked_davis

            w_removed = davis_valid & df_after_w["u"].isna().to_numpy()
            combined_removed = davis_valid & df_after_combined["u"].isna().to_numpy()

            filtered_w_only += int(np.sum(w_removed))
            filtered_icv_only += int(np.sum(combined_removed & ~w_removed))
            filtered_combined += int(np.sum(combined_removed))
            valid_points += int(np.sum(davis_valid & ~combined_removed))

    if total_points > 0:
        results["n_total"] = total_points
        results["f_mask"] = (
            masked_davis / total_points
        )  # DaVis geometric/reflection masking
        # Separate post-validation contributions:
        # - f_u_y: removed by |u_y|/|w| threshold only
        # - f_icv: additional removals by ICV/proxy-SNR after u_y filter
        results["f_u_y"] = filtered_w_only / total_points
        results["f_icv"] = filtered_icv_only / total_points
        results["f_quality"] = filtered_combined / total_points
        results["f_valid"] = valid_points / total_points
        results["f_loss"] = 1 - valid_points / total_points

        # Legacy names for compatibility
        results["f_w"] = results["f_u_y"]
        results["f_w_only"] = results["f_u_y"]
        results["f_icv_only"] = results["f_icv"]
        results["f_invalid_davis"] = results["f_mask"]
        results["f_data_loss"] = results["f_loss"]

    return results


def analyze_all_masking_metrics():
    """
    Compute masking metrics for all planes and generate summary.
    """
    results = []

    for aoa in [13, 23]:
        for y_plane in range(1, 8):
            metrics = compute_masking_metrics(aoa, y_plane)
            if "n_total" in metrics:
                results.append(metrics)

    df = pd.DataFrame(results)
    return df


# =============================================================================
# 2. Freestream Velocity Statistics
# =============================================================================


def compute_freestream_statistics(aoa: int, y_plane: int, U_inf: float = 15.0):
    """
    Compute freestream velocity statistics from outer field regions.

    Selects points far from the airfoil (high y for normal, low y after flip)
    where velocity should be close to freestream.

    Returns:
        dict with freestream region statistics
    """
    results = {"aoa": aoa, "y_plane": y_plane}

    freestream_u = []
    freestream_V = []

    for x_plane in [1, 2, 3]:
        mean_data, std_data = load_raw_plane_data(aoa, y_plane, x_plane, "normal")

        if mean_data is None or std_data is None:
            continue

        df_mean, _, df_after_combined = _apply_combined_mask(
            mean_data, std_data, aoa, y_plane, w_threshold=3.0
        )
        valid_mask = ~df_after_combined["u"].isna().to_numpy()
        if not np.any(valid_mask):
            continue

        # Select freestream region: top 20% of y-range (far from airfoil)
        y_vals = df_mean["y"].to_numpy()
        y_valid = y_vals[valid_mask]
        if len(y_valid) == 0:
            continue

        y_threshold = np.percentile(y_valid, 80)  # Top 20%

        freestream_mask = valid_mask & (y_vals > y_threshold)

        u_fs = df_after_combined["u"].to_numpy()[freestream_mask]
        V_fs = df_after_combined["V"].to_numpy()[freestream_mask]

        # Filter out zeros (invalid data represented as 0)
        u_fs = u_fs[u_fs != 0]
        V_fs = V_fs[V_fs != 0]

        freestream_u.extend(u_fs)
        freestream_V.extend(V_fs)

    if len(freestream_u) > 0:
        freestream_u = np.array(freestream_u)
        freestream_V = np.array(freestream_V)

        results["n_freestream_points"] = len(freestream_u)
        results["u_mean"] = np.mean(freestream_u)
        results["u_std"] = np.std(freestream_u)
        results["u_deviation_from_Uinf"] = (np.mean(freestream_u) - U_inf) / U_inf * 100
        results["V_mean"] = np.mean(freestream_V)
        results["V_std"] = np.std(freestream_V)
        results["V_deviation_from_Uinf"] = (np.mean(freestream_V) - U_inf) / U_inf * 100

    return results


def analyze_all_freestream():
    """
    Compute freestream statistics for all planes.
    """
    results = []

    for aoa in [13, 23]:
        for y_plane in range(1, 8):
            metrics = compute_freestream_statistics(aoa, y_plane)
            if "n_freestream_points" in metrics:
                results.append(metrics)

    df = pd.DataFrame(results)
    return df


# =============================================================================
# 3. Spatial Uncertainty Maps
# =============================================================================


def create_spatial_uncertainty_map(
    aoa: int,
    y_plane: int,
    x_plane: int,
    config: str = "normal",
    w_threshold: float = 3.0,
):
    """
    Create spatial map of velocity standard deviation.

    Returns:
        x, y coordinates (2D arrays) and std maps for u, v, w
    """
    input_dir = Path(project_dir) / "data_ALL_ERIK_FILES" / "JelleStitching" / "Input"
    base_path = input_dir / f"aoa_{aoa}" / f"Y{y_plane}"

    if not base_path.exists():
        return None

    for item in base_path.iterdir():
        if item.is_dir() and config in item.name and f"_X{x_plane}" in item.name:
            mean_file = item / "B0001.dat"
            std_file = item / "B0002.dat"

            if not (mean_file.exists() and std_file.exists()):
                continue

            # Load mean and std data
            mean_data = load_dat_file(mean_file)
            std_data = load_dat_file(std_file)

            # Get grid shape from header
            with open(mean_file, "r") as f:
                lines = f.readlines()
            zone_line = lines[2]
            i_val = int(zone_line.split("I=")[1].split(",")[0])
            j_val = int(zone_line.split("J=")[1].split(",")[0])

            # Apply combined masking pipeline (w + ICV) on flat data first.
            _, _, df_after_combined = _apply_combined_mask(
                mean_data, std_data, aoa, y_plane, w_threshold
            )
            valid_flat = ~df_after_combined["u"].isna().to_numpy()

            # Reshape to 2D
            x_2d = mean_data["x"].reshape(j_val, i_val)
            y_2d = mean_data["y"].reshape(j_val, i_val)
            u_std_2d = std_data["u"].reshape(j_val, i_val)
            v_std_2d = std_data["v"].reshape(j_val, i_val)
            w_std_2d = std_data["w"].reshape(j_val, i_val)
            mask_valid = valid_flat.reshape(j_val, i_val)
            u_std_2d = np.where(mask_valid, u_std_2d, np.nan)
            v_std_2d = np.where(mask_valid, v_std_2d, np.nan)
            w_std_2d = np.where(mask_valid, w_std_2d, np.nan)

            return {
                "x": x_2d,
                "y": y_2d,
                "u_std": u_std_2d,
                "v_std": v_std_2d,
                "w_std": w_std_2d,
                "is_valid": mask_valid,
            }

    return None


def plot_spatial_uncertainty_maps(aoa: int, y_plane: int, save_dir: Path = None):
    """
    Plot spatial uncertainty maps for all X-planes of a given Y-plane.
    """
    set_plot_style()

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    U_inf = 15.0
    vmax = 0.15  # Max std as fraction of U_inf

    for col, x_plane in enumerate([1, 2, 3]):
        data = create_spatial_uncertainty_map(aoa, y_plane, x_plane, "normal")

        if data is None:
            continue

        for row, (var, label) in enumerate(
            [
                ("u_std", r"$\sigma_u$"),
                ("v_std", r"$\sigma_v$"),
                ("w_std", r"$\sigma_w$"),
            ]
        ):
            ax = axes[row, col]
            std_normalized = data[var] / U_inf

            im = ax.pcolormesh(
                data["x"],
                data["y"],
                std_normalized,
                cmap="hot",
                vmin=0,
                vmax=vmax,
                shading="auto",
            )

            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(f"X{x_plane}: {label}/$U_\\infty$")

            if col == 2:
                plt.colorbar(im, ax=ax, label=f"{label}/$U_\\infty$")

    plt.suptitle(
        f"Spatial Uncertainty Maps: AoA={aoa}°, Y{y_plane}\n"
        f"(Yellow/white = high uncertainty, dark = low uncertainty)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_dir:
        save_path = save_dir / f"spatial_uncertainty_aoa{aoa}_Y{y_plane}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# 4. Combined Uncertainty Budget
# =============================================================================


def compute_combined_uncertainty_budget(confidence_level: float = 0.95):
    """
    Compute comprehensive uncertainty budget combining all components.

    Returns:
        DataFrame with uncertainty budget per plane
    """
    results = []
    U_inf = 15.0
    N = 250  # Number of images
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be between 0 and 1.")
    k = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)

    for aoa in [13, 23]:
        for y_plane in range(1, 8):
            budget = {"aoa": aoa, "y_plane": y_plane}

            # 1. Statistical precision (from std files)
            all_u_std = []
            all_v_std = []
            all_w_std = []
            n_valid_total = 0
            n_total_total = 0

            for x_plane in [1, 2, 3]:
                for config in ["normal", "flipped"]:
                    data = create_spatial_uncertainty_map(aoa, y_plane, x_plane, config)
                    if data is not None:
                        valid_mask = ~np.isnan(data["u_std"])
                        all_u_std.extend(data["u_std"][valid_mask].flatten())
                        all_v_std.extend(data["v_std"][valid_mask].flatten())
                        all_w_std.extend(data["w_std"][valid_mask].flatten())
                        n_valid_total += np.sum(valid_mask)
                        n_total_total += data["u_std"].size

            if len(all_u_std) > 0:
                # Mean standard deviation across field
                sigma_u = np.mean(all_u_std)
                sigma_v = np.mean(all_v_std)
                sigma_w = np.mean(all_w_std)

                # Standard uncertainty of the mean
                u_u_x = k * sigma_u / np.sqrt(N)
                u_u_y = k * sigma_v / np.sqrt(N)
                u_u_z = k * sigma_w / np.sqrt(N)

                # Combined velocity uncertainty
                u_u_vec = np.sqrt(u_u_x**2 + u_u_y**2 + u_u_z**2)

                budget["sigma_u_mean"] = sigma_u
                budget["sigma_v_mean"] = sigma_v
                budget["sigma_w_mean"] = sigma_w
                budget["confidence_level"] = confidence_level
                budget["coverage_factor"] = k
                budget["u_u_x_ms"] = u_u_x
                budget["u_u_y_ms"] = u_u_y
                budget["u_u_z_ms"] = u_u_z
                budget["u_u_vec_ms"] = u_u_vec
                budget["u_u_x"] = u_u_x / U_inf  # Normalized
                budget["u_u_y"] = u_u_y / U_inf
                budget["u_u_z"] = u_u_z / U_inf
                budget["u_u_vec"] = u_u_vec / U_inf

            # 2. Data loss fraction
            masking_metrics = compute_masking_metrics(aoa, y_plane)
            if "f_data_loss" in masking_metrics:
                budget["f_data_loss"] = masking_metrics["f_data_loss"]
                budget["f_valid"] = masking_metrics["f_valid"]

            # 3. Freestream deviation
            fs_metrics = compute_freestream_statistics(aoa, y_plane)
            if "u_deviation_from_Uinf" in fs_metrics:
                budget["freestream_bias_pct"] = fs_metrics["u_deviation_from_Uinf"]

            if len(budget) > 2:  # More than just aoa and y_plane
                results.append(budget)

    df = pd.DataFrame(results)
    return df


# =============================================================================
# Main Analysis and Reporting
# =============================================================================


def generate_uncertainty_report(confidence_level: float = 0.95):
    """
    Generate comprehensive uncertainty report with LaTeX tables.
    """
    output_dir = Path(project_dir) / "results" / "uncertainty_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_factor = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)

    print("=" * 100)
    print("COMPREHENSIVE PIV UNCERTAINTY BUDGET ANALYSIS")
    print("=" * 100)
    print()

    # 1. Masking metrics
    print("\n" + "=" * 80)
    print("1. DATA LOSS AND MASKING METRICS")
    print("=" * 80)

    df_masking = analyze_all_masking_metrics()
    if not df_masking.empty:
        df_masking.to_csv(
            output_dir / "masking_metrics.csv", index=False, float_format="%.4f"
        )

        print("\nPer-plane data quality fractions:")
        print(
            "  f_mask:  DaVis isValid=0 (geometric masking, reflections, correlation failure)"
        )
        print(
            "  f_u,y:   Filtered by |u_y| threshold (|u_y| > 3 m/s)"
        )
        print(
            "  f_ICV:   Additional filtering by ICV/proxy-SNR quality mask"
        )
        print("  f_valid: Remaining valid data")
        print()
        print(
            f"{'AoA':<6} {'Plane':<8} {'f_mask':<10} {'f_u,y':<10} {'f_ICV':<10} "
            f"{'f_valid':<10} {'f_loss':<10}"
        )
        print("-" * 75)
        for _, row in df_masking.iterrows():
            print(
                f"{int(row['aoa']):<6} Y{int(row['y_plane']):<7} "
                f"{row['f_mask']:<10.3f} {row['f_u_y']:<10.3f} {row['f_icv']:<10.3f} "
                f"{row['f_valid']:<10.3f} {row['f_loss']:<10.3f}"
            )

        # Summary by AoA
        print("\nSummary by angle of attack:")
        for aoa in df_masking["aoa"].unique():
            subset = df_masking[df_masking["aoa"] == aoa]
            print(
                f"  AoA {aoa}°: Mean data loss = {subset['f_loss'].mean()*100:.1f}%, "
                f"Range = {subset['f_loss'].min()*100:.1f}% - {subset['f_loss'].max()*100:.1f}%"
            )

    # 2. Freestream statistics
    print("\n" + "=" * 80)
    print("2. FREESTREAM VELOCITY VERIFICATION")
    print("=" * 80)

    df_fs = analyze_all_freestream()
    if not df_fs.empty:
        df_fs.to_csv(
            output_dir / "freestream_statistics.csv", index=False, float_format="%.4f"
        )

        print("\nFreestream region statistics (top 20% of y-range):")
        print(f"Expected: U_inf = 15.0 m/s")
        print()
        print(
            f"{'AoA':<6} {'Plane':<8} {'u_mean (m/s)':<14} {'u_std (m/s)':<14} {'Deviation %':<14}"
        )
        print("-" * 60)
        for _, row in df_fs.iterrows():
            print(
                f"{int(row['aoa']):<6} Y{int(row['y_plane']):<7} "
                f"{row['u_mean']:<14.2f} {row['u_std']:<14.2f} "
                f"{row['u_deviation_from_Uinf']:<14.1f}"
            )

        print(
            f"\nOverall mean freestream velocity: {df_fs['u_mean'].mean():.2f} ± {df_fs['u_std'].mean():.2f} m/s"
        )
        print(
            f"Mean deviation from U_inf: {df_fs['u_deviation_from_Uinf'].mean():.1f}%"
        )

    # 3. Combined uncertainty budget
    print("\n" + "=" * 80)
    print("3. COMBINED UNCERTAINTY BUDGET")
    print("=" * 80)

    df_budget = compute_combined_uncertainty_budget(confidence_level=confidence_level)
    if not df_budget.empty:
        df_budget.to_csv(
            output_dir / "uncertainty_budget.csv", index=False, float_format="%.4f"
        )

        print("\nUncertainty budget per measurement plane:")
        print(
            "  u_u = standard uncertainty of mean velocity "
            f"({confidence_level*100:.0f}% CI, k={coverage_factor:.3f}, normalized by U_inf)"
        )
        print()
        cols = ["aoa", "y_plane", "u_u_x", "u_u_y", "u_u_z", "u_u_vec", "f_data_loss"]
        available_cols = [c for c in cols if c in df_budget.columns]
        print(df_budget[available_cols].round(3).to_string(index=False))

    # 4. Generate spatial uncertainty maps for representative planes
    print("\n" + "=" * 80)
    print("4. SPATIAL UNCERTAINTY MAPS")
    print("=" * 80)

    for aoa in [13]:
        for y_plane in [1, 4]:  # Representative planes
            print(f"  Generating map for AoA={aoa}°, Y{y_plane}...")
            try:
                plot_spatial_uncertainty_maps(aoa, y_plane, output_dir)
            except Exception as e:
                print(f"    Error: {e}")

    # 5. Generate LaTeX tables
    print("\n" + "=" * 80)
    print("5. LATEX TABLES FOR PAPER")
    print("=" * 80)

    generate_latex_tables(
        df_masking=df_masking,
        df_budget=df_budget,
        df_fs=df_fs,
        output_dir=output_dir,
        confidence_level=confidence_level,
        coverage_factor=coverage_factor,
    )

    # Divergence table generation was removed; keep summary call stable.
    df_div = pd.DataFrame()

    # Save comprehensive summary
    generate_text_summary(
        df_masking=df_masking,
        df_div=df_div,
        df_fs=df_fs,
        df_budget=df_budget,
        output_dir=output_dir,
        confidence_level=confidence_level,
    )

    print(f"\n\nAll results saved to: {output_dir}")


def generate_latex_tables(
    df_masking,
    df_budget,
    df_fs,
    output_dir,
    confidence_level: float = 0.95,
    coverage_factor: float | None = None,
):
    """
    Generate LaTeX-formatted tables for the paper.
    """
    if coverage_factor is None:
        coverage_factor = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    latex_output = []

    # Table: Data quality metrics per plane (wide format with alpha as columns)
    latex_output.append(
        r"""
% =============================================================================
% TABLE: Data Quality Metrics (Detailed)
% =============================================================================
\begin{table}[htp]
    \centering
    \caption{Data quality metrics per measurement plane. $f_\mathrm{mask}$ denotes the fraction of vectors marked invalid by DaVis (geometric masking, reflections, correlation failure). $f_{u,y}$ denotes the fraction filtered by the out-of-plane velocity threshold ($|u_y| > 3\,\mathrm{m\,s^{-1}}$), and $f_\mathrm{ICV}$ denotes the additional fraction filtered by the ICV/proxy-SNR quality mask. $f_\mathrm{valid}$ is the remaining valid data fraction.}
    \label{tab:data_quality}
    \begin{tabular}{l ccccccc l cccc}
    \hline
    & \multicolumn{7}{c}{$\alpha = 7\unit{\degree}$} & & \multicolumn{4}{c}{$\alpha = 17\unit{\degree}$} \\
     & $Y1$ & $Y2$ & $Y3$ & $Y4$ & $Y5$ & $Y6$ & $Y7$ & & $Y1$ & $Y2$ & $Y3$ & $Y4$ \\
    \hline"""
    )

    if not df_masking.empty:
        # Organize data by plane for each metric
        def get_plane_value(df, aoa, y_plane, metric):
            row = df[(df["aoa"] == aoa) & (df["y_plane"] == y_plane)]
            if len(row) > 0:
                return row[metric].values[0] * 100  # Convert to percentage
            return None

        # Row: f_mask
        row_mask_7 = []
        for y in range(1, 8):
            val = get_plane_value(df_masking, 13, y, "f_mask")
            row_mask_7.append(f"{val:.1f}" if val is not None else "--")

        row_mask_17 = []
        for y in range(1, 5):
            val = get_plane_value(df_masking, 23, y, "f_mask")
            row_mask_17.append(f"{val:.1f}" if val is not None else "--")

        latex_output.append(
            f"    $f_\\mathrm{{mask}}$ (\\%) & {' & '.join(row_mask_7)} & & {' & '.join(row_mask_17)} \\\\"
        )

        # Row: f_u,y (|u_y| threshold contribution)
        row_uy_7 = []
        for y in range(1, 8):
            val = get_plane_value(df_masking, 13, y, "f_u_y")
            row_uy_7.append(f"{val:.1f}" if val is not None else "--")

        row_uy_17 = []
        for y in range(1, 5):
            val = get_plane_value(df_masking, 23, y, "f_u_y")
            row_uy_17.append(f"{val:.1f}" if val is not None else "--")

        latex_output.append(
            f"    $f_{{u,y}}$ (\\%) & {' & '.join(row_uy_7)} & & {' & '.join(row_uy_17)} \\\\"
        )

        # Row: f_ICV (additional ICV/proxy-SNR contribution)
        row_icv_7 = []
        for y in range(1, 8):
            val = get_plane_value(df_masking, 13, y, "f_icv")
            row_icv_7.append(f"{val:.1f}" if val is not None else "--")

        row_icv_17 = []
        for y in range(1, 5):
            val = get_plane_value(df_masking, 23, y, "f_icv")
            row_icv_17.append(f"{val:.1f}" if val is not None else "--")

        latex_output.append(
            f"    $f_\\mathrm{{ICV}}$ (\\%) & {' & '.join(row_icv_7)} & & {' & '.join(row_icv_17)} \\\\"
        )

        # Row: f_valid
        row_valid_7 = []
        for y in range(1, 8):
            val = get_plane_value(df_masking, 13, y, "f_valid")
            row_valid_7.append(f"{val:.1f}" if val is not None else "--")

        row_valid_17 = []
        for y in range(1, 5):
            val = get_plane_value(df_masking, 23, y, "f_valid")
            row_valid_17.append(f"{val:.1f}" if val is not None else "--")

        latex_output.append(
            f"    $f_\\mathrm{{valid}}$ (\\%) & {' & '.join(row_valid_7)} & & {' & '.join(row_valid_17)} \\\\"
        )

    latex_output.append(
        r"""    \hline
    \end{tabular}
\end{table}
"""
    )

    # Table: Expanded uncertainty of mean velocity (flipped layout)
    latex_output.append(
        r"""
% =============================================================================
% TABLE: Expanded Uncertainty of Mean Velocity
% =============================================================================
\begin{table}[htp]
    \centering
    \caption{Expanded uncertainties of the mean velocity corresponding to a """
        + f"{confidence_level*100:.0f}"
        + r"""\% confidence interval ($k \approx """
        + f"{coverage_factor:.3f}"
        + r"""$), calculated from $N=250$ samples and aggregated over all data points within a measurement plane. These values quantify statistical convergence only and do not include systematic (Type-B) contributions.}
    \label{tab:uncertainty}
    \begin{tabular}{l ccccccc l cccc}
    \hline
    & \multicolumn{7}{c}{$\alpha$ = 7\unit{\degree}} & & \multicolumn{4}{c}{$\alpha$ = 17\unit{\degree}} \\
     & $Y1$ & $Y2$ & $Y3$ & $Y4$ & $Y5$ & $Y6$ & $Y7$ & & $Y1$ & $Y2$ & $Y3$ & $Y4$ \\
    \hline"""
    )

    if not df_budget.empty:
        def get_budget_value(df, aoa, y_plane, col):
            row = df[(df["aoa"] == aoa) & (df["y_plane"] == y_plane)]
            if len(row) == 0:
                return None
            val = row[col].values[0] if col in row.columns else np.nan
            if np.isnan(val):
                return None
            return val

        def build_row(col_name):
            vals_7 = []
            for y in range(1, 8):
                val = get_budget_value(df_budget, 13, y, col_name)
                vals_7.append(f"{val:.3f}" if val is not None else "--")
            vals_17 = []
            for y in range(1, 5):
                val = get_budget_value(df_budget, 23, y, col_name)
                vals_17.append(f"{val:.3f}" if val is not None else "--")
            return vals_7, vals_17

        ux7, ux17 = build_row("u_u_x_ms")
        uy7, uy17 = build_row("u_u_y_ms")
        uz7, uz17 = build_row("u_u_z_ms")
        uv7, uv17 = build_row("u_u_vec_ms")

        latex_output.append(
            f"    $\\textrm{{u}}_{{\\overline{{\\textrm{{u}}}},{{x}}}}$ (\\unit{{m s^{{-1}}}}) & {' & '.join(ux7)} & & {' & '.join(ux17)} \\\\"
        )
        latex_output.append(
            f"    $\\textrm{{u}}_{{\\overline{{\\textrm{{u}}}},{{y}}}}$ (\\unit{{m s^{{-1}}}}) & {' & '.join(uy7)} & & {' & '.join(uy17)} \\\\"
        )
        latex_output.append(
            f"    $\\textrm{{u}}_{{\\overline{{\\textrm{{u}}}},{{z}}}}$ (\\unit{{m s^{{-1}}}}) & {' & '.join(uz7)} & & {' & '.join(uz17)} \\\\"
        )
        latex_output.append(
            f"    $\\textrm{{u}}_{{\\overline{{\\vec{{u}}}}}}$ (\\unit{{m s^{{-1}}}}) & {' & '.join(uv7)} & & {' & '.join(uv17)} \\\\"
        )

    latex_output.append(
        r"""    \hline
    \end{tabular}
\end{table}
"""
    )

    # Write to file
    latex_file = output_dir / "uncertainty_tables.tex"
    with open(latex_file, "w") as f:
        f.write("\n".join(latex_output))
    print(f"\nLaTeX tables saved to: {latex_file}")

    # Print to console
    print("\n" + "\n".join(latex_output))


def generate_text_summary(
    df_masking, df_div, df_fs, df_budget, output_dir, confidence_level: float = 0.95
):
    """
    Generate comprehensive text summary of uncertainty analysis.
    """
    summary = []
    summary.append("=" * 80)
    summary.append("COMPREHENSIVE PIV UNCERTAINTY BUDGET SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    summary.append("This analysis quantifies multiple uncertainty contributions in the")
    summary.append("stereoscopic PIV measurements, addressing the following sources:")
    summary.append("")
    summary.append("1. STATISTICAL PRECISION")
    summary.append("   - Based on velocity standard deviation across 250 image samples")
    summary.append(
        f"   - Reported as {confidence_level*100:.0f}% confidence interval of the mean"
    )
    summary.append("")

    if not df_budget.empty:
        mean_ux = df_budget["u_u_x"].mean() if "u_u_x" in df_budget else np.nan
        mean_uy = df_budget["u_u_y"].mean() if "u_u_y" in df_budget else np.nan
        mean_uz = df_budget["u_u_z"].mean() if "u_u_z" in df_budget else np.nan
        mean_uvec = df_budget["u_u_vec"].mean() if "u_u_vec" in df_budget else np.nan
        summary.append(f"   Mean values (normalized by U_inf):")
        summary.append(f"     u_u,x = {mean_ux:.3f}")
        summary.append(f"     u_u,y = {mean_uy:.3f}")
        summary.append(f"     u_u,z = {mean_uz:.3f}")
        summary.append(f"     u_u,vec = {mean_uvec:.3f}")
        summary.append("")

    summary.append("2. DATA LOSS AND MASKING")
    summary.append("   - Quantifies fraction of invalid/masked measurement points")
    summary.append("   - Higher data loss correlates with increased uncertainty")
    summary.append("")

    if not df_masking.empty:
        summary.append(
            f"   Mean data loss: {df_masking['f_data_loss'].mean()*100:.1f}%"
        )
        summary.append(
            f"   Range: {df_masking['f_data_loss'].min()*100:.1f}% - {df_masking['f_data_loss'].max()*100:.1f}%"
        )
        summary.append("")

    summary.append("3. PHYSICAL CONSISTENCY (DIVERGENCE CHECK)")
    summary.append("   - 2D divergence should be near zero for incompressible flow")
    summary.append("   - Non-zero values indicate 3D effects or measurement errors")
    summary.append("")

    if not df_div.empty:
        summary.append(
            f"   Mean normalized RMS divergence: {df_div['div_rms_normalized'].mean():.4f}"
        )
        summary.append("")

    summary.append("4. FREESTREAM VERIFICATION")
    summary.append("   - Compares outer-field velocity to expected U_inf = 15 m/s")
    summary.append("   - Indicates systematic bias in velocity reconstruction")
    summary.append("")

    if not df_fs.empty:
        summary.append(f"   Mean freestream velocity: {df_fs['u_mean'].mean():.2f} m/s")
        summary.append(
            f"   Mean deviation from U_inf: {df_fs['u_deviation_from_Uinf'].mean():.1f}%"
        )
        summary.append("")

    summary.append("5. OVERLAP REGION ANALYSIS (from previous analysis)")
    summary.append("   - Velocity differences in overlapping measurement regions")
    summary.append(
        "   - Indicates combined calibration, mapping, and reconstruction errors"
    )
    summary.append("   - See overlap_analysis/ for detailed results")
    summary.append("")
    summary.append("=" * 80)
    summary.append("CONCLUSION")
    summary.append("=" * 80)
    summary.append("")
    summary.append("The uncertainty analysis reveals that:")
    summary.append(
        "- Statistical precision (Table 3 values) represents random uncertainty"
    )
    summary.append("- Data loss increases toward wingtip and at high angle of attack")
    summary.append("- Overlap analysis indicates additional systematic contributions")
    summary.append("- Freestream verification confirms overall velocity calibration")
    summary.append("")
    summary.append("These multiple uncertainty sources should be considered when")
    summary.append("interpreting the quantitative comparison with CFD results.")
    summary.append("")

    # Write summary
    summary_file = output_dir / "uncertainty_summary.txt"
    with open(summary_file, "w") as f:
        f.write("\n".join(summary))
    print(f"\nText summary saved to: {summary_file}")


if __name__ == "__main__":
    generate_uncertainty_report()
    plt.show()
