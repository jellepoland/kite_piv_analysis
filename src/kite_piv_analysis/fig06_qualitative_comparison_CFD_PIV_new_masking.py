from kite_piv_analysis.plotting import *
from kite_piv_analysis.plot_styling import set_plot_style
from matplotlib.gridspec import GridSpec
from scipy.ndimage import generic_filter
import warnings


PIV_COLUMNS_TO_NAN = [
    "u",
    "v",
    "w",
    "V",
    "dudx",
    "dudy",
    "dvdx",
    "dvdy",
    "dwdx",
    "dwdy",
    "vort_z",
]


def _count_valid_vectors(df: pd.DataFrame) -> int:
    """Count vectors considered valid for plotting diagnostics."""
    if "u" in df.columns:
        return int(df["u"].notna().sum())
    for col in ["V", "v", "w"]:
        if col in df.columns:
            return int(df[col].notna().sum())
    return int(len(df))


def _enforce_monotonic_masking(
    df_before: pd.DataFrame, df_after: pd.DataFrame
) -> pd.DataFrame:
    """
    Ensure masking can only remove data, never re-enable previously masked values.
    """
    cols = [
        col
        for col in PIV_COLUMNS_TO_NAN
        if col in df_before.columns and col in df_after.columns
    ]
    if not cols:
        return df_after
    prev_nan = df_before[cols].isna()
    out = df_after.copy()
    for col in cols:
        out.loc[prev_nan[col], col] = np.nan
    return out


def _resolve_alpha_pair_param(value, alpha, param_name: str):
    """
    Resolve scalar or 2-element [alpha6, alpha16] parameter values.

    If a 2-element list/tuple/array is provided:
    - index 0 is used for alpha = 6
    - index 1 is used for alpha = 16
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 2:
            raise ValueError(
                f"{param_name} list/tuple/array must have exactly 2 elements: [alpha6, alpha16]."
            )

        alpha_int = int(round(float(alpha)))
        if alpha_int == 6:
            return value[0]
        if alpha_int == 16:
            return value[1]

        warnings.warn(
            f"{param_name} received a 2-element alpha mapping, but alpha={alpha} is not 6 or 16. "
            "Falling back to the first element.",
            RuntimeWarning,
            stacklevel=2,
        )
        return value[0]

    return value


def _resolve_alpha_pair_components(value, alpha):
    """
    Resolve proxy SNR component config.

    Accepts either:
    - a flat component list, e.g. ["u", "v"]
    - an alpha-pair nested list, e.g. [["u", "v"], ["u", "v", "w"]]
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2 and all(
            isinstance(v, (list, tuple, np.ndarray)) for v in value
        ):
            resolved = _resolve_alpha_pair_param(value, alpha, "proxy_snr_components")
            return list(resolved)
    return value


def _nanmad(values: np.ndarray) -> float:
    """Median absolute deviation for a 1D neighbourhood (NaN-safe)."""
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    values_finite = values[finite]
    med = np.median(values_finite)
    return np.median(np.abs(values_finite - med))


def _nanmedian_no_warn(values: np.ndarray) -> float:
    """Median for a 1D neighbourhood that avoids all-NaN warnings."""
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    return float(np.median(values[finite]))


def compute_proxy_snr(df: pd.DataFrame, plot_params: dict) -> np.ndarray | None:
    """
    Compute a proxy SNR from mean/std fields.

    Proxy per component: |u|/u_std, |v|/v_std, |w|/w_std.
    Final proxy per vector is reduced across components (default: min).
    """
    csv_file_path_std = plot_params.get("csv_file_path_std")
    if csv_file_path_std is None:
        if plot_params.get("snr_mask_strict", False):
            raise ValueError(
                "Proxy SNR requested but csv_file_path_std is unavailable."
            )
        warnings.warn(
            "Proxy SNR masking skipped: std CSV path unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    df_std = pd.read_csv(csv_file_path_std)
    components = plot_params.get("proxy_snr_components", ["u", "v", "w"])
    if isinstance(components, str):
        components = [components]
    components = [c for c in components if c in df.columns and c in df_std.columns]
    if not components:
        if plot_params.get("snr_mask_strict", False):
            raise KeyError(
                "Proxy SNR requested, but no configured components exist in both mean and std data."
            )
        warnings.warn(
            "Proxy SNR masking skipped: no shared mean/std components for proxy computation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    min_std = float(plot_params.get("proxy_snr_min_std", 1e-6))
    ratio_stack = []
    for comp in components:
        mean_vals = np.abs(pd.to_numeric(df[comp], errors="coerce").to_numpy())
        std_vals = np.abs(pd.to_numeric(df_std[comp], errors="coerce").to_numpy())
        ratio = np.full(len(df), np.nan, dtype=float)
        valid = np.isfinite(mean_vals) & np.isfinite(std_vals) & (std_vals > min_std)
        ratio[valid] = mean_vals[valid] / std_vals[valid]
        ratio_stack.append(ratio)

    ratios = np.vstack(ratio_stack)
    reduction = plot_params.get("proxy_snr_reduction", "min")
    proxy = np.full(ratios.shape[1], np.nan, dtype=float)
    has_valid = np.isfinite(ratios).any(axis=0)
    if not np.any(has_valid):
        return proxy

    valid_slice = ratios[:, has_valid]
    if reduction == "max":
        proxy[has_valid] = np.nanmax(valid_slice, axis=0)
    elif reduction == "mean":
        proxy[has_valid] = np.nanmean(valid_slice, axis=0)
    elif reduction == "median":
        proxy[has_valid] = np.nanmedian(valid_slice, axis=0)
    else:
        # Conservative default: all selected components must be sufficiently clean.
        proxy[has_valid] = np.nanmin(valid_slice, axis=0)

    return proxy


def apply_snr_masking(df: pd.DataFrame, plot_params: dict) -> pd.DataFrame:
    """Apply SNR-based masking if enabled (proxy SNR by default)."""
    alpha = plot_params.get("alpha", 6)
    is_with_snr_mask = _resolve_alpha_pair_param(
        plot_params.get("is_with_snr_mask", False), alpha, "is_with_snr_mask"
    )
    if not is_with_snr_mask:
        return df

    snr_params = plot_params | {
        "snr_use_proxy": _resolve_alpha_pair_param(
            plot_params.get("snr_use_proxy", True), alpha, "snr_use_proxy"
        ),
        "snr_column_to_mask": _resolve_alpha_pair_param(
            plot_params.get("snr_column_to_mask", "snr"), alpha, "snr_column_to_mask"
        ),
        "snr_mask_lower_bound": _resolve_alpha_pair_param(
            plot_params.get("snr_mask_lower_bound", 2.0),
            alpha,
            "snr_mask_lower_bound",
        ),
        "snr_mask_upper_bound": _resolve_alpha_pair_param(
            plot_params.get("snr_mask_upper_bound", np.inf),
            alpha,
            "snr_mask_upper_bound",
        ),
        "proxy_snr_components": _resolve_alpha_pair_components(
            plot_params.get("proxy_snr_components", ["u", "v", "w"]),
            alpha,
        ),
        "proxy_snr_reduction": _resolve_alpha_pair_param(
            plot_params.get("proxy_snr_reduction", "min"),
            alpha,
            "proxy_snr_reduction",
        ),
        "proxy_snr_min_std": _resolve_alpha_pair_param(
            plot_params.get("proxy_snr_min_std", 1e-6), alpha, "proxy_snr_min_std"
        ),
        "snr_mask_nan_as_invalid": _resolve_alpha_pair_param(
            plot_params.get("snr_mask_nan_as_invalid", True),
            alpha,
            "snr_mask_nan_as_invalid",
        ),
        "snr_mask_strict": _resolve_alpha_pair_param(
            plot_params.get("snr_mask_strict", False), alpha, "snr_mask_strict"
        ),
    }

    if snr_params.get("snr_use_proxy", True):
        snr_values = compute_proxy_snr(df, snr_params)
        if snr_values is None:
            return df
    else:
        snr_column = snr_params.get("snr_column_to_mask", "snr")
        if snr_column not in df.columns:
            if snr_params.get("snr_mask_strict", False):
                raise KeyError(
                    f"SNR masking requested, but column '{snr_column}' was not found in the PIV data."
                )
            warnings.warn(
                f"SNR masking skipped: column '{snr_column}' not found in PIV data.",
                RuntimeWarning,
                stacklevel=2,
            )
            return df
        snr_values = pd.to_numeric(df[snr_column], errors="coerce").to_numpy()

    lower = snr_params.get("snr_mask_lower_bound", 2.0)
    upper = snr_params.get("snr_mask_upper_bound", np.inf)
    lower = -np.inf if lower is None else float(lower)
    upper = np.inf if upper is None else float(upper)

    mask = (snr_values < lower) | (snr_values > upper)
    if snr_params.get("snr_mask_nan_as_invalid", True):
        mask |= ~np.isfinite(snr_values)

    df_masked = df.copy()
    columns_to_nan = [col for col in PIV_COLUMNS_TO_NAN if col in df_masked.columns]
    df_masked.loc[mask, columns_to_nan] = np.nan
    return df_masked


def apply_spatial_coherence_masking(
    df: pd.DataFrame, plot_params: dict
) -> pd.DataFrame:
    """Mask vectors that deviate strongly from their local neighbourhood median."""
    if not plot_params.get("is_with_spatial_coherence_mask", False):
        return df

    columns = plot_params.get("spatial_coherence_columns", ["u", "v", "w"])
    if isinstance(columns, str):
        columns = [columns]
    columns = [col for col in columns if col in df.columns]
    if not columns:
        raise KeyError(
            "Spatial coherence masking requested, but none of the configured columns exist in the PIV data."
        )

    window_size = int(plot_params.get("spatial_median_window_size", 3))
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1

    mad_scale = float(plot_params.get("spatial_mad_scale", 3.0))
    abs_threshold = plot_params.get("spatial_abs_threshold", 0.25)
    abs_threshold = None if abs_threshold is None else float(abs_threshold)

    x_unique = df["x"].unique()
    y_unique = df["y"].unique()
    shape = (len(y_unique), len(x_unique))
    outlier_mask_2d = np.zeros(shape, dtype=bool)

    for column in columns:
        values_2d = df[column].values.reshape(shape)
        median_2d = generic_filter(
            values_2d,
            _nanmedian_no_warn,
            size=window_size,
            mode="nearest",
        )
        mad_2d = generic_filter(
            values_2d,
            _nanmad,
            size=window_size,
            mode="nearest",
        )
        robust_sigma = 1.4826 * mad_2d
        threshold_2d = mad_scale * robust_sigma
        if abs_threshold is not None:
            threshold_2d = np.maximum(threshold_2d, abs_threshold)

        deviation_2d = np.abs(values_2d - median_2d)
        column_outliers = (deviation_2d > threshold_2d) & ~np.isnan(values_2d)
        outlier_mask_2d |= column_outliers

    outlier_mask_flat = outlier_mask_2d.reshape(-1)
    columns_to_nan = [col for col in PIV_COLUMNS_TO_NAN if col in df.columns]
    df_masked = df.copy()
    df_masked.loc[outlier_mask_flat, columns_to_nan] = np.nan
    return df_masked


def apply_piv_masking(df: pd.DataFrame, plot_params: dict) -> pd.DataFrame:
    """Apply PIV masking with is_valid by default and optional |w| filtering."""
    if not plot_params.get("is_with_mask", False):
        return df

    mask_debug_print = plot_params.get("mask_debug_print", False)

    df_masked = df.copy()
    n_in = _count_valid_vectors(df_masked)

    # DaVis validity masking.
    if plot_params.get("is_with_valid_mask", True):
        valid_mask_params = plot_params | {
            "column_to_mask": "is_valid",
            "mask_lower_bound": 0.5,
            "mask_upper_bound": 1.5,
        }
        df_stage = apply_mask(df_masked, valid_mask_params)
        df_masked = _enforce_monotonic_masking(df_masked, df_stage)
        n_valid = _count_valid_vectors(df_masked)
    else:
        n_valid = n_in

    # Optional additional |w| mask.
    if plot_params.get("is_with_w_mask", False):
        w_mask_params = plot_params | {
            "column_to_mask": "w",
            "mask_lower_bound": plot_params.get("w_mask_lower_bound", -3.0),
            "mask_upper_bound": plot_params.get("w_mask_upper_bound", 3.0),
        }
        df_stage = apply_mask(df_masked, w_mask_params)
        df_masked = _enforce_monotonic_masking(df_masked, df_stage)
    n_w = _count_valid_vectors(df_masked)

    # Optional SNR threshold mask.
    df_stage = apply_snr_masking(df_masked, plot_params)
    df_masked = _enforce_monotonic_masking(df_masked, df_stage)
    n_snr = _count_valid_vectors(df_masked)

    # Optional local-median spatial coherence mask.
    df_stage = apply_spatial_coherence_masking(df_masked, plot_params)
    df_masked = _enforce_monotonic_masking(df_masked, df_stage)
    n_spatial = _count_valid_vectors(df_masked)

    if mask_debug_print:
        print(
            "Mask stack counts"
            f" | input: {n_in}"
            f" | valid: {n_valid}"
            f" | w: {n_w}"
            f" | snr: {n_snr}"
            f" | spatial: {n_spatial}"
        )

    return df_masked


def plotting_qualitative_CFD_PIV(alphas, y_nums, file_name, plot_params: dict) -> None:

    set_plot_style()

    # Set up alpha and y_num values
    # alphas = [6, 6, 6, 16]
    # y_nums = [1, 3, 4, 1]
    is_with_xlabel = False

    n_rows = len(alphas)
    n_cols = 2

    """Create a comparison of CFD and PIV data for different Y positions and alphas."""
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10.5, 13.6),  # int(20 * (n_rows / 6))),
        gridspec_kw={
            "hspace": -0.01,  # A bit more vertical space for labels
            "wspace": 0.07,
            # "height_ratios": [1, 1, 1, 1, 1, 2],
            # "hspace": 0.005,
        },
    )

    for row, (alpha, y_num) in enumerate(zip(alphas, y_nums)):
        # Update plot_params for current alpha and y_num
        current_params = plot_params.copy()
        current_params["alpha"] = alpha
        current_params["y_num"] = y_num

        alpha += 1  # corrections
        print(f"Plotting α = {alpha}, Y = {y_num}")

        # Add main figure title for first and last rows
        if row == 0 or row == (n_rows - 1):
            axes[row, 0].set_title(
                rf"CFD for $\alpha$ = {alpha}" + r"$^{\circ}$",
                fontsize=14,
                fontweight="bold",
                pad=5,
            )
            axes[row, 1].set_title(
                rf"PIV for $\alpha$ = {alpha}" + r"$^{\circ}$",
                fontsize=14,
                fontweight="bold",
                pad=5,
            )
        if row == (n_rows - 1):
            is_with_xlabel = True

        # Load and plot CFD data for this row
        current_params_cfd = current_params | {"is_CFD": True}
        df_cfd, x_mesh_cfd, y_mesh_cfd, current_params_cfd = load_data(
            current_params_cfd
        )
        current_params_cfd = plotting_on_ax(
            fig,
            axes[row, 0],
            df_cfd,
            x_mesh_cfd,
            y_mesh_cfd,
            current_params_cfd,
            is_with_xlabel=is_with_xlabel,
            is_with_ylabel=False,
        )

        # Load and plot PIV data for this row
        current_params_piv = current_params | {"is_CFD": False}
        df_piv, x_mesh_piv, y_mesh_piv, current_params_piv = load_data(
            current_params_piv
        )
        df_piv = apply_piv_masking(df_piv, current_params_piv)
        current_params_piv = current_params_piv | {"is_with_mask": False}
        current_params_piv = plotting_on_ax(
            fig,
            axes[row, 1],
            df_piv,
            x_mesh_piv,
            y_mesh_piv,
            current_params_piv,
            is_with_xlabel=is_with_xlabel,
            is_label_left=False,
        )

        # add cbar
        add_vertical_colorbar_for_row(
            fig,
            axes[row, :],
            current_params_piv,
            label=f"$Y{y_num}$\n$u$\n" + r"(ms$^{-1}$)",
            labelpad=21,
            fontsize=13,
        )

    # add a horizotnal line to each subplot at y=0.25
    for ax_row in axes:
        for ax in ax_row:
            ax.axhline(
                y=0.25,
                color="black",
                linestyle="--",
                linewidth=0.8,
                label="y = 0.25 m",
            )

    # Save the plot
    save_path = (
        Path(project_dir) / "results" / "paper_plots_21_10_2025" / f"{file_name}.pdf"
    )
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close()


def main():
    plot_params: PlotParams = {
        # Basic configuration
        "is_CFD": False,
        "y_num": 3,
        "alpha": 6,
        "project_dir": project_dir,
        "plot_type": ".pdf",
        "title": None,
        "spanwise_CFD": False,
        "is_CFD_PIV_comparison": True,
        "color_data_col_name": "V",
        "is_CFD_PIV_comparison_multicomponent_masked": False,
        "run_for_all_planes": False,
        # Plot_settings
        "xlim": (-0.2, 0.8),
        "ylim": (-0.2, 0.4),
        # Color and contour settings
        "is_with_cbar": True,
        "cbar_value_factor_of_std": 2.0,
        "min_cbar_value": None,
        "max_cbar_value": None,
        "subsample_color": 1,
        "countour_levels": 100,
        "cmap": "coolwarm",
        # Quiver settings
        "is_with_quiver": False,
        "subsample_quiver": 5,
        "u_inf": 15.0,
        # PIV specific settings
        "d_alpha_rod": 7.25,
        # Overlay settings
        "is_with_overlay": False,
        "overlay_alpha": 0.4,
        # Airfoil settings
        "is_with_airfoil": True,
        "airfoil_transparency": 1.0,
        # Raw image settings
        "subsample_factor_raw_images": 1,
        "intensity_lower_bound": 10000,
        # Boundary settings
        "is_with_bound": False,
        "drot": 0.0,
        "iP": 65,
        ##
        "ellipse_color": "red",
        "rectangle_color": "green",
        "bound_linewidth": 2.0,
        "bound_alpha": 1.0,
        # Circulation analysis
        "is_with_circulation_analysis": False,
        "rho": 1.225,
        "mu": 1.7894e-5,
        "is_with_maximim_vorticity_location_correction": True,
        "chord": 0.37,
        # Mask settings
        "is_with_mask": True,
        "is_with_valid_mask": False,
        "mask_debug_print": False,
        "column_to_mask": "is_valid",
        "mask_lower_bound": 0.5,
        "mask_upper_bound": 1.5,
        "is_with_w_mask": True,
        "w_mask_lower_bound": -3,
        "w_mask_upper_bound": 3,
        "is_with_snr_mask": [True, True],
        "snr_use_proxy": [True, True],
        "snr_column_to_mask": ["snr", "snr"],
        "snr_mask_lower_bound": [2.0, 0.8],
        "snr_mask_upper_bound": [np.inf, np.inf],
        "proxy_snr_components": [["u", "v"], ["u", "v"]],
        "proxy_snr_reduction": ["median", "mean"],
        "proxy_snr_min_std": [1e-6, 1e-6],
        "snr_mask_nan_as_invalid": [True, True],
        "snr_mask_strict": [False, False],
        "is_with_spatial_coherence_mask": False,
        "spatial_coherence_columns": ["u", "v"],
        "spatial_median_window_size": 3,
        "spatial_mad_scale": 3.0,
        "spatial_abs_threshold": 0.25,
        "normal_masked_interpolated": False,
        ## Interpolation settings
        "is_with_interpolation": False,
        "interpolation_method": "nearest",
        "rectangle_size": 0.05,
    }
    # alphas = [6, 6, 6, 16]
    # y_nums = [1, 3, 4, 1]
    alphas = [6, 16]
    y_nums = [1, 1]
    # alphas = [6, 6, 6, 6, 6, 6, 16, 16, 16, 16]
    # y_nums = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
    file_name = "fig06_qualitative_comparison_CFD_PIV_new_masking"
    plotting_qualitative_CFD_PIV(alphas, y_nums, file_name, plot_params)

    # alphas = [6, 6, 6, 6, 6, 6, 16, 16, 16, 16]
    # y_nums = [1, 3, 4, 5, 6, 7, 1, 2, 3, 4]
    # file_name = "fig15_checking_line_interference_CFD_PIV"
    # plotting_qualitative_CFD_PIV(alphas, y_nums, file_name, plot_params)


if __name__ == "__main__":
    main()
