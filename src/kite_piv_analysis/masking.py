import warnings

import numpy as np
import pandas as pd


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


def compute_proxy_snr(df: pd.DataFrame, plot_params: dict) -> np.ndarray | None:
    """
    Compute a proxy SNR from mean/std fields.

    Proxy per component: |u|/u_std, |v|/v_std, |w|/w_std.
    Final proxy per vector is reduced across components (default: min).
    """
    df_std = plot_params.get("df_std")
    if df_std is not None:
        if not isinstance(df_std, pd.DataFrame):
            df_std = pd.DataFrame(df_std)
        if len(df_std) != len(df):
            if plot_params.get("snr_mask_strict", False):
                raise ValueError(
                    "Proxy SNR requested with in-memory std data, but length does not match mean data."
                )
            warnings.warn(
                "Proxy SNR masking skipped: in-memory std data length mismatch.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
    else:
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


def _fit_rear_airfoil_line(
    plot_params: dict, rear_fraction: float = 0.5
) -> tuple[float, float] | None:
    """
    Fit y = m*x + b through the rear fraction of plotted airfoil surface points.
    """
    try:
        # Local import keeps this helper module lightweight for non-airfoil workflows.
        from kite_piv_analysis.plotting import plot_airfoil

        airfoil_x, airfoil_y = plot_airfoil(
            None, plot_params, is_return_surface_points=True
        )
    except Exception as exc:
        warnings.warn(
            f"Rear-airfoil line fit skipped: unable to get airfoil points ({exc}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    x_vals = pd.to_numeric(np.asarray(airfoil_x), errors="coerce").astype(float)
    y_vals = pd.to_numeric(np.asarray(airfoil_y), errors="coerce").astype(float)
    finite = np.isfinite(x_vals) & np.isfinite(y_vals)
    if finite.sum() < 2:
        return None

    x_vals = x_vals[finite]
    y_vals = y_vals[finite]

    rear_fraction = float(rear_fraction)
    rear_fraction = min(max(rear_fraction, 0.05), 1.0)
    x_threshold = np.quantile(x_vals, 1.0 - rear_fraction)
    rear_mask = x_vals >= x_threshold
    if rear_mask.sum() < 2:
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
        slope, intercept = np.polyfit(x_vals[rear_mask], y_vals[rear_mask], 1)

    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    return float(slope), float(intercept)


def _snr_region_below_rear_airfoil_line(
    df: pd.DataFrame, plot_params: dict, rear_fraction: float = 0.5
) -> np.ndarray | None:
    """
    Build a boolean mask for points below a fitted rear-airfoil diagonal line.
    """
    line = _fit_rear_airfoil_line(plot_params, rear_fraction=rear_fraction)
    if line is None:
        return None

    slope, intercept = line
    x_vals = pd.to_numeric(df["x"], errors="coerce").to_numpy()
    y_vals = pd.to_numeric(df["y"], errors="coerce").to_numpy()
    return np.isfinite(x_vals) & np.isfinite(y_vals) & (
        y_vals <= (slope * x_vals + intercept)
    )


def apply_w_masking(df: pd.DataFrame, plot_params: dict) -> pd.DataFrame:
    """
    Apply |w|-based masking equivalent to fig06_qualitative..._new_masking.

    Required columns:
    - target: "w"
    - masked outputs: subset of PIV_COLUMNS_TO_NAN present in df
    """
    alpha = plot_params.get("alpha", 6)
    is_with_w_mask = _resolve_alpha_pair_param(
        plot_params.get("is_with_w_mask", False), alpha, "is_with_w_mask"
    )
    if not is_with_w_mask:
        return df

    if "w" not in df.columns:
        if plot_params.get("w_mask_strict", False):
            raise KeyError("W masking requested, but column 'w' was not found.")
        warnings.warn(
            "W masking skipped: column 'w' not found in PIV data.",
            RuntimeWarning,
            stacklevel=2,
        )
        return df

    lower = _resolve_alpha_pair_param(
        plot_params.get("w_mask_lower_bound", -3.0), alpha, "w_mask_lower_bound"
    )
    upper = _resolve_alpha_pair_param(
        plot_params.get("w_mask_upper_bound", 3.0), alpha, "w_mask_upper_bound"
    )
    lower = -np.inf if lower is None else float(lower)
    upper = np.inf if upper is None else float(upper)

    w_values = pd.to_numeric(df["w"], errors="coerce").to_numpy()
    mask = (w_values < lower) | (w_values > upper)

    df_masked = df.copy()
    columns_to_nan = [col for col in PIV_COLUMNS_TO_NAN if col in df_masked.columns]
    df_masked.loc[mask, columns_to_nan] = np.nan
    return df_masked


def apply_snr_masking(df: pd.DataFrame, plot_params: dict) -> pd.DataFrame:
    """
    Apply SNR-based masking equivalent to fig06_qualitative..._new_masking.

    Supports:
    - native SNR column or proxy SNR from mean/std
    - alpha-pair config [alpha6, alpha16]
    - optional region gating below a rear-airfoil fitted diagonal line
    """
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
        "snr_apply_only_below_rear_airfoil_line": _resolve_alpha_pair_param(
            plot_params.get("snr_apply_only_below_rear_airfoil_line", [False, True]),
            alpha,
            "snr_apply_only_below_rear_airfoil_line",
        ),
        "snr_rear_airfoil_fraction": _resolve_alpha_pair_param(
            plot_params.get("snr_rear_airfoil_fraction", [0.5, 0.5]),
            alpha,
            "snr_rear_airfoil_fraction",
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

    if snr_params.get("snr_apply_only_below_rear_airfoil_line", False):
        below_line = _snr_region_below_rear_airfoil_line(
            df,
            snr_params,
            rear_fraction=snr_params.get("snr_rear_airfoil_fraction", 0.5),
        )
        if below_line is None:
            if snr_params.get("snr_mask_strict", False):
                raise ValueError(
                    "SNR rear-airfoil line masking requested, but line fitting failed."
                )
            warnings.warn(
                "SNR rear-airfoil line masking skipped: unable to fit rear-airfoil line.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            # Restrict SNR filtering to the region below the fitted rear-airfoil line.
            mask &= below_line

    df_masked = df.copy()
    columns_to_nan = [col for col in PIV_COLUMNS_TO_NAN if col in df_masked.columns]
    df_masked.loc[mask, columns_to_nan] = np.nan
    return df_masked
