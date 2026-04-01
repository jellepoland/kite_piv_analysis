import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from kite_piv_analysis.utils import (
    project_dir,
)
from kite_piv_analysis.calculating_circulation import calculate_circulation
from kite_piv_analysis.utils import reading_optimal_bound_placement
from kite_piv_analysis import calculating_airfoil_centre
from kite_piv_analysis.defining_bound_volume import boundary_ellipse, boundary_rectangle

# from VSM.core.WingGeometry import Wing
# from VSM.core.BodyAerodynamics import BodyAerodynamics
# from VSM.core.Solver import Solver
# from plot_styling import set_plot_style, plot_on_ax
from kite_piv_analysis import force_from_noca
from kite_piv_analysis.plotting import (
    load_data,
    find_areas_needing_interpolation,
    interpolate_missing_data,
)
from kite_piv_analysis.masking import apply_snr_masking, apply_w_masking


def _apply_fig06_masking(df: pd.DataFrame, plot_params: dict) -> pd.DataFrame:
    """
    Apply the same PIV masking pipeline/settings as fig06 before analysis.
    """
    if plot_params.get("is_CFD", False):
        return df
    if not plot_params.get("is_with_mask", False):
        return df
    df = apply_w_masking(df, plot_params)
    df = apply_snr_masking(df, plot_params)
    return df


def _required_mask_from_interpolation_zones(
    df: pd.DataFrame,
    interpolation_zones: list[dict],
) -> pd.Series:
    """Return boolean mask for the union of all interpolation-zone squares."""
    required_mask = pd.Series(False, index=df.index)
    for zone in interpolation_zones:
        x_min, x_max, y_min, y_max = zone["bounds"]
        in_zone = (
            (df["x"] >= x_min)
            & (df["x"] <= x_max)
            & (df["y"] >= y_min)
            & (df["y"] <= y_max)
        )
        required_mask |= in_zone
    return required_mask


def _compute_interpolation_metrics(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    required_mask: pd.Series,
) -> dict[str, float]:
    """
    Compute interpolation metrics on unique required points.

    Definitions:
    - required points: union of all points inside interpolation-zone squares.
    - invalid before: required points where u or v is NaN before interpolation.
    - filled successfully: invalid-before points where u and v are finite after.
    """
    required_mask = required_mask.astype(bool)
    n_required = int(required_mask.sum())
    if n_required == 0:
        return {
            "n_required_points": 0,
            "n_invalid_before_points": 0,
            "n_filled_success_points": 0,
            "perc_missing_before_required": 0.0,
            "perc_of_interpolated_points": 0.0,
            "perc_filled_of_missing": 0.0,
        }

    invalid_before_mask = required_mask & (
        df_before["u"].isna() | df_before["v"].isna()
    )
    n_invalid_before = int(invalid_before_mask.sum())

    filled_success_mask = (
        invalid_before_mask & df_after["u"].notna() & df_after["v"].notna()
    )
    n_filled_success = int(filled_success_mask.sum())

    perc_missing_before_required = 100.0 * n_invalid_before / n_required
    perc_of_interpolated_points = 100.0 * n_filled_success / n_required
    perc_filled_of_missing = (
        100.0 * n_filled_success / n_invalid_before if n_invalid_before > 0 else 0.0
    )

    return {
        "n_required_points": n_required,
        "n_invalid_before_points": n_invalid_before,
        "n_filled_success_points": n_filled_success,
        "perc_missing_before_required": perc_missing_before_required,
        "perc_of_interpolated_points": perc_of_interpolated_points,
        "perc_filled_of_missing": perc_filled_of_missing,
    }


def _quantitative_csv_path(alpha: int) -> Path:
    return (
        Path(project_dir)
        / "processed_data"
        / f"quantitative_chordwise_analysis_alpha_{alpha}_with_std.csv"
    )


def _load_existing_rows_by_y(alpha: int, y_num_list: list[int]) -> dict[int, pd.Series]:
    csv_path = _quantitative_csv_path(alpha)
    if not csv_path.exists():
        return {}

    df_existing = pd.read_csv(csv_path)
    if "y_num" not in df_existing.columns:
        return {}

    rows_by_y: dict[int, pd.Series] = {}
    df_existing = df_existing.copy()
    df_existing["y_num"] = pd.to_numeric(df_existing["y_num"], errors="coerce")
    for y_num in y_num_list:
        row = df_existing[df_existing["y_num"] == int(y_num)]
        if len(row) > 0:
            rows_by_y[int(y_num)] = row.iloc[0]
    return rows_by_y


def _merge_quantitative_rows_by_y(alpha: int, df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge newly computed rows into the quantitative cache by y_num.

    This prevents partial runs (e.g. only Y6) from overwriting the whole
    alpha-specific CSV.
    """
    if "y_num" not in df_new.columns:
        raise ValueError("Expected 'y_num' column in new quantitative results.")

    csv_path = _quantitative_csv_path(alpha)

    df_new_clean = df_new.copy()
    df_new_clean["y_num"] = pd.to_numeric(df_new_clean["y_num"], errors="coerce")
    df_new_clean = df_new_clean.dropna(subset=["y_num"])
    df_new_clean["y_num"] = df_new_clean["y_num"].astype(int)

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        if "y_num" in df_existing.columns:
            df_existing = df_existing.copy()
            df_existing["y_num"] = pd.to_numeric(df_existing["y_num"], errors="coerce")
            df_existing = df_existing.dropna(subset=["y_num"])
            df_existing["y_num"] = df_existing["y_num"].astype(int)
            combined = pd.concat([df_existing, df_new_clean], ignore_index=True, sort=False)
        else:
            combined = df_new_clean
    else:
        combined = df_new_clean

    combined = combined.drop_duplicates(subset=["y_num"], keep="last")
    combined = combined.sort_values("y_num").reset_index(drop=True)
    return combined


def _extract_cached_values(
    row: pd.Series | None,
    prefix: str,
    suffixes: list[str],
    alpha: int,
    y_num: int,
) -> list[float]:
    if row is None:
        raise ValueError(
            f"Requested cached values for alpha={alpha}, Y{y_num}, '{prefix}', "
            "but no existing row was found. Run once with full computation first."
        )

    values: list[float] = []
    for suffix in suffixes:
        col = f"{prefix}_{suffix}"
        if col not in row.index:
            raise ValueError(
                f"Requested cached column '{col}' for alpha={alpha}, Y{y_num}, "
                "but it was not found in existing quantitative CSV."
            )
        values.append(row[col])
    return values


CFD_SUFFIXES = ["gamma", "fx", "fy", "gamma_std", "fx_std", "fy_std"]
PIV_SUFFIXES = [
    "gamma",
    "fx",
    "fy",
    "gamma_std",
    "fx_std",
    "fy_std",
]

NOCA_TERM_BREAKDOWN_RAW_PATH = (
    Path(project_dir) / "processed_data" / "noca_term_breakdown_raw.csv"
)
NOCA_TERM_BREAKDOWN_AGG_PATH = (
    Path(project_dir) / "processed_data" / "noca_term_breakdown_agg.csv"
)


def _build_analysis_plot_params(alpha: int, y_num: int, is_CFD: bool) -> dict:
    params = {
        "is_CFD": is_CFD,
        "y_num": y_num,
        "alpha": alpha,
        "d_alpha_rod": 7.25,
        "project_dir": project_dir,
        "plot_type": ".pdf",
        "title": None,
        "spanwise_CFD": False,
        "rectangle_size": 0.05,
    }
    if not is_CFD:
        params.update(
            {
                # Keep masking consistent with fig06 and convergence study.
                "is_with_mask": True,
                "is_with_w_mask": True,
                "w_mask_lower_bound": -3,
                "w_mask_upper_bound": 3,
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
        )
    return params


def _noca_term_group_cols() -> list[str]:
    return [
        "alpha",
        "y_num",
        "data_type",
        "shape",
        "is_CFD",
        "n_points",
        "d_perc",
        "iP",
        "dLx_optimal",
        "dLy_optimal",
        "ref_chord",
        "rho",
        "U_inf",
        "mu",
        "is_with_smoothing",
        "is_with_vort_correction",
    ]


def _aggregate_noca_term_breakdown(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame()

    group_cols = [c for c in _noca_term_group_cols() if c in df_raw.columns]
    value_cols = [
        c
        for c in df_raw.columns
        if (
            c.startswith("Fx_")
            or c.startswith("Fy_")
            or c.startswith("Cd_")
            or c.startswith("Cl_")
        )
    ]
    if not value_cols:
        return pd.DataFrame()

    agg = (
        df_raw.groupby(group_cols, dropna=False)[value_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = [
        "_".join([p for p in col if p]).rstrip("_") if isinstance(col, tuple) else col
        for col in agg.columns
    ]

    counts = (
        df_raw.groupby(group_cols, dropna=False).size().reset_index(name="n_samples")
    )
    agg = agg.merge(counts, on=group_cols, how="left")
    agg = _ensure_rotational_total_columns(agg)
    return agg


def _ensure_rotational_total_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure rotational (3+4) aggregate columns exist for raw and aggregated tables.
    Fills from available term-3 and optional term-4 columns.
    """
    if df.empty:
        return df

    out = df.copy()
    for prefix in ["Fx", "Fy", "Cd", "Cl"]:
        suffixes = [
            suffix
            for suffix in ["", "_mean", "_std"]
            if (
                f"{prefix}_term_3{suffix}" in out.columns
                or f"{prefix}_term_4{suffix}" in out.columns
            )
        ]
        for suffix in suffixes:
            col_term3 = f"{prefix}_term_3{suffix}"
            col_term4 = f"{prefix}_term_4{suffix}"
            col_rot = f"{prefix}_rotational_total_3_4{suffix}"

            if col_rot not in out.columns:
                out[col_rot] = np.nan
            if col_term3 not in out.columns:
                continue

            term4_vals = out[col_term4] if col_term4 in out.columns else 0.0
            rot_vals = out[col_term3] + term4_vals
            out[col_rot] = out[col_rot].where(out[col_rot].notna(), rot_vals)
    return out


def _merge_write_noca_breakdown(
    df_new: pd.DataFrame,
    save_path: Path,
    key_cols: list[str],
) -> None:
    if df_new.empty:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        existing = pd.read_csv(save_path)
        combined = pd.concat([existing, df_new], ignore_index=True)
    else:
        combined = df_new.copy()

    combined = _ensure_rotational_total_columns(combined)

    valid_keys = [c for c in key_cols if c in combined.columns]
    if valid_keys:
        combined = combined.drop_duplicates(subset=valid_keys, keep="last")
    combined.to_csv(save_path, index=False)


def _save_noca_term_breakdowns(df_raw: pd.DataFrame) -> tuple[Path, Path]:
    if df_raw.empty:
        return NOCA_TERM_BREAKDOWN_RAW_PATH, NOCA_TERM_BREAKDOWN_AGG_PATH

    raw_key_cols = _noca_term_group_cols() + ["dLx_current", "dLy_current"]
    _merge_write_noca_breakdown(
        df_new=df_raw,
        save_path=NOCA_TERM_BREAKDOWN_RAW_PATH,
        key_cols=raw_key_cols,
    )

    df_agg = _aggregate_noca_term_breakdown(df_raw)
    agg_key_cols = _noca_term_group_cols()
    _merge_write_noca_breakdown(
        df_new=df_agg,
        save_path=NOCA_TERM_BREAKDOWN_AGG_PATH,
        key_cols=agg_key_cols,
    )
    return NOCA_TERM_BREAKDOWN_RAW_PATH, NOCA_TERM_BREAKDOWN_AGG_PATH


def computing_gamma_and_noca_fx_fy(
    df,
    plot_params,
    is_ellipse=True,
    n_points=10,
    d_perc=5,
    U_inf=15,
    mu=1.7894e-5,
    rho=1.2,
    drot=0,
    ref_chord=0.396,
    is_with_smoothing: bool = True,
    is_with_vort_correction: bool = True,
    max_perc_interpolated_zones=25.0,
    n_datapoints=104304,
    noca_term_rows_collector: list[dict] | None = None,
):

    alpha = plot_params["alpha"]
    y_num = plot_params["y_num"]

    # Calculate airfoil center
    x_airfoil, y_airfoil, chord = calculating_airfoil_centre.main(
        alpha, y_num, is_with_chord=True
    )
    d1centre = (x_airfoil, y_airfoil)

    # Get initial bound placement
    dLx, dLy, iP = reading_optimal_bound_placement(
        alpha, y_num, is_with_N_datapoints=True
    )

    # Initialize list to store circulation values
    gamma_list = []
    F_x_list = []
    F_y_list = []

    # Create ranges for dLx and dLy: ±5% with 10 datapoints
    dmin = 1 - (d_perc / 100)
    dmax = 1 + (d_perc / 100)
    dLx_range = np.linspace(dLx * dmin, dLx * dmax, n_points)
    dLy_range = np.linspace(dLy * dmin, dLy * dmax, n_points)

    # Nested loop to go through 10x10 options
    for current_dLx in dLx_range:
        for current_dLy in dLy_range:
            # Create a copy of the dataframe to avoid modifying the original
            current_df = df.copy()

            # Interpolation check (assuming these functions and parameters exist)
            perc_of_interpolated_points = 0
            perc_missing_before_required = 0
            perc_missing_before_global = 0
            perc_filled_of_missing = 0
            n_required_points = 0
            n_invalid_before_points = 0
            n_filled_success_points = 0

            # Check if not CFD data and interpolation is needed
            if not plot_params.get("is_CFD", False):
                # Find areas needing interpolation
                plot_params["interpolation_zones"] = find_areas_needing_interpolation(
                    current_df,
                    alpha,
                    y_num,
                    plot_params.get("rectangle_size"),
                    dLx=current_dLx,
                    dLy=current_dLy,
                )

                required_mask = _required_mask_from_interpolation_zones(
                    df,
                    plot_params["interpolation_zones"],
                )
                pre_metrics = _compute_interpolation_metrics(df, df, required_mask)
                n_required_points = int(pre_metrics["n_required_points"])
                n_invalid_before_points = int(pre_metrics["n_invalid_before_points"])
                perc_missing_before_required = float(
                    pre_metrics["perc_missing_before_required"]
                )
                # Backward-compatible alias for historical output columns.
                perc_missing_before_global = perc_missing_before_required

                # Skip criterion based on required-region missing fraction.
                if perc_missing_before_required < max_perc_interpolated_zones:
                    for interpolation_zone in plot_params["interpolation_zones"]:
                        current_df, _ = interpolate_missing_data(
                            current_df,
                            interpolation_zone,
                        )
                    post_metrics = _compute_interpolation_metrics(
                        df,
                        current_df,
                        required_mask,
                    )
                    n_filled_success_points = int(
                        post_metrics["n_filled_success_points"]
                    )
                    perc_of_interpolated_points = float(
                        post_metrics["perc_of_interpolated_points"]
                    )
                    perc_filled_of_missing = float(
                        post_metrics["perc_filled_of_missing"]
                    )
                else:
                    # Skip this iteration if too many points need interpolation
                    print(
                        f"---> SKIPPED perc_missing_before_required: {perc_missing_before_required:.2f}% "
                        f"(threshold={max_perc_interpolated_zones:.2f}%), "
                        f"dLx: {current_dLx:.2f}, dLy: {current_dLy:.2f}"
                    )
                    continue

            # Determine boundary based on shape (ellipse or rectangle)
            if is_ellipse:
                d2curve = boundary_ellipse(
                    d1centre,
                    drot,
                    current_dLx,
                    current_dLy,
                    iP,
                )
            else:
                d2curve = boundary_rectangle(
                    d1centre,
                    drot,
                    current_dLx,
                    current_dLy,
                    iP,
                )

            # Calculate circulation for current parameters
            circulation = calculate_circulation(current_df, d2curve, is_with_smoothing)
            gamma_list.append(circulation)

            # Computing NOCA
            F_x, F_y, C_l, C_d, term_breakdown = force_from_noca.main(
                current_df,
                d2curve,
                mu=mu,
                is_with_maximim_vorticity_location_correction=is_with_vort_correction,
                is_with_smoothing=is_with_smoothing,
                rho=rho,
                U_inf=U_inf,
                ref_chord=ref_chord,
                return_term_breakdown=True,
            )
            F_x_list.append(F_x)
            F_y_list.append(F_y)
            if noca_term_rows_collector is not None:
                data_type = "CFD" if plot_params.get("is_CFD", False) else "PIV"
                shape = "ellipse" if is_ellipse else "rectangle"
                noca_term_rows_collector.append(
                    {
                        "alpha": int(alpha),
                        "y_num": int(y_num),
                        "data_type": data_type,
                        "shape": shape,
                        "is_CFD": bool(plot_params.get("is_CFD", False)),
                        "n_points": int(n_points),
                        "d_perc": float(d_perc),
                        "iP": int(iP),
                        "dLx_optimal": float(dLx),
                        "dLy_optimal": float(dLy),
                        "dLx_current": float(current_dLx),
                        "dLy_current": float(current_dLy),
                        # (invalid-before and valid-after) / required.
                        "perc_interpolated_points": float(perc_of_interpolated_points),
                        "perc_missing_before_required": float(
                            perc_missing_before_required
                        ),
                        "perc_missing_before_global": float(perc_missing_before_global),
                        "perc_filled_of_missing": float(perc_filled_of_missing),
                        "n_required_points": int(n_required_points),
                        "n_invalid_before_points": int(n_invalid_before_points),
                        "n_filled_success_points": int(n_filled_success_points),
                        "ref_chord": float(ref_chord),
                        "rho": float(rho),
                        "U_inf": float(U_inf),
                        "mu": float(mu),
                        "is_with_smoothing": bool(is_with_smoothing),
                        "is_with_vort_correction": bool(is_with_vort_correction),
                        **term_breakdown,
                    }
                )
            print(
                f"C_kutta:{(circulation*rho*U_inf)/((0.5 * rho * U_inf**2 * ref_chord)):.2f}, "
                f"NOCA: C_l:{C_l:.2f}, C_d:{C_d:.2f}, "
                f"% required points filled by interpolation: {perc_of_interpolated_points:.2f}%"
            )

    # Return average circulation and forces (use lists and handle empty lists)
    if len(gamma_list) > 0:
        gamma_mean = np.mean(gamma_list)
        gamma_std = np.std(gamma_list)
    else:
        gamma_mean = np.nan
        gamma_std = np.nan

    if len(F_x_list) > 0:
        Fx_mean = np.mean(F_x_list)
        Fx_std = np.std(F_x_list)
    else:
        Fx_mean = np.nan
        Fx_std = np.nan

    if len(F_y_list) > 0:
        Fy_mean = np.mean(F_y_list)
        Fy_std = np.std(F_y_list)
    else:
        Fy_mean = np.nan
        Fy_std = np.nan

    return (gamma_mean, Fx_mean, Fy_mean, gamma_std, Fx_std, Fy_std)


def get_PIV_and_CFD_gamma_distribution_for_single_alpha(
    alpha: int = 6,
    y_num_list: list = [1],
    n_points: int = 10,
    include_cfd: bool = True,
    include_piv: bool = True,
    reuse_existing_when_skipped: bool = True,
    save_noca_terms: bool = True,
):

    Re_cfd = 1e6
    Re_piv = 3.8e5
    rho = 1.20
    ref_chord = 0.396
    U_inf = 15
    mu_cfd = (rho * ref_chord * U_inf) / Re_cfd
    mu_piv = (rho * ref_chord * U_inf) / Re_piv
    print(f"mu_cfd: {mu_cfd}, mu_piv: {mu_piv}")
    # mu_piv = 1.7894e-5

    cfd_ellipse_list = []
    cfd_rectangle_list = []
    piv_ellipse_list = []
    piv_rectangle_list = []
    noca_term_rows_all: list[dict] | None = [] if save_noca_terms else None

    rows_by_y_existing: dict[int, pd.Series] = {}
    if reuse_existing_when_skipped and ((not include_cfd) or (not include_piv)):
        rows_by_y_existing = _load_existing_rows_by_y(alpha, y_num_list)

    for y_num in y_num_list:
        print(f"\n-----> y_num: {y_num}")
        existing_row = rows_by_y_existing.get(int(y_num))

        # ## CFD
        if include_cfd:
            print(f"CFD")
            plot_params_cfd = {
                "is_CFD": True,
                "y_num": y_num,
                "alpha": alpha,
                "d_alpha_rod": 7.25,
                "project_dir": project_dir,
                "plot_type": ".pdf",
                "title": None,
                "spanwise_CFD": False,
                "rectangle_size": 0.05,
            }
            df, x_mesh, y_mesh, plot_params_cfd = load_data(plot_params_cfd)
            print(f"Computing CFD ellipse")
            (
                cfd_gamma_ellipse,
                cfd_fx_ellipse,
                cfd_fy_ellipse,
                cfd_gamma_ellipse_std,
                cfd_fx_ellipse_std,
                cfd_fy_ellipse_std,
            ) = computing_gamma_and_noca_fx_fy(
                df,
                plot_params_cfd,
                is_ellipse=True,
                mu=mu_cfd,
                n_points=n_points,
                noca_term_rows_collector=noca_term_rows_all,
            )
            print(f"Computing CFD rectangle")
            (
                cfd_gamma_rectangle,
                cfd_fx_rectangle,
                cfd_fy_rectangle,
                cfd_gamma_rectangle_std,
                cfd_fx_rectangle_std,
                cfd_fy_rectangle_std,
            ) = computing_gamma_and_noca_fx_fy(
                df,
                plot_params_cfd,
                is_ellipse=False,
                mu=mu_cfd,
                n_points=n_points,
                noca_term_rows_collector=noca_term_rows_all,
            )
            cfd_ellipse_list.append(
                [
                    cfd_gamma_ellipse,
                    cfd_fx_ellipse,
                    cfd_fy_ellipse,
                    cfd_gamma_ellipse_std,
                    cfd_fx_ellipse_std,
                    cfd_fy_ellipse_std,
                ]
            )
            cfd_rectangle_list.append(
                [
                    cfd_gamma_rectangle,
                    cfd_fx_rectangle,
                    cfd_fy_rectangle,
                    cfd_gamma_rectangle_std,
                    cfd_fx_rectangle_std,
                    cfd_fy_rectangle_std,
                ]
            )
        else:
            print("Skipping CFD computation; reusing cached CFD values.")
            cfd_ellipse_list.append(
                _extract_cached_values(
                    existing_row, "ellipse_cfd", CFD_SUFFIXES, alpha, y_num
                )
            )
            cfd_rectangle_list.append(
                _extract_cached_values(
                    existing_row, "rectangle_cfd", CFD_SUFFIXES, alpha, y_num
                )
            )

        ## PIV
        if not include_piv:
            print("Skipping PIV computation; reusing cached PIV values.")
            if y_num >= 7 and existing_row is None:
                piv_ellipse_list.append([0] * len(PIV_SUFFIXES))
                piv_rectangle_list.append([0] * len(PIV_SUFFIXES))
            else:
                piv_ellipse_list.append(
                    _extract_cached_values(
                        existing_row, "ellipse_piv", PIV_SUFFIXES, alpha, y_num
                    )
                )
                piv_rectangle_list.append(
                    _extract_cached_values(
                        existing_row, "rectangle_piv", PIV_SUFFIXES, alpha, y_num
                    )
                )
        elif y_num >= 7:
            piv_ellipse_list.append(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            piv_rectangle_list.append(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
        else:
            print(f"PIV")
            plot_params_piv = {
                "is_CFD": False,
                "y_num": y_num,
                "alpha": alpha,
                "d_alpha_rod": 7.25,
                "project_dir": project_dir,
                "plot_type": ".pdf",
                "title": None,
                "spanwise_CFD": False,
                "rectangle_size": 0.05,
                # Keep masking consistent with fig06_qualitative_comparison_CFD_PIV
                "is_with_mask": True,
                "is_with_w_mask": True,
                "w_mask_lower_bound": -3,
                "w_mask_upper_bound": 3,
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
            df, x_mesh, y_mesh, plot_params_piv = load_data(plot_params_piv)

            # Apply masking before circulation/NOCA analyses.
            df = _apply_fig06_masking(df, plot_params_piv)
            ## Normal
            print(f"Computing PIV ellipse")
            (
                piv_gamma_ellipse,
                piv_fx_ellipse,
                piv_fy_ellipse,
                piv_gamma_ellipse_std,
                piv_fx_ellipse_std,
                piv_fy_ellipse_std,
            ) = computing_gamma_and_noca_fx_fy(
                df,
                plot_params_piv,
                is_ellipse=True,
                mu=mu_piv,
                n_points=n_points,
                noca_term_rows_collector=noca_term_rows_all,
            )
            print(f"Computing PIV rectangle")
            (
                piv_gamma_rectangle,
                piv_fx_rectangle,
                piv_fy_rectangle,
                piv_gamma_rectangle_std,
                piv_fx_rectangle_std,
                piv_fy_rectangle_std,
            ) = computing_gamma_and_noca_fx_fy(
                df,
                plot_params_piv,
                is_ellipse=False,
                mu=mu_piv,
                n_points=n_points,
                noca_term_rows_collector=noca_term_rows_all,
            )

            piv_ellipse_list.append(
                [
                    piv_gamma_ellipse,
                    piv_fx_ellipse,
                    piv_fy_ellipse,
                    piv_gamma_ellipse_std,
                    piv_fx_ellipse_std,
                    piv_fy_ellipse_std,
                ]
            )
            piv_rectangle_list.append(
                [
                    piv_gamma_rectangle,
                    piv_fx_rectangle,
                    piv_fy_rectangle,
                    piv_gamma_rectangle_std,
                    piv_fx_rectangle_std,
                    piv_fy_rectangle_std,
                ]
            )

    if noca_term_rows_all:
        raw_path, agg_path = _save_noca_term_breakdowns(
            pd.DataFrame(noca_term_rows_all)
        )
        print(
            "Saved NOCA term breakdown caches: " f"raw='{raw_path}', agg='{agg_path}'"
        )

    return (
        cfd_ellipse_list,
        cfd_rectangle_list,
        piv_ellipse_list,
        piv_rectangle_list,
    )


def save_results_single_alpha(
    alpha,
    y_num_list,
    n_points=10,
    include_cfd: bool = True,
    include_piv: bool = True,
    reuse_existing_when_skipped: bool = True,
    save_noca_terms: bool = True,
):
    print(
        f"\n\n=====> Saving results for alpha: {alpha}\n, for y_num_list: {y_num_list}"
    )
    ## acquiring data
    cfd_ellipse, cfd_rectangle, piv_ellipse, piv_rectangle = (
        get_PIV_and_CFD_gamma_distribution_for_single_alpha(
            alpha,
            y_num_list,
            n_points=n_points,
            include_cfd=include_cfd,
            include_piv=include_piv,
            reuse_existing_when_skipped=reuse_existing_when_skipped,
            save_noca_terms=save_noca_terms,
        )
    )
    df = pd.DataFrame(
        {
            "y_num": y_num_list,
            "ellipse_cfd_gamma": [x[0] for x in cfd_ellipse],
            "ellipse_cfd_fx": [x[1] for x in cfd_ellipse],
            "ellipse_cfd_fy": [x[2] for x in cfd_ellipse],
            "ellipse_cfd_gamma_std": [x[3] for x in cfd_ellipse],
            "ellipse_cfd_fx_std": [x[4] for x in cfd_ellipse],
            "ellipse_cfd_fy_std": [x[5] for x in cfd_ellipse],
            "rectangle_cfd_gamma": [x[0] for x in cfd_rectangle],
            "rectangle_cfd_fx": [x[1] for x in cfd_rectangle],
            "rectangle_cfd_fy": [x[2] for x in cfd_rectangle],
            "rectangle_cfd_gamma_std": [x[3] for x in cfd_rectangle],
            "rectangle_cfd_fx_std": [x[4] for x in cfd_rectangle],
            "rectangle_cfd_fy_std": [x[5] for x in cfd_rectangle],
            "ellipse_piv_gamma": [x[0] for x in piv_ellipse],
            "ellipse_piv_fx": [x[1] for x in piv_ellipse],
            "ellipse_piv_fy": [x[2] for x in piv_ellipse],
            "ellipse_piv_gamma_std": [x[3] for x in piv_ellipse],
            "ellipse_piv_fx_std": [x[4] for x in piv_ellipse],
            "ellipse_piv_fy_std": [x[5] for x in piv_ellipse],
            "rectangle_piv_gamma": [x[0] for x in piv_rectangle],
            "rectangle_piv_fx": [x[1] for x in piv_rectangle],
            "rectangle_piv_fy": [x[2] for x in piv_rectangle],
            "rectangle_piv_gamma_std": [x[3] for x in piv_rectangle],
            "rectangle_piv_fx_std": [x[4] for x in piv_rectangle],
            "rectangle_piv_fy_std": [x[5] for x in piv_rectangle],
        }
    )
    csv_path = _quantitative_csv_path(alpha)
    df_merged = _merge_quantitative_rows_by_y(alpha, df)
    df_merged.to_csv(csv_path, index=False)
    print(
        f"Saved quantitative cache '{csv_path.name}': "
        f"updated {len(df)} row(s), total {len(df_merged)} row(s)."
    )
    return df


def saving_results(n_points=10):
    alpha = 6
    y_num_list = [1, 2, 3, 4, 5, 6, 7]
    save_results_single_alpha(alpha, y_num_list, n_points=n_points)

    alpha = 16
    y_num_list = [1]
    save_results_single_alpha(alpha, y_num_list, n_points=n_points)


def main(n_points=10):

    # takes very long time
    print(f"---> Starting calculations and saving results")
    print(f"---> This may take extremely long")
    saving_results(n_points=n_points)

    ## reading results
    alpha = 6
    df_alpha_6 = pd.read_csv(_quantitative_csv_path(alpha))
    alpha = 16
    df_alpha_16 = pd.read_csv(_quantitative_csv_path(alpha))

    ## printing results
    rho = 1.20
    U_inf = 15
    ref_chord = 0.396
    q_infc = 0.5 * rho * (U_inf**2) * ref_chord
    print(f"rho: {rho}, U_inf: {U_inf}, ref_chord: {ref_chord} --> q_infc: {q_infc}")
    print(f"\nalpha: 6")
    for y_num in [1, 2, 3, 4, 5]:
        print(f"\n-----> y_num: {y_num}")
        for name_prefix in [
            "ellipse_cfd",
            "ellipse_piv",
            "rectangle_cfd",
            "rectangle_piv",
        ]:
            fx = df_alpha_6[df_alpha_6["y_num"] == y_num][f"{name_prefix}_fx"].values[0]
            fy = df_alpha_6[df_alpha_6["y_num"] == y_num][f"{name_prefix}_fy"].values[0]
            gamma = df_alpha_6[df_alpha_6["y_num"] == y_num][
                f"{name_prefix}_gamma"
            ].values[0]
            print(
                f"{name_prefix} -- cl: {fy/q_infc:.2f}, cd: {fx/q_infc:.2f}, c_kutta: {gamma*rho*U_inf/q_infc:.2f}"
            )

    print(f"\nalpha: 16")
    for y_num in [1]:
        print(f"\n-----> y_num: {y_num}")
        for name_prefix in [
            "ellipse_cfd",
            "ellipse_piv",
            "rectangle_cfd",
            "rectangle_piv",
        ]:
            fx = df_alpha_16[df_alpha_16["y_num"] == y_num][f"{name_prefix}_fx"].values[
                0
            ]
            fy = df_alpha_16[df_alpha_16["y_num"] == y_num][f"{name_prefix}_fy"].values[
                0
            ]
            gamma = df_alpha_16[df_alpha_16["y_num"] == y_num][
                f"{name_prefix}_gamma"
            ].values[0]
            print(
                f"{name_prefix} -- cl: {fy/q_infc:.2f}, cd: {fx/q_infc:.2f}, c_kutta: {gamma*rho*U_inf/q_infc:.2f}"
            )


if __name__ == "__main__":
    # main()

    # alpha = 6
    # y_num_list = [1, 2, 3, 4, 5, 6, 7]
    # save_results_single_alpha(alpha, y_num_list)

    alpha = 6
    y_num_list = [7]
    save_results_single_alpha(alpha, y_num_list)

    # Re_cfd = 1e6
    # Re_piv = 3.8e5
    # rho = 1.20
    # ref_chord = 0.396
    # U_inf = 15
    # mu_cfd = (rho * ref_chord * U_inf) / Re_cfd
    # mu_piv = (rho * ref_chord * U_inf) / Re_piv
    # print(f"mu_cfd: {mu_cfd}, mu_piv: {mu_piv}")
    # breakpoint()
