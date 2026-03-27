import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
from kite_piv_analysis.utils import project_dir
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

BERNOULLI_CACHE_PATH = (
    Path(project_dir) / "processed_data" / "bernoulli_contour_force_coefficients.csv"
)
DEFAULT_BERNOULLI_ALPHA_TO_Y_NUMS = {
    6: [1, 2, 3, 4, 5],
    16: [1],
}


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


def _prepare_closed_curve(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError(f"Expected contour shape (N,2), got {curve.shape}")
    if len(curve) < 3:
        raise ValueError("Contour needs at least 3 points.")
    if not np.allclose(curve[0], curve[-1], rtol=0, atol=1e-12):
        curve = np.vstack([curve, curve[0]])
    return curve


def _signed_area(closed_curve: np.ndarray) -> float:
    x = closed_curve[:, 0]
    y = closed_curve[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def _integrate_bernoulli_contour_forces(
    df: pd.DataFrame,
    contour: np.ndarray,
    U_inf: float,
    rho: float,
    q_infc: float,
    chord_for_vorticity_scale: float,
) -> dict:
    closed = _prepare_closed_curve(contour)
    seg = closed[1:] - closed[:-1]
    ds = np.linalg.norm(seg, axis=1)
    mid = 0.5 * (closed[1:] + closed[:-1])
    valid_seg = ds > 1e-12
    n_total_segments = int(np.count_nonzero(valid_seg))
    if n_total_segments == 0:
        raise ValueError("All contour segments have zero length.")

    seg = seg[valid_seg]
    ds = ds[valid_seg]
    mid = mid[valid_seg]

    area = _signed_area(closed)
    is_ccw = area > 0
    if is_ccw:
        nx = seg[:, 1] / ds
        ny = -seg[:, 0] / ds
    else:
        nx = -seg[:, 1] / ds
        ny = seg[:, 0] / ds

    xy = np.column_stack((df["x"].values, df["y"].values))
    u_vals = df["u"].values
    v_vals = df["v"].values
    u_mask = np.isfinite(u_vals)
    v_mask = np.isfinite(v_vals)
    if np.count_nonzero(u_mask) < 3 or np.count_nonzero(v_mask) < 3:
        raise ValueError("Not enough valid u/v samples for contour interpolation.")

    u_i = griddata(xy[u_mask], u_vals[u_mask], mid, method="linear")
    v_i = griddata(xy[v_mask], v_vals[v_mask], mid, method="linear")
    if "vort_z" in df.columns:
        vort_vals = df["vort_z"].values
        vort_mask = np.isfinite(vort_vals)
        if np.count_nonzero(vort_mask) >= 3:
            vort_i = griddata(xy[vort_mask], vort_vals[vort_mask], mid, method="linear")
        else:
            vort_i = np.full_like(u_i, np.nan, dtype=float)
    else:
        vort_i = np.full_like(u_i, np.nan, dtype=float)

    valid = np.isfinite(u_i) & np.isfinite(v_i)
    if np.count_nonzero(valid) < 3:
        raise ValueError(
            "Too few valid contour interpolation points for Bernoulli integration "
            f"({np.count_nonzero(valid)} valid segments)."
        )

    u_i = u_i[valid]
    v_i = v_i[valid]
    vort_i = vort_i[valid]
    nx = nx[valid]
    ny = ny[valid]
    ds = ds[valid]

    speed2 = u_i**2 + v_i**2
    un = u_i * nx + v_i * ny
    bern_term = 0.5 * (U_inf**2 - speed2)

    # Pressure-traction contribution follows the body-force convention:
    # -p * n with p - p_inf = 0.5 * rho * (U_inf^2 - |u|^2).
    fx_per_rho = np.sum((-u_i * un - bern_term * nx) * ds)
    fz_per_rho = np.sum((-v_i * un - bern_term * ny) * ds)
    F_x = rho * fx_per_rho
    F_z = rho * fz_per_rho

    abs_vort = np.abs(vort_i)
    vort_scale = U_inf / chord_for_vorticity_scale
    return {
        "Fx_bernoulli": float(F_x),
        "Fz_bernoulli": float(F_z),
        "C_d_bernoulli": float(F_x / q_infc),
        "C_l_bernoulli": float(F_z / q_infc),
        "vort_abs_mean": float(np.nanmean(abs_vort)),
        "vort_abs_max": float(np.nanmax(abs_vort)),
        "vort_rel_mean": float(np.nanmean(abs_vort) / vort_scale),
        "vort_rel_max": float(np.nanmax(abs_vort) / vort_scale),
        "n_valid_segments": int(len(ds)),
        "valid_fraction": float(len(ds) / n_total_segments),
    }


def _aggregate_bernoulli_samples(
    sample_dicts: list[dict],
) -> dict:
    """
    Aggregate Bernoulli contour-sweep samples into mean/std metrics.
    Keeps legacy keys as means for downstream compatibility.
    """
    if not sample_dicts:
        raise ValueError("No Bernoulli samples provided for aggregation.")

    def _arr(key: str) -> np.ndarray:
        return np.array([s[key] for s in sample_dicts], dtype=float)

    fx = _arr("Fx_bernoulli")
    fz = _arr("Fz_bernoulli")
    cl = _arr("C_l_bernoulli")
    cd = _arr("C_d_bernoulli")
    vort_abs_mean = _arr("vort_abs_mean")
    vort_abs_max = _arr("vort_abs_max")
    vort_rel_mean = _arr("vort_rel_mean")
    vort_rel_max = _arr("vort_rel_max")
    n_valid_segments = _arr("n_valid_segments")
    valid_fraction = _arr("valid_fraction")

    return {
        # Keep these names for compatibility with existing table loader.
        "Fx_bernoulli": float(np.mean(fx)),
        "Fz_bernoulli": float(np.mean(fz)),
        "C_l_bernoulli": float(np.mean(cl)),
        "C_d_bernoulli": float(np.mean(cd)),
        "vort_abs_mean": float(np.mean(vort_abs_mean)),
        "vort_abs_max": float(np.mean(vort_abs_max)),
        "vort_rel_mean": float(np.mean(vort_rel_mean)),
        "vort_rel_max": float(np.mean(vort_rel_max)),
        "n_valid_segments": int(round(np.mean(n_valid_segments))),
        "valid_fraction": float(np.mean(valid_fraction)),
        # New dispersion outputs from contour-size sweep.
        "Fx_bernoulli_std": float(np.std(fx)),
        "Fz_bernoulli_std": float(np.std(fz)),
        "C_l_bernoulli_std": float(np.std(cl)),
        "C_d_bernoulli_std": float(np.std(cd)),
        "vort_abs_mean_std": float(np.std(vort_abs_mean)),
        "vort_abs_max_std": float(np.std(vort_abs_max)),
        "vort_rel_mean_std": float(np.std(vort_rel_mean)),
        "vort_rel_max_std": float(np.std(vort_rel_max)),
        "n_valid_segments_std": float(np.std(n_valid_segments)),
        "valid_fraction_std": float(np.std(valid_fraction)),
        "n_contours_used": int(len(sample_dicts)),
    }


def compute_bernoulli_contour_coefficients(
    alpha_to_y_nums: dict[int, list[int]] | None = None,
    data_types: tuple[str, ...] = ("CFD", "PIV"),
    shapes: tuple[str, ...] = ("ellipse", "rectangle"),
    n_points: int = 10,
    d_perc: float = 5.0,
    rho: float = 1.2,
    U_inf: float = 15.0,
    ref_chord_alpha6: float = 0.39834712,
) -> pd.DataFrame:
    """
    Compute steady momentum + Bernoulli contour forces on NOCA control contours.
    """
    if alpha_to_y_nums is None:
        alpha_to_y_nums = DEFAULT_BERNOULLI_ALPHA_TO_Y_NUMS

    if n_points <= 0:
        raise ValueError(f"n_points must be > 0, got {n_points}")
    if d_perc < 0:
        raise ValueError(f"d_perc must be >= 0, got {d_perc}")

    rows: list[dict] = []
    for alpha in sorted(alpha_to_y_nums.keys()):
        for y_num in alpha_to_y_nums[alpha]:
            dLx, dLy, iP = reading_optimal_bound_placement(
                alpha, y_num, is_with_N_datapoints=True
            )
            x_airfoil, y_airfoil, chord_local = calculating_airfoil_centre.main(
                alpha, y_num, is_with_chord=True
            )
            chord_for_scale = ref_chord_alpha6 if alpha == 6 else chord_local
            q_infc = 0.5 * rho * U_inf**2 * chord_for_scale
            d1centre = (x_airfoil, y_airfoil)

            for data_type in data_types:
                is_CFD = str(data_type).upper() == "CFD"
                plot_params = _build_analysis_plot_params(alpha, y_num, is_CFD=is_CFD)
                df, _, _, _ = load_data(plot_params)
                if not is_CFD:
                    df = _apply_fig06_masking(df, plot_params)

                for shape in shapes:
                    shape_l = shape.lower()
                    print(
                        "Processing Bernoulli contour sweep: "
                        f"alpha={alpha}, Y{y_num}, data={data_type}, shape={shape_l}, "
                        f"n_points={n_points}, d_perc={d_perc}"
                    )
                    dmin = 1 - (d_perc / 100.0)
                    dmax = 1 + (d_perc / 100.0)
                    dLx_range = np.linspace(dLx * dmin, dLx * dmax, n_points)
                    dLy_range = np.linspace(dLy * dmin, dLy * dmax, n_points)

                    samples: list[dict] = []
                    failed = 0
                    for current_dLx in dLx_range:
                        for current_dLy in dLy_range:
                            if shape_l == "ellipse":
                                contour = boundary_ellipse(
                                    d1centre=d1centre,
                                    drot=0,
                                    dLx=current_dLx,
                                    dLy=current_dLy,
                                    iP=iP,
                                )
                            elif shape_l == "rectangle":
                                contour = boundary_rectangle(
                                    d1centre=d1centre,
                                    drot=0,
                                    dLx=current_dLx,
                                    dLy=current_dLy,
                                    iP=iP,
                                )
                            else:
                                raise ValueError(f"Unknown shape '{shape}'.")

                            try:
                                vals = _integrate_bernoulli_contour_forces(
                                    df=df,
                                    contour=contour,
                                    U_inf=U_inf,
                                    rho=rho,
                                    q_infc=q_infc,
                                    chord_for_vorticity_scale=chord_local,
                                )
                            except ValueError:
                                failed += 1
                                continue
                            samples.append(vals)

                    if not samples:
                        raise ValueError(
                            "No valid Bernoulli contour samples after sweep for "
                            f"alpha={alpha}, Y{y_num}, data={data_type}, shape={shape_l}."
                        )

                    agg = _aggregate_bernoulli_samples(samples)
                    total = int(len(dLx_range) * len(dLy_range))
                    rows.append(
                        {
                            "alpha": int(alpha),
                            "y_num": int(y_num),
                            "data_type": str(data_type).upper(),
                            "shape": shape_l,
                            "dLx": float(dLx),  # optimal/base
                            "dLy": float(dLy),  # optimal/base
                            "iP": int(iP),
                            "q_infc": float(q_infc),
                            "n_points": int(n_points),
                            "d_perc": float(d_perc),
                            "n_contours_total": total,
                            "n_contours_failed": int(failed),
                            **agg,
                        }
                    )
    return pd.DataFrame(rows)


def save_bernoulli_contour_coefficients(
    df: pd.DataFrame, save_path: Path = BERNOULLI_CACHE_PATH
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    return save_path


def compute_and_save_bernoulli_contour_coefficients(
    alpha_to_y_nums: dict[int, list[int]] | None = None,
    n_points: int = 10,
    d_perc: float = 5.0,
    save_path: Path = BERNOULLI_CACHE_PATH,
) -> tuple[pd.DataFrame, Path]:
    df = compute_bernoulli_contour_coefficients(
        alpha_to_y_nums=alpha_to_y_nums,
        n_points=n_points,
        d_perc=d_perc,
    )
    path = save_bernoulli_contour_coefficients(df, save_path=save_path)
    return df, path


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
    ref_chord=0.39834712,
    is_with_smoothing: bool = True,
    max_perc_interpolated_zones=1.0000001,
    n_datapoints=104304,
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

                # Determine number of poitns that are interpolated
                n_points_nan = 0
                for interpolation_zone_i in plot_params["interpolation_zones"]:
                    x_min, x_max, y_min, y_max = interpolation_zone_i["bounds"]
                    subset = df[
                        (df["x"] >= x_min)
                        & (df["x"] <= x_max)
                        & (df["y"] >= y_min)
                        & (df["y"] <= y_max)
                    ]
                    dropped_by_nan = subset[subset["u"].isna()]
                    n_points_nan += len(dropped_by_nan)

                # Calculate percentage of interpolated points
                perc_of_interpolated_points = 100 * (n_points_nan / n_datapoints)

                # Interpolate if within acceptable range
                if perc_of_interpolated_points < max_perc_interpolated_zones:
                    for interpolation_zone in plot_params["interpolation_zones"]:
                        current_df, _ = interpolate_missing_data(
                            current_df,
                            interpolation_zone,
                        )
                else:
                    # Skip this iteration if too many points need interpolation
                    print(
                        f"---> SKIPPED perc_of_interpolated_points: {perc_of_interpolated_points:.2f}%, "
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
            F_x, F_y, C_l, C_d = force_from_noca.main(
                current_df,
                d2curve,
                mu=mu,
                is_with_maximim_vorticity_location_correction=True,
                is_with_smoothing=is_with_smoothing,
                rho=rho,
                U_inf=U_inf,
                ref_chord=ref_chord,
            )
            F_x_list.append(F_x)
            F_y_list.append(F_y)
            print(
                f"C_kutta:{(circulation*rho*U_inf)/((0.5 * rho * U_inf**2 * ref_chord)):.2f}, NOCA: C_l:{C_l:.2f}, C_d:{C_d:.2f}, % interpolated: {perc_of_interpolated_points:.2f}%"
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
):

    Re_cfd = 1e6
    Re_piv = 4.2e5
    rho = 1.20
    ref_chord = 0.39834712
    U_inf = 15
    mu_cfd = (rho * ref_chord * U_inf) / Re_cfd
    mu_piv = (rho * ref_chord * U_inf) / Re_piv
    print(f"mu_cfd: {mu_cfd}, mu_piv: {mu_piv}")
    # mu_piv = 1.7894e-5

    cfd_ellipse_list = []
    cfd_rectangle_list = []
    piv_ellipse_list = []
    piv_rectangle_list = []

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
                df, plot_params_cfd, is_ellipse=True, mu=mu_cfd, n_points=n_points
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
                df, plot_params_cfd, is_ellipse=False, mu=mu_cfd, n_points=n_points
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
                df, plot_params_piv, is_ellipse=True, mu=mu_piv, n_points=n_points
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
                df, plot_params_piv, is_ellipse=False, mu=mu_piv, n_points=n_points
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
    df.to_csv(csv_path, index=False)
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
    ref_chord = 0.39834712
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
    # ref_chord = 0.39834712
    # U_inf = 15
    # mu_cfd = (rho * ref_chord * U_inf) / Re_cfd
    # mu_piv = (rho * ref_chord * U_inf) / Re_piv
    # print(f"mu_cfd: {mu_cfd}, mu_piv: {mu_piv}")
    # breakpoint()
