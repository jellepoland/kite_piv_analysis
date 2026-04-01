from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kite_piv_analysis.utils import reading_optimal_bound_placement
from kite_piv_analysis.utils import project_dir


ALPHA_TO_Y_NUMS = {
    6: [1, 2, 3, 4, 5, 6],
    16: [1],
}
PRESSURE_CACHE_PATH = (
    Path(project_dir) / "processed_data" / "pressure_integration_coefficients.csv"
)
REQUIRED_PRESSURE_COLUMNS = ("alpha", "y_num", "C_l_p", "C_d_p")
NOCA_AGG_CACHE_PATH = (
    Path(project_dir) / "processed_data" / "noca_term_breakdown_agg.csv"
)
NOCA_RAW_CACHE_PATH = (
    Path(project_dir) / "processed_data" / "noca_term_breakdown_raw.csv"
)
REQUIRED_NOCA_AGG_COLUMNS = (
    "alpha",
    "y_num",
    "data_type",
    "shape",
    "Cl_bernoulli_like_1_2_mean",
)
REQUIRED_NOCA_RAW_FILTER_COLUMNS = (
    "alpha",
    "y_num",
    "data_type",
    "shape",
    "perc_missing_before_required",
)
MAX_PIV_MISSING_BEFORE_REQUIRED_PCT = 20.0
RHO_REF = 1.20
U_INF_REF = 15.0
REF_CHORD = 0.396


def _compute_kutta_cl_from_gamma(
    gamma: float,
    rho: float = RHO_REF,
    U_inf: float = U_INF_REF,
    ref_chord: float = REF_CHORD,
) -> float:
    """Compute C_l via Kutta-Joukowski with fixed reference chord."""
    if not (np.isfinite(gamma) and np.isfinite(rho) and np.isfinite(U_inf)):
        return np.nan
    denom = 0.5 * rho * U_inf**2 * ref_chord
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return np.nan
    return float(gamma * rho * U_inf / denom)


def _load_allowed_piv_shape_keys(
    max_piv_missing_before_required_pct: float = MAX_PIV_MISSING_BEFORE_REQUIRED_PCT,
) -> set[tuple[int, int, str]] | None:
    """
    Return allowed (alpha, y_num, shape) keys for PIV based on raw NOCA cache.

    Returns None when no raw cache (or required columns) is available so callers
    can safely fall back to the previous unfiltered behavior.
    """
    if not NOCA_RAW_CACHE_PATH.exists():
        return None

    df_raw = pd.read_csv(NOCA_RAW_CACHE_PATH)
    if not set(REQUIRED_NOCA_RAW_FILTER_COLUMNS).issubset(df_raw.columns):
        return None

    data_type = df_raw["data_type"].astype(str).str.upper()
    perc_missing = pd.to_numeric(
        df_raw["perc_missing_before_required"], errors="coerce"
    )
    keep = (
        (data_type == "PIV")
        & np.isfinite(perc_missing)
        & (perc_missing < float(max_piv_missing_before_required_pct))
    )
    df_keep = df_raw.loc[keep, ["alpha", "y_num", "shape"]].copy()
    if df_keep.empty:
        return set()

    out: set[tuple[int, int, str]] = set()
    for _, row in df_keep.iterrows():
        alpha = pd.to_numeric(row["alpha"], errors="coerce")
        y_num = pd.to_numeric(row["y_num"], errors="coerce")
        shape = str(row["shape"]).strip().lower()
        if not (np.isfinite(alpha) and np.isfinite(y_num)):
            continue
        if shape not in {"ellipse", "rectangle"}:
            continue
        out.add((int(alpha), int(y_num), shape))
    return out


def _piv_shape_allowed_by_default_missing_filter(
    alpha: int,
    y_num: int,
    shape: str,
    max_piv_missing_before_required_pct: float = MAX_PIV_MISSING_BEFORE_REQUIRED_PCT,
) -> bool:
    allowed_keys = _load_allowed_piv_shape_keys(
        max_piv_missing_before_required_pct=max_piv_missing_before_required_pct
    )
    if allowed_keys is None:
        return True
    return (int(alpha), int(y_num), str(shape).lower()) in allowed_keys


def _empty_results_payload() -> dict[str, float]:
    return {
        "C_l": np.nan,
        "C_d": np.nan,
        "d1centre_x": np.nan,
        "d1centre_y": np.nan,
        "perc_of_interpolated_points": np.nan,
        "Gamma": np.nan,
        "F_kutta_cached": np.nan,
        "F_kutta": np.nan,
    }


def read_parameter_sweep_results(
    alpha: float, y_num: int, is_ellipse: bool, data_type: str = "PIV"
) -> pd.DataFrame:
    """
    Read the 2D (dLx, dLy) sweep results CSV for a specific configuration.
    """
    shape = "Ellipse" if is_ellipse else "Rectangle"
    save_folder = (
        Path(project_dir) / "processed_data" / "convergence_study" / "PIV_sweep"
    )
    file_path = Path(save_folder) / f"alpha_{alpha}_Y{y_num}_{data_type}_{shape}.csv"
    if not file_path.exists():
        raise FileNotFoundError(
            "Missing 2D contour sweep cache for Table06: "
            f"'{file_path}'.\n"
            "Regenerate processing caches with:\n"
            "python -m kite_piv_analysis._main_process_and_plot --skip-plots"
        )
    return pd.read_csv(file_path)


def read_PIV_parameter_sweep_results(
    alpha: float, y_num: int, is_ellipse: bool, data_type: str = "PIV"
) -> pd.DataFrame:
    """
    Backward-compatible alias for historical callers.
    """
    return read_parameter_sweep_results(alpha, y_num, is_ellipse, data_type=data_type)


def read_results(alpha: int, y_num: int, is_CFD: bool, is_ellipse: bool) -> dict:
    """
    Read Cl/Cd/Gamma and compute Kutta C_l with fixed c_ref.
    """
    shape = "Ellipse" if is_ellipse else "Rectangle"
    shape_key = shape.lower()
    data_type = "CFD" if is_CFD else "PIV"

    if not is_CFD and not _piv_shape_allowed_by_default_missing_filter(
        alpha, y_num, shape_key
    ):
        return _empty_results_payload()

    results = _empty_results_payload()
    filtered_df = read_parameter_sweep_results(
        alpha, y_num, is_ellipse, data_type=data_type
    )

    if not filtered_df.empty:
        filtered_df = filtered_df.mean(numeric_only=True).to_frame().T

    if not filtered_df.empty:
        results["C_l"] = filtered_df["C_l"].values[0]
        results["C_d"] = filtered_df["C_d"].values[0]
        results["d1centre_x"] = filtered_df["d1centre_x"].values[0]
        results["d1centre_y"] = filtered_df["d1centre_y"].values[0]
        results["perc_of_interpolated_points"] = filtered_df[
            "perc_of_interpolated_points"
        ].values[0]
        gamma_val = (
            float(filtered_df["Gamma"].values[0])
            if "Gamma" in filtered_df.columns
            else np.nan
        )
        cached_f_kutta = (
            float(filtered_df["F_kutta"].values[0])
            if "F_kutta" in filtered_df.columns
            else np.nan
        )
        results["Gamma"] = gamma_val
        results["F_kutta_cached"] = cached_f_kutta
        results["F_kutta"] = _compute_kutta_cl_from_gamma(gamma_val)
        if not np.isfinite(results["F_kutta"]):
            results["F_kutta"] = cached_f_kutta
    else:
        if is_CFD:
            raise ValueError(
                "No matching data found for "
                f"alpha={alpha}, Y{y_num}, data_type={data_type}, shape={shape}."
            )

    return results


def collect_quantitative_table_data() -> list[dict]:
    """
    Collect NOCA/Kutta values + boundary settings for all table rows.
    """
    rows: list[dict] = []
    required_nonrot_keys = {
        (int(alpha), int(y_num), shape, data_type)
        for alpha, y_nums in ALPHA_TO_Y_NUMS.items()
        for y_num in y_nums
        for shape in ["ellipse", "rectangle"]
        for data_type in ["CFD", "PIV"]
    }
    nonrot_map = load_inviscid_nonrot_coefficients(
        required_keys=required_nonrot_keys,
        strict=False,
    )

    for alpha, y_num_range in ALPHA_TO_Y_NUMS.items():
        for y_num in y_num_range:
            dLx, dLy, iP = reading_optimal_bound_placement(
                alpha, y_num, is_with_N_datapoints=True
            )
            ref = read_results(alpha, y_num, is_CFD=True, is_ellipse=True)

            row = {
                "alpha": alpha,
                "y_num": y_num,
                "x_b": ref["d1centre_x"],
                "z_b": ref["d1centre_y"],
                "W_b": dLx,
                "H_b": dLy,
                "N_b": iP,
            }

            for is_CFD in [True, False]:
                for is_ellipse in [True, False]:
                    res = read_results(
                        alpha, y_num, is_CFD=is_CFD, is_ellipse=is_ellipse
                    )
                    data_key = "cfd" if is_CFD else "piv"
                    data_type = "CFD" if is_CFD else "PIV"
                    shape_key = "ellipse" if is_ellipse else "rectangle"
                    row[f"{shape_key}_{data_key}_cl_noca"] = res["C_l"]
                    row[f"{shape_key}_{data_key}_cd_noca"] = res["C_d"]
                    row[f"{shape_key}_{data_key}_cl_kutta"] = res["F_kutta"]
                    nonrot_key = (alpha, y_num, shape_key, data_type)
                    cl_nonrot = nonrot_map.get(nonrot_key, np.nan)
                    row[f"{shape_key}_{data_key}_cl_nonrot"] = cl_nonrot
                    if np.isfinite(cl_nonrot) and np.isfinite(res["F_kutta"]):
                        row[f"{shape_key}_{data_key}_cl_nonrot_minus_kutta"] = (
                            cl_nonrot - res["F_kutta"]
                        )
                    else:
                        row[f"{shape_key}_{data_key}_cl_nonrot_minus_kutta"] = np.nan

            rows.append(row)

    return rows


def _select_best_agg_row(df_subset: pd.DataFrame) -> pd.Series | None:
    if df_subset.empty:
        return None
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "n_samples" in df_subset.columns:
        sort_cols.append("n_samples")
        ascending.append(False)
    if "n_points" in df_subset.columns:
        sort_cols.append("n_points")
        ascending.append(False)
    if sort_cols:
        df_subset = df_subset.sort_values(sort_cols, ascending=ascending)
    return df_subset.iloc[0]


def load_inviscid_nonrot_coefficients(
    required_keys: set[tuple[int, int, str, str]] | None = None,
    strict: bool = True,
) -> dict[tuple[int, int, str, str], float]:
    """
    Read inviscid, non-rotational lift coefficients from NOCA term cache.

    Returns map:
    (alpha, y_num, shape, data_type) -> Cl_bernoulli_like_1_2_mean
    """
    if not NOCA_AGG_CACHE_PATH.exists():
        raise FileNotFoundError(
            "Missing NOCA aggregation cache: "
            f"'{NOCA_AGG_CACHE_PATH}'.\n"
            "Run this once first:\n"
            "python -m kite_piv_analysis._process_calculating_noca_and_kutta__convergence_study"
        )

    df = pd.read_csv(NOCA_AGG_CACHE_PATH)
    missing_columns = [c for c in REQUIRED_NOCA_AGG_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"NOCA agg cache '{NOCA_AGG_CACHE_PATH}' is missing required columns: "
            f"{missing_columns}"
        )

    df = df.copy()
    df["shape"] = df["shape"].astype(str).str.lower()
    df["data_type"] = df["data_type"].astype(str).str.upper()
    allowed_piv_keys = _load_allowed_piv_shape_keys(
        max_piv_missing_before_required_pct=MAX_PIV_MISSING_BEFORE_REQUIRED_PCT
    )
    if allowed_piv_keys is not None:
        is_piv = df["data_type"] == "PIV"
        if bool(is_piv.any()):
            keep_mask = ~is_piv
            piv_subset = df.loc[is_piv, ["alpha", "y_num", "shape"]]
            piv_keep = [
                (
                    int(alpha),
                    int(y_num),
                    str(shape).lower(),
                )
                in allowed_piv_keys
                for alpha, y_num, shape in zip(
                    piv_subset["alpha"], piv_subset["y_num"], piv_subset["shape"]
                )
            ]
            keep_mask.loc[is_piv] = piv_keep
            df = df.loc[keep_mask].copy()

    if required_keys is None:
        required_keys = {
            (int(alpha), int(y_num), shape, data_type)
            for alpha, y_nums in ALPHA_TO_Y_NUMS.items()
            for y_num in y_nums
            for shape in ["ellipse", "rectangle"]
            for data_type in ["CFD", "PIV"]
        }

    out: dict[tuple[int, int, str, str], float] = {}
    missing_keys: list[tuple[int, int, str, str]] = []
    for alpha, y_num, shape, data_type in sorted(required_keys):
        subset = df[
            (df["alpha"] == int(alpha))
            & (df["y_num"] == int(y_num))
            & (df["shape"] == str(shape).lower())
            & (df["data_type"] == str(data_type).upper())
        ]
        best = _select_best_agg_row(subset)
        if best is None:
            missing_keys.append((alpha, y_num, shape, data_type))
            continue
        out[(alpha, y_num, shape, data_type)] = float(
            best["Cl_bernoulli_like_1_2_mean"]
        )

    if strict and missing_keys:
        missing_txt = ", ".join(
            f"(alpha={a}, Y{y}, {s}, {d})" for a, y, s, d in missing_keys
        )
        raise ValueError(
            "NOCA agg cache is incomplete for inviscid, non-rotational extraction. "
            f"Missing: {missing_txt}"
        )
    return out


def load_pressure_integration_coefficients(
    required_pairs: set[tuple[int, int]] | None = None,
    strict: bool = True,
) -> dict[tuple[int, int], dict[str, float]]:
    """
    Read precomputed CFD pressure-integration coefficients C_l,p and C_d,p.
    The expensive integration must be run once beforehand and saved to processed_data.
    """
    if not PRESSURE_CACHE_PATH.exists():
        raise FileNotFoundError(
            "Missing precomputed pressure-integration cache: "
            f"'{PRESSURE_CACHE_PATH}'.\n"
            "Run this once first:\n"
            "python -m kite_piv_analysis.calculating_integrated_surface_pressure"
        )

    df = pd.read_csv(PRESSURE_CACHE_PATH)
    missing_columns = [c for c in REQUIRED_PRESSURE_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Pressure cache '{PRESSURE_CACHE_PATH}' is missing required columns: "
            f"{missing_columns}"
        )

    if df.duplicated(subset=["alpha", "y_num"]).any():
        duplicates = df[df.duplicated(subset=["alpha", "y_num"], keep=False)][
            ["alpha", "y_num"]
        ]
        raise ValueError(
            "Pressure cache contains duplicate alpha/y_num rows:\n"
            f"{duplicates.to_string(index=False)}"
        )

    if required_pairs is None:
        required_pairs = {
            (int(alpha), int(y_num))
            for alpha, y_nums in ALPHA_TO_Y_NUMS.items()
            for y_num in y_nums
        }
    available_pairs = {
        (int(alpha), int(y_num)) for alpha, y_num in zip(df["alpha"], df["y_num"])
    }
    missing_pairs = sorted(required_pairs - available_pairs)
    if strict and missing_pairs:
        missing_txt = ", ".join(f"(alpha={a}, Y{y})" for a, y in missing_pairs)
        raise ValueError(
            "Pressure cache is incomplete for Table 6. Missing: "
            f"{missing_txt}\n"
            "Regenerate with:\n"
            "python -m kite_piv_analysis.calculating_integrated_surface_pressure"
        )

    results: dict[tuple[int, int], dict[str, float]] = {}
    for _, row in df.iterrows():
        key = (int(row["alpha"]), int(row["y_num"]))
        if required_pairs is not None and key not in required_pairs:
            continue
        results[key] = {
            "C_l_p": float(row["C_l_p"]),
            "C_d_p": float(row["C_d_p"]),
        }
    return results


def _fmt(v: float, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{digits}f}"


def build_latex_table_no_bernoulli(
    rows: list[dict],
    pressure_map: dict[tuple[int, int], dict[str, float]],
) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table}[htp]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{Aerodynamic coefficients for the measured PIV and simulated CFD of different planes, using NOCA, Kutta, and pressure-integration reference values.}"
    )
    lines.append(
        r"    \begin{tabular}{p{0.5cm} p{0.8cm} p{.8cm} p{2.6cm} p{1.8cm} c c c p{1cm} p{1.3cm} c}"
    )
    lines.append(r"    \hline")
    lines.append(
        r"    \multirow{2}{*}{Plane} & \multirow{2}{*}{$\alpha$} & \multicolumn{2}{c}{\multirow{2}{*}{Boundary Setting}} & \multirow{2}{*}{} & \multicolumn{2}{c}{Ellipse} & \multicolumn{2}{c}{Rectangle} &  & $P$-integration \\"
    )
    lines.append(r"    & & & & & CFD & PIV & CFD & PIV & & CFD \\")
    lines.append(r"    \hline")

    for row in rows:
        alpha = int(row["alpha"])
        y_num = int(row["y_num"])
        alpha_display = alpha + 1
        p_vals = pressure_map.get((alpha, y_num), {})
        cl_p = p_vals.get("C_l_p", np.nan)
        cd_p = p_vals.get("C_d_p", np.nan)

        lines.append(
            "    "
            + f"\\multirow{{3}}{{*}}{{$Y{y_num}$}} & \\multirow{{3}}{{*}}{{{alpha_display}\\unit{{\\degree}}}}"
            + f" & $x_{{\\textrm{{b}}}},z_{{\\textrm{{b}}}}$ & {_fmt(row['x_b'])}, {_fmt(row['z_b'])} (\\unit{{m}})"
            + f" & $C_{{\\textrm{{l, Noca}}}}$ (\\unit{{-}}) & {_fmt(row['ellipse_cfd_cl_noca'])} & {_fmt(row['ellipse_piv_cl_noca'])}"
            + f" & {_fmt(row['rectangle_cfd_cl_noca'])} & {_fmt(row['rectangle_piv_cl_noca'])}"
            + f" & $C_{{\\textrm{{l, p}}}}$ (\\unit{{-}})& {_fmt(cl_p)} \\\\"
        )
        lines.append(
            "    "
            + f"& & $W_{{\\textrm{{b}}}}, H_{{\\textrm{{b}}}}$ & {_fmt(row['W_b'])}, {_fmt(row['H_b'])} (\\unit{{m}})"
            + f" & $C_{{\\textrm{{d, Noca}}}}$ (\\unit{{-}})& {_fmt(row['ellipse_cfd_cd_noca'])} & {_fmt(row['ellipse_piv_cd_noca'])}"
            + f" & {_fmt(row['rectangle_cfd_cd_noca'])} & {_fmt(row['rectangle_piv_cd_noca'])}"
            + f" & $C_{{\\textrm{{d, p}}}}$ (\\unit{{-}})& {_fmt(cd_p)} \\\\"
        )
        lines.append(
            "    "
            + f"& & $N_{{\\textrm{{b}}}}$ & {int(row['N_b'])} (\\unit{{-}})"
            + f" & $C_{{\\textrm{{l,Kutta}}}}$ (\\unit{{-}})& {_fmt(row['ellipse_cfd_cl_kutta'])} & {_fmt(row['ellipse_piv_cl_kutta'])}"
            + f" & {_fmt(row['rectangle_cfd_cl_kutta'])} & {_fmt(row['rectangle_piv_cl_kutta'])}"
            + r" & & \\"
        )
        lines.append(r"    \hline")

    lines.append(r"    \end{tabular}")
    lines.append(r"    \label{tab:quantitative_results}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_latex_table_nonrot_minus_kutta(rows: list[dict]) -> str:
    """
    Diagnostic table:
    Delta Cl = Cl_inviscid_nonrot - Cl_kutta
    """
    lines: list[str] = []
    lines.append(r"\begin{table}[htp]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{Diagnostic comparison between the inviscid, non-rotational Noca contribution and Kutta--Joukowski lift: $\Delta C_{\textrm{l}} = C_{\textrm{l,inv,nonrot}} - C_{\textrm{l,Kutta}}$. Negative values indicate that $C_{\textrm{l,Kutta}}$ is larger.}"
    )
    lines.append(r"    \begin{tabular}{c c c c c c c}")
    lines.append(r"    \hline")
    lines.append(
        r"    Plane & $\alpha$ & Ellipse CFD & Ellipse PIV & Rectangle CFD & Rectangle PIV & All negative \\"
    )
    lines.append(r"    \hline")

    for row in rows:
        alpha = int(row["alpha"])
        y_num = int(row["y_num"])
        alpha_display = alpha + 1
        values = [
            row.get("ellipse_cfd_cl_nonrot_minus_kutta", np.nan),
            row.get("ellipse_piv_cl_nonrot_minus_kutta", np.nan),
            row.get("rectangle_cfd_cl_nonrot_minus_kutta", np.nan),
            row.get("rectangle_piv_cl_nonrot_minus_kutta", np.nan),
        ]
        finite_vals = [float(v) for v in values if np.isfinite(v)]
        all_negative = (
            "yes"
            if len(finite_vals) == len(values) and all(v < 0 for v in finite_vals)
            else "no"
        )
        lines.append(
            "    "
            + f"$Y{y_num}$ & {alpha_display}\\unit{{\\degree}}"
            + f" & {_fmt(values[0], 3)} & {_fmt(values[1], 3)}"
            + f" & {_fmt(values[2], 3)} & {_fmt(values[3], 3)} & {all_negative} \\\\"
        )

    lines.append(r"    \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \label{tab:nonrot_minus_kutta}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def plot_alpha6_cl_noca_y_sweep() -> Path:
    """
    Backward-compatible wrapper; plotting moved to fig09_quantitative_results.py.
    """
    from kite_piv_analysis.fig09_quantitative_results import (
        plot_alpha6_quantitative_results,
    )

    save_path_cl, _ = plot_alpha6_quantitative_results()
    return save_path_cl


def plot_alpha6_cd_noca_y_sweep() -> Path:
    """
    Backward-compatible wrapper; plotting moved to fig09_quantitative_results.py.
    """
    from kite_piv_analysis.fig09_quantitative_results import (
        plot_alpha6_quantitative_results,
    )

    _, save_path_cd = plot_alpha6_quantitative_results()
    return save_path_cd


def main():
    rows = collect_quantitative_table_data()
    pressure_map = load_pressure_integration_coefficients()
    latex_table_main = build_latex_table_no_bernoulli(rows, pressure_map)
    latex_table_delta = build_latex_table_nonrot_minus_kutta(rows)

    save_dir = Path(project_dir) / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path_main = save_dir / "quantitative_results_table.tex"
    save_path_delta = save_dir / "quantitative_results_nonrot_minus_kutta_table.tex"
    save_path_main.write_text(latex_table_main)
    save_path_delta.write_text(latex_table_delta)
    print(latex_table_main)
    print("\n")
    print(latex_table_delta)
    print(f"\nSaved LaTeX table to: {save_path_main}")
    print(f"Saved diagnostic LaTeX table to: {save_path_delta}")


if __name__ == "__main__":
    main()
