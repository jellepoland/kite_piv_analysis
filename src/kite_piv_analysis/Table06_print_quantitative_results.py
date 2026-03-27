from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kite_piv_analysis.utils import reading_optimal_bound_placement
from kite_piv_analysis.utils import project_dir


ALPHA_TO_Y_NUMS = {
    6: [1, 2, 3, 4, 5],
    16: [1],
}
PRESSURE_CACHE_PATH = (
    Path(project_dir) / "processed_data" / "pressure_integration_coefficients.csv"
)
REQUIRED_PRESSURE_COLUMNS = ("alpha", "y_num", "C_l_p", "C_d_p")
BERNOULLI_CACHE_PATH = (
    Path(project_dir) / "processed_data" / "bernoulli_contour_force_coefficients.csv"
)
REQUIRED_BERNOULLI_COLUMNS = (
    "alpha",
    "y_num",
    "data_type",
    "shape",
    "C_l_bernoulli",
    "C_d_bernoulli",
    "vort_rel_mean",
    "vort_rel_max",
)


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
    Read Cl/Cd/Gamma/F_kutta from saved convergence-study files.
    """
    shape = "Ellipse" if is_ellipse else "Rectangle"
    data_type = "CFD" if is_CFD else "PIV"

    results = {"C_l": None, "C_d": None}
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
        results["Gamma"] = filtered_df["Gamma"].values[0]
        results["F_kutta"] = filtered_df["F_kutta"].values[0]
    else:
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
                    res = read_results(alpha, y_num, is_CFD=is_CFD, is_ellipse=is_ellipse)
                    data_key = "cfd" if is_CFD else "piv"
                    shape_key = "ellipse" if is_ellipse else "rectangle"
                    row[f"{shape_key}_{data_key}_cl_noca"] = res["C_l"]
                    row[f"{shape_key}_{data_key}_cd_noca"] = res["C_d"]
                    row[f"{shape_key}_{data_key}_cl_kutta"] = res["F_kutta"]

            rows.append(row)

    return rows


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


def load_bernoulli_contour_coefficients(
    required_keys: set[tuple[int, int, str, str]] | None = None,
    strict: bool = True,
) -> dict[tuple[int, int, str, str], dict[str, float]]:
    """
    Read precomputed Bernoulli contour coefficients C_l and C_d.
    """
    if not BERNOULLI_CACHE_PATH.exists():
        raise FileNotFoundError(
            "Missing precomputed Bernoulli contour cache: "
            f"'{BERNOULLI_CACHE_PATH}'.\n"
            "Run this once first:\n"
            "python -c \"from kite_piv_analysis.calculating_noca_and_kutta import "
            "compute_and_save_bernoulli_contour_coefficients as run; run()\""
        )

    df = pd.read_csv(BERNOULLI_CACHE_PATH)
    missing_columns = [c for c in REQUIRED_BERNOULLI_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Bernoulli cache '{BERNOULLI_CACHE_PATH}' is missing required columns: "
            f"{missing_columns}"
        )

    df = df.copy()
    df["shape"] = df["shape"].astype(str).str.lower()
    df["data_type"] = df["data_type"].astype(str).str.upper()
    dup_cols = ["alpha", "y_num", "shape", "data_type"]
    if df.duplicated(subset=dup_cols).any():
        duplicates = df[df.duplicated(subset=dup_cols, keep=False)][dup_cols]
        raise ValueError(
            "Bernoulli cache contains duplicate alpha/y_num/shape/data_type rows:\n"
            f"{duplicates.to_string(index=False)}"
        )

    if required_keys is None:
        required_keys = {
            (int(alpha), int(y_num), shape, data_type)
            for alpha, y_nums in ALPHA_TO_Y_NUMS.items()
            for y_num in y_nums
            for shape in ["ellipse", "rectangle"]
            for data_type in ["CFD", "PIV"]
        }
    available_keys = {
        (int(row["alpha"]), int(row["y_num"]), row["shape"], row["data_type"])
        for _, row in df.iterrows()
    }
    missing_keys = sorted(required_keys - available_keys)
    if strict and missing_keys:
        missing_txt = ", ".join(
            f"(alpha={a}, Y{y}, {s}, {d})" for a, y, s, d in missing_keys
        )
        raise ValueError(
            "Bernoulli cache is incomplete for Table 6. Missing: "
            f"{missing_txt}\n"
            "Regenerate with:\n"
            "python -c \"from kite_piv_analysis.calculating_noca_and_kutta import "
            "compute_and_save_bernoulli_contour_coefficients as run; run()\""
        )

    out: dict[tuple[int, int, str, str], dict[str, float]] = {}
    for _, row in df.iterrows():
        key = (int(row["alpha"]), int(row["y_num"]), row["shape"], row["data_type"])
        if key not in required_keys:
            continue
        out[key] = {
            "C_l_bernoulli": float(row["C_l_bernoulli"]),
            "C_d_bernoulli": float(row["C_d_bernoulli"]),
            "vort_rel_mean": float(row["vort_rel_mean"]),
            "vort_rel_max": float(row["vort_rel_max"]),
        }
    return out


def _fmt(v: float, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{digits}f}"


def _shape_sensitivity_percent(ellipse_val: float, rectangle_val: float) -> float:
    if not (np.isfinite(ellipse_val) and np.isfinite(rectangle_val)):
        return np.nan
    denom = 0.5 * (abs(ellipse_val) + abs(rectangle_val))
    if denom < 1e-12:
        return 0.0
    return 100.0 * abs(rectangle_val - ellipse_val) / denom


def _fmt_percent(v: float, digits: int = 1) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:.{digits}f}\\%"


def _fmt_diff_percent_vs_reference(value: float, reference: float, digits: int = 1) -> str:
    if not (np.isfinite(value) and np.isfinite(reference)):
        return "n/a"
    if abs(reference) < 1e-12:
        return "n/a"

    signed_percent = 100.0 * (value - reference) / abs(reference)
    sign_disagree = (abs(value) > 1e-12) and (abs(reference) > 1e-12) and (
        np.sign(value) != np.sign(reference)
    )
    suffix = " (sign)" if sign_disagree else ""
    return f"{signed_percent:+.{digits}f}\\%{suffix}"


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
    lines.append(
        r"    & & & & & CFD & PIV & CFD & PIV & & CFD \\"
    )
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


def build_latex_table_bernoulli_only(
    rows: list[dict],
    bernoulli_map: dict[tuple[int, int, str, str], dict[str, float]],
    pressure_map: dict[tuple[int, int], dict[str, float]],
) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table}[htp]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{Bernoulli contour-based aerodynamic coefficients and diagnostics. Percentage differences are computed relative to CFD pressure-integration ($C_{\textrm{l,p}}, C_{\textrm{d,p}}$); entries marked '(sign)' indicate opposite sign to the pressure reference. Shape sensitivity is the relative ellipse-rectangle delta.}"
    )
    lines.append(
        r"    \begin{tabular}{p{0.7cm} p{1.0cm} p{2.9cm} c c c c c c}"
    )
    lines.append(r"    \hline")
    lines.append(
        r"    Plane & $\alpha$ & Quantity & Ellipse CFD & Ellipse PIV & Rectangle CFD & Rectangle PIV & Shape sens CFD & Shape sens PIV \\"
    )
    lines.append(r"    \hline")

    for row in rows:
        alpha = int(row["alpha"])
        y_num = int(row["y_num"])
        alpha_display = alpha + 1
        p_vals = pressure_map.get((alpha, y_num), {})
        cl_p = p_vals.get("C_l_p", np.nan)
        cd_p = p_vals.get("C_d_p", np.nan)

        be_ell_cfd = bernoulli_map[(alpha, y_num, "ellipse", "CFD")]
        be_ell_piv = bernoulli_map[(alpha, y_num, "ellipse", "PIV")]
        be_rec_cfd = bernoulli_map[(alpha, y_num, "rectangle", "CFD")]
        be_rec_piv = bernoulli_map[(alpha, y_num, "rectangle", "PIV")]

        cl_be_ell_cfd = be_ell_cfd["C_l_bernoulli"]
        cl_be_ell_piv = be_ell_piv["C_l_bernoulli"]
        cl_be_rec_cfd = be_rec_cfd["C_l_bernoulli"]
        cl_be_rec_piv = be_rec_piv["C_l_bernoulli"]
        cd_be_ell_cfd = be_ell_cfd["C_d_bernoulli"]
        cd_be_ell_piv = be_ell_piv["C_d_bernoulli"]
        cd_be_rec_cfd = be_rec_cfd["C_d_bernoulli"]
        cd_be_rec_piv = be_rec_piv["C_d_bernoulli"]

        vort_rel_mean_ell_cfd = be_ell_cfd["vort_rel_mean"]
        vort_rel_mean_ell_piv = be_ell_piv["vort_rel_mean"]
        vort_rel_mean_rec_cfd = be_rec_cfd["vort_rel_mean"]
        vort_rel_mean_rec_piv = be_rec_piv["vort_rel_mean"]
        vort_rel_max_ell_cfd = be_ell_cfd["vort_rel_max"]
        vort_rel_max_ell_piv = be_ell_piv["vort_rel_max"]
        vort_rel_max_rec_cfd = be_rec_cfd["vort_rel_max"]
        vort_rel_max_rec_piv = be_rec_piv["vort_rel_max"]

        cl_shape_sens_cfd = _shape_sensitivity_percent(cl_be_ell_cfd, cl_be_rec_cfd)
        cl_shape_sens_piv = _shape_sensitivity_percent(cl_be_ell_piv, cl_be_rec_piv)
        cd_shape_sens_cfd = _shape_sensitivity_percent(cd_be_ell_cfd, cd_be_rec_cfd)
        cd_shape_sens_piv = _shape_sensitivity_percent(cd_be_ell_piv, cd_be_rec_piv)
        vort_mean_shape_sens_cfd = _shape_sensitivity_percent(
            vort_rel_mean_ell_cfd, vort_rel_mean_rec_cfd
        )
        vort_mean_shape_sens_piv = _shape_sensitivity_percent(
            vort_rel_mean_ell_piv, vort_rel_mean_rec_piv
        )
        vort_max_shape_sens_cfd = _shape_sensitivity_percent(
            vort_rel_max_ell_cfd, vort_rel_max_rec_cfd
        )
        vort_max_shape_sens_piv = _shape_sensitivity_percent(
            vort_rel_max_ell_piv, vort_rel_max_rec_piv
        )

        lines.append(
            "    "
            + f"\\multirow{{6}}{{*}}{{$Y{y_num}$}}"
            + f" & \\multirow{{6}}{{*}}{{{alpha_display}\\unit{{\\degree}}}}"
            + f" & $C_{{\\textrm{{l, Bern}}}}$ (\\unit{{-}})"
            + f" & {_fmt(cl_be_ell_cfd)} & {_fmt(cl_be_ell_piv)}"
            + f" & {_fmt(cl_be_rec_cfd)} & {_fmt(cl_be_rec_piv)}"
            + f" & {_fmt_percent(cl_shape_sens_cfd)} & {_fmt_percent(cl_shape_sens_piv)} \\\\"
        )
        lines.append(
            "    "
            + f"& & $\\Delta C_{{\\textrm{{l}}}}$ vs $C_{{\\textrm{{l,p}}}}$ (\\unit{{\\%}})"
            + f" & {_fmt_diff_percent_vs_reference(cl_be_ell_cfd, cl_p)}"
            + f" & {_fmt_diff_percent_vs_reference(cl_be_ell_piv, cl_p)}"
            + f" & {_fmt_diff_percent_vs_reference(cl_be_rec_cfd, cl_p)}"
            + f" & {_fmt_diff_percent_vs_reference(cl_be_rec_piv, cl_p)}"
            + r" & -- & -- \\"
        )
        lines.append(
            "    "
            + f"& & $C_{{\\textrm{{d, Bern}}}}$ (\\unit{{-}})"
            + f" & {_fmt(cd_be_ell_cfd)} & {_fmt(cd_be_ell_piv)}"
            + f" & {_fmt(cd_be_rec_cfd)} & {_fmt(cd_be_rec_piv)}"
            + f" & {_fmt_percent(cd_shape_sens_cfd)} & {_fmt_percent(cd_shape_sens_piv)} \\\\"
        )
        lines.append(
            "    "
            + f"& & $\\Delta C_{{\\textrm{{d}}}}$ vs $C_{{\\textrm{{d,p}}}}$ (\\unit{{\\%}})"
            + f" & {_fmt_diff_percent_vs_reference(cd_be_ell_cfd, cd_p)}"
            + f" & {_fmt_diff_percent_vs_reference(cd_be_ell_piv, cd_p)}"
            + f" & {_fmt_diff_percent_vs_reference(cd_be_rec_cfd, cd_p)}"
            + f" & {_fmt_diff_percent_vs_reference(cd_be_rec_piv, cd_p)}"
            + r" & -- & -- \\"
        )
        lines.append(
            "    "
            + f"& & $\\langle |\\omega| \\rangle /(U_\\infty/c)$ (\\unit{{-}})"
            + f" & {_fmt(vort_rel_mean_ell_cfd, 3)} & {_fmt(vort_rel_mean_ell_piv, 3)}"
            + f" & {_fmt(vort_rel_mean_rec_cfd, 3)} & {_fmt(vort_rel_mean_rec_piv, 3)}"
            + f" & {_fmt_percent(vort_mean_shape_sens_cfd)} & {_fmt_percent(vort_mean_shape_sens_piv)} \\\\"
        )
        lines.append(
            "    "
            + f"& & $\\max |\\omega| /(U_\\infty/c)$ (\\unit{{-}})"
            + f" & {_fmt(vort_rel_max_ell_cfd, 3)} & {_fmt(vort_rel_max_ell_piv, 3)}"
            + f" & {_fmt(vort_rel_max_rec_cfd, 3)} & {_fmt(vort_rel_max_rec_piv, 3)}"
            + f" & {_fmt_percent(vort_max_shape_sens_cfd)} & {_fmt_percent(vort_max_shape_sens_piv)} \\\\"
        )
        lines.append(r"    \hline")

    lines.append(r"    \end{tabular}")
    lines.append(r"    \label{tab:bernoulli_results}")
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
    bernoulli_map = load_bernoulli_contour_coefficients()
    latex_table_main = build_latex_table_no_bernoulli(rows, pressure_map)
    latex_table_bernoulli = build_latex_table_bernoulli_only(
        rows, bernoulli_map, pressure_map
    )

    save_dir = Path(project_dir) / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path_main = save_dir / "quantitative_results_table.tex"
    save_path_bernoulli = save_dir / "bernoulli_results_table.tex"
    save_path_main.write_text(latex_table_main)
    save_path_bernoulli.write_text(latex_table_bernoulli)
    print(latex_table_main)
    print("\n")
    print(latex_table_bernoulli)
    print(f"\nSaved LaTeX table to: {save_path_main}")
    print(f"Saved Bernoulli LaTeX table to: {save_path_bernoulli}")


if __name__ == "__main__":
    main()
