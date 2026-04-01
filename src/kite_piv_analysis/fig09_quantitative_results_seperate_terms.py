from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from kite_piv_analysis.Table06_print_quantitative_results import (
    load_pressure_integration_coefficients,
)
from kite_piv_analysis.plot_styling import set_plot_style
from kite_piv_analysis.utils import project_dir

ZORDER_SHAPE = 0.0
ZORDER_SHAPE_MARKERS = 1.0
ZORDER_KUTTA = 1.5
ZORDER_PRESSURE = 2.0
ZORDER_CFD = 3.0
ZORDER_PIV = 4.0
MAX_PIV_MISSING_BEFORE_REQUIRED_PCT = 20.0


def _mean_and_spread(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), np.nan
    return float(np.mean(arr)), float(0.5 * (np.max(arr) - np.min(arr)))


def _load_y_locations_m(y_nums: list[int]) -> np.ndarray:
    y_path = Path(project_dir) / "data" / "gamma_distribution" / "y_locations.csv"
    if not y_path.exists():
        return np.asarray(y_nums, dtype=float)

    df = pd.read_csv(y_path)
    if "Y" not in df.columns or "PIV_mm" not in df.columns:
        return np.asarray(y_nums, dtype=float)

    y_map: dict[int, float] = {}
    for _, row in df.iterrows():
        y_label = str(row["Y"]).strip()
        if not y_label.startswith("Y"):
            continue
        try:
            y_num = int(y_label.replace("Y", ""))
        except ValueError:
            continue
        y_map[y_num] = float(row["PIV_mm"]) / 1000.0

    return np.asarray([y_map.get(y, np.nan) for y in y_nums], dtype=float)


def _noca_agg_path() -> Path:
    return Path(project_dir) / "processed_data" / "noca_term_breakdown_agg.csv"


def _noca_raw_path() -> Path:
    return Path(project_dir) / "processed_data" / "noca_term_breakdown_raw.csv"


def _load_noca_term_agg() -> pd.DataFrame:
    path = _noca_agg_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Noca term aggregation cache: '{path}'. "
            "Run the Noca pipeline first."
        )
    return pd.read_csv(path)


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
    return agg.merge(counts, on=group_cols, how="left")


def _load_noca_term_agg_filtered(
    max_piv_missing_before_required_pct: float = MAX_PIV_MISSING_BEFORE_REQUIRED_PCT,
) -> pd.DataFrame:
    raw_path = _noca_raw_path()
    if raw_path.exists():
        df_raw = pd.read_csv(raw_path)

        if (
            "data_type" in df_raw.columns
            and "perc_missing_before_required" in df_raw.columns
        ):
            data_type = df_raw["data_type"].astype(str).str.upper()
            piv_rows = data_type == "PIV"
            perc_missing = pd.to_numeric(
                df_raw["perc_missing_before_required"], errors="coerce"
            )
            keep_rows = (~piv_rows) | (
                np.isfinite(perc_missing)
                & (perc_missing < float(max_piv_missing_before_required_pct))
            )
            df_raw = df_raw.loc[keep_rows].copy()

        df_agg = _aggregate_noca_term_breakdown(df_raw)
        if not df_agg.empty:
            return df_agg

    return _load_noca_term_agg()


def _select_best_shape_row(df_shape: pd.DataFrame) -> pd.Series | None:
    if df_shape.empty:
        return None
    # Prefer richest sweep representation.
    sort_cols = []
    ascending = []
    if "n_samples" in df_shape.columns:
        sort_cols.append("n_samples")
        ascending.append(False)
    if "n_points" in df_shape.columns:
        sort_cols.append("n_points")
        ascending.append(False)
    if sort_cols:
        df_shape = df_shape.sort_values(by=sort_cols, ascending=ascending)
    return df_shape.iloc[0]


def _get_shape_rows_for_case(
    df_agg: pd.DataFrame,
    alpha: int,
    y_num: int,
    data_type: str,
) -> dict[str, pd.Series | None]:
    out: dict[str, pd.Series | None] = {"ellipse": None, "rectangle": None}
    mask_case = (
        (df_agg["alpha"] == int(alpha))
        & (df_agg["y_num"] == int(y_num))
        & (df_agg["data_type"].astype(str).str.upper() == str(data_type).upper())
    )
    df_case = df_agg[mask_case]
    for shape in ["ellipse", "rectangle"]:
        df_shape = df_case[df_case["shape"].astype(str).str.lower() == shape]
        out[shape] = _select_best_shape_row(df_shape)
    return out


def _metric_col_name(coef_key: str, metric_key: str) -> str:
    prefix = "Cl" if coef_key == "Cl" else "Cd"
    mapping = {
        "inviscid_nonrot": f"{prefix}_bernoulli_like_1_2_mean",
        "inviscid_rot": f"{prefix}_rotational_total_3_4_mean",
        "viscous": f"{prefix}_viscous_total_8_9_10_mean",
        "delta": f"{prefix}_delta_noca_minus_bernoulli_mean",
        "full": f"{prefix}_total_noca_mean",
    }
    return mapping[metric_key]


def _metric_col_candidates(coef_key: str, metric_key: str) -> list[str]:
    """
    Candidate columns in preferred order.
    Keeps temporary backward compatibility while caches are regenerated.
    """
    primary = _metric_col_name(coef_key, metric_key)
    if metric_key == "inviscid_rot":
        prefix = "Cl" if coef_key == "Cl" else "Cd"
        return [primary, f"{prefix}_term_3_mean"]
    return [primary]


def _closure_coeff_from_row(row: pd.Series, coef_key: str) -> float:
    if coef_key == "Cl":
        force_col = "Fy_closure_residual_all_terms_mean"
    else:
        force_col = "Fx_closure_residual_all_terms_mean"
    if force_col not in row.index:
        return np.nan

    rho = float(row.get("rho", np.nan))
    u_inf = float(row.get("U_inf", np.nan))
    ref_chord = float(row.get("ref_chord", np.nan))
    if not np.isfinite(rho) or not np.isfinite(u_inf) or not np.isfinite(ref_chord):
        return np.nan
    denom = 0.5 * rho * u_inf**2 * ref_chord
    if abs(denom) < 1e-14:
        return np.nan
    return float(row[force_col]) / denom


def _extract_metric_from_row(
    row: pd.Series | None, coef_key: str, metric_key: str
) -> float:
    if row is None:
        return np.nan
    if metric_key == "closure":
        return _closure_coeff_from_row(row, coef_key)
    for col in _metric_col_candidates(coef_key, metric_key):
        if col in row.index:
            return float(row[col])
    return np.nan


def _collect_noca_metric_series(
    df_agg: pd.DataFrame,
    alpha: int,
    y_nums: list[int],
    coef_key: str,
    metric_key: str,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for data_type in ["CFD", "PIV"]:
        means: list[float] = []
        spreads: list[float] = []
        for y_num in y_nums:
            rows = _get_shape_rows_for_case(df_agg, alpha, y_num, data_type)
            vals = [
                _extract_metric_from_row(rows["ellipse"], coef_key, metric_key),
                _extract_metric_from_row(rows["rectangle"], coef_key, metric_key),
            ]
            mean_val, spread_val = _mean_and_spread(vals)
            means.append(mean_val)
            spreads.append(spread_val)
        out[data_type] = {
            "mean": np.asarray(means, dtype=float),
            "spread": np.asarray(spreads, dtype=float),
        }
    return out


def _collect_pressure_series(
    alpha: int, y_nums: list[int], coef_key: str
) -> np.ndarray:
    required_pairs = {(alpha, y_num) for y_num in y_nums}
    pressure_map = load_pressure_integration_coefficients(
        required_pairs=required_pairs,
        strict=False,
    )
    field = "C_l_p" if coef_key == "Cl" else "C_d_p"
    return np.asarray(
        [
            (
                float(pressure_map[(alpha, y_num)][field])
                if (alpha, y_num) in pressure_map
                else np.nan
            )
            for y_num in y_nums
        ],
        dtype=float,
    )


def _collect_kutta_series(
    y_nums: list[int],
    alpha: int = 6,
    rho: float = 1.20,
    U_inf: float = 15.0,
    ref_chord: float = 0.396,
) -> dict[str, dict[str, np.ndarray]]:
    """Load gamma and convert to Kutta-Joukowski Cl for CFD and PIV.

    Uses the same processed-data CSV as fig08 (authoritative source).
    """
    gamma_path = (
        Path(project_dir)
        / "processed_data"
        / f"quantitative_chordwise_analysis_alpha_{alpha}_with_std.csv"
    )
    if not gamma_path.exists():
        nan_arr = np.full(len(y_nums), np.nan)
        return {
            "CFD": {"mean": nan_arr.copy(), "spread": nan_arr.copy()},
            "PIV": {"mean": nan_arr.copy(), "spread": nan_arr.copy()},
        }

    df_gamma = pd.read_csv(gamma_path)
    q_c = 0.5 * rho * U_inf**2 * ref_chord

    out: dict[str, dict[str, np.ndarray]] = {}
    for src_key, csv_key in [("CFD", "cfd"), ("PIV", "piv")]:
        means: list[float] = []
        spreads: list[float] = []
        for y_num in y_nums:
            row_idx = y_num - 1  # CSV rows are ordered Y1..Y7
            if row_idx < 0 or row_idx >= len(df_gamma):
                means.append(np.nan)
                spreads.append(np.nan)
                continue
            vals = []
            for shape in ["ellipse", "rectangle"]:
                col = f"{shape}_{csv_key}_gamma"
                if col in df_gamma.columns:
                    g = df_gamma[col].iloc[row_idx]
                    if pd.notna(g) and float(g) > 0:
                        vals.append(float(g) * rho * U_inf / q_c)
            mean_val, spread_val = _mean_and_spread(vals)
            means.append(mean_val)
            spreads.append(spread_val)
        out[src_key] = {
            "mean": np.asarray(means, dtype=float),
            "spread": np.asarray(spreads, dtype=float),
        }
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
    ax: plt.Axes,
    x_shape: np.ndarray,
    z_shape: np.ndarray,
    z_lim: tuple[float, float],
    show_right_axis: bool,
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
            zorder=ZORDER_SHAPE,
        )
    ax_right.set_xlim(0.0, 0.7)
    ax_right.set_ylim(*z_lim)
    if show_right_axis:
        ax_right.set_ylabel(r"$z$ (m)")
        ax_right.tick_params(axis="y", labelright=True, right=True)
    else:
        ax_right.set_ylabel("")
        ax_right.tick_params(axis="y", labelright=False, right=False)
        ax_right.spines["right"].set_visible(False)
    ax_right.grid(False)
    return ax_right


def _shape_vertical_center_at_x(
    x0: float, x_shape: np.ndarray, z_shape: np.ndarray
) -> float:
    """Return the vertical center of the overlaid shape at x = x0."""
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
    enabled: bool = True,
    line_height_m: float = 0.09,
    line_width: float = 2.0,
    color: str = "#00FF00",
    alpha: float = 1.0,
    zorder: float = ZORDER_SHAPE_MARKERS,
    legend_label: str | None = None,
) -> None:
    """Draw short vertical lines centered on the overlaid shape at given x-positions."""
    if not enabled or len(x_positions) == 0 or len(x_shape) == 0 or len(z_shape) == 0:
        return

    half_height = 0.5 * float(line_height_m)
    legend_used = False
    for idx, x0 in enumerate(np.asarray(x_positions, dtype=float)):
        if not np.isfinite(x0):
            continue
        z_center = _shape_vertical_center_at_x(float(x0), x_shape, z_shape)
        if not np.isfinite(z_center):
            continue
        marker_label = "_nolegend_"
        if legend_label is not None and not legend_used:
            marker_label = legend_label
            legend_used = True
        ax_right.plot(
            [x0, x0],
            [z_center - half_height, z_center + half_height],
            color=color,
            linewidth=line_width,
            alpha=alpha,
            solid_capstyle="round",
            zorder=zorder,
            label=marker_label,
        )


def _plot_curve(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    spread: np.ndarray,
    color: str,
    linestyle: str,
    marker: str,
    label: str,
    band_alpha: float,
    markersize: float = 5,
    markerfacecolor: str | None = None,
    markeredgewidth: float | None = None,
    zorder: float | None = None,
) -> None:
    plot_kwargs: dict[str, object] = {
        "color": color,
        "linestyle": linestyle,
        "marker": marker,
        "linewidth": 1.2,
        "markersize": markersize,
        "label": label,
    }
    if markerfacecolor is not None:
        plot_kwargs["markerfacecolor"] = markerfacecolor
    if markeredgewidth is not None:
        plot_kwargs["markeredgewidth"] = markeredgewidth
    if zorder is not None:
        plot_kwargs["zorder"] = zorder
    ax.plot(x, y, **plot_kwargs)
    spread_arr = np.asarray(spread, dtype=float)
    valid = np.isfinite(y) & np.isfinite(spread_arr)
    if np.any(valid):
        ax.fill_between(
            x[valid],
            y[valid] - spread_arr[valid],
            y[valid] + spread_arr[valid],
            color=color,
            alpha=band_alpha,
            linewidth=0,
        )


def plot_fig09_2col(
    save_path: Path | None = None,
    show_shape_marker_lines: bool = False,
    alpha: int = 6,
    y_nums: list[int] | None = None,
) -> Path:
    df_agg = _load_noca_term_agg_filtered()
    default_overlay_y_nums = [1, 2, 3, 4, 5, 6, 7]
    requested_y_nums = (
        default_overlay_y_nums if y_nums is None else [int(y) for y in y_nums]
    )
    if len(requested_y_nums) == 0:
        raise ValueError("y_nums must contain at least one y-plane index.")

    # Overlay markers: always include Y7.
    overlay_y_nums = sorted(set(requested_y_nums + [7]))
    data_y_nums = requested_y_nums.copy()

    if y_nums is None:
        available_y_nums = sorted(
            df_agg.loc[df_agg["alpha"] == int(alpha), "y_num"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if len(available_y_nums) > 0:
            data_y_nums = [y for y in data_y_nums if y in available_y_nums]
        if len(data_y_nums) == 0:
            raise ValueError(
                f"No y-plane data found in Noca aggregation for alpha={alpha}."
            )

    # Plot data excludes Y7, while overlay still shows Y7 marker.
    y_nums = [y for y in data_y_nums if y != 7]
    if len(y_nums) == 0:
        raise ValueError("No y-planes remain for CFD/PIV data after excluding Y7.")

    y_m = _load_y_locations_m(y_nums)
    y_m_overlay = _load_y_locations_m(overlay_y_nums)
    x_shape, z_shape = _load_shape_outline(alpha=alpha, cm_offset=25)
    if len(x_shape) > 0 and len(z_shape) > 0:
        z_min = float(np.nanmin(z_shape))
        z_max = float(np.nanmax(z_shape))
    else:
        z_min, z_max = -0.6, 0.1
    z_max += 0.02
    z_span = max(z_max - z_min, 1e-6)
    if show_shape_marker_lines:
        z_pad = 0.09 * z_span
        figheight = 10.5
        bottom_adjust = 0.17
    else:
        z_pad = 0.01 * z_span
        figheight = 8.9
        bottom_adjust = 0.18
        z_max -= 0.02

    z_lim = (z_min - z_pad, z_max + z_pad)

    set_plot_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, figheight),
        sharex=True,
        sharey="row",
        gridspec_kw={"hspace": 0.08, "wspace": 0.08},
    )

    source_style = {
        "CFD": {"color": "blue", "linestyle": "-"},
        "PIV": {"color": "red", "linestyle": "--"},
    }
    cfd_label_inviscid_nonrot = (
        r"CFD Noca inviscid, non-rotational"
        # r": $\frac{\vec{F}}{\rho} = \oint_{{S}} \vec{n}\cdot \left( \frac{1}{2}u^2{I}-{u}{u} \right) \textrm{d}s$"
    )
    cfd_label_inviscid_rot = (
        r"CFD Noca inviscid, rotational"
        # r": $\frac{\vec{F}}{\rho} = \oint_{{S}} \vec{n}\cdot \left( -{u}({x}\times{\omega})+{\omega}({x}\times{u})\right) \textrm{d}s$"
    )
    cfd_label_viscous = (
        r"CFD Noca viscous"
        # r": $\frac{\vec{F}}{\rho} = \oint_{{S}} \vec{n}\cdot \left( \left[{x}\cdot({\nabla}\cdot{\tau})\right]{I} -{x}({\nabla}\cdot{\tau})+{\tau}\right) \textrm{d}s$"
    )

    # Add shape overlay on all panels; keep right z-axis only on right column.
    overlay_axes: dict[tuple[int, int], plt.Axes] = {}
    for row_idx in [0, 1]:
        for col_idx in [0, 1]:
            overlay_axes[(row_idx, col_idx)] = _add_shape_overlay(
                axes[row_idx, col_idx],
                x_shape=x_shape,
                z_shape=z_shape,
                z_lim=z_lim,
                show_right_axis=(col_idx == 1),
            )
            # Draw overlay axis below data axis so main curves can stay on top.
            overlay_axes[(row_idx, col_idx)].set_zorder(
                axes[row_idx, col_idx].get_zorder() - 1
            )
            axes[row_idx, col_idx].patch.set_visible(False)
            _add_shape_centered_marker_lines(
                overlay_axes[(row_idx, col_idx)],
                x_positions=y_m_overlay,
                x_shape=x_shape,
                z_shape=z_shape,
                enabled=show_shape_marker_lines,
                zorder=ZORDER_SHAPE_MARKERS,
                legend_label=(
                    "Measurement planes"
                    if (show_shape_marker_lines and row_idx == 0 and col_idx == 1)
                    else None
                ),
            )

    for row_idx, coef_key in enumerate(["Cl", "Cd"]):
        pressure = _collect_pressure_series(alpha, y_nums, coef_key)
        inviscid_nonrot = _collect_noca_metric_series(
            df_agg, alpha, y_nums, coef_key, "inviscid_nonrot"
        )
        inviscid_rot = _collect_noca_metric_series(
            df_agg, alpha, y_nums, coef_key, "inviscid_rot"
        )
        viscous = _collect_noca_metric_series(
            df_agg, alpha, y_nums, coef_key, "viscous"
        )
        full = _collect_noca_metric_series(df_agg, alpha, y_nums, coef_key, "full")
        kutta = _collect_kutta_series(y_nums, alpha=alpha) if coef_key == "Cl" else None

        ax_left = axes[row_idx, 0]
        for src in ["CFD", "PIV"]:
            band_alpha = 0.12 if src == "CFD" else 0.08
            source_zorder = ZORDER_CFD if src == "CFD" else ZORDER_PIV
            _plot_curve(
                ax_left,
                y_m,
                inviscid_rot[src]["mean"],
                inviscid_rot[src]["spread"],
                color=source_style[src]["color"],
                linestyle=source_style[src]["linestyle"],
                marker="s",
                label=(
                    cfd_label_inviscid_rot
                    if src == "CFD"
                    else "PIV Noca inviscid, rotational"
                ),
                band_alpha=band_alpha,
                markersize=9,
                markerfacecolor="none",
                markeredgewidth=1.4,
                zorder=source_zorder,
            )
            _plot_curve(
                ax_left,
                y_m,
                viscous[src]["mean"],
                viscous[src]["spread"],
                color=source_style[src]["color"],
                linestyle=source_style[src]["linestyle"],
                marker="D",
                label=(cfd_label_viscous if src == "CFD" else "PIV Noca viscous"),
                band_alpha=band_alpha,
                zorder=source_zorder,
            )
            _plot_curve(
                ax_left,
                y_m,
                inviscid_nonrot[src]["mean"],
                inviscid_nonrot[src]["spread"],
                color=source_style[src]["color"],
                linestyle="-" if src == "CFD" else "--",
                marker="o",
                label=(
                    cfd_label_inviscid_nonrot
                    if src == "CFD"
                    else "PIV Noca inviscid, non-rotational"
                ),
                band_alpha=band_alpha,
                markersize=9,
                markerfacecolor="none",
                markeredgewidth=1.4,
                zorder=source_zorder,
            )
        # Add Kutta-Joukowski lift on left top panel as requested.
        if coef_key == "Cl" and kutta is not None:
            for src in ["CFD", "PIV"]:
                kutta_color = source_style[src]["color"]
                _plot_curve(
                    ax_left,
                    y_m,
                    kutta[src]["mean"],
                    kutta[src]["spread"],
                    color=kutta_color,
                    linestyle=":",
                    marker="^",
                    label=f"{src} Kutta-Joukowski",
                    band_alpha=0.12 if src == "CFD" else 0.08,
                    markersize=9,
                    markerfacecolor="none",
                    markeredgewidth=1.4,
                    zorder=ZORDER_KUTTA,
                )
        ax_left.plot(
            y_m,
            pressure,
            color="black",  # pressure
            linestyle="-",
            marker="x",
            linewidth=1.2,
            label="CFD pressure integration",
            zorder=ZORDER_PRESSURE,
        )
        if row_idx == 1:
            ax_left.axhline(0.0, color="k", linewidth=0.8, alpha=0.35)
        ax_left.set_ylabel(
            r"$C_\mathrm{l}$ (-)" if coef_key == "Cl" else r"$C_\mathrm{d}$ (-)"
        )

        ax_right = axes[row_idx, 1]
        for src in ["CFD", "PIV"]:
            source_zorder = ZORDER_CFD if src == "CFD" else ZORDER_PIV
            _plot_curve(
                ax_right,
                y_m,
                full[src]["mean"],
                full[src]["spread"],
                color=source_style[src]["color"],
                linestyle=source_style[src]["linestyle"],
                marker="o",
                label=f"{src} Noca total",
                band_alpha=0.12 if src == "CFD" else 0.08,
                zorder=source_zorder,
            )
        # Kutta-Joukowski lift (only meaningful for Cl, not Cd)
        if coef_key == "Cl" and kutta is not None:
            for src in ["CFD", "PIV"]:
                kutta_color = source_style[src]["color"]
                _plot_curve(
                    ax_right,
                    y_m,
                    kutta[src]["mean"],
                    kutta[src]["spread"],
                    color=kutta_color,
                    linestyle=":",
                    marker="^",
                    label=f"{src} Kutta-Joukowski",
                    band_alpha=0.12 if src == "CFD" else 0.08,
                    markersize=9,
                    markerfacecolor="none",
                    markeredgewidth=1.4,
                    zorder=ZORDER_KUTTA,
                )
        ax_right.plot(
            y_m,
            pressure,
            color="black",  # pressure
            linestyle="-",
            marker="x",
            linewidth=1.2,
            label="CFD pressure integration",
            zorder=ZORDER_PRESSURE,
        )
        if row_idx == 1:
            ax_right.axhline(0.0, color="k", linewidth=0.8, alpha=0.35)

        for col_idx in [0, 1]:
            ax = axes[row_idx, col_idx]
            ax.set_xlim(0.0, 0.7)
            ax.grid(True, alpha=0.0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Only left column has left y-axis labels/ticks.
    for ax in axes[:, 1]:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)

    # Only bottom row has x labels.
    for ax in axes[0, :]:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    for ax in axes[1, :]:
        ax.set_xlabel(r"$y$ (m)")

    # Requested common y-range for both rows.
    for ax in axes[0, :]:
        ax.set_ylim(-0.2, 1.0)
    for ax in axes[1, :]:
        ax.set_ylim(-0.2, 0.2)

    # Collect all handles/labels from all axes for a single external legend.
    handle_map = {}
    for row_idx in [0, 1]:
        for col_idx in [0, 1]:
            for ax_obj in [axes[row_idx, col_idx], overlay_axes[(row_idx, col_idx)]]:
                for handle, label in zip(*ax_obj.get_legend_handles_labels()):
                    if label not in handle_map:
                        handle_map[label] = handle
    if "Shape outline" not in handle_map:
        handle_map["Shape outline"] = Patch(
            facecolor="black",
            edgecolor="black",
            alpha=0.2,
        )
    if "Measurement planes" not in handle_map and show_shape_marker_lines:
        handle_map["Measurement planes"] = Line2D(
            [0],
            [0],
            color="#00FF00",
            linewidth=2.0,
        )
    if "CFD contour shape variation" not in handle_map:
        handle_map["CFD contour shape variation"] = Patch(
            facecolor="blue",
            edgecolor="blue",
            alpha=0.12,
        )
    if "PIV contour shape variation" not in handle_map:
        handle_map["PIV contour shape variation"] = Patch(
            facecolor="red",
            edgecolor="red",
            alpha=0.08,
        )

    # Keep grouped blocks so legend columns are:
    # col 0 -> CFD, col 1 -> PIV, col 2 -> shared items.
    cfd_order = [
        cfd_label_inviscid_nonrot,
        cfd_label_inviscid_rot,
        cfd_label_viscous,
        "CFD Noca total",
        "CFD contour shape variation",
    ]
    piv_order = [
        "PIV Noca inviscid, non-rotational",
        "PIV Noca inviscid, rotational",
        "PIV Noca viscous",
        "PIV Noca total",
        "PIV contour shape variation",
    ]
    shared_order = [
        "CFD pressure integration",
        "CFD Kutta-Joukowski",
        "PIV Kutta-Joukowski",
        "Shape outline",
        "Measurement planes",
    ]
    cfd_labels = [label for label in cfd_order if label in handle_map]
    piv_labels = [label for label in piv_order if label in handle_map]
    shared_labels = [label for label in shared_order if label in handle_map]

    cfd_handles = [handle_map[label] for label in cfd_labels]
    piv_handles = [handle_map[label] for label in piv_labels]
    shared_handles = [handle_map[label] for label in shared_labels]

    # Matplotlib fills multi-column legends column-wise with near-equal column sizes.
    # Add invisible shared placeholders so col 0 can hold all CFD entries and
    # col 1 can hold all PIV entries deterministically.
    def _col_lengths(n_items: int, n_cols: int) -> tuple[int, ...]:
        q, r = divmod(n_items, n_cols)
        return tuple(q + 1 if col_idx < r else q for col_idx in range(n_cols))

    n_cols = 3
    total_items = len(cfd_labels) + len(piv_labels) + len(shared_labels)
    while True:
        col_lengths = _col_lengths(total_items, n_cols)
        if col_lengths[0] >= len(cfd_labels) and col_lengths[1] >= len(piv_labels):
            break
        total_items += 1

    n_padding = total_items - (len(cfd_labels) + len(piv_labels) + len(shared_labels))
    for _ in range(n_padding):
        shared_handles.append(Line2D([0], [0], linestyle="None", alpha=0.0))
        shared_labels.append("")

    legend_handles = cfd_handles + piv_handles + shared_handles
    legend_labels = cfd_labels + piv_labels + shared_labels

    for axes_ in axes:
        for ax in axes_:
            ax.set_xlim(-0.03, 0.67)
    # Keep x-limits fixed and enforce equal x/z data scale on the overlay axis.
    for overlay_ax in overlay_axes.values():
        overlay_ax.set_aspect("equal", adjustable="datalim")

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        fontsize=13,
        framealpha=1.0,
        fancybox=False,
        facecolor="white",
        frameon=False,
    )

    fig.subplots_adjust(left=0.09, right=0.91, top=0.93, bottom=bottom_adjust)

    if save_path is None:
        default_name = (
            "fig09.pdf" if int(alpha) == 6 else f"fig09_2col_alpha_{alpha}.pdf"
        )
        save_path = (
            Path(project_dir) / "results" / "paper_plots_24_03_2026" / default_name
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.0001)
    plt.close(fig)
    print(f"Saved fig09 to: {save_path}")
    return save_path


def main() -> None:
    # plot_fig09_separate_terms()
    plot_fig09_2col()


if __name__ == "__main__":
    main()
