from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from kite_piv_analysis.Table06_print_quantitative_results import (
    load_pressure_integration_coefficients,
)
from kite_piv_analysis.plot_styling import set_plot_style
from kite_piv_analysis.utils import project_dir

ZORDER_SHAPE = 0.0
ZORDER_SHAPE_MARKERS = 1.0
ZORDER_PRESSURE = 2.0
ZORDER_CFD = 3.0
ZORDER_PIV = 4.0


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


def _load_noca_term_agg() -> pd.DataFrame:
    path = _noca_agg_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing NOCA term aggregation cache: '{path}'. "
            "Run the NOCA pipeline first."
        )
    return pd.read_csv(path)


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
        "core": f"{prefix}_bernoulli_like_1_2_mean",
        "rotational": f"{prefix}_term_3_mean",
        "viscous": f"{prefix}_viscous_total_8_9_10_mean",
        "delta": f"{prefix}_delta_noca_minus_bernoulli_mean",
        "full": f"{prefix}_total_noca_mean",
        "reduced": f"{prefix}_reduced_total_1_2_3_8_9_10_mean",
        "omitted": f"{prefix}_omitted_terms_4_5_6_7_mean",
    }
    return mapping[metric_key]


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
    col = _metric_col_name(coef_key, metric_key)
    if col not in row.index:
        return np.nan
    return float(row[col])


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
        ax_right.plot(
            x_shape,
            z_shape,
            color="black",
            alpha=0.5,
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
    alpha: float = 0.6,
    zorder: float = ZORDER_SHAPE_MARKERS,
    legend_label: str | None = None,
) -> None:
    """Draw short vertical lines centered on the overlaid shape at given x-positions."""
    if not enabled or len(x_positions) == 0 or len(x_shape) == 0 or len(z_shape) == 0:
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
            zorder=zorder,
            label=(
                legend_label if idx == 0 and legend_label is not None else "_nolegend_"
            ),
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


def _sum_series_with_spread(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Sum two mean/spread series with RSS uncertainty propagation."""
    return {
        "mean": np.asarray(a["mean"], dtype=float) + np.asarray(b["mean"], dtype=float),
        "spread": np.sqrt(
            np.asarray(a["spread"], dtype=float) ** 2
            + np.asarray(b["spread"], dtype=float) ** 2
        ),
    }


# def _plot_column_left(
#     ax: plt.Axes,
#     x: np.ndarray,
#     coef_key: str,
#     source_style: dict[str, dict[str, str]],
#     rotational: dict[str, dict[str, np.ndarray]],
#     viscous: dict[str, dict[str, np.ndarray]],
#     delta: dict[str, dict[str, np.ndarray]],
# ) -> None:
#     metrics = [
#         ("rotational (3)", rotational, "o"),
#         ("viscous (8+9+10)", viscous, "p"),
#         ("delta (total - inviscid)", delta, "D"),
#     ]
#     for src in ["CFD", "PIV"]:
#         for label_suffix, metric_dict, marker in metrics:
#             _plot_curve(
#                 ax,
#                 x,
#                 metric_dict[src]["mean"],
#                 metric_dict[src]["spread"],
#                 color=source_style[src]["color"],
#                 linestyle=source_style[src]["linestyle"],
#                 marker=marker,
#                 label=f"{src} NOCA {label_suffix}",
#                 band_alpha=0.12 if src == "CFD" else 0.08,
#             )

#     ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.35)
#     ax.set_ylabel(
#         r"$C_l$ contribution (-)" if coef_key == "Cl" else r"$C_d$ contribution (-)"
#     )


# def _plot_column_middle(
#     ax: plt.Axes,
#     x: np.ndarray,
#     coef_key: str,
#     source_style: dict[str, dict[str, str]],
#     pressure: np.ndarray,
#     noca_core: dict[str, dict[str, np.ndarray]],
# ) -> None:
#     for src in ["CFD", "PIV"]:
#         _plot_curve(
#             ax,
#             x,
#             noca_core[src]["mean"],
#             noca_core[src]["spread"],
#             color=source_style[src]["color"],
#             linestyle=source_style[src]["linestyle"],
#             marker="o",
#             label=f"{src} NOCA inviscid (1+2)",
#             band_alpha=0.12 if src == "CFD" else 0.08,
#         )

#     ax.plot(
#         x,
#         pressure,
#         color="black",
#         linestyle="-",
#         marker="x",
#         linewidth=1.2,
#         label="CFD pressure integration",
#     )
#     ax.set_ylabel(r"$C_l$ (-)" if coef_key == "Cl" else r"$C_d$ (-)")


# def _plot_column_right(
#     ax: plt.Axes,
#     x: np.ndarray,
#     coef_key: str,
#     source_style: dict[str, dict[str, str]],
#     pressure: np.ndarray,
#     full: dict[str, dict[str, np.ndarray]],
# ) -> None:
#     for src in ["CFD", "PIV"]:
#         _plot_curve(
#             ax,
#             x,
#             full[src]["mean"],
#             full[src]["spread"],
#             color=source_style[src]["color"],
#             linestyle=source_style[src]["linestyle"],
#             marker="s",
#             label=f"{src} NOCA total (all)",
#             band_alpha=0.12 if src == "CFD" else 0.08,
#         )
#     ax.plot(
#         x,
#         pressure,
#         color="black",
#         linestyle="-",
#         marker="x",
#         linewidth=1.2,
#         label="CFD pressure integration",
#     )
#     ax.set_ylabel(r"$C_l$ (-)" if coef_key == "Cl" else r"$C_d$ (-)")


# def plot_fig09_separate_terms(
#     save_path: Path | None = None, show_shape_marker_lines: bool = True
# ) -> Path:
#     alpha = 6
#     y_nums = [1, 2, 3, 4, 5, 6, 7]
#     y_m = _load_y_locations_m(y_nums)
#     x_shape, z_shape = _load_shape_outline(alpha=alpha, cm_offset=25)
#     if len(x_shape) > 0 and len(z_shape) > 0:
#         z_min = float(np.nanmin(z_shape))
#         z_max = float(np.nanmax(z_shape))
#     else:
#         z_min, z_max = -0.6, 0.1
#     z_span = max(z_max - z_min, 1e-6)
#     z_pad = 0.05 * z_span
#     z_lim = (z_min - z_pad, z_max + z_pad)

#     df_agg = _load_noca_term_agg()

#     set_plot_style()
#     fig, axes = plt.subplots(
#         2,
#         3,
#         figsize=(18, 9),
#         sharex=True,
#         sharey=False,
#         gridspec_kw={"hspace": 0.16, "wspace": 0.15},
#     )

#     # Add shape overlay on all panels; show z-axis only on right column.
#     for row_idx in [0, 1]:
#         for col in [0, 1, 2]:
#             overlay_ax = _add_shape_overlay(
#                 axes[row_idx, col],
#                 x_shape=x_shape,
#                 z_shape=z_shape,
#                 z_lim=z_lim,
#                 show_right_axis=(col == 2),
#             )
#             _add_shape_centered_marker_lines(
#                 overlay_ax,
#                 x_positions=y_m,
#                 x_shape=x_shape,
#                 z_shape=z_shape,
#                 enabled=show_shape_marker_lines,
#             )

#     source_style = {
#         "CFD": {"color": "blue", "linestyle": "-"},
#         "PIV": {"color": "red", "linestyle": "--"},
#     }

#     for row_idx, coef_key in enumerate(["Cl", "Cd"]):
#         pressure = _collect_pressure_series(alpha, y_nums, coef_key)

#         noca_core = _collect_noca_metric_series(df_agg, alpha, y_nums, coef_key, "core")
#         rotational = _collect_noca_metric_series(
#             df_agg, alpha, y_nums, coef_key, "rotational"
#         )
#         viscous = _collect_noca_metric_series(
#             df_agg, alpha, y_nums, coef_key, "viscous"
#         )
#         delta = _collect_noca_metric_series(df_agg, alpha, y_nums, coef_key, "delta")
#         full = _collect_noca_metric_series(df_agg, alpha, y_nums, coef_key, "full")

#         _plot_column_left(
#             axes[row_idx, 0],
#             y_m,
#             coef_key,
#             source_style,
#             rotational,
#             viscous,
#             delta,
#         )
#         _plot_column_middle(
#             axes[row_idx, 1],
#             y_m,
#             coef_key,
#             source_style,
#             pressure,
#             noca_core,
#         )
#         _plot_column_right(
#             axes[row_idx, 2],
#             y_m,
#             coef_key,
#             source_style,
#             pressure,
#             full,
#         )

#     axes[0, 0].set_title("Small Terms (Zoom)")
#     axes[0, 1].set_title("Inviscid vs Pressure")
#     axes[0, 2].set_title("Total vs Pressure")

#     for ax in axes.flat:
#         ax.set_xlim(0.0, 0.7)
#         ax.grid(True, alpha=0.01)
#         ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

#     for ax in axes[1, :]:
#         ax.set_xlabel(r"$y$ (m)")

#     # Legends on top row only to reduce clutter.
#     for col in [0, 1, 2]:
#         axes[0, col].legend(loc="lower left", fontsize=10, ncol=2)

#     fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.08)

#     if save_path is None:
#         save_path = (
#             Path(project_dir)
#             / "results"
#             / "paper_plots_24_03_2026"
#             / "fig09_separate_terms.pdf"
#         )
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
#     plt.close(fig)
#     print(f"Saved fig09 separate terms to: {save_path}")
#     return save_path


def plot_fig09_2col(
    save_path: Path | None = None, show_shape_marker_lines: bool = True
) -> Path:
    alpha = 6
    y_nums = [1, 2, 3, 4, 5, 6, 7]
    y_m = _load_y_locations_m(y_nums)
    df_agg = _load_noca_term_agg()
    x_shape, z_shape = _load_shape_outline(alpha=alpha, cm_offset=25)
    if len(x_shape) > 0 and len(z_shape) > 0:
        z_min = float(np.nanmin(z_shape))
        z_max = float(np.nanmax(z_shape))
    else:
        z_min, z_max = -0.6, 0.1
    z_max += 0.02
    z_span = max(z_max - z_min, 1e-6)
    z_pad = 0.09 * z_span
    z_lim = (z_min - z_pad, z_max + z_pad)

    set_plot_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        sharex=True,
        sharey="row",
        gridspec_kw={"hspace": 0.08, "wspace": 0.08},
    )

    source_style = {
        "CFD": {"color": "blue", "linestyle": "-"},
        "PIV": {"color": "red", "linestyle": "--"},
    }

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
                x_positions=y_m,
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
        inviscid = _collect_noca_metric_series(df_agg, alpha, y_nums, coef_key, "core")
        rotational = _collect_noca_metric_series(
            df_agg, alpha, y_nums, coef_key, "rotational"
        )
        viscous = _collect_noca_metric_series(
            df_agg, alpha, y_nums, coef_key, "viscous"
        )
        full = _collect_noca_metric_series(df_agg, alpha, y_nums, coef_key, "full")

        rot_plus_visc_cfd = _sum_series_with_spread(rotational["CFD"], viscous["CFD"])
        rot_plus_visc_piv = _sum_series_with_spread(rotational["PIV"], viscous["PIV"])

        ax_left = axes[row_idx, 0]
        for src, rpv, inv, band_alpha in [
            ("CFD", rot_plus_visc_cfd, inviscid["CFD"], 0.12),
            ("PIV", rot_plus_visc_piv, inviscid["PIV"], 0.08),
        ]:
            source_zorder = ZORDER_CFD if src == "CFD" else ZORDER_PIV
            _plot_curve(
                ax_left,
                y_m,
                rpv["mean"],
                rpv["spread"],
                color=source_style[src]["color"],
                linestyle=source_style[src]["linestyle"],
                marker="s",
                label=f"{src} NOCA rotational + viscous",
                band_alpha=band_alpha,
                zorder=source_zorder,
            )
            _plot_curve(
                ax_left,
                y_m,
                inv["mean"],
                inv["spread"],
                color=source_style[src]["color"],
                linestyle="-." if src == "CFD" else "--",
                marker="o",
                label=f"{src} NOCA inviscid",
                band_alpha=band_alpha,
                markersize=9,
                markerfacecolor="none",
                markeredgewidth=1.4,
                zorder=source_zorder,
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
        ax_left.set_ylabel(r"$C_l$ (-)" if coef_key == "Cl" else r"$C_d$ (-)")

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
                label=f"{src} NOCA total",
                band_alpha=0.12 if src == "CFD" else 0.08,
                zorder=source_zorder,
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

    # Split legends across top-row columns.
    handles_left, labels_left = axes[0, 0].get_legend_handles_labels()
    handles_right, labels_right = axes[0, 1].get_legend_handles_labels()
    handles_shape, labels_shape = overlay_axes[(0, 1)].get_legend_handles_labels()
    handle_map = {}
    for handle, label in (
        list(zip(handles_left, labels_left))
        + list(zip(handles_right, labels_right))
        + list(zip(handles_shape, labels_shape))
    ):
        if label not in handle_map:
            handle_map[label] = handle
    if "Shape outline" not in handle_map:
        handle_map["Shape outline"] = Line2D(
            [0], [0], color="black", alpha=0.5, linewidth=1.1
        )

    legend_left_order = [
        "CFD NOCA rotational + viscous",
        "PIV NOCA rotational + viscous",
        "CFD NOCA inviscid",
        "PIV NOCA inviscid",
    ]
    legend_right_order = [
        "CFD NOCA total",
        "PIV NOCA total",
        "CFD pressure integration",
        "Shape outline",
        "Measurement planes",
    ]

    legend_left = overlay_axes[(0, 0)].legend(
        handles=[
            handle_map[label] for label in legend_left_order if label in handle_map
        ],
        labels=[label for label in legend_left_order if label in handle_map],
        loc="upper left",
        fontsize=11,
        framealpha=1.0,
        fancybox=False,
        facecolor="white",
        # edgecolor="black",
    )
    legend_left.set_zorder(1000)

    legend_right = overlay_axes[(0, 1)].legend(
        handles=[
            handle_map[label] for label in legend_right_order if label in handle_map
        ],
        labels=[label for label in legend_right_order if label in handle_map],
        loc="lower left",
        fontsize=11,
        framealpha=1.0,
        fancybox=False,
        facecolor="white",
        # edgecolor="black",
    )
    legend_right.set_zorder(1000)

    for axes_ in axes:
        for ax in axes_:
            ax.set_xlim(-0.03, 0.67)
    # Keep x-limits fixed and enforce equal x/z data scale on the overlay axis.
    for overlay_ax in overlay_axes.values():
        overlay_ax.set_aspect("equal", adjustable="datalim")

    fig.subplots_adjust(left=0.09, right=0.91, top=0.93, bottom=0.08)

    if save_path is None:
        save_path = (
            Path(project_dir) / "results" / "paper_plots_24_03_2026" / "fig09_2col.pdf"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved fig09 2-column to: {save_path}")
    return save_path


def main() -> None:
    # plot_fig09_separate_terms()
    plot_fig09_2col()


if __name__ == "__main__":
    main()
