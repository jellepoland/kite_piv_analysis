from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kite_piv_analysis.Table06_print_quantitative_results import (
    load_bernoulli_contour_coefficients,
    load_pressure_integration_coefficients,
    read_results,
)
from kite_piv_analysis.plot_styling import set_plot_style
from kite_piv_analysis.utils import project_dir


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


def _collect_series(
    alpha: int,
    y_nums: list[int],
    noca_field: str,
    pressure_field: str,
    bernoulli_field: str,
) -> dict:
    noca = {
        "CFD": {"mean": [], "spread": []},
        "PIV": {"mean": [], "spread": []},
    }
    bernoulli = {
        "CFD": {"mean": [], "spread": []},
        "PIV": {"mean": [], "spread": []},
    }

    for y_num in y_nums:
        for data_label, is_cfd in [("CFD", True), ("PIV", False)]:
            noca_vals: list[float] = []
            for is_ellipse in [True, False]:
                try:
                    res = read_results(
                        alpha=alpha, y_num=y_num, is_CFD=is_cfd, is_ellipse=is_ellipse
                    )
                    noca_vals.append(float(res[noca_field]))
                except (FileNotFoundError, ValueError):
                    noca_vals.append(np.nan)
            mean_val, spread_val = _mean_and_spread(noca_vals)
            noca[data_label]["mean"].append(mean_val)
            noca[data_label]["spread"].append(spread_val)

    plot_pairs = {(alpha, y_num) for y_num in y_nums}
    pressure_map = load_pressure_integration_coefficients(
        required_pairs=plot_pairs, strict=False
    )
    pressure = np.asarray(
        [
            (
                float(pressure_map[(alpha, y)][pressure_field])
                if (alpha, y) in pressure_map
                else np.nan
            )
            for y in y_nums
        ],
        dtype=float,
    )

    bernoulli_required_keys = {
        (alpha, y_num, shape, data_type)
        for y_num in y_nums
        for shape in ["ellipse", "rectangle"]
        for data_type in ["CFD", "PIV"]
    }
    bernoulli_map = load_bernoulli_contour_coefficients(
        required_keys=bernoulli_required_keys, strict=False
    )
    for y_num in y_nums:
        for data_label in ["CFD", "PIV"]:
            b_vals: list[float] = []
            for shape in ["ellipse", "rectangle"]:
                key = (alpha, y_num, shape, data_label)
                if key in bernoulli_map:
                    b_vals.append(float(bernoulli_map[key][bernoulli_field]))
                else:
                    b_vals.append(np.nan)
            mean_val, spread_val = _mean_and_spread(b_vals)
            bernoulli[data_label]["mean"].append(mean_val)
            bernoulli[data_label]["spread"].append(spread_val)

    return {
        "noca": noca,
        "bernoulli": bernoulli,
        "pressure": pressure,
    }


def _plot_single(
    ax: plt.Axes,
    x: np.ndarray,
    data: dict,
    y_label_tex: str,
    with_legend: bool = True,
) -> None:
    # Method-based colors requested by user.
    noca_color = "red"
    bernoulli_color = "blue"
    pressure_color = "black"

    # NOCA (solid)
    ax.errorbar(
        x,
        data["noca"]["CFD"]["mean"],
        yerr=data["noca"]["CFD"]["spread"],
        color=noca_color,
        linestyle="-",
        marker="o",
        capsize=3,
        label="NOCA CFD",
    )
    ax.errorbar(
        x,
        data["noca"]["PIV"]["mean"],
        yerr=data["noca"]["PIV"]["spread"],
        color=noca_color,
        linestyle="-",
        marker="s",
        capsize=3,
        label="NOCA PIV",
    )

    # Bernoulli (dashed)
    ax.errorbar(
        x,
        data["bernoulli"]["CFD"]["mean"],
        yerr=data["bernoulli"]["CFD"]["spread"],
        color=bernoulli_color,
        linestyle="--",
        marker="^",
        capsize=3,
        label="Bernoulli CFD",
    )
    ax.errorbar(
        x,
        data["bernoulli"]["PIV"]["mean"],
        yerr=data["bernoulli"]["PIV"]["spread"],
        color=bernoulli_color,
        linestyle="--",
        marker="v",
        capsize=3,
        label="Bernoulli PIV",
    )

    # Pressure (dash-dot, CFD only)
    ax.plot(
        x,
        data["pressure"],
        color=pressure_color,
        linestyle="-.",
        marker="x",
        linewidth=1.2,
        label="Pressure CFD",
    )

    ax.set_xlim(0.0, 0.7)
    ax.set_xlabel(r"$y$ (m)")
    ax.set_ylabel(y_label_tex)
    ax.grid(True, alpha=0.4)
    if with_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, ncol=2, loc="best")


def _plot_shape_outline_panel(
    ax: plt.Axes,
    alpha: int = 6,
    cm_offset: int = 25,
) -> None:
    from kite_piv_analysis import extract_spanwise_contour

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

    if outline_path.exists():
        extract_spanwise_contour.main(ax, outline_path)
    else:
        ax.text(
            0.5,
            0.5,
            "shape outline\\nnot found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # Geometric subplot keeps equal axis because x/y units are the same.
    x_min, x_max = 0.0, 0.7
    y_auto_min, y_auto_max = ax.get_ylim()
    y_center = 0.5 * (y_auto_min + y_auto_max)
    y_half_span = 0.5 * (x_max - x_min)  # equal physical step-size in x/y
    ax.set_xlim(x_min, x_max)
    factor = 0.7
    ax.set_ylim(y_center - factor * y_half_span + 0.02, y_center + factor * y_half_span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$y$ (m)")
    ax.set_ylabel(r"$z$ (m)")
    ax.grid(True)


def _plot_circulation_panel(ax: plt.Axes) -> None:
    from kite_piv_analysis.fig08_gamma_distribution import get_VSM_gamma_distribution

    df_y_locations = pd.read_csv(
        Path(project_dir) / "data" / "gamma_distribution" / "y_locations.csv",
        index_col=False,
    )
    y_numbers = df_y_locations["PIV_mm"] / 1000

    csv_path = (
        Path(project_dir)
        / "processed_data"
        / "quantitative_chordwise_analysis_alpha_6_with_std.csv"
    )
    if not csv_path.exists():
        ax.text(
            0.5,
            0.5,
            "circulation cache\\nnot found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    df = pd.read_csv(csv_path)
    cfd_gamma_ellipse = df["ellipse_cfd_gamma"]
    cfd_gamma_rectangle = df["rectangle_cfd_gamma"]
    piv_gamma_ellipse = df["ellipse_piv_gamma"]
    piv_gamma_rectangle = df["rectangle_piv_gamma"]

    # Match current fig08 behavior: show CFD/PIV through Y6.
    plot_count = min(
        6,
        len(y_numbers),
        len(cfd_gamma_ellipse),
        len(cfd_gamma_rectangle),
        len(piv_gamma_ellipse),
        len(piv_gamma_rectangle),
    )
    y_plot = np.asarray(y_numbers.iloc[:plot_count], dtype=float)
    cfd_ellipse_plot = np.asarray(cfd_gamma_ellipse.iloc[:plot_count], dtype=float)
    cfd_rectangle_plot = np.asarray(cfd_gamma_rectangle.iloc[:plot_count], dtype=float)
    piv_ellipse_plot = np.asarray(piv_gamma_ellipse.iloc[:plot_count], dtype=float)
    piv_rectangle_plot = np.asarray(piv_gamma_rectangle.iloc[:plot_count], dtype=float)

    cfd_gamma_mean = (cfd_ellipse_plot + cfd_rectangle_plot) / 2
    cfd_gamma_min = np.minimum(cfd_ellipse_plot, cfd_rectangle_plot)
    cfd_gamma_max = np.maximum(cfd_ellipse_plot, cfd_rectangle_plot)

    piv_gamma_mean = (piv_ellipse_plot + piv_rectangle_plot) / 2
    piv_gamma_min = np.minimum(piv_ellipse_plot, piv_rectangle_plot)
    piv_gamma_max = np.maximum(piv_ellipse_plot, piv_rectangle_plot)

    try:
        vsm_gamma, cad_y = get_VSM_gamma_distribution()
        mask = cad_y > -1e-6
        ax.plot(cad_y[mask], vsm_gamma[mask], color="black", linestyle="-", label="VSM")
    except Exception:
        pass

    ax.plot(
        y_plot, cfd_gamma_mean, color="blue", marker="*", linestyle="-", label="CFD"
    )
    ax.fill_between(
        y_plot,
        cfd_gamma_min,
        cfd_gamma_max,
        color="blue",
        alpha=0.2,
        label="CFD range",
    )
    ax.plot(y_plot, piv_gamma_mean, color="red", marker=".", linestyle="-", label="PIV")
    ax.fill_between(
        y_plot,
        piv_gamma_min,
        piv_gamma_max,
        color="red",
        alpha=0.2,
        label="PIV range",
    )

    ax.set_xlim(0.0, 0.7)
    ax.set_ylim(0, 2.5)
    ax.set_xlabel(r"$y$ (m)")
    ax.set_ylabel(r"$\Gamma$ (m$^2$s$^{-1}$)")
    ax.grid(True, alpha=0.4)
    ax.legend(ncol=2, loc="best")


def plot_alpha6_quantitative_results(
    save_path_cl: Path | None = None,
    save_path_cd: Path | None = None,
) -> tuple[Path, Path]:
    alpha = 6
    y_nums = [1, 2, 3, 4, 5, 6, 7]
    y_m = _load_y_locations_m(y_nums)

    cl_data = _collect_series(
        alpha=alpha,
        y_nums=y_nums,
        noca_field="C_l",
        pressure_field="C_l_p",
        bernoulli_field="C_l_bernoulli",
    )
    cd_data = _collect_series(
        alpha=alpha,
        y_nums=y_nums,
        noca_field="C_d",
        pressure_field="C_d_p",
        bernoulli_field="C_d_bernoulli",
    )

    set_plot_style()

    if save_path_cl is None:
        save_path_cl = (
            Path(project_dir) / "results" / "paper_plots_24_03_2026" / "new_figure.pdf"
        )
    if save_path_cd is None:
        save_path_cd = (
            Path(project_dir)
            / "results"
            / "paper_plots_24_03_2026"
            / "new_figure_CD.pdf"
        )

    save_path_cl.parent.mkdir(parents=True, exist_ok=True)
    save_path_cd.parent.mkdir(parents=True, exist_ok=True)

    fig_cl, ax_cl = plt.subplots(figsize=(6, 6))
    _plot_single(ax_cl, y_m, cl_data, y_label_tex=r"$C_l$ (-)", with_legend=True)
    ax_cl.set_title(r"$\alpha = 6^\circ$ quantitative comparison")
    fig_cl.tight_layout(pad=0.02)
    fig_cl.savefig(save_path_cl, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig_cl)

    fig_cd, ax_cd = plt.subplots(figsize=(6, 6))
    _plot_single(ax_cd, y_m, cd_data, y_label_tex=r"$C_d$ (-)", with_legend=True)
    ax_cd.set_title(r"$\alpha = 6^\circ$ quantitative comparison")
    fig_cd.tight_layout(pad=0.02)
    fig_cd.savefig(save_path_cd, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig_cd)

    print(f"Saved CL quantitative figure to: {save_path_cl}")
    print(f"Saved CD quantitative figure to: {save_path_cd}")
    return save_path_cl, save_path_cd


def plot_stacked_figure(save_path: Path | None = None) -> Path:
    alpha = 6
    y_nums = [1, 2, 3, 4, 5, 6, 7]
    y_m = _load_y_locations_m(y_nums)

    cl_data = _collect_series(
        alpha=alpha,
        y_nums=y_nums,
        noca_field="C_l",
        pressure_field="C_l_p",
        bernoulli_field="C_l_bernoulli",
    )
    cd_data = _collect_series(
        alpha=alpha,
        y_nums=y_nums,
        noca_field="C_d",
        pressure_field="C_d_p",
        bernoulli_field="C_d_bernoulli",
    )

    set_plot_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9.5),
        gridspec_kw={"hspace": 0.01, "wspace": 0.1},
    )

    _plot_shape_outline_panel(axes[0, 0], alpha=6, cm_offset=25)
    _plot_circulation_panel(axes[0, 1])
    _plot_single(axes[1, 0], y_m, cl_data, y_label_tex=r"$C_l$ (-)", with_legend=True)
    _plot_single(axes[1, 1], y_m, cd_data, y_label_tex=r"$C_d$ (-)", with_legend=True)

    # Enforce common x-range for all subplots after all plotting calls.
    for ax in axes.flat:
        ax.set_xlim(0.0, 0.65)

    # Match subplot box aspect ratio to the top-left panel.
    # This lets you control all panel aspect ratios by changing row0/col0 y-limits.
    x0_min, x0_max = axes[0, 0].get_xlim()
    y0_min, y0_max = axes[0, 0].get_ylim()
    x0_span = abs(x0_max - x0_min)
    y0_span = abs(y0_max - y0_min)
    target_box_aspect = (y0_span / x0_span) if x0_span > 1e-12 else 1.0
    for ax in axes.flat:
        ax.set_box_aspect(target_box_aspect)

    # Top row: no x-axis label/tick labels.
    for ax in axes[0, :]:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.08)

    if save_path is None:
        save_path = (
            Path(project_dir) / "results" / "paper_plots_24_03_2026" / "fig_stacked.pdf"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved stacked quantitative figure to: {save_path}")
    return save_path


def plot_col_stacked_figure(save_path: Path | None = None) -> Path:
    from matplotlib.patches import Patch

    alpha = 6
    y_nums = [1, 2, 3, 4, 5, 6, 7]
    y_m = _load_y_locations_m(y_nums)

    cl_data = _collect_series(
        alpha=alpha,
        y_nums=y_nums,
        noca_field="C_l",
        pressure_field="C_l_p",
        bernoulli_field="C_l_bernoulli",
    )
    cd_data = _collect_series(
        alpha=alpha,
        y_nums=y_nums,
        noca_field="C_d",
        pressure_field="C_d_p",
        bernoulli_field="C_d_bernoulli",
    )

    set_plot_style()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.5, 13),
        sharex=True,
        gridspec_kw={"hspace": 0.06},
    )

    # Top: shape outline with its own legend.
    _plot_shape_outline_panel(axes[0], alpha=6, cm_offset=25)
    axes[0].legend(
        handles=[Patch(facecolor="black", edgecolor="black", label="Wing outline")],
        ncol=1,
        loc="best",
    )

    # Middle: CL with legend.
    _plot_single(axes[1], y_m, cl_data, y_label_tex=r"$C_l$ (-)", with_legend=True)

    # Bottom: CD without legend.
    _plot_single(axes[2], y_m, cd_data, y_label_tex=r"$C_d$ (-)", with_legend=False)

    # Shared x-axis range across all rows.
    for ax in axes:
        ax.set_xlim(0.0, 0.7)

    # Match subplot box aspect ratio to shape-outline panel.
    x0_min, x0_max = axes[0].get_xlim()
    y0_min, y0_max = axes[0].get_ylim()
    x0_span = abs(x0_max - x0_min)
    y0_span = abs(y0_max - y0_min)
    target_box_aspect = (y0_span / x0_span) if x0_span > 1e-12 else 1.0
    for ax in axes:
        ax.set_box_aspect(target_box_aspect)

    # Only bottom row gets x tick labels and x-axis label.
    for ax in axes[:2]:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    axes[2].set_xlabel(r"$y$ (m)")

    fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.07)

    if save_path is None:
        save_path = (
            Path(project_dir)
            / "results"
            / "paper_plots_24_03_2026"
            / "fig_col_stacked.pdf"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved column-stacked quantitative figure to: {save_path}")
    return save_path


def main():
    plot_alpha6_quantitative_results()
    plot_stacked_figure()
    plot_col_stacked_figure()


if __name__ == "__main__":
    main()
