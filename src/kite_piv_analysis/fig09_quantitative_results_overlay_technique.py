from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
            vals: list[float] = []
            for is_ellipse in [True, False]:
                try:
                    res = read_results(
                        alpha=alpha,
                        y_num=y_num,
                        is_CFD=is_cfd,
                        is_ellipse=is_ellipse,
                    )
                    vals.append(float(res[noca_field]))
                except (FileNotFoundError, ValueError):
                    vals.append(np.nan)
            mean_val, spread_val = _mean_and_spread(vals)
            noca[data_label]["mean"].append(mean_val)
            noca[data_label]["spread"].append(spread_val)

    required_pairs = {(alpha, y_num) for y_num in y_nums}
    pressure_map = load_pressure_integration_coefficients(
        required_pairs=required_pairs,
        strict=False,
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

    required_keys = {
        (alpha, y_num, shape, data_type)
        for y_num in y_nums
        for shape in ["ellipse", "rectangle"]
        for data_type in ["CFD", "PIV"]
    }
    bernoulli_map = load_bernoulli_contour_coefficients(
        required_keys=required_keys,
        strict=False,
    )
    for y_num in y_nums:
        for data_label in ["CFD", "PIV"]:
            vals: list[float] = []
            for shape in ["ellipse", "rectangle"]:
                key = (alpha, y_num, shape, data_label)
                if key in bernoulli_map:
                    vals.append(float(bernoulli_map[key][bernoulli_field]))
                else:
                    vals.append(np.nan)
            mean_val, spread_val = _mean_and_spread(vals)
            bernoulli[data_label]["mean"].append(mean_val)
            bernoulli[data_label]["spread"].append(spread_val)

    return {
        "noca": noca,
        "bernoulli": bernoulli,
        "pressure": pressure,
    }


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


def _plot_pressure_noca(
    ax: plt.Axes, x: np.ndarray, series: dict, y_label: str
) -> None:
    noca_cfd_mean = np.asarray(series["noca"]["CFD"]["mean"], dtype=float)
    noca_cfd_spread = np.asarray(series["noca"]["CFD"]["spread"], dtype=float)
    noca_piv_mean = np.asarray(series["noca"]["PIV"]["mean"], dtype=float)
    noca_piv_spread = np.asarray(series["noca"]["PIV"]["spread"], dtype=float)
    ax.plot(
        x,
        noca_cfd_mean,
        color="blue",
        linestyle="-",
        marker="o",
        label="NOCA CFD",
    )
    ax.fill_between(
        x,
        noca_cfd_mean - noca_cfd_spread,
        noca_cfd_mean + noca_cfd_spread,
        color="blue",
        alpha=0.2,
    )
    ax.plot(
        x,
        noca_piv_mean,
        color="red",
        linestyle="--",
        marker="^",
        label="NOCA PIV",
    )
    ax.fill_between(
        x,
        noca_piv_mean - noca_piv_spread,
        noca_piv_mean + noca_piv_spread,
        color="red",
        alpha=0.1,
    )
    ax.plot(
        x,
        series["pressure"],
        color="green",
        linestyle="-",
        marker="x",
        linewidth=1.2,
        label="Pressure CFD",
    )
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.2)


def _plot_pressure_bernoulli(
    ax: plt.Axes,
    x: np.ndarray,
    series: dict,
    y_label: str,
) -> None:
    bern_cfd_mean = np.asarray(series["bernoulli"]["CFD"]["mean"], dtype=float)
    bern_cfd_spread = np.asarray(series["bernoulli"]["CFD"]["spread"], dtype=float)
    bern_piv_mean = np.asarray(series["bernoulli"]["PIV"]["mean"], dtype=float)
    bern_piv_spread = np.asarray(series["bernoulli"]["PIV"]["spread"], dtype=float)
    ax.plot(
        x,
        bern_cfd_mean,
        color="blue",
        linestyle="-",
        marker="o",
        label="Bernoulli CFD",
    )
    ax.fill_between(
        x,
        bern_cfd_mean - bern_cfd_spread,
        bern_cfd_mean + bern_cfd_spread,
        color="blue",
        alpha=0.2,
    )
    ax.plot(
        x,
        bern_piv_mean,
        color="red",
        linestyle="--",
        marker="^",
        label="Bernoulli PIV",
    )
    ax.fill_between(
        x,
        bern_piv_mean - bern_piv_spread,
        bern_piv_mean + bern_piv_spread,
        color="red",
        alpha=0.1,
    )
    ax.plot(
        x,
        series["pressure"],
        color="green",
        linestyle="-",
        marker="x",
        linewidth=1.2,
        label="Pressure CFD",
    )
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.2)


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


def plot_fig09(save_path: Path | None = None) -> Path:
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

    x_shape, z_shape = _load_shape_outline(alpha=6, cm_offset=25)
    if len(x_shape) > 0 and len(z_shape) > 0:
        z_min = float(np.nanmin(z_shape))
        z_max = float(np.nanmax(z_shape))
    else:
        z_min, z_max = -0.6, 0.1
    z_span = max(z_max - z_min, 1e-6)
    z_pad = 0.05 * z_span
    z_lim = (z_min - z_pad, z_max + z_pad)

    set_plot_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        sharex=True,
        sharey="row",
        gridspec_kw={"hspace": 0.04, "wspace": 0.08},
    )

    # Row 0: C_l; Row 1: C_d
    # Col 0: pressure + NOCA; Col 1: pressure + Bernoulli
    _plot_pressure_noca(axes[0, 0], y_m, cl_data, y_label=r"$C_l$ (-)")
    _plot_pressure_bernoulli(axes[0, 1], y_m, cl_data, y_label=r"$C_l$ (-)")
    _plot_pressure_noca(axes[1, 0], y_m, cd_data, y_label=r"$C_d$ (-)")
    _plot_pressure_bernoulli(axes[1, 1], y_m, cd_data, y_label=r"$C_d$ (-)")

    # Enforce identical left-y scales per row.
    for r in [0, 1]:
        y0 = axes[r, 0].get_ylim()
        y1 = axes[r, 1].get_ylim()
        y_min = min(y0[0], y1[0])
        y_max = max(y0[1], y1[1])
        axes[r, 0].set_ylim(y_min, y_max)
        axes[r, 1].set_ylim(y_min, y_max)

    for ax in axes.flat:
        ax.set_xlim(0.0, 0.7)

    # Add shape overlay with right-side z-axis only on right column.
    for r in [0, 1]:
        for c in [0, 1]:
            _add_shape_overlay(
                axes[r, c],
                x_shape=x_shape,
                z_shape=z_shape,
                z_lim=z_lim,
                show_right_axis=(c == 1),
            )

    # Shape-driven panel aspect ratio.
    x_span = 0.7
    y_span_shape = z_lim[1] - z_lim[0]
    target_box_aspect = (y_span_shape / x_span) if x_span > 1e-12 else 1.0
    for ax in axes.flat:
        ax.set_box_aspect(target_box_aspect)

    # Left y-axis labels/ticks only in col 0.
    for ax in axes[:, 1]:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)

    # x-axis labels/ticks only in bottom row.
    for ax in axes[0, :]:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    for ax in axes[1, :]:
        ax.set_xlabel(r"$y$ (m)")

    # adjust the ylims of cl
    for ax in axes[0, :]:
        ax.set_ylim(0, 1.0)
    # adjust for ylim of cd
    for ax in axes[1, :]:
        ax.set_ylim(-0.4, 0.4)

    for ax in axes.flat:
        ax.grid(True, alpha=0.2)

    # Single external legend.
    legend_handles = [
        Line2D([0], [0], color="blue", linestyle="-", marker="o", label="NOCA CFD"),
        Line2D([0], [0], color="red", linestyle="--", marker="^", label="NOCA PIV"),
        Line2D(
            [0],
            [0],
            color="blue",
            linestyle="-",
            marker="o",
            label="Bernoulli CFD",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            marker="^",
            label="Bernoulli PIV",
        ),
        Line2D(
            [0],
            [0],
            color="green",
            linestyle="-",
            marker="x",
            label="Pressure CFD",
        ),
        Line2D(
            [0], [0], color="black", alpha=0.5, linestyle="-", label="Shape outline"
        ),
        Patch(
            facecolor="blue",
            edgecolor="none",
            alpha=0.2,
            label="CFD shape uncertainty band",
        ),
        Patch(
            facecolor="red",
            edgecolor="none",
            alpha=0.1,
            label="PIV shape uncertainty band",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.subplots_adjust(left=0.09, right=0.91, top=0.98, bottom=0.16)

    if save_path is None:
        save_path = (
            Path(project_dir) / "results" / "paper_plots_24_03_2026" / "fig09.pdf"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved fig09 to: {save_path}")
    return save_path


def main():
    plot_fig09()


if __name__ == "__main__":
    main()
