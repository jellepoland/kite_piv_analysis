"""
Minimal driver to generate the suction‑side overlap plots for aoa=23.
Focuses only on Y1, following the MATLAB stitching procedure.
Shows only raw differences in overlap regions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from kite_piv_analysis.utils import project_dir
from kite_piv_analysis.compute_overlap_error import (
    load_dat_file,
    interpolate_to_grid,
)
from kite_piv_analysis.masking import apply_snr_masking, apply_w_masking
from kite_piv_analysis.plot_styling import set_plot_style
from kite_piv_analysis.plotting import plot_airfoil


def _alpha_from_aoa(aoa: int) -> int:
    return {13: 6, 23: 16}.get(int(aoa), int(aoa))


def _mask_plot_params(aoa: int, y_num: int, w_threshold: float = 3.0) -> dict:
    return {
        "alpha": _alpha_from_aoa(aoa),
        "y_num": y_num,
        "is_with_w_mask": True,
        "w_mask_lower_bound": -abs(float(w_threshold)),
        "w_mask_upper_bound": abs(float(w_threshold)),
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


def _apply_combined_mask_to_plane(
    mean_data: dict,
    std_data: dict,
    aoa: int,
    y_num: int,
    w_threshold: float = 3.0,
) -> dict:
    """
    Apply combined quality mask (w + ICV/proxy-SNR) to raw plane data.
    """
    df_mean = pd.DataFrame(
        {
            "x": mean_data["x"],
            "y": mean_data["y"],
            "u": mean_data["u"],
            "v": mean_data["v"],
            "w": mean_data["w"],
            "V": mean_data["V"],
            "is_valid": mean_data["is_valid"],
        }
    )
    df_std = pd.DataFrame(
        {
            "u": std_data["u"],
            "v": std_data["v"],
            "w": std_data["w"],
        }
    )
    mask_params = _mask_plot_params(aoa, y_num, w_threshold=w_threshold)
    mask_params["df_std"] = df_std

    df_masked = apply_w_masking(df_mean, mask_params)
    df_masked = apply_snr_masking(df_masked, mask_params)

    out = dict(mean_data)
    for key in ["x", "y", "u", "v", "w", "V", "is_valid"]:
        out[key] = df_masked[key].to_numpy()
    return out


def collect_y1_data(
    q=0.6,
    aoa=23,
    y_num=1,
    override_x3_shift=None,
    override_dx_normal=None,
    override_dx_flipped=None,
    extra_dy_flipped=None,
):
    """
    Load flipped (top) and normal (bottom) planes for the requested AoA and
    return interpolated data + metadata for Y1 only.
    Applies per-plane delta_x/delta_y from planes_location.csv.
    """
    # Primary input source only.
    input_dir = Path(project_dir) / "data_ALL_ERIK_FILES" / "JelleStitching" / "Input"

    # plane offsets table (alpha stored as 6/16; map AoA 23 -> alpha 16)
    planes_loc = pd.read_csv(
        Path(project_dir)
        / "data_ALL_ERIK_FILES"
        / "JelleStitching"
        / "Input"
        / "planes_location.csv"
    )
    planes_loc = planes_loc.rename(columns={"Unnamed: 0": "idx"})
    alpha_map = {23: 16, 13: 6}
    alpha = alpha_map.get(aoa, aoa)

    aoa_dir = input_dir / f"aoa_{aoa}"
    if not aoa_dir.exists():
        print(f"aoa_{aoa} folder not found")
        return None

    y_planes = sorted(int(d.name[1:]) for d in aoa_dir.iterdir() if d.is_dir())

    for y_plane in y_planes:
        if y_plane != y_num:
            continue
        base_path = aoa_dir / f"Y{y_plane}"

        configs_data = {}
        for config in ["flipped", "normal"]:
            plane_dirs = {}
            for item in base_path.iterdir():
                if item.is_dir() and config in item.name:
                    if "_X1" in item.name:
                        plane_dirs.setdefault("X1", item)
                    elif "_X2" in item.name:
                        plane_dirs.setdefault("X2", item)
                    elif "_X3" in item.name:
                        plane_dirs.setdefault("X3", item)

            if len(plane_dirs) < 2:
                continue

            planes_data = {}
            planes_std_data = {}
            for plane_name, plane_dir in plane_dirs.items():
                mean_file = plane_dir / "B0001.dat"
                std_file = plane_dir / "B0002.dat"
                if mean_file.exists() and std_file.exists():
                    planes_data[plane_name] = load_dat_file(mean_file)
                    planes_std_data[plane_name] = load_dat_file(std_file)
                else:
                    continue

                # apply per-plane offsets
                row = planes_loc[
                    (planes_loc["alpha"] == alpha)
                    & (planes_loc["config"] == config)
                    & (planes_loc["Y"] == y_plane)
                    & (planes_loc["X"] == int(plane_name[1]))
                ]
                if not row.empty:
                    dx_shift = float(row.iloc[0]["delta_x"]) / 1000.0
                    dy_shift = float(row.iloc[0]["delta_y"]) / 1000.0

                    # Optional additional per-config streamwise shifts (mm) applied on top
                    if config == "normal" and override_dx_normal is not None:
                        dx_shift += override_dx_normal / 1000.0
                    if config == "flipped" and override_dx_flipped is not None:
                        dx_shift += override_dx_flipped / 1000.0
                    # Optional extra vertical shift for flipped (mm) applied on top
                    if config == "flipped" and extra_dy_flipped is not None:
                        dy_shift += extra_dy_flipped / 1000.0

                    if plane_name == "X3" and override_x3_shift is not None:
                        planes_data[plane_name]["x"] += override_x3_shift / 1000.0
                    else:
                        planes_data[plane_name]["x"] += dx_shift
                    planes_data[plane_name]["y"] += dy_shift
                    planes_data[plane_name]["_dy_shift"] = dy_shift

            if len(planes_data) < 2:
                continue

            # flip y for suction side
            for plane_name, data in planes_data.items():
                if config == "flipped":
                    # Mirror vertically and flip the vertical velocity component
                    dy_local = data.get("_dy_shift", 0.0)
                    # Mirror vertically, then shift up by 2*dy_local, and flip v component
                    data["y"] = -data["y"] + 2 * dy_local
                    if "v" in data:
                        data["v"] = -data["v"]

                # Apply combined quality mask (w + ICV) after geometric transforms.
                if plane_name in planes_std_data:
                    planes_data[plane_name] = _apply_combined_mask_to_plane(
                        data,
                        planes_std_data[plane_name],
                        aoa=aoa,
                        y_num=y_plane,
                        w_threshold=3.0,
                    )

            configs_data[config] = planes_data

        if not configs_data:
            continue

        # build global grid from both configs
        all_x = []
        all_y = []
        for planes in configs_data.values():
            all_x.append(np.concatenate([p["x"] for p in planes.values()]))
            all_y.append(np.concatenate([p["y"] for p in planes.values()]))
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        dx = dy = 0.005
        grid_x_1d = np.arange(all_x.min(), all_x.max() + dx, dx)
        grid_y_1d = np.arange(all_y.min(), all_y.max() + dy, dy)
        grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

        interpolated_by_config = {}
        for config, planes in configs_data.items():
            interpolated_by_config[config] = {
                plane: interpolate_to_grid(data, grid_x, grid_y)
                for plane, data in planes.items()
            }

        if y_plane == y_num:
            return {
                "interpolated_by_config": interpolated_by_config,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "y_threshold": grid_y.min() + q * (grid_y.max() - grid_y.min()),
            }

    print(f"No Y1 data found for aoa={aoa}")
    return None


def plot_y1_overlap_grid(
    y1_data,
    output_dir,
    q=0.6,
    aoa=23,
    y_num=1,
    w_threshold=3.0,
    vmin=False,
    vmax=False,
    y_mask_min_height=0.2,
):
    """
    Build a 2x2 figure showing raw velocity differences in overlap regions.
    Following MATLAB procedure: just show plane1 - plane2 in overlaps.
    Merges top (flipped) and bottom (normal) configurations.
    Shows both X1-X2 and X2-X3 overlaps.
    """
    set_plot_style()
    interp_by_cfg = y1_data.get("interpolated_by_config", {})
    grid_x = y1_data["grid_x"]
    grid_y = y1_data["grid_y"]

    top_cfg = interp_by_cfg.get("flipped")  # top/suction side
    bottom_cfg = interp_by_cfg.get("normal")  # bottom/pressure side

    if top_cfg is None and bottom_cfg is None:
        print("No interpolated data found for flipped or normal configurations.")
        return

    def magnitude(plane):
        """Compute |V|; keep NaNs where any component is missing."""
        u = plane["u"]
        v = plane["v"]
        w = plane["w"]
        mag = np.full_like(u, np.nan)
        m = ~np.isnan(u) & ~np.isnan(v) & ~np.isnan(w)
        mag[m] = np.sqrt(u[m] ** 2 + v[m] ** 2 + w[m] ** 2)
        return mag

    def apply_filters(plane):
        """Apply a final |w| guard before diffs (data already pre-masked by w+ICV)."""
        out = {k: np.array(v, copy=True) for k, v in plane.items()}
        mask = np.ones_like(out["w"], dtype=bool)
        if "is_valid" in out:
            mask &= out["is_valid"] != 0
        mask &= np.abs(out["w"]) <= w_threshold
        for k in ["u", "v", "w"]:
            out[k] = np.where(mask, out[k], np.nan)
        return out

    def compute_raw_diffs(cfg_planes):
        """
        Compute raw differences (plane1 - plane2) in overlap regions.
        This is the actual stitching uncertainty, following MATLAB approach.
        """
        has_x3_local = "X3" in cfg_planes
        rows_labels_local = ["u", "v", "w", "V"]

        plane1 = apply_filters({k: cfg_planes["X1"][k] for k in ["u", "v", "w"]})
        plane2 = apply_filters({k: cfg_planes["X2"][k] for k in ["u", "v", "w"]})
        if has_x3_local:
            plane3 = apply_filters({k: cfg_planes["X3"][k] for k in ["u", "v", "w"]})
        else:
            plane3 = None

        # Add velocity magnitude
        plane1["V"] = magnitude(plane1)
        plane2["V"] = magnitude(plane2)
        if has_x3_local:
            plane3["V"] = magnitude(plane3)

        # Compute raw differences in overlap regions
        raw12 = {}
        raw23 = {} if has_x3_local else None

        for key in rows_labels_local:
            c1 = plane1[key]
            c2 = plane2[key]

            # X1-X2 overlap: raw difference
            mask12 = ~np.isnan(c1) & ~np.isnan(c2)
            r12 = np.full_like(c1, np.nan)
            r12[mask12] = c1[mask12] - c2[mask12]
            raw12[key] = r12

            # X2-X3 overlap: raw difference
            if has_x3_local:
                c3 = plane3[key]
                mask23 = ~np.isnan(c2) & ~np.isnan(c3)
                r23 = np.full_like(c2, np.nan)
                r23[mask23] = c2[mask23] - c3[mask23]
                raw23[key] = r23

        return raw12, raw23

    # Compute raw differences for both configurations
    top_raw12 = top_raw23 = None
    if top_cfg is not None:
        top_raw12, top_raw23 = compute_raw_diffs(top_cfg)

    bot_raw12 = bot_raw23 = None
    if bottom_cfg is not None:
        bot_raw12, bot_raw23 = compute_raw_diffs(bottom_cfg)

    has_x3 = (top_raw23 is not None) or (bot_raw23 is not None)
    rows_labels = ["u", "v", "w", "V"]

    def blend(top_arr, bottom_arr):
        """
        Blend top (flipped/suction) and bottom (normal/pressure) data
        along vertical overlap using sigmoid weighting.
        """
        if top_arr is None and bottom_arr is None:
            return None
        if top_arr is None:
            return bottom_arr.copy()
        if bottom_arr is None:
            return top_arr.copy()

        merged = np.full_like(top_arr, np.nan)
        top_mask = ~np.isnan(top_arr)
        bot_mask = ~np.isnan(bottom_arr)

        # Non-overlapping regions: use available data
        merged[top_mask & ~bot_mask] = top_arr[top_mask & ~bot_mask]
        merged[bot_mask & ~top_mask] = bottom_arr[bot_mask & ~top_mask]

        # Overlapping region: sigmoid blend
        both = top_mask & bot_mask
        if both.any():
            y_vals = grid_y[both]
            y_min = y_vals.min()
            y_max = y_vals.max()
            span = max(y_max - y_min, 1e-9)
            # t goes from 0 (at y_min) to 1 (at y_max)
            t = np.clip((grid_y - y_min) / span, 0.0, 1.0)
            # w_top increases with y (higher y → more top/suction side weight)
            w_top = 1.0 / (1.0 + np.exp(-10.0 * (t - 0.5)))
            merged[both] = (
                w_top[both] * top_arr[both] + (1.0 - w_top[both]) * bottom_arr[both]
            )
        return merged

    # Merge top and bottom configurations
    merged_raw12 = {}
    merged_raw23 = {} if has_x3 else None

    for key in rows_labels:
        top_r12 = top_raw12[key] if top_raw12 else None
        bot_r12 = bot_raw12[key] if bot_raw12 else None
        merged_raw12[key] = blend(top_r12, bot_r12)

        if has_x3:
            top_r23_val = top_raw23[key] if top_raw23 else None
            bot_r23_val = bot_raw23[key] if bot_raw23 else None
            merged_raw23[key] = blend(top_r23_val, bot_r23_val)

    # Compute amplitude scale for colormap
    def concat_valid(arrs):
        vals = []
        for arr in arrs:
            if arr is not None:
                vals.append(arr[~np.isnan(arr)])
        return np.concatenate(vals) if vals else np.array([])

    all_vals = concat_valid(
        [merged_raw12[k] for k in rows_labels]
        + ([merged_raw23[k] for k in rows_labels] if has_x3 else [])
    )
    if (vmin or vmax) is False:
        vmax = max(np.percentile(np.abs(all_vals), 95), 1.0) if all_vals.size else 5.0

    # Create 2x2 figure
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 10), sharex=True, sharey=True, squeeze=True
    )
    axes = axes.flatten()  # Make indexing easier

    # Preload airfoil geometry
    alpha_map = {23: 16, 13: 6}
    airfoil_alpha = alpha_map.get(aoa, aoa)
    airfoil_params = {
        "y_num": y_num,
        "alpha": airfoil_alpha,
        "airfoil_transparency": 0.25,
    }
    try:
        airfoil_x, airfoil_y = plot_airfoil(
            axes[0], airfoil_params, is_return_surface_points=True
        )
    except IndexError:
        # Airfoil translation not available for this alpha/Y; skip overlay
        airfoil_x = airfoil_y = None

    # Plot each component
    component_labels = {
        "u": r"$u$ (m/s)",
        "v": r"$v$ (m/s)",
        "w": r"$w$ (m/s)",
        "V": r"$|\mathbf{V}|$ (m/s)",
    }

    for i, key in enumerate(rows_labels):
        ax = axes[i]

        # Plot X1-X2 overlap differences
        im = ax.pcolormesh(
            grid_x,
            grid_y,
            merged_raw12[key],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            shading="auto",
        )

        # Plot X2-X3 overlap differences if available
        if has_x3:
            ax.pcolormesh(
                grid_x,
                grid_y,
                merged_raw23[key],
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                shading="auto",
            )

        # add a horizontal line at the y_mask_min_height
        if y_mask_min_height is not None:
            ax.axhline(
                y=y_mask_min_height,
                color="black",
                linestyle="--",
                linewidth=0.8,
                label=f"y = {y_mask_min_height} m",
            )

        # Formatting
        ax.set_aspect("equal")
        ax.set_xlim(-0.2, 0.8)
        ax.set_ylim(-0.2, 0.4)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.grid(False)

        # Labels
        ax.set_title(f"{component_labels[key]} overlap difference", fontsize=11)

        # Overlay airfoil outline
        if airfoil_x is not None:
            ax.plot(airfoil_x, airfoil_y, color="black", linewidth=0.4)
            ax.fill(
                airfoil_x,
                airfoil_y,
                color="black",
                alpha=airfoil_params["airfoil_transparency"],
            )

        # X-axis labels for bottom row
        if i >= 2:
            ax.set_xlabel("x (m)")

        # Y-axis labels for left column
        if i % 2 == 0:
            ax.set_ylabel("y (m)")

    # Add colorbar
    fig.colorbar(
        im,
        ax=axes,
        location="right",
        fraction=0.02,
        pad=0.02,
        label="Velocity difference (m/s)",
    )

    fig.suptitle(
        f"Y{y_num} overlap differences (AoA={aoa}°)", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "overlap_Y1_suction_raw.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    return fig


def plot_raw_planes(
    y1_data,
    output_dir,
    aoa=23,
    y_num=1,
    comp="u",
    w_threshold=3.0,
    y_min_mask=0.2,
):
    """
    Plot raw interpolated planes (no stitching) for flipped and normal configs.
    Plane fields are pre-masked by the combined quality mask (w + ICV);
    this function applies an additional |w| guard and y-cutoff for display.
    Layout: 2 rows (flipped, normal) x 3 columns (X1,X2,X3).
    """
    set_plot_style()
    grid_x = y1_data["grid_x"]
    grid_y = y1_data["grid_y"]
    interp_by_cfg = y1_data.get("interpolated_by_config", {})

    def filt(p):
        out = np.array(p[comp], copy=True)
        mask = ~np.isnan(out)
        if "is_valid" in p:
            mask &= p["is_valid"] != 0
        mask &= np.abs(p["w"]) <= w_threshold
        if y_min_mask is not None:
            mask &= grid_y > y_min_mask
        out = np.where(mask, out, np.nan)
        return out

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    cfg_order = [("flipped", "Flipped (suction)"), ("normal", "Normal (pressure)")]
    v_all = []
    for cfg_name, _ in cfg_order:
        cfg = interp_by_cfg.get(cfg_name, {})
        for plane in ["X1", "X2", "X3"]:
            if plane in cfg:
                data = filt(cfg[plane])
                v_all.append(np.nanmax(np.abs(data)))
    vmax = np.nanmax(v_all) if v_all else 1.0

    for i, (cfg_name, cfg_label) in enumerate(cfg_order):
        cfg = interp_by_cfg.get(cfg_name, {})
        for j, plane in enumerate(["X1", "X2", "X3"]):
            ax = axes[i, j]
            if plane not in cfg:
                ax.axis("off")
                continue
            data = filt(cfg[plane])
            im = ax.pcolormesh(
                grid_x,
                grid_y,
                data,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                shading="auto",
            )
            ax.set_title(f"{cfg_label} {plane}")
            ax.set_aspect("equal")
            ax.set_xlim(grid_x.min(), grid_x.max())
            ax.set_ylim(grid_y.min(), grid_y.max())
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
            ax.grid(False)

    fig.colorbar(
        im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label=f"{comp} (m/s)"
    )
    axes[-1, 0].set_ylabel("y (m)")
    axes[-1, 1].set_xlabel("x (m)")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"raw_planes_Y{y_num}_aoa{aoa}_{comp}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved raw planes plot: {save_path}")
    return fig


def plot_raw_planes_overlay(
    y1_data,
    output_dir,
    aoa=23,
    y_num=1,
    comp="u",
    w_threshold=3.0,
    y_min_mask=0.2,
    alpha=0.6,
):
    """
    Overlay all six planes (flipped & normal, X1/X2/X3) in a single plot.
    Uses distinct colors per plane; data is pre-masked by w + ICV and then
    additionally filtered by |w| and y>y_min_mask for display.
    """
    set_plot_style()
    grid_x = y1_data["grid_x"]
    grid_y = y1_data["grid_y"]
    interp_by_cfg = y1_data.get("interpolated_by_config", {})

    def filt(p):
        out = np.array(p[comp], copy=True)
        mask = ~np.isnan(out)
        if "is_valid" in p:
            mask &= p["is_valid"] != 0
        mask &= np.abs(p["w"]) <= w_threshold
        if y_min_mask is not None:
            mask &= grid_y > y_min_mask
        out = np.where(mask, out, np.nan)
        return out

    colors = {
        ("flipped", "X1"): "Reds",
        ("flipped", "X2"): "Oranges",
        ("flipped", "X3"): "YlOrBr",
        ("normal", "X1"): "Blues",
        ("normal", "X2"): "PuBu",
        ("normal", "X3"): "Greens",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    vmax_list = []
    data_store = []
    for cfg_name, cfg in interp_by_cfg.items():
        for plane in ["X1", "X2", "X3"]:
            if plane not in cfg:
                continue
            data = filt(cfg[plane])
            data_store.append(((cfg_name, plane), data))
            vmax_list.append(np.nanmax(np.abs(data)))
    vmax = np.nanmax(vmax_list) if vmax_list else 1.0

    for (cfg_name, plane), data in data_store:
        cmap = colors.get((cfg_name, plane), "RdBu_r")
        ax.pcolormesh(
            grid_x,
            grid_y,
            data,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            shading="auto",
            alpha=alpha,
            label=f"{cfg_name} {plane}",
        )

    ax.set_aspect("equal")
    ax.set_xlim(grid_x.min(), grid_x.max())
    ax.set_ylim(grid_y.min(), grid_y.max())
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.grid(False)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="RdBu_r"),
        ax=ax,
        fraction=0.04,
        pad=0.02,
        label=f"{comp} (m/s)",
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"raw_planes_overlay_Y{y_num}_aoa{aoa}_{comp}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved raw planes overlay plot: {save_path}")
    return fig


def print_stitching_uncertainty_table(y1_data, aoa=23, w_threshold=3.0):
    """
    Compute stitching uncertainty following MATLAB procedure.
    Reports RMS and P95 of raw differences (plane1 - plane2) in overlap regions.
    This is the actual measurement uncertainty in the overlap.
    """
    interp_by_cfg = y1_data.get("interpolated_by_config", {})
    rows = []

    def rms_and_p95(a, b):
        """Compute RMS and 95th percentile of raw difference a - b."""
        mask = ~np.isnan(a) & ~np.isnan(b)
        if mask.sum() == 0:
            return None
        diff = a[mask] - b[mask]
        rms = float(np.sqrt(np.mean(diff**2)))
        p95 = float(np.percentile(np.abs(diff), 95))
        npts = int(mask.sum())
        return rms, p95, npts

    def apply_filters(plane):
        out = {k: np.array(v, copy=True) for k, v in plane.items()}
        mask = np.ones_like(out["w"], dtype=bool)
        if "is_valid" in out:
            mask &= out["is_valid"] != 0
        mask &= np.abs(out["w"]) <= w_threshold
        for k in ["u", "v", "w"]:
            out[k] = np.where(mask, out[k], np.nan)
        return out

    for cfg_name, planes in interp_by_cfg.items():
        for pair in [("X1", "X2"), ("X2", "X3")]:
            if pair[0] not in planes or pair[1] not in planes:
                continue
            pA = apply_filters(planes[pair[0]])
            pB = apply_filters(planes[pair[1]])

            stats = {}
            n_ref = None

            # Compute statistics for each velocity component
            for comp in ["u", "v", "w"]:
                res = rms_and_p95(pA[comp], pB[comp])
                if res is None:
                    stats[comp] = (np.nan, np.nan)
                else:
                    rms, p95, npts = res
                    stats[comp] = (rms, p95)
                    n_ref = npts if n_ref is None else n_ref
            # Combined magnitude RMS
            magA = np.sqrt(pA["u"] ** 2 + pA["v"] ** 2 + pA["w"] ** 2)
            magB = np.sqrt(pB["u"] ** 2 + pB["v"] ** 2 + pB["w"] ** 2)
            res_mag = rms_and_p95(magA, magB)
            if res_mag:
                stats["mag"] = res_mag[:2]

            rows.append(
                {
                    "config": cfg_name,
                    "overlap": f"{pair[0]}-{pair[1]}",
                    "RMS u [m/s]": stats["u"][0],
                    "RMS v [m/s]": stats["v"][0],
                    "RMS w [m/s]": stats["w"][0],
                    "RMS |V| [m/s]": stats.get("mag", (np.nan, np.nan))[0],
                    "P95 |u| [m/s]": stats["u"][1],
                    "P95 |v| [m/s]": stats["v"][1],
                    "P95 |w| [m/s]": stats["w"][1],
                    "P95 |V| [m/s]": stats.get("mag", (np.nan, np.nan))[1],
                    "N pts": n_ref or 0,
                }
            )

    if not rows:
        print("No overlap regions found to compute stitching uncertainty.")
        return

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print(f"STITCHING UNCERTAINTY (Overlap Mismatch) — AoA {aoa}°")
    print("=" * 80)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nDefinition:")
    print("  RMS  = Root-mean-square of (plane1 - plane2) in overlap region")
    print("  P95  = 95th percentile of |plane1 - plane2| in overlap region")
    print("  N pts = Number of points in overlap region")
    print("=" * 80 + "\n")


def print_stitching_uncertainty_table_flipped(
    y1_data, aoa=23, w_threshold=3.0, y_mask_min_height=-0.5
):
    """
    Print RMS for u, v, w for flipped config only.
    Combines X1-X2 and X2-X3 overlaps into a single RMS per component.
    """
    interp_by_cfg = y1_data.get("interpolated_by_config", {})
    flipped = interp_by_cfg.get("flipped")
    if flipped is None:
        print("No flipped configuration found.")
        return

    y_grid = y1_data.get("grid_y")

    def filt(p):
        out = {k: np.array(v, copy=True) for k, v in p.items()}
        mask = np.ones_like(out["w"], dtype=bool)
        if "is_valid" in out:
            mask &= out["is_valid"] != 0
        mask &= np.abs(out["w"]) <= w_threshold
        # apply vertical cutoff using global grid if available
        if y_grid is not None:
            mask &= y_grid > y_mask_min_height
        for k in ["u", "v", "w"]:
            out[k] = np.where(mask, out[k], np.nan)
        return out

    pairs = []
    if "X1" in flipped and "X2" in flipped:
        pairs.append(("X1", "X2"))
    if "X2" in flipped and "X3" in flipped:
        pairs.append(("X2", "X3"))

    if not pairs:
        print("No flipped overlaps available.")
        return

    rms_accum = {"u": [], "v": [], "w": []}

    for a, b in pairs:
        pA = filt(flipped[a])
        pB = filt(flipped[b])
        mask = ~np.isnan(pA["u"]) & ~np.isnan(pB["u"])
        if mask.sum() == 0:
            continue
        for comp in ["u", "v", "w"]:
            diff = pA[comp][mask] - pB[comp][mask]
            rms_accum[comp].append(float(np.sqrt(np.nanmean(diff**2))))

    print("\nFlipped stitching uncertainty (combined overlaps) — AoA", aoa)
    for comp in ["u", "v", "w"]:
        if rms_accum[comp]:
            combined = float(np.mean(rms_accum[comp]))
            print(f"  RMS {comp}: {combined:.4f} m/s (averaged over X1-X2 & X2-X3)")
        else:
            print(f"  RMS {comp}: n/a (no valid overlap)")
    print("")


def sweep_offsets_and_report(
    q=0.6,
    aoa=23,
    y_num=1,
    dx_flipped_mm=(0, 5, 10),  # applied to X1/X2/X3 of flipped
    dx_normal_mm=(0, 5, 10),  # applied to X1/X2/X3 of normal
    dy_flipped_mm=(0,),  # extra vertical shift for flipped
    dx3_override_mm=None,  # optional list of absolute overrides for X3 only
):
    """
    Sweep streamwise (dx) and vertical (dy) offsets and report RMS of X2-X3 overlaps.

    - Flipped config: same dx applied to X1/X2/X3.
    - Normal config: its own dx applied to X1/X2/X3.
    - Optional extra dy for flipped (added before mirroring).
    - Optional explicit X3 override list (if given, supersedes per-config dx for X3).
    """
    results = []

    candidates_x3 = dx3_override_mm if dx3_override_mm is not None else [None]

    for dx_f in dx_flipped_mm:
        for dx_n in dx_normal_mm:
            for dy_f in dy_flipped_mm:
                for x3_override in candidates_x3:
                    y1_data = collect_y1_data(
                        q=q,
                        aoa=aoa,
                        y_num=y_num,
                        override_dx_flipped=dx_f,
                        override_dx_normal=dx_n,
                        extra_dy_flipped=dy_f,
                        override_x3_shift=x3_override,
                    )
                    if y1_data is None:
                        continue
                    interp_by_cfg = y1_data["interpolated_by_config"]
                    row = {
                        "dx_f_mm": dx_f,
                        "dx_n_mm": dx_n,
                        "dy_f_mm": dy_f,
                        "x3_override_mm": x3_override,
                    }

                    def filt(p):
                        out = {k: np.array(v, copy=True) for k, v in p.items()}
                        mask = np.ones_like(out["w"], dtype=bool)
                        if "is_valid" in out:
                            mask &= out["is_valid"] != 0
                        mask &= np.abs(out["w"]) <= 3.0
                        for k in ["u", "v", "w"]:
                            out[k] = np.where(mask, out[k], np.nan)
                        out["V"] = np.sqrt(
                            out["u"] ** 2 + out["v"] ** 2 + out["w"] ** 2
                        )
                        return out

                    for cfg_name, cfg in interp_by_cfg.items():
                        if "X2" not in cfg or "X3" not in cfg:
                            continue
                        p2 = filt(cfg["X2"])
                        p3 = filt(cfg["X3"])
                        mask = ~np.isnan(p2["u"]) & ~np.isnan(p3["u"])
                        if mask.sum() == 0:
                            continue
                        for comp in ["u", "v", "w"]:
                            diff = p2[comp] - p3[comp]
                            rms = float(np.sqrt(np.nanmean(diff[mask] ** 2)))
                            row[f"{cfg_name}_{comp}_rms"] = rms
                        du_mag = np.sqrt(
                            (p2["u"] - p3["u"]) ** 2
                            + (p2["v"] - p3["v"]) ** 2
                            + (p2["w"] - p3["w"]) ** 2
                        )
                        du_mag_valid = du_mag[mask]
                        row[f"{cfg_name}_mag_rms"] = float(
                            np.sqrt(np.nanmean(du_mag_valid**2))
                        )
                    results.append(row)

    if results:
        df = pd.DataFrame(results)
        print("\nOffset sweep — RMS of X2-X3 overlaps (m/s):")
        print(df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    else:
        print("No results from offset sweep (missing overlaps).")


def main():

    y_mask_min_height = 0.25

    # q = 0.6
    # aoa = 13
    # y_num = 1
    # for y_num in [1, 2, 3, 4, 5, 6, 7]:
    #     print(f"\nLoading Y{y_num} data for AoA={aoa}°...")
    #     y1_data = collect_y1_data(q=q, aoa=aoa, y_num=y_num)
    #     if y1_data is None:
    #         print("Failed to load data.")
    #         return

    #     output_dir = Path(project_dir) / "results" / "overlap_analysis"

    #     # # Print uncertainty table
    #     print_stitching_uncertainty_table_flipped(
    #         y1_data, aoa=aoa, w_threshold=3.0, y_mask_min_height=y_mask_min_height
    #     )

    ### Generate plot
    # plot_y1_overlap_grid(
    #     y1_data,
    #     output_dir,
    #     q=q,
    #     aoa=aoa,
    #     y_num=y_num,
    #     w_threshold=3.0,
    #     vmin=-0.5,  # -2.5,
    #     vmax=0.5,  # 2.5,
    #     y_mask_min_height=y_mask_min_height,
    # )
    # plt.show()

    q = 0.6
    aoa = 23
    y_num = 1
    for y_num in [1, 2, 3, 4]:
        print(f"\nLoading Y{y_num} data for AoA={aoa}°...")
        y1_data = collect_y1_data(q=q, aoa=aoa, y_num=y_num)
        if y1_data is None:
            print("Failed to load data.")
            return

        output_dir = Path(project_dir) / "results" / "overlap_analysis"

        # # Print uncertainty table
        print_stitching_uncertainty_table_flipped(
            y1_data, aoa=aoa, w_threshold=3.0, y_mask_min_height=y_mask_min_height
        )

        ### Generate plot
        # plot_y1_overlap_grid(
        #     y1_data,
        #     output_dir,
        #     q=q,
        #     aoa=aoa,
        #     y_num=y_num,
        #     w_threshold=3.0,
        #     vmin=-0.5,  # -2.5,
        #     vmax=0.5,  # 2.5,
        #     y_mask_min_height=y_mask_min_height,
        # )
        # plt.show()

    # # Optional: sweep X3 shifts and report RMS
    # print("Sweeping X3 shifts and reporting RMS values...")
    # sweep_offsets_and_report(
    #     q=q,
    #     aoa=aoa,
    #     y_num=y_num,
    # )

    # plot raw data
    # print("Generating raw planes plot...")
    # plot_raw_planes_overlay(
    #     y1_data,
    #     output_dir,
    #     aoa=aoa,
    #     y_num=y_num,
    #     comp="u",
    #     w_threshold=3.0,
    #     y_min_mask=-0.5,
    # )
    # plt.show()


if __name__ == "__main__":
    main()
