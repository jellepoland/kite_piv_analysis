"""
Offset optimizer for PIV plane stitching.
Minimizes RMS error in X2-X3 overlaps by adjusting plane offsets.

Rules from CSV:
1. Z-value (Y coordinate) should be consistent
2. X-displacement between planes should be ~300mm
3. X-offset can differ between flipped and normal configs
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from functools import lru_cache
import hashlib
import pickle

from kite_piv_analysis.utils import project_dir
import csv
from kite_piv_analysis.compute_overlap_error import (
    load_dat_file_with_std,
    interpolate_to_grid,
)
from kite_piv_analysis.TableD1_stitching_uncertainty_RMS import (
    print_stitching_uncertainty_table_flipped,
)

# Global cache for expensive computations
_RAW_DATA_CACHE = {}
_EVAL_COUNT = 0
_EVAL_START_TIME = None


def clear_raw_cache():
    _RAW_DATA_CACHE.clear()


def load_base_offsets_from_csv(aoa, y_num):
    """
    Read planes_location.csv and return base offsets dict in meters.
    """
    csv_path = (
        Path(project_dir)
        / "data_ALL_ERIK_FILES"
        / "JelleStitching"
        / "Input"
        / "planes_location.csv"
    )
    df = pd.read_csv(csv_path).rename(columns={"Unnamed: 0": "idx"})
    alpha_map = {23: 16, 13: 6}
    alpha = alpha_map.get(aoa, aoa)

    def get_offset(config, X):
        row = df[
            (df["alpha"] == alpha)
            & (df["config"] == config)
            & (df["Y"] == y_num)
            & (df["X"] == X)
        ]
        if row.empty:
            return (0.0, 0.0)
        return (
            float(row.iloc[0]["delta_x"]) / 1000.0,
            float(row.iloc[0]["delta_y"]) / 1000.0,
        )

    return {
        "flipped": {"X1": get_offset("flipped", 1)},
        "normal": {"X1": get_offset("normal", 1)},
    }


def offsets_dict_from_csv(aoa, y_num):
    """
    Build full offsets_dict (X1/X2/X3 for flipped/normal) in meters from planes_location.csv.
    """
    csv_path = (
        Path(project_dir)
        / "data_ALL_ERIK_FILES"
        / "JelleStitching"
        / "Input"
        / "planes_location.csv"
    )
    df = pd.read_csv(csv_path).rename(columns={"Unnamed: 0": "idx"})
    alpha_map = {23: 16, 13: 6}
    alpha = alpha_map.get(aoa, aoa)

    offsets = {"flipped": {}, "normal": {}}
    for config in ["flipped", "normal"]:
        for X in [1, 2, 3]:
            row = df[
                (df["alpha"] == alpha)
                & (df["config"] == config)
                & (df["Y"] == y_num)
                & (df["X"] == X)
            ]
            if row.empty:
                continue
            dx_m = float(row.iloc[0]["delta_x"]) / 1000.0
            dy_m = float(row.iloc[0]["delta_y"]) / 1000.0
            offsets[config][f"X{X}"] = (dx_m, dy_m)
    return offsets


def write_offsets_to_csv(aoa, y_num, dx_n, dx_f, dy_f, base_offsets):
    """
    Write updated offsets into planes_location.csv (mm).
    This overwrites delta_x/delta_y for the specified aoa and y_num.
    """
    csv_path = (
        Path(project_dir)
        / "data_ALL_ERIK_FILES"
        / "JelleStitching"
        / "Input"
        / "planes_location.csv"
    )
    df = pd.read_csv(csv_path).rename(columns={"Unnamed: 0": "idx"})
    alpha_map = {23: 16, 13: 6}
    alpha = alpha_map.get(aoa, aoa)

    base_n_x = base_offsets["normal"]["X1"][0] * 1000  # mm
    base_n_y = base_offsets["normal"]["X1"][1] * 1000  # mm
    base_f_x = base_offsets["flipped"]["X1"][0] * 1000  # mm
    base_f_y = base_offsets["flipped"]["X1"][1] * 1000  # mm

    for config, dx in [("normal", dx_n), ("flipped", dx_f)]:
        dy_add = dy_f if config == "flipped" else 0
        for X, base_dx in [(1, 0), (2, 300), (3, 600)]:
            mask = (
                (df["alpha"] == alpha)
                & (df["config"] == config)
                & (df["Y"] == y_num)
                & (df["X"] == X)
            )
            if config == "normal":
                df.loc[mask, "delta_x"] = base_n_x + base_dx + dx
                df.loc[mask, "delta_y"] = base_n_y
            else:
                df.loc[mask, "delta_x"] = base_f_x + base_dx + dx
                df.loc[mask, "delta_y"] = base_f_y + dy_add

    df.to_csv(csv_path, index=False)


def load_raw_data_cached(aoa=23, y_num=1):
    """
    Load raw PIV data once and cache it.
    This avoids repeated file I/O which is the slowest part.
    """
    cache_key = (aoa, y_num)

    if cache_key in _RAW_DATA_CACHE:
        return _RAW_DATA_CACHE[cache_key]

    input_dir = Path(project_dir) / "data_ALL_ERIK_FILES" / "JelleStitching" / "Input"
    U_inf = 15.0
    std_threshold = 0.1

    aoa_dir = input_dir / f"aoa_{aoa}"
    if not aoa_dir.exists():
        return None

    base_path = aoa_dir / f"Y{y_num}"
    if not base_path.exists():
        return None

    raw_data = {}
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
        for plane_name, plane_dir in plane_dirs.items():
            mean_file = plane_dir / "B0001.dat"
            std_file = plane_dir / "B0002.dat"
            if mean_file.exists() and std_file.exists():
                planes_data[plane_name] = load_dat_file_with_std(
                    mean_file, std_file, U_inf=U_inf, std_threshold=std_threshold
                )

        if planes_data:
            raw_data[config] = planes_data

    # Cache it
    _RAW_DATA_CACHE[cache_key] = raw_data
    return raw_data


def collect_y1_data_with_offsets(
    q=0.6,
    aoa=23,
    y_num=1,
    offsets_dict=None,
):
    """
    Load and process Y1 data with custom offsets.
    Now uses cached raw data to avoid repeated file I/O.

    offsets_dict format:
    {
        'flipped': {'X1': (dx, dy), 'X2': (dx, dy), 'X3': (dx, dy)},
        'normal': {'X1': (dx, dy), 'X2': (dx, dy), 'X3': (dx, dy)}
    }

    All offsets in meters.
    """
    # Load raw data from cache
    raw_data = load_raw_data_cached(aoa, y_num)
    if raw_data is None:
        return None

    # Deep copy and apply offsets
    configs_data = {}
    for config in ["flipped", "normal"]:
        if config not in raw_data:
            continue

        planes_data = {}
        for plane_name, raw_plane in raw_data[config].items():
            # Copy arrays
            plane = {k: np.array(v, copy=True) for k, v in raw_plane.items()}

            # Apply custom offsets if provided
            if (
                offsets_dict
                and config in offsets_dict
                and plane_name in offsets_dict[config]
            ):
                dx_custom, dy_custom = offsets_dict[config][plane_name]
                plane["x"] += dx_custom
                plane["y"] += dy_custom
                plane["_dy_shift"] = dy_custom

            planes_data[plane_name] = plane

        # Flip y for suction side
        for plane_name, data in planes_data.items():
            if config == "flipped":
                dy_local = data.get("_dy_shift", 0.0)
                data["y"] = -data["y"] + 2 * dy_local
                if "v" in data:
                    data["v"] = -data["v"]

        configs_data[config] = planes_data

    if not configs_data:
        return None

    # Build global grid
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

    return {
        "interpolated_by_config": interpolated_by_config,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "y_threshold": grid_y.min() + q * (grid_y.max() - grid_y.min()),
    }


def compute_overlap_rms(
    y1_data, filter_outliers=True, min_overlap_pts=50, y_min_mask=None
):
    """
    Compute RMS differences in X1-X2 and X2-X3 overlaps.
    Returns dict with RMS values per config and overlap.
    """
    if y1_data is None:
        return None

    interp_by_cfg = y1_data["interpolated_by_config"]
    results = {}

    for cfg_name, cfg in interp_by_cfg.items():
        results[cfg_name] = {}

        # Check both X1-X2 and X2-X3 overlaps
        for pair in [("X1", "X2"), ("X2", "X3")]:
            if pair[0] not in cfg or pair[1] not in cfg:
                continue

            p1 = cfg[pair[0]]
            p2 = cfg[pair[1]]

            # Apply filtering if requested
            if filter_outliers:
                mask = ~np.isnan(p1["u"]) & ~np.isnan(p2["u"])
                # Filter extreme w values (likely outliers)
                mask &= np.abs(p1["w"]) <= 3.0
                mask &= np.abs(p2["w"]) <= 3.0
            else:
                mask = ~np.isnan(p1["u"]) & ~np.isnan(p2["u"])

            # Optional vertical mask using global grid
            if y_min_mask is not None and "grid_y" in y1_data:
                mask &= y1_data["grid_y"] > y_min_mask

        npts = int(mask.sum())
        if npts < min_overlap_pts:
            continue

        pair_results = {}
        for comp in ["u", "v", "w"]:
            diff = p1[comp][mask] - p2[comp][mask]
            rms = float(np.sqrt(np.mean(diff**2)))
            pair_results[comp] = rms

            # Vector magnitude RMS
            du_mag = np.sqrt(
                (p1["u"][mask] - p2["u"][mask]) ** 2
                + (p1["v"][mask] - p2["v"][mask]) ** 2
                + (p1["w"][mask] - p2["w"][mask]) ** 2
            )
        pair_results["mag"] = float(np.sqrt(np.mean(du_mag**2)))
        pair_results["n_pts"] = npts

        results[cfg_name][f"{pair[0]}-{pair[1]}"] = pair_results

    # Drop configs with no valid overlaps
    results = {k: v for k, v in results.items() if v}
    return results


def offset_params_to_dict(params, base_offsets):
    """
    Convert optimization parameters to offset dictionary.

    Parameters:
    - params[0]: dx shift for flipped X2 (relative to X1)
    - params[1]: dx shift for flipped X3 (relative to X2)
    - params[2]: dx shift for normal X2 (relative to X1)
    - params[3]: dx shift for normal X3 (relative to X2)
    - params[4]: dy shift for flipped (applied to all planes)

    All in millimeters, converted to meters.
    """
    # Base spacing (from CSV rules): ~300mm between planes
    base_dx_12 = 0.300  # m
    base_dx_23 = 0.300  # m

    # Convert mm to m
    dx_f_12 = params[0] / 1000.0
    dx_f_23 = params[1] / 1000.0
    dx_n_12 = params[2] / 1000.0
    dx_n_23 = params[3] / 1000.0
    dy_f = params[4] / 1000.0

    offsets_dict = {
        "flipped": {
            "X1": (
                base_offsets["flipped"]["X1"][0],
                base_offsets["flipped"]["X1"][1] + dy_f,
            ),
            "X2": (
                base_offsets["flipped"]["X1"][0] + base_dx_12 + dx_f_12,
                base_offsets["flipped"]["X1"][1] + dy_f,
            ),
            "X3": (
                base_offsets["flipped"]["X1"][0]
                + base_dx_12
                + dx_f_12
                + base_dx_23
                + dx_f_23,
                base_offsets["flipped"]["X1"][1] + dy_f,
            ),
        },
        "normal": {
            "X1": (base_offsets["normal"]["X1"][0], base_offsets["normal"]["X1"][1]),
            "X2": (
                base_offsets["normal"]["X1"][0] + base_dx_12 + dx_n_12,
                base_offsets["normal"]["X1"][1],
            ),
            "X3": (
                base_offsets["normal"]["X1"][0]
                + base_dx_12
                + dx_n_12
                + base_dx_23
                + dx_n_23,
                base_offsets["normal"]["X1"][1],
            ),
        },
    }

    return offsets_dict


def objective_function(params, aoa, y_num, base_offsets, weight_x2x3=2.0, verbose=True):
    """
    Objective function to minimize: weighted sum of RMS errors.

    Gives more weight to X2-X3 overlap (typically worse).
    """
    global _EVAL_COUNT, _EVAL_START_TIME

    if _EVAL_START_TIME is None:
        _EVAL_START_TIME = time.time()

    _EVAL_COUNT += 1
    # Snap parameters to nearest integer millimeter to enforce integer offsets
    params_int = np.round(np.asarray(params, dtype=float))

    offsets_dict = offset_params_to_dict(params_int, base_offsets)

    y1_data = collect_y1_data_with_offsets(
        aoa=aoa, y_num=y_num, offsets_dict=offsets_dict
    )

    if y1_data is None:
        return 1e6  # Penalty for failed data loading

    rms_results = compute_overlap_rms(
        y1_data, filter_outliers=True, min_overlap_pts=50, y_min_mask=-0.5
    )

    if not rms_results:
        return 1e6

    # Compute weighted cost
    cost = 0.0
    n_terms = 0

    for cfg_name, cfg_results in rms_results.items():
        for overlap_name, overlap_results in cfg_results.items():
            if overlap_results.get("n_pts", 0) < 50:
                return 1e6  # Penalize if overlap too small
            # Weight X2-X3 more heavily (typically the problematic overlap)
            weight = weight_x2x3 if "X2-X3" in overlap_name else 1.0

            # Use magnitude RMS as primary metric
            cost += weight * overlap_results["mag"]
            n_terms += weight

    # Average cost
    if n_terms > 0:
        cost /= n_terms
    else:
        return 1e6  # no valid overlaps considered

    # Progress reporting
    if verbose and _EVAL_COUNT % 5 == 0:
        elapsed = time.time() - _EVAL_START_TIME
        rate = _EVAL_COUNT / elapsed
        print(
            f"  Eval {_EVAL_COUNT}: cost={cost:.6f} m/s  "
            f"[{rate:.1f} eval/s, {elapsed:.1f}s elapsed]"
        )

    return cost


def sweep_offsets_grid(
    aoa=23,
    y_num=1,
    base_offsets=None,
    dx_normal_range_mm=(-150, 150, 50),  # (start, stop, step)
    dx_flipped_range_mm=(-150, 150, 50),
    dy_flipped_range_mm=(-100, 100, 35),
):
    """
    Grid search over three scalars:
      - dx_normal (mm): applied to X1/X2/X3 of normal config (relative to 0/300/600)
      - dx_flipped (mm): applied to X1/X2/X3 of flipped config (relative to 0/300/600)
      - dy_flipped (mm): extra vertical shift for flipped (added before mirroring)
    """
    if base_offsets is None:
        base_offsets = load_base_offsets_from_csv(aoa, y_num)

    dx_n_vals = np.arange(*dx_normal_range_mm)
    dx_f_vals = np.arange(*dx_flipped_range_mm)
    dy_f_vals = np.arange(*dy_flipped_range_mm)

    results = []

    total = len(dx_n_vals) * len(dx_f_vals) * len(dy_f_vals)
    count = 0

    print(f"\nGrid search: {total} evaluations (integer mm)")
    print("-" * 60)

    for dx_n in dx_n_vals:
        for dx_f in dx_f_vals:
            for dy_f in dy_f_vals:
                count += 1
                # Params: [dx_f_12, dx_f_23, dx_n_12, dx_n_23, dy_f]
                params = [dx_f, dx_f, dx_n, dx_n, dy_f]
                cost = objective_function(
                    params, aoa, y_num, base_offsets, verbose=False
                )
                results.append(
                    {
                        "dx_normal_mm": dx_n,
                        "dx_flipped_mm": dx_f,
                        "dy_flipped_mm": dy_f,
                        "cost": cost,
                    }
                )
                if count % 50 == 0:
                    print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")

    df = pd.DataFrame(results)

    # Find best
    best_idx = df["cost"].idxmin()
    best_row = df.loc[best_idx]

    print("\n" + "=" * 60)
    print("GRID SEARCH RESULTS")
    print("=" * 60)
    print(f"Best dx_normal:  {best_row['dx_normal_mm']:.1f} mm")
    print(f"Best dx_flipped: {best_row['dx_flipped_mm']:.1f} mm")
    print(f"Best dy_flipped: {best_row['dy_flipped_mm']:.1f} mm")
    print(f"Best cost: {best_row['cost']:.6f} m/s")

    # Also print as planes_location-style rows for convenience
    alpha_map = {23: 16, 13: 6}
    alpha = alpha_map.get(aoa, aoa)
    dx_n = best_row["dx_normal_mm"]
    dx_f = best_row["dx_flipped_mm"]
    dy_f = best_row["dy_flipped_mm"]

    base_n_x = base_offsets["normal"]["X1"][0] * 1000  # mm
    base_n_y = base_offsets["normal"]["X1"][1] * 1000  # mm
    base_f_x = base_offsets["flipped"]["X1"][0] * 1000  # mm
    base_f_y = base_offsets["flipped"]["X1"][1] * 1000  # mm

    lines = []
    # flipped
    lines.append(
        f",{alpha},flipped,{y_num},1,{base_f_x + dx_f:.0f},{base_f_y + dy_f:.0f}"
    )
    lines.append(
        f",{alpha},flipped,{y_num},2,{base_f_x + 300 + dx_f:.0f},{base_f_y + dy_f:.0f}"
    )
    lines.append(
        f",{alpha},flipped,{y_num},3,{base_f_x + 600 + dx_f:.0f},{base_f_y + dy_f:.0f}"
    )
    # normal
    lines.append(f", {alpha},normal,{y_num},1,{base_n_x + dx_n:.0f},{base_n_y:.0f}")
    lines.append(
        f",{alpha},normal,{y_num},2,{base_n_x + 300 + dx_n:.0f},{base_n_y:.0f}"
    )
    lines.append(
        f",{alpha},normal,{y_num},3,{base_n_x + 600 + dx_n:.0f},{base_n_y:.0f}"
    )

    print("\nPlanes_location-style output (mm):")
    for ln in lines:
        print(ln)

    return df


def sweep_offsets_grid_fine(
    coarse_df,
    aoa=23,
    y_num=1,
    base_offsets=None,
    dx_window_mm=30,
    dy_window_mm=20,
    dx_step_mm=5,
    dy_step_mm=4,
):
    """
    Run a finer sweep around the best point from a coarse sweep_offsets_grid result.
    """
    if coarse_df is None or coarse_df.empty:
        print("coarse_df is empty; run sweep_offsets_grid first.")
        return None

    best = coarse_df.loc[coarse_df["cost"].idxmin()]
    dx_n_best = int(round(best["dx_normal_mm"]))
    dx_f_best = int(round(best["dx_flipped_mm"]))
    dy_f_best = int(round(best["dy_flipped_mm"]))

    dx_n_range = (
        dx_n_best - dx_window_mm,
        dx_n_best + dx_window_mm + dx_step_mm,
        dx_step_mm,
    )
    dx_f_range = (
        dx_f_best - dx_window_mm,
        dx_f_best + dx_window_mm + dx_step_mm,
        dx_step_mm,
    )
    dy_f_range = (
        dy_f_best - dy_window_mm,
        dy_f_best + dy_window_mm + dy_step_mm,
        dy_step_mm,
    )

    print(
        f"Fine sweep around best: dx_n≈{dx_n_best}, dx_f≈{dx_f_best}, dy_f≈{dy_f_best} (step {dx_step_mm} mm)"
    )

    return sweep_offsets_grid(
        aoa=aoa,
        y_num=y_num,
        base_offsets=base_offsets,
        dx_normal_range_mm=dx_n_range,
        dx_flipped_range_mm=dx_f_range,
        dy_flipped_range_mm=dy_f_range,
    )


if __name__ == "__main__":
    # Example usage

    aoa = 13
    y_num = 1

    # Option 1: Quick grid search
    print("Running grid search...")
    grid_results = sweep_offsets_grid(
        aoa=aoa,
        y_num=y_num,
        dx_normal_range_mm=(-50, 50, 30),  # (start, stop, step)
        dx_flipped_range_mm=(-50, 50, 30),
        dy_flipped_range_mm=(-50, 50, 30),
    )

    # Option 2: Fine sweep around the coarse optimum
    print("\nRunning fine grid search around coarse optimum...")
    grid_results_fine = sweep_offsets_grid_fine(
        grid_results,
        aoa=aoa,
        y_num=y_num,
        dx_window_mm=5,
        dy_window_mm=5,
        dx_step_mm=5,
        dy_step_mm=5,
    )

    # # Option 2: Fine sweep around the coarse optimum
    # print("\nRunning fine grid search around coarse optimum...")
    # grid_results_fine = sweep_offsets_grid_fine(
    #     grid_results_fine,
    #     aoa=aoa,
    #     y_num=y_num,
    #     dx_window_mm=5,
    #     dy_window_mm=5,
    #     dx_step_mm=5,
    #     dy_step_mm=5,
    # )

    # Optionally write best fine offsets back to planes_location.csv
    if grid_results_fine is not None and not grid_results_fine.empty:
        best = grid_results_fine.loc[grid_results_fine["cost"].idxmin()]
        base_offsets = load_base_offsets_from_csv(aoa, y_num)
        write_offsets_to_csv(
            aoa=aoa,
            y_num=y_num,
            dx_n=best["dx_normal_mm"],
            dx_f=best["dx_flipped_mm"],
            dy_f=best["dy_flipped_mm"],
            base_offsets=base_offsets,
        )
        print("\nUpdated planes_location.csv with best fine-sweep offsets.")

    # Print stitching uncertainty (flipped) using current planes_location offsets
    print("\nStitching uncertainty (flipped) with current planes_location.csv offsets:")
    clear_raw_cache()
    offsets_dict = offsets_dict_from_csv(aoa, y_num)
    y1_data = collect_y1_data_with_offsets(
        aoa=aoa, y_num=y_num, offsets_dict=offsets_dict
    )
    if y1_data:
        print_stitching_uncertainty_table_flipped(y1_data, aoa=aoa, w_threshold=3.0)
    else:
        print("Could not load Y data.")
