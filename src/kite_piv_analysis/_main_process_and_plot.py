from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

# Support both:
# 1) python -m kite_piv_analysis._main_process_and_plot
# 2) python src/kite_piv_analysis/_main_process_and_plot.py
try:
    from kite_piv_analysis.utils import project_dir
except ModuleNotFoundError:
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kite_piv_analysis.utils import project_dir


PROJECT_DIR = Path(project_dir)
BASE_PACKAGE = __package__ or "kite_piv_analysis"

# Ensure matplotlib cache is writable (some systems have a non-writable ~/.cache).
if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = PROJECT_DIR / ".cache" / "matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

FULL_CONVERGENCE_PAIRS = [(6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (16, 1)]
SMOKE_CONVERGENCE_PAIRS = [(6, 1)]

FULL_PARAMETER_NAMES = ["iP", "dLx", "dLy"]
SMOKE_PARAMETER_NAMES = ["iP"]

FULL_FAST_FACTOR = 10
SMOKE_FAST_FACTOR = 50

FULL_SWEEP_N_POINTS = 10
SMOKE_SWEEP_N_POINTS = 2

FULL_GAMMA_Y_NUM_LIST = [1, 2, 3, 4, 5, 6, 7]
SMOKE_GAMMA_Y_NUM_LIST = [1]

FULL_NOCA_ALPHA6_Y = [1, 2, 3, 4, 5, 6, 7]
FULL_NOCA_ALPHA16_Y = [1]
SMOKE_NOCA_ALPHA6_Y = [1]
SMOKE_NOCA_ALPHA16_Y = [1]

FULL_INCLUDE_CFD_NOCA = True
SMOKE_INCLUDE_CFD_NOCA = True
FULL_INCLUDE_CFD_CONVERGENCE = True
SMOKE_INCLUDE_CFD_CONVERGENCE = False

FULL_BERNOULLI_D_PERC = 5.0
SMOKE_BERNOULLI_D_PERC = 5.0


def _import_package_module(module_name: str):
    return importlib.import_module(f"{BASE_PACKAGE}.{module_name}")


def processed_cfd_path(alpha: int, y_num: int) -> Path:
    return (
        PROJECT_DIR
        / "processed_data"
        / "CFD"
        / f"alpha_{alpha}"
        / f"Y{y_num}_paraview_corrected.csv"
    )


def piv_stitched_path(aoa: int, y_num: int) -> Path:
    return (
        PROJECT_DIR
        / "data"
        / "piv_stichted_planes"
        / f"aoa_{aoa}"
        / f"aoa_{aoa}_Y{y_num}_stitched.csv"
    )


def spanwise_raw_path(alpha: int, cm_offset: int) -> Path:
    return (
        PROJECT_DIR
        / "data"
        / "CFD_slices"
        / "spanwise_slices"
        / f"alpha_{alpha}_CFD_spanwise_slice_{cm_offset}cm_1.csv"
    )


def spanwise_outline_data_path(alpha: int, cm_offset: int) -> Path:
    return (
        PROJECT_DIR
        / "data"
        / "CFD_slices"
        / "spanwise_slices_outline_wing"
        / f"alpha_{alpha}_CFD_{cm_offset}cm_outline_wing.csv"
    )


def convergence_paths(
    alpha: int,
    y_num: int,
    parameter_names: list[str] | None = None,
    include_cfd_sweep: bool = True,
) -> list[Path]:
    if parameter_names is None:
        parameter_names = FULL_PARAMETER_NAMES

    files: list[Path] = []
    for data_type in ["CFD", "PIV"]:
        for shape in ["Ellipse", "Rectangle"]:
            for parameter in parameter_names:
                files.append(
                    PROJECT_DIR
                    / "processed_data"
                    / "convergence_study"
                    / f"alpha_{alpha}_Y{y_num}_{data_type}_{shape}_{parameter}.csv"
                )
    files.append(
        PROJECT_DIR
        / "processed_data"
        / "convergence_study"
        / "PIV_sweep"
        / f"alpha_{alpha}_Y{y_num}_PIV_Ellipse.csv"
    )
    files.append(
        PROJECT_DIR
        / "processed_data"
        / "convergence_study"
        / "PIV_sweep"
        / f"alpha_{alpha}_Y{y_num}_PIV_Rectangle.csv"
    )
    if include_cfd_sweep:
        files.append(
            PROJECT_DIR
            / "processed_data"
            / "convergence_study"
            / "PIV_sweep"
            / f"alpha_{alpha}_Y{y_num}_CFD_Ellipse.csv"
        )
        files.append(
            PROJECT_DIR
            / "processed_data"
            / "convergence_study"
            / "PIV_sweep"
            / f"alpha_{alpha}_Y{y_num}_CFD_Rectangle.csv"
        )
    return files


def required_processed_cfd_files(smoke: bool = False) -> list[Path]:
    if smoke:
        # Minimal CFD cache required for smoke computations.
        return [processed_cfd_path(6, 1)]

    files: list[Path] = []
    for y_num in range(1, 8):
        files.append(processed_cfd_path(6, y_num))
    for y_num in range(1, 5):
        files.append(processed_cfd_path(16, y_num))
    return files


def required_source_files(smoke: bool = False) -> list[Path]:
    files: list[Path] = []

    if smoke:
        files.append(piv_stitched_path(13, 1))
    else:
        # PIV stitched input used by fig06/07/10/12/13/15
        for y_num in range(1, 8):
            files.append(piv_stitched_path(13, y_num))
        for y_num in range(1, 5):
            files.append(piv_stitched_path(23, y_num))

    # Airfoil and bound placement data used by plotting helpers
    files.append(PROJECT_DIR / "data" / "airfoils" / "airfoil_translation_values.csv")
    airfoil_y_nums = [1] if smoke else range(1, 8)
    for y_num in airfoil_y_nums:
        files.append(PROJECT_DIR / "data" / "airfoils" / f"y{y_num}.dat")
    files.append(PROJECT_DIR / "data" / "optimal_bound_placement.csv")

    # fig08 and fig14 source inputs
    files.append(PROJECT_DIR / "data" / "gamma_distribution" / "y_locations.csv")

    # Spanwise raw source data used to generate missing outlines
    if smoke:
        files.append(spanwise_raw_path(6, 25))
    else:
        for cm_offset in [10, 15, 20, 25, 30]:
            files.append(spanwise_raw_path(6, cm_offset))
        files.append(spanwise_raw_path(16, 25))

    return files


def required_spanwise_outline_files(smoke: bool = False) -> list[Path]:
    if smoke:
        return [spanwise_outline_data_path(6, 25)]

    files: list[Path] = []
    for cm_offset in [10, 15, 20, 25, 30]:
        files.append(spanwise_outline_data_path(6, cm_offset))
    files.append(spanwise_outline_data_path(16, 25))
    return files


def missing_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def ensure_result_directories() -> None:
    (PROJECT_DIR / "results" / "paper_plots_21_10_2025").mkdir(
        parents=True, exist_ok=True
    )
    (PROJECT_DIR / "results" / "convergence_study" / "alpha_6").mkdir(
        parents=True, exist_ok=True
    )
    (PROJECT_DIR / "results" / "convergence_study" / "alpha_16").mkdir(
        parents=True, exist_ok=True
    )
    (PROJECT_DIR / "processed_data" / "gamma_distribution").mkdir(
        parents=True, exist_ok=True
    )
    (PROJECT_DIR / "processed_data" / "CFD" / "spanwise_slices").mkdir(
        parents=True, exist_ok=True
    )
    (PROJECT_DIR / "data" / "CFD_slices" / "spanwise_slices_outline_wing").mkdir(
        parents=True, exist_ok=True
    )
    for alpha in [6, 16]:
        (PROJECT_DIR / "processed_data" / "CFD" / f"alpha_{alpha}").mkdir(
            parents=True, exist_ok=True
        )


def validate_source_data(smoke: bool = False) -> None:
    missing = missing_paths(required_source_files(smoke=smoke))

    if not smoke:
        raw_dat_dir = (
            PROJECT_DIR
            / "data"
            / "raw_dat_files_convergence"
            / "flipped_aoa_23_vw_15_H_918_Z1_Y4_X2"
        )
        if not raw_dat_dir.exists() or not any(raw_dat_dir.glob("*.dat")):
            missing.append(raw_dat_dir)

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required source data. Add these first:\n" + formatted
        )


def _generate_processed_cfd_subset(cases: list[tuple[int, int]]) -> None:
    transforming_paraview_output = _import_package_module(
        "transforming_paraview_output"
    )
    for alpha, y_num in cases:
        input_dir = PROJECT_DIR / "data" / "CFD_slices" / f"alpha_{alpha}"
        candidates = sorted(
            [
                path
                for path in input_dir.glob(f"Y{y_num}_*.csv")
                if not path.name.endswith("_0.csv")
            ]
        )
        if not candidates:
            raise FileNotFoundError(
                f"Cannot generate processed CFD for alpha={alpha}, Y={y_num}; "
                f"no matching input in {input_dir}"
            )

        output_path = processed_cfd_path(alpha, y_num)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        transforming_paraview_output.process_csv(
            input_path=candidates[0],
            output_path=output_path,
            spatial_scale=2.584,
            velocity_scale=15,
            y_num=y_num,
            alpha=alpha,
        )


def ensure_processed_cfd(smoke: bool = False) -> None:
    required = required_processed_cfd_files(smoke=smoke)
    missing_before = missing_paths(required)
    if not missing_before:
        print("Processed CFD files already available.")
        return

    print(
        f"Generating missing processed CFD files ({len(missing_before)} missing before run)..."
    )
    if smoke:
        _generate_processed_cfd_subset(cases=[(6, 1)])
    else:
        transforming_paraview_output = _import_package_module(
            "transforming_paraview_output"
        )
        transforming_paraview_output.main()

    missing_after = missing_paths(required)
    if missing_after:
        formatted = "\n".join(f"  - {path}" for path in missing_after)
        raise RuntimeError(
            "Could not generate all processed CFD files. Still missing:\n" + formatted
        )
    print("Processed CFD files generated.")


@contextmanager
def _suppress_matplotlib_show():
    import matplotlib.pyplot as plt

    original_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt.show = original_show


def ensure_spanwise_outlines(smoke: bool = False) -> None:
    required = required_spanwise_outline_files(smoke=smoke)
    missing = missing_paths(required)
    if not missing:
        print("Spanwise outline files already available.")
        return

    print(
        f"Generating missing spanwise outlines ({len(missing)} missing before run)..."
    )
    extract_spanwise_contour = _import_package_module("extract_spanwise_contour")

    for data_outline_path in missing:
        stem = data_outline_path.stem  # alpha_6_CFD_25cm_outline_wing
        parts = stem.split("_")
        alpha = int(parts[1])
        cm_offset = int(parts[3].replace("cm", ""))

        raw_path = spanwise_raw_path(alpha, cm_offset)
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Cannot generate outline; missing raw spanwise slice: {raw_path}"
            )

        with _suppress_matplotlib_show():
            points = extract_spanwise_contour.transform_raw_csv_to_processed_df(
                alpha=alpha,
                cm_offset=cm_offset,
            )
            generated_path = extract_spanwise_contour.save_contour_points_to_csv(
                points, alpha, cm_offset
            )

        generated_path = Path(generated_path)
        if not generated_path.exists():
            raise RuntimeError(
                f"Outline generation returned missing file: {generated_path}"
            )

        shutil.copy2(generated_path, data_outline_path)
        print(f"Generated: {data_outline_path}")

    missing_after = missing_paths(required)
    if missing_after:
        formatted = "\n".join(f"  - {path}" for path in missing_after)
        raise RuntimeError(
            "Could not generate all spanwise outline files. Still missing:\n"
            + formatted
        )
    print("Spanwise outline files generated.")


def _quantitative_file_has_y_nums(csv_path: Path, y_num_list: list[int]) -> bool:
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    if "y_num" not in df.columns:
        return False
    observed = set(pd.to_numeric(df["y_num"], errors="coerce").dropna().astype(int))
    return set(y_num_list).issubset(observed)


def ensure_quantitative_gamma_file(smoke: bool = False) -> None:
    target = (
        PROJECT_DIR
        / "processed_data"
        / "quantitative_chordwise_analysis_alpha_6_with_std.csv"
    )
    y_num_list = SMOKE_GAMMA_Y_NUM_LIST if smoke else FULL_GAMMA_Y_NUM_LIST
    n_points = SMOKE_SWEEP_N_POINTS if smoke else FULL_SWEEP_N_POINTS

    if _quantitative_file_has_y_nums(target, y_num_list):
        print("Quantitative chordwise analysis file already available.")
        return

    print(
        f"Generating quantitative chordwise analysis (alpha=6, Y={y_num_list}, n_points={n_points})..."
    )
    calculating_noca_and_kutta = _import_package_module("calculating_noca_and_kutta")

    calculating_noca_and_kutta.save_results_single_alpha(
        alpha=6,
        y_num_list=y_num_list,
        n_points=n_points,
    )
    if not _quantitative_file_has_y_nums(target, y_num_list):
        raise RuntimeError(f"Failed to generate required file: {target}")
    print(f"Generated: {target}")


def ensure_vsm_gamma_cache(smoke: bool = False) -> None:
    if smoke:
        print("Skipping VSM gamma cache generation in smoke mode.")
        return

    target = (
        PROJECT_DIR
        / "processed_data"
        / "gamma_distribution"
        / "VSM_gamma_distribution.csv"
    )
    if target.exists():
        print("VSM gamma cache already available.")
        return

    print("Generating VSM gamma cache...")
    fig08_gamma_distribution = _import_package_module("fig08_gamma_distribution")

    fig08_gamma_distribution.get_VSM_gamma_distribution()
    if not target.exists():
        raise RuntimeError(f"Failed to generate required file: {target}")
    print(f"Generated: {target}")


def run_processing_pipeline(smoke: bool = False) -> None:
    process_module = _import_package_module(
        "_process_calculating_noca_and_kutta__convergence_study"
    )

    if smoke:
        alpha6_y_nums = SMOKE_NOCA_ALPHA6_Y
        alpha16_y_nums = SMOKE_NOCA_ALPHA16_Y
        noca_n_points = SMOKE_SWEEP_N_POINTS
        include_cfd_noca = SMOKE_INCLUDE_CFD_NOCA
        conv_pairs = SMOKE_CONVERGENCE_PAIRS
        conv_parameter_names = SMOKE_PARAMETER_NAMES
        conv_data_types = ["PIV"]
        conv_fast_factor = SMOKE_FAST_FACTOR
        conv_is_small_piv = True
        conv_piv_sweep_n_points = SMOKE_SWEEP_N_POINTS
        include_cfd_convergence = SMOKE_INCLUDE_CFD_CONVERGENCE
        bernoulli_d_perc = SMOKE_BERNOULLI_D_PERC
    else:
        alpha6_y_nums = FULL_NOCA_ALPHA6_Y
        alpha16_y_nums = FULL_NOCA_ALPHA16_Y
        noca_n_points = FULL_SWEEP_N_POINTS
        include_cfd_noca = FULL_INCLUDE_CFD_NOCA
        conv_pairs = FULL_CONVERGENCE_PAIRS
        conv_parameter_names = FULL_PARAMETER_NAMES
        conv_data_types = ["CFD", "PIV"]
        conv_fast_factor = FULL_FAST_FACTOR
        conv_is_small_piv = False
        conv_piv_sweep_n_points = FULL_SWEEP_N_POINTS
        include_cfd_convergence = FULL_INCLUDE_CFD_CONVERGENCE
        bernoulli_d_perc = FULL_BERNOULLI_D_PERC

    print("Running processing pipeline (NOCA/Kutta, Bernoulli, convergence)...")
    process_module.run_calculating_noca_and_kutta(
        alpha6_y_nums=alpha6_y_nums,
        alpha16_y_nums=alpha16_y_nums,
        n_points=noca_n_points,
        include_cfd=include_cfd_noca,
    )
    process_module.run_bernoulli_contour_force_cache(
        alpha6_y_nums=alpha6_y_nums,
        alpha16_y_nums=alpha16_y_nums,
        n_points=noca_n_points,
        d_perc=bernoulli_d_perc,
    )
    process_module.run_convergence_study(
        pairs=conv_pairs,
        parameter_names=conv_parameter_names,
        data_types=conv_data_types,
        is_small_piv=conv_is_small_piv,
        fast_factor=conv_fast_factor,
        piv_sweep_n_points=conv_piv_sweep_n_points,
        include_cfd=include_cfd_convergence,
    )


def ensure_convergence_cache(smoke: bool = False) -> None:
    required_pairs = SMOKE_CONVERGENCE_PAIRS if smoke else FULL_CONVERGENCE_PAIRS
    parameter_names = SMOKE_PARAMETER_NAMES if smoke else FULL_PARAMETER_NAMES
    fast_factor = SMOKE_FAST_FACTOR if smoke else FULL_FAST_FACTOR
    sweep_n_points = SMOKE_SWEEP_N_POINTS if smoke else FULL_SWEEP_N_POINTS

    missing_pairs: list[tuple[int, int]] = []
    for alpha, y_num in required_pairs:
        pair_missing = missing_paths(
            convergence_paths(
                alpha,
                y_num,
                parameter_names,
                include_cfd_sweep=not smoke,
            )
        )
        if pair_missing:
            missing_pairs.append((alpha, y_num))

    if not missing_pairs:
        print("Convergence cache files already available.")
        return

    print(f"Generating convergence cache for {len(missing_pairs)} alpha/Y pairs...")
    convergence_study = _import_package_module("convergence_study")

    for alpha, y_num in missing_pairs:
        print(f"  - alpha={alpha}, Y={y_num}")
        convergence_study.storing_and_collecting_results(
            alpha=alpha,
            y_num=y_num,
            parameter_names=parameter_names,
            data_types=["CFD", "PIV"],
            is_small_piv=smoke,
            fast_factor=fast_factor,
        )
        convergence_study.storing_PIV_percentage_sweep(
            alpha=alpha,
            y_num=y_num,
            n_points=sweep_n_points,
            data_type="PIV",
            fast_factor=fast_factor,
        )
        if not smoke:
            convergence_study.storing_PIV_percentage_sweep(
                alpha=alpha,
                y_num=y_num,
                n_points=sweep_n_points,
                data_type="CFD",
                fast_factor=fast_factor,
            )

    missing_after: list[Path] = []
    for alpha, y_num in required_pairs:
        missing_after.extend(
            missing_paths(
                convergence_paths(
                    alpha,
                    y_num,
                    parameter_names,
                    include_cfd_sweep=not smoke,
                )
            )
        )
    if missing_after:
        formatted = "\n".join(f"  - {path}" for path in missing_after)
        raise RuntimeError(
            "Could not generate all convergence cache files. Still missing:\n"
            + formatted
        )
    print("Convergence cache files generated.")


def run_plot_scripts(smoke: bool = False) -> None:
    print("Running figure/table scripts...")
    runner = _import_package_module("_plot_all_figures_and_print_all_tables")
    runner.main(run_figures=True, run_tables=True, smoke=smoke)
    print("All requested figure/table scripts completed.")


def main(
    run_plots: bool = True, smoke: bool = False, skip_process: bool = False
) -> None:
    print("Starting dependency-aware processing pipeline...")
    if smoke and not skip_process:
        print(
            "Smoke mode enabled: reduced workload "
            "(n_points=2, y_num_list=[1], reduced convergence sweep, reduced plotting/tables)."
        )
    if skip_process:
        print(
            "Skipping processing stage (--skip-process). "
            "Only running plotting/table scripts."
        )
    else:
        ensure_result_directories()
        validate_source_data(smoke=smoke)

        ensure_processed_cfd(smoke=smoke)
        ensure_spanwise_outlines(smoke=smoke)
        run_processing_pipeline(smoke=smoke)
        ensure_vsm_gamma_cache(smoke=smoke)

    if run_plots:
        run_plot_scripts(smoke=smoke)
    else:
        print("Skipping figure/table execution (--skip-plots).")

    print("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Prepare processed_data caches from raw/source data, then optionally run "
            "all figures (fig04..figE2) and tables (Table04..TableD1)."
        )
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only generate/check dependencies; do not run figure/table scripts.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a reduced-fidelity pipeline for quick error checking.",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip preprocessing/cache generation and only run figure/table scripts.",
    )
    args = parser.parse_args()
    main(
        run_plots=not args.skip_plots,
        smoke=args.smoke,
        skip_process=args.skip_process,
    )
