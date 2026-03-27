from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


if __package__ in (None, ""):
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    BASE_PACKAGE = "kite_piv_analysis"
else:
    BASE_PACKAGE = __package__


FIGURE_MODULES = [
    "fig04_plane_location",
    "fig06_qualitative_comparison_CFD_PIV",
    "fig07_bounds_CFD_PIV_single_row",
    "fig08_gamma_distribution",
    "fig09_spanwise_CFD_comparison_v_and_p_x_10_to_50",
    "figA1_convergence_250im_uvw",
    "figC1_PIV_normal_masked_Y1",
    "figC2_spanwise_CFD_alpha_comparison",
    "figD1_line_interference_PIV",
    "figE1_bounds_CFD_PIV",
    "figE2_convergence_study",
]

TABLE_ENTRYPOINTS: list[tuple[str, str]] = [
    ("Table04_and_Table05_uncertainty_analysis", "generate_uncertainty_report"),
    ("Table06_print_quantitative_results", "main"),
    ("TableD1_stitching_uncertainty_RMS", "main"),
]

SMOKE_FIGURE_MODULES = ["fig04_plane_location"]
SMOKE_TABLE_ENTRYPOINTS: list[tuple[str, str]] = []


def _import_package_module(module_name: str):
    return importlib.import_module(f"{BASE_PACKAGE}.{module_name}")


def _run_entrypoint(module_name: str, entrypoint: str) -> None:
    module = _import_package_module(module_name)
    fn = getattr(module, entrypoint, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"{module_name} has no callable {entrypoint}()")
    print(f"  - {module_name}.{entrypoint}()")
    fn()


def run_all_figures(smoke: bool = False) -> None:
    modules = SMOKE_FIGURE_MODULES if smoke else FIGURE_MODULES
    print("Running figure scripts...")
    for module_name in modules:
        _run_entrypoint(module_name, "main")


def run_all_tables(smoke: bool = False) -> None:
    entrypoints = SMOKE_TABLE_ENTRYPOINTS if smoke else TABLE_ENTRYPOINTS
    if not entrypoints:
        print("Skipping table scripts in smoke mode.")
        return
    print("Running table scripts...")
    for module_name, entrypoint in entrypoints:
        _run_entrypoint(module_name, entrypoint)


def main(
    run_figures: bool = True,
    run_tables: bool = True,
    smoke: bool = False,
) -> None:
    if run_figures:
        run_all_figures(smoke=smoke)
    else:
        print("Skipping figures.")

    if run_tables:
        run_all_tables(smoke=smoke)
    else:
        print("Skipping tables.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run all paper figure scripts (fig04..figE2) and table scripts "
            "(Table04..TableD1)."
        )
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Do not run figure scripts.",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Do not run table scripts.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a minimal subset for quick validation.",
    )
    args = parser.parse_args()
    main(
        run_figures=not args.skip_figures,
        run_tables=not args.skip_tables,
        smoke=args.smoke,
    )
